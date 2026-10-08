"""Prepared original-array evaluation primitives, independent of model selection.

Directions use fixed weights, not a Capon spectrum with recomputed weights.
The default grid covers the entire declared domain at 0.001 degree spacing.
No training or test-data generation occurs in this module.
"""
import numpy as np
import torch

from beamnas.strict_dnnabf import steering


def numpy_reference(angles, mask):
    angles=np.asarray(angles,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    a=np.exp(1j*np.pi*np.sin(np.deg2rad(angles))[...,None]*np.arange(12))
    ai=a[:,1:]*mask[:,1:,None]
    covariance=1000*np.einsum('bsm,bsn->bmn',ai,ai.conj())+np.eye(12)
    v=np.linalg.solve(covariance,a[:,0,:,None])[...,0]
    weights=v/np.einsum('bm,bm->b',a[:,0].conj(),v)[:,None]
    return weights,covariance,a


def numpy_metrics(weights, angles, mask, oracle=None, sample=None):
    """Recompute per-scene powers in complex128; component and complex MSE differ."""
    w=np.asarray(weights,dtype=np.complex128)
    angles=np.asarray(angles,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if oracle is None:
        oracle,_,a=numpy_reference(angles,mask)
    else:
        a=np.exp(1j*np.pi*np.sin(np.deg2rad(angles))[...,None]*np.arange(12))
    def powers(z):
        response=np.einsum('bm,bsm->bs',z.conj(),a)
        signal=10*abs(response[:,0])**2
        interference=1000*(abs(response[:,1:])**2*np.asarray(mask)[:,1:]).sum(-1)
        noise=(abs(z)**2).sum(-1)
        sinr=10*np.log10(np.maximum(signal/np.maximum(interference+noise,1e-30),1e-30))
        return sinr,signal,interference,noise,response[:,0]
    sinr,signal,interference,noise,response=powers(w)
    bound=powers(np.asarray(oracle,dtype=np.complex128))[0]
    gap=bound-sinr
    assert np.isfinite(gap).all() and gap.min()>-1e-7
    error=np.sum(abs(w-oracle)**2,axis=-1)
    out={'sinr_db':sinr,'gap_db':gap,'desired_power':signal,
         'interference_power':interference,'noise_power':noise,
         'distortionless_residual':abs(response-1),
         'component_mse_vs_population':error/24,
         'complex_mse_vs_population':error/12,
         'nmse_vs_population':error/np.maximum(np.sum(abs(oracle)**2,axis=-1),1e-30)}
    if sample is not None:
        error=np.sum(abs(w-np.asarray(sample,dtype=np.complex128))**2,axis=-1)
        out['component_mse_vs_sample']=error/24
        out['complex_mse_vs_sample']=error/12
    return out


def minimum_distance_assignment(targets, minima):
    """Exact rectangular one-to-one angular assignment, without SciPy.

    Sorted one-dimensional absolute-distance costs admit a noncrossing optimum.
    Match every member of the smaller set; retain unmatched interference targets.
    """
    targets=np.asarray(targets,dtype=np.float64)
    minima=np.asarray(minima,dtype=np.float64)
    ti=np.argsort(targets,kind='stable')
    mi=np.argsort(minima,kind='stable')
    reverse=len(ti)>len(mi)
    small,large=(minima[mi],targets[ti]) if reverse else (targets[ti],minima[mi])
    m,n=len(small),len(large)
    if not m:return np.empty(0,dtype=int),np.empty(0,dtype=int)
    dp=np.full((m+1,n+1),np.inf)
    dp[0]=0.
    take=np.zeros((m+1,n+1),dtype=bool)
    for i in range(1,m+1):
        for j in range(i,n+1):
            chosen=dp[i-1,j-1]+abs(small[i-1]-large[j-1])
            skipped=dp[i,j-1]
            take[i,j]=chosen<=skipped
            dp[i,j]=chosen if take[i,j] else skipped
    first,second=[],[]
    i,j=m,n
    while i:
        assert j>0
        if take[i,j]:
            first.append(i-1);second.append(j-1);i-=1
        j-=1
    first,second=np.asarray(first[::-1]),np.asarray(second[::-1])
    return (ti[second],mi[first]) if reverse else (ti[first],mi[second])


@torch.no_grad()
def directions(weights, angles, mask, device='cpu', step_deg=.001, batch_size=128):
    """Full-grid fixed-pattern maxima and globally assigned interior minima.

    The two endfire directions have identical ULA steering. If the discrete
    global maximum is an endpoint, use -90 degrees as the fixed tie rule and
    report the ambiguity. Do not choose an endpoint using the desired AOA.
    """
    assert step_deg>0 and abs(180/step_deg-round(180/step_deg))<1e-7
    wn=np.asarray(weights,dtype=np.complex128)
    an=np.asarray(angles,dtype=np.float64)
    mn=np.asarray(mask,dtype=bool)
    assert wn.shape==(len(an),12) and an.shape==mn.shape
    grid=torch.linspace(-90,90,round(180/step_deg)+1,device=device,dtype=torch.float64)
    ag=steering(grid)
    result={'peak_deg':np.empty(len(wn)),
            'main_error_deg':np.empty(len(wn)),
            'null_deg':np.full(mn[:,1:].shape,np.nan),
            'null_error_deg':np.full(mn[:,1:].shape,np.nan),
            'null_depth_db_relative_peak':np.full(mn[:,1:].shape,np.nan),
            'interior_minima_count':np.zeros(len(wn),dtype=int),
            'endfire_global_ambiguity':np.zeros(len(wn),dtype=bool),
            'flat_pattern':np.zeros(len(wn),dtype=bool)}
    for start in range(0,len(wn),batch_size):
        w=torch.as_tensor(wn[start:start+batch_size],device=device)
        power=abs(w.conj()@ag.T).square()
        index=power.argmax(-1)
        maximum=power.gather(1,index[:,None]).squeeze(1)
        endpoint=(index==0)|(index==len(grid)-1)
        index=torch.where(endpoint,torch.zeros_like(index),index)
        peak=grid[index].cpu().numpy()
        result['peak_deg'][start:start+len(w)]=peak
        result['main_error_deg'][start:start+len(w)]=abs(peak-an[start:start+len(w),0])
        result['endfire_global_ambiguity'][start:start+len(w)]=endpoint.cpu().numpy()
        flat=power.amax(-1)-power.amin(-1)<=maximum*1e-12
        result['flat_pattern'][start:start+len(w)]=flat.cpu().numpy()
        # Choose the left point of an exactly tied two-point minimum.
        interior=(power[:,1:-1]<power[:,:-2])&(power[:,1:-1]<=power[:,2:])&~flat[:,None]
        rows,cols=interior.nonzero(as_tuple=True)
        # Reject roundoff-induced extrema near endfire using the analytic
        # derivative with respect to u=sin(theta), not differences of powers.
        n=torch.arange(12,device=device,dtype=torch.float64)
        def derivative(indices):
            term=w[rows].conj()*ag[indices]
            amplitude=term.sum(-1)
            slope=(term*(1j*np.pi*n)).sum(-1)
            return 2*(amplitude.conj()*slope).real
        verified=(derivative(cols)<=0)&(derivative(cols+2)>=0)
        rows,cols=rows[verified],cols[verified]
        minima=grid[cols+1].cpu().numpy()
        depths=(10*torch.log10((power[rows,cols+1]/maximum[rows]).clamp_min(1e-30))).cpu().numpy()
        rows=rows.cpu().numpy()
        for local in range(len(w)):
            i=start+local
            valid=np.flatnonzero(mn[i,1:])
            found=rows==local
            possible=minima[found]
            result['interior_minima_count'][i]=len(possible)
            targets,selected=minimum_distance_assignment(an[i,valid+1],possible)
            target_columns=valid[targets]
            result['null_deg'][i,target_columns]=possible[selected]
            result['null_error_deg'][i,target_columns]=abs(an[i,target_columns+1]-possible[selected])
            result['null_depth_db_relative_peak'][i,target_columns]=depths[found][selected]
    return result


def summarize_directions(measured, mask, oracle_peaks=None):
    valid=np.asarray(mask,dtype=bool)[:,1:]
    errors=measured['null_error_deg'][valid]
    assigned=np.isfinite(errors)
    out={'main_mean_abs_error_deg':float(measured['main_error_deg'].mean()),
         'main_p95_abs_error_deg':float(np.quantile(measured['main_error_deg'],.95)),
         'null_mean_abs_error_deg_assigned_only':float(errors[assigned].mean()) if assigned.any() else None,
         'null_total_count':int(valid.sum()),'null_unassigned_count':int((~assigned).sum()),
         'endfire_global_ambiguity_count':int(measured['endfire_global_ambiguity'].sum()),
         'flat_pattern_count':int(measured['flat_pattern'].sum())}
    if oracle_peaks is not None:
        out['mean_abs_peak_difference_vs_MVDR_deg']=float(abs(measured['peak_deg']-oracle_peaks).mean())
    return out
