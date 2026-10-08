"""Verify actual GPU MVDR work, cache independence, and synchronized timings."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import benchmark_restored_latency as bench
from beamnas.common import setup,save_json,sha256
from beamnas.strict_dnnabf import steering
from original_array_evaluation import numpy_reference


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    root=Path('runs/ch3_grid_v11')
    frozen=json.loads((root/'test/freeze.json').read_text())
    setup(117);bench.require_idle_gpu()
    d=np.load('data/ch3_grid_good_v11/test.npz')
    counts=d['mask'].sum(1)-1
    indices=np.array([np.flatnonzero(counts==k)[0] for k in range(3,9)])
    angles=d['angles'][indices].astype(float);mask=d['mask'][indices]
    inputs=bench.make_inputs(angles,mask,'cuda')
    models=bench.load_models({'candidate':frozen['models']['candidate']},'cuda')
    traces=[]
    original_covariance,original_solve=bench.covariance,torch.linalg.solve
    def checked_covariance(a,m):
        result=original_covariance(a,m)
        traces.append({'operation':'construct_covariance','AOA_shape':list(a.shape),'R_shape':list(result.shape),
            'device':str(result.device),'dtype':str(result.dtype)})
        return result
    def checked_solve(r,a,*v,**kw):
        traces.append({'operation':'torch.linalg.solve','R_shape':list(r.shape),'rhs_shape':list(a.shape),
            'device':str(r.device),'dtype':str(r.dtype)})
        return original_solve(r,a,*v,**kw)
    verified=[]
    with patch.object(bench,'covariance',checked_covariance),patch.object(torch.linalg,'solve',checked_solve):
        for aa,mm,ad,r,x in inputs:
            # Poison every precomputed object. The AOA path must recompute what
            # it uses from the angles/mask rather than read these cached values.
            poisoned=(aa,mm,torch.full_like(ad,float('nan')),torch.full_like(r,float('nan')),torch.full_like(x,float('nan')))
            w=bench.call_method('population_AOA_to_weights',poisoned,models)
            reference,_,_=numpy_reference(aa.cpu().numpy(),mm.cpu().numpy())
            np.testing.assert_allclose(w.cpu().numpy(),reference,rtol=1e-8,atol=1e-9)
            verified.append(float(abs(w.cpu().numpy()-reference).max()))
    assert len(traces)==12
    aa,mm,ad,r,x=inputs[0]
    changed=aa.clone();changed[:,0]+=1
    changed_w=bench.call_method('population_AOA_to_weights',(changed,mm,ad,r,x),models)
    expected,_,_=numpy_reference(changed.cpu().numpy(),mm.cpu().numpy())
    np.testing.assert_allclose(changed_w.cpu().numpy(),expected,rtol=1e-8,atol=1e-9)
    original_w=bench.call_method('population_AOA_to_weights',inputs[0],models)
    changed_distance=float(abs(changed_w-original_w).max());assert changed_distance>1e-6
    def explicit_inverse(inp):
        aa,mm=inp[:2]
        rr=bench.covariance(aa,mm);a=steering(aa[:,0])
        v=(torch.linalg.inv(rr)@a[...,None]).squeeze(-1)
        return v/(a.conj()*v).sum(-1,keepdim=True)
    for inp in inputs:
        torch.testing.assert_close(explicit_inverse(inp),bench.call_method('population_AOA_to_weights',inp,models),rtol=1e-8,atol=1e-9)
    methods={
        'NN_AOA_to_weights':lambda inp:bench.call_method('candidate',inp,models),
        'MVDR_AOA_covariance_solve':lambda inp:bench.call_method('population_AOA_to_weights',inp,models),
        'MVDR_AOA_covariance_explicit_inverse':explicit_inverse,
        'sample_MVDR_1024_snapshots_to_weights':lambda inp:bench.call_method('sample_snapshots_given',inp,models)}
    for fn in methods.values():
        for _ in range(50):fn(inputs[0])
    rows=[]
    jobs=[(name,i,j) for name in methods for i in range(6) for j in range(30)]
    np.random.default_rng(2026100851).shuffle(jobs)
    for name,i,j in jobs:
        torch.cuda.synchronize()
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter_ns();begin.record()
        result=methods[name](inputs[i]);end.record();end.synchronize()
        elapsed=(time.perf_counter_ns()-wall)/1e6
        assert result.is_cuda and result.shape==(1,12)
        rows.append({'method':name,'scene':int(indices[i]),'repeat':j,'wall_ms':elapsed,'CUDA_event_ms':begin.elapsed_time(end)})
    # A second scope starts with a CPU-resident AOA and ends with CPU weights.
    # No precomputed steering/covariance is passed to either measured function.
    host_rows=[]
    host=[(torch.tensor(a[None],dtype=torch.float64),torch.tensor(m[None])) for a,m in zip(angles,mask)]
    for name in ['NN_AOA_to_weights','MVDR_AOA_covariance_solve']:
        for repeat in range(30):
            for i,(aa,mm) in enumerate(host):
                torch.cuda.synchronize();start=time.perf_counter_ns()
                at,mt=aa.cuda(),mm.cuda()
                weights=methods[name]((at,mt,None,None,None)).cpu()
                elapsed=(time.perf_counter_ns()-start)/1e6
                assert weights.shape==(1,12)
                host_rows.append({'method':name,'scene':int(indices[i]),'repeat':repeat,'wall_ms':elapsed})
    bench.require_idle_gpu()
    summary={}
    for name in methods:
        rr=[row for row in rows if row['method']==name]
        summary[name]={key:{'mean':float(np.mean([r[key] for r in rr])),
            'median':float(np.median([r[key] for r in rr])),'p95':float(np.quantile([r[key] for r in rr],.95))} for key in ['wall_ms','CUDA_event_ms']}
    host_summary={name:float(np.mean([r['wall_ms'] for r in host_rows if r['method']==name])) for name in ['NN_AOA_to_weights','MVDR_AOA_covariance_solve']}
    result={'scope':'Post-test timing implementation audit; no retraining or model selection',
        'GPU':torch.cuda.get_device_name(),'sources':{s:sha256(s) for s in [__file__,'benchmark_restored_latency.py','beamnas/original_modules.py','beamnas/strict_dnnabf.py']},
        'trace':traces,'poisoned_precomputed_objects_ignored':True,'max_weight_difference_vs_independent_NumPy':max(verified),
        'changed_AOA_changed_weights':changed_distance,'explicit_inverse_matches_solve':True,
        'ready_GPU_summary':summary,'CPU_to_GPU_to_CPU_mean_ms':host_summary,'ready_GPU_observations':rows,
        'host_roundtrip_observations':host_rows,'covariance_matrix_size':[12,12],
        'snapshots_not_matrix_dimension':1024,'calls_per_method':180,'ready_GPU_timing_note':'Events recorded around each call, end-event synchronization included in wall time; includes CPU launch overhead and GPU work.',
        'AOA_path_covariance':'analytic interference-plus-noise population covariance, constructed anew in every call',
        'sample_path_covariance':'X X^H /1024 constructed from given 12x1024 snapshots inside every call',
        'simulation_waveform_generation_in_timing':False,'neural_checkpoint_sha256':frozen['models']['candidate']['checkpoint_sha256']}
    save_json(out/'report.json',result)
    print(json.dumps({k:result[k] for k in ['poisoned_precomputed_objects_ignored','max_weight_difference_vs_independent_NumPy','changed_AOA_changed_weights','explicit_inverse_matches_solve','ready_GPU_summary','CPU_to_GPU_to_CPU_mean_ms']}))


if __name__=='__main__':main()
