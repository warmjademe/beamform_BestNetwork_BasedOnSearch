"""Development-only MSE pilot for explicit modules under original array physics."""
import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from beamnas.common import save_json, setup, sha256, unpacked
from beamnas.original_modules import (ModuleNetwork, features, canonical_target,
    physical_weights, covariance, sinr_db)
from beamnas.strict_dnnabf import mvdr, steering


def snapshot(out, entrypoint):
    files = [entrypoint,'configs/original_modules_v4.json']+[str(p) for p in Path('beamnas').glob('*.py')]
    hashes = {}
    for source in files:
        path = Path(source)
        rel = path.resolve().relative_to(Path.cwd())
        hashes[str(rel)] = sha256(path)
        dest = out/'source_snapshot'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    save_json(out/'source_hashes.json',hashes)
    save_json(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'pid':os.getpid(),
        'cpu_threads':torch.get_num_threads(),'tf32':False})
    return hashes


def load_data(root,split,n=None):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    path=root/(split+'.npz')
    assert manifest['complete'] and sha256(path)==manifest['splits'][split]['sha256']
    with np.load(path) as d:
        angles=torch.tensor(d['angles'][:n],device='cuda')
        mask=torch.tensor(d['mask'][:n],device='cuda')
        raw=torch.tensor(d['weights'][:n],device='cuda')
    target=canonical_target(raw,angles)
    # Confirm loss-preserving coordinate conversion independently of the network.
    restored=physical_weights(target,angles)
    torch.testing.assert_close(restored,unpacked(raw),rtol=2e-5,atol=5e-6)
    return features(angles,mask),mask,target,angles,raw


@torch.no_grad()
def context(data):
    angles,mask=data[3].double(),data[1]
    a=steering(angles[:,0])
    w=mvdr(covariance(angles,mask),a)
    ref=sinr_db(w,angles,mask)
    teacher=sinr_db(unpacked(data[4].double()),angles,mask)
    return ref, {'sample_teacher_mean_sinr_gap_db':float((ref-teacher).mean()),
                 'population_mvdr_mean_sinr_db':float(ref.mean())}


@torch.no_grad()
def validate(model,data,reference):
    model.eval()
    predictions=torch.cat([model(data[0][i:i+1024],data[1][i:i+1024]) for i in range(0,len(data[0]),1024)])
    w=physical_weights(predictions.double(),data[3].double())
    gap=reference-sinr_db(w,data[3].double(),data[1])
    assert float(gap.min()) > -1e-7
    counts=data[1].sum(-1)-1
    mse=float(torch.nn.functional.mse_loss(predictions,data[2]))
    physical_mse=float((torch.view_as_real(w-unpacked(data[4].double()))).square().mean())
    assert abs(mse-physical_mse)<max(1e-6,1e-5*mse)
    return {'mse':mse,'mean_sinr_gap_db':float(gap.mean()),'median_sinr_gap_db':float(gap.median()),
        'p95_sinr_gap_db':float(gap.quantile(.95)),
        'by_k':{str(k):float(gap[counts==k].mean()) for k in range(3,9)},
        'max_abs_output_component':float(predictions.abs().max()),
        'max_distortionless_residual':float(abs((w.conj()*steering(data[3].double()[:,0])).sum(-1)-1).max())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--family',choices=['fixed_dnnabf_modules','residual_pilot'],required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    out=Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    config=json.loads(Path('configs/original_modules_v4.json').read_text())
    pc,oc=config['pilot'],config['optimizer']
    if args.smoke:pc.update(train_scenes=96,epochs=2,batch_size=32,residual_width=32,residual_depth=2)
    setup(pc['seed'])
    assert torch.cuda.is_available()
    hashes=snapshot(out,__file__)
    save_json(out/'config.json',{'protocol':config,'arguments':vars(args),'data_manifest_sha256':sha256(Path(args.data)/'manifest.json')})
    try:
        train=load_data(args.data,'train',pc['train_scenes'])
        validation=load_data(args.data,'selection_validation')
        reference,reference_info=context(validation)
        save_json(out/'reference.json',reference_info)
        space={'width':pc['residual_width'],'depth':pc['residual_depth']}
        model=ModuleNetwork(args.family,space).cuda()
        assert not model.architecture_parameters()
        optimizer=torch.optim.AdamW(model.parameters(),lr=oc['lr'],weight_decay=oc['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,pc['epochs'],eta_min=oc['cosine_min_lr'])
        generator=torch.Generator(device='cuda').manual_seed(pc['seed']+3000)
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        best=float('inf')
        steps=0
        for epoch in range(1,pc['epochs']+1):
            model.train()
            total=0.
            for idx in torch.randperm(len(train[0]),device='cuda',generator=generator).split(pc['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.mse_loss(model(train[0][idx],train[1][idx]),train[2][idx])
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),oc['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
                total+=float(loss.detach())*len(idx)
                steps+=1
            metrics=validate(model,validation,reference)
            if metrics['mse']<best:
                best=metrics['mse']
                torch.save({'state_dict':model.state_dict(),'family':args.family,'space':space,'genotype':None,
                    'epoch':epoch,'validation':metrics,'config':config,'optimizer':optimizer.state_dict()},out/'best.pt')
            row={'epoch':epoch,'train_mse':total/len(train[0]),'validation':metrics,'steps':steps,'seconds':time.perf_counter()-start}
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            save_json(out/'status.json',{'state':'training','pid':os.getpid(),**row})
            if epoch%5==0 or args.smoke:print(json.dumps(row),flush=True)
            scheduler.step()
        saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True)
        verified=validate(model,validation,reference)
        assert verified==saved['validation']
        assert all(sha256(s)==h for s,h in hashes.items())
        result={'scope':'development_selection_validation_only','family':args.family,'best_epoch':saved['epoch'],
            'validation':verified,'test_accessed':False,'parameters':sum(p.numel() for p in model.parameters()),
            'epochs':pc['epochs'],'steps':steps,'training_seconds':time.perf_counter()-start,
            'checkpoint_sha256':sha256(out/'best.pt'),'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'loss':'plain_component_MSE_equivalent_to_original_physical_weight_MSE','label_scaling':'none',
            'new_modules':config['module_changes'],'overall_goal_complete':False}
        save_json(out/'summary.json',result)
        save_json(out/'status.json',{'state':'complete','source_hashes_verified':True,'checkpoint_reload_verified':True,'test_accessed':False})
        print(json.dumps(result),flush=True)
    except Exception:
        save_json(out/'status.json',{'state':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
