"""Original-form DNNABF alongside matched module controls on the new data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from beamnas.common import save_json,sha256,setup
from beamnas.strict_dnnabf import StrictDense,unpack
from run_strict_dnnabf import fit,physics_context,validate


def load_models(item,device):
    result={}
    for k,info in item['checkpoints'].items():
        assert sha256(info['path'])==info['sha256']
        saved=torch.load(info['path'],map_location=device,weights_only=False)
        g=saved['genotype']
        model=StrictDense(saved['input_dim'],g['widths'],g['skips']).to(device).eval()
        model.load_state_dict(saved['state_dict'],strict=True)
        result[int(k)]=model
    return result


@torch.no_grad()
def predict(item,angles,mask,device='cuda'):
    models=load_models(item,device)
    result=np.empty((len(angles),12),complex)
    counts=mask.sum(1)-1
    for k,model in models.items():
        chosen=np.flatnonzero(counts==k)
        for ii in np.array_split(chosen,max(1,(len(chosen)+1023)//1024)):
            if not len(ii):continue
            aa=torch.tensor(angles[ii,:k+1],device=device,dtype=torch.float32)
            result[ii]=unpack(model(aa).double()).cpu().numpy()
    del models
    if device=='cuda':torch.cuda.empty_cache()
    return result


def inspect(root,manifest_sha):
    root=Path(root)
    report=json.loads((root/'summary.json').read_text())
    assert report['complete'] and report['data_manifest_sha256']==manifest_sha and not report['test_accessed']
    for p,h in report['source_hashes'].items():
        assert sha256(p)==h and sha256(root/'source_snapshot'/p)==h
    checkpoints={}
    for k,s in report['by_k'].items():
        path=root/('K'+k)/'best.pt'
        assert sha256(path)==s['checkpoint_sha256']
        history=[json.loads(row) for row in (path.parent/'history.jsonl').read_text().splitlines()]
        assert len(history)==report['protocol']['training']['epochs']
        best=min(history,key=lambda row:row['selection_validation']['mse'])
        assert best['epoch']==s['best_epoch']
        checkpoints[k]={'path':str(path),'sha256':sha256(path)}
    return {'kind':'strict_DNNABF_by_K','role':'strict_DNNABF','checkpoint':str(root/'summary.json'),
        'checkpoint_sha256':sha256(root/'summary.json'),'checkpoints':checkpoints,
        'parameters_by_k':{k:s['parameters'] for k,s in report['by_k'].items()},
        'training_summary':report,'genotype':{'widths':report['protocol']['fixed_widths'],'activation':'PReLU','output':'tanh','input':'raw_K_plus_1_AOA'}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--data',default='data/ch3_grid_good_v11')
    p.add_argument('--protocol',default='configs/ch3_strict_baseline_v11.json')
    args=p.parse_args()
    out=Path(args.out);assert not out.exists();out.mkdir(parents=True)
    cfg=json.loads(Path(args.protocol).read_text())
    manifest=json.loads((Path(args.data)/'manifest.json').read_text());assert manifest['complete']
    setup(cfg['seed'])
    sources=[Path(__file__).resolve().relative_to(Path.cwd()),Path(args.protocol),Path('run_strict_dnnabf.py'),*Path('beamnas').glob('*.py')]
    hashes={str(p):sha256(p) for p in sources}
    for source in sources:
        target=out/'source_snapshot'/source;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    loaded={}
    for split in ['train','selection_validation']:
        path=Path(args.data)/(split+'.npz')
        assert sha256(path)==manifest['splits'][split]['sha256']
        with np.load(path) as d:loaded[split]={key:d[key] for key in ['angles','mask','weights']}
    report={'complete':False,'protocol':cfg,'source_hashes':hashes,'data_manifest_sha256':sha256(Path(args.data)/'manifest.json'),
        'by_k':{},'test_accessed':False}
    save_json(out/'summary.json',report)
    start=time.perf_counter()
    for k in range(3,9):
        save_json(out/'status.json',{'state':'running','K':k,'test_accessed':False})
        config=dict(cfg);config['data']={**manifest['config'],'interferers':k}
        data={}
        for split,d in loaded.items():
            chosen=d['mask'].sum(1)-1==k
            data[split]=(torch.tensor(d['angles'][chosen,:k+1],device='cuda'),torch.tensor(d['weights'][chosen],device='cuda'))
        assert len(data['train'][0])==10000 and len(data['selection_validation'][0])==1000
        context=physics_context(data['selection_validation'],config['data'])
        genotype={'family':'raw_aoa_dense','widths':cfg['fixed_widths'],'skips':['none']*4,'activation':'PReLU','output':'tanh'}
        summary=fit(config,data,context,genotype,'K'+str(k),out)
        saved=torch.load(out/('K'+str(k))/'best.pt',map_location='cuda',weights_only=False)
        model=StrictDense(k+1,cfg['fixed_widths']).cuda();model.load_state_dict(saved['state_dict'],strict=True)
        actual=validate(model,data['selection_validation'],context,config['data'])
        assert actual==saved['validation']
        summary.update(validation=actual,checkpoint_reload_verified=True)
        report['by_k'][str(k)]=summary
        save_json(out/'summary.json',report)
        del model,data,context,saved
        torch.cuda.empty_cache()
    assert all(sha256(p)==h for p,h in hashes.items())
    report.update(complete=True,total_training_seconds=time.perf_counter()-start)
    save_json(out/'summary.json',report)
    save_json(out/'status.json',{'state':'complete','test_accessed':False})


if __name__=='__main__':main()
