"""Verify published artifacts and reproduce frozen test predictions, without training."""
import argparse
import json
from pathlib import Path

import numpy as np

from beamnas.common import save_json,sha256
from ch3_strict_baseline import predict as predict_strict
from evaluate_ch3_reconstruction import inspect_run
from evaluate_restored_holdout import predict
from original_array_evaluation import numpy_reference,numpy_metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--out',default='results/release/VALIDATION.json')
    args=parser.parse_args()
    root=Path('data/ch3_grid_good_v11')
    manifest=json.loads((root/'manifest.json').read_text())
    groups=set();splits={}
    for split,info in manifest['splits'].items():
        p=root/(split+'.npz')
        assert sha256(p)==info['sha256'],p
        with np.load(p) as data:
            a,m=data['angles'],data['mask'];g=set(a[:,0].tolist())
            assert not groups.intersection(g);groups.update(g)
            assert len(a)==info['n'] and a.shape==m.shape
            counts=m.sum(1)-1
            assert all(int(np.sum(counts==int(k)))==v for k,v in info['counts_by_k'].items())
            splits[split]={'samples':len(a),'SHA256_verified':True}
    for audit in manifest['audit_files']:
        assert sha256(audit['path'])==audit['sha256'],audit['path']
    prior=json.loads(Path('results/ch3_grid_v11/FINAL.json').read_text())
    inspected={}
    for role in ['candidate','fixed','mse','random']:
        inspected[role]=inspect_run(Path('runs/ch3_grid_v11')/role,role,manifest['protocol'],sha256(root/'manifest.json'))
    for role,item in prior['models'].items():
        assert sha256(item['checkpoint'])==item['checkpoint_sha256']
        for value in item.get('checkpoints',{}).values():assert sha256(value['path'])==value['sha256']
    with np.load(root/'test.npz') as data:
        angles,mask=data['angles'].astype(float),data['mask']
        oracle=data['population_weights']
    independent,_,_=numpy_reference(angles,mask)
    np.testing.assert_allclose(independent,oracle,rtol=1e-9,atol=1e-9)
    models={}
    for role,item in prior['models'].items():
        function=predict_strict if role=='strict_DNNABF' else predict
        actual=function(item,angles,mask,device=args.device)
        p=Path('runs/ch3_grid_v11/test')/(role+'.npz')
        assert sha256(p)==prior['metrics'][role]['predictions_sha256']
        with np.load(p) as saved:
            max_difference=float(np.max(abs(actual-saved['weights'])))
            np.testing.assert_allclose(actual,saved['weights'],rtol=2e-4,atol=2e-5)
        physical=numpy_metrics(actual,angles,mask,oracle)
        mean=float(physical['sinr_db'].mean())
        assert abs(mean-prior['metrics'][role]['mean_sinr_db'])<.001
        models[role]={'test_scenes':len(actual),'max_abs_weight_difference':max_difference,'mean_sinr_db':mean}
        print(role,json.dumps(models[role]),flush=True)
    published=Path('docs/RELEASE_MANIFEST.json')
    checked_files=0
    if published.exists():
        for entry in json.loads(published.read_text())['files']:
            assert sha256(entry['path'])==entry['sha256'],entry['path']
            checked_files+=1
    report={'pass':True,'device':args.device,'splits':splits,'audit_batches':len(manifest['audit_files']),
            'desired_angle_group_overlap':0,'model_checkpoints_verified':True,'models':models,
            'publication_files_hashed':checked_files,'scope':'Artifact integrity and frozen prediction replay; no retraining or new selection'}
    save_json(args.out,report)
    print('RELEASE_VALIDATION_PASS',flush=True)


if __name__=='__main__':main()
