"""Audit development or frozen holdout for the Chapter 3 angle reconstruction."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from beamnas.common import save_json, sha256, setup
from beamnas.pointing_boundary import nearest_peak_directions
from benchmark_ch3_latency import measure_latency
from evaluate_restored_holdout import predict as predict_modules, summarize, independent_torch_check
from ch3_strict_baseline import inspect as inspect_strict, predict as predict_strict
from freeze_restored_candidates import passes, canonical_keys
from original_array_evaluation import numpy_reference, numpy_metrics, directions, summarize_directions


def predict(item,angles,mask,device='cuda'):
    fn=predict_strict if item.get('kind')=='strict_DNNABF_by_K' else predict_modules
    return fn(item,angles,mask,device)


def inspect_run(path, role, protocol, manifest_sha):
    path = Path(path)
    status = json.loads((path/'status.json').read_text())
    assert status['state'] == 'complete' and status['source_hashes_verified'] and not status['test_accessed']
    cfg = json.loads((path/'config.json').read_text())
    assert cfg['experiment_protocol'] == protocol
    assert cfg['data_manifest_sha256'] == manifest_sha and not cfg['arguments']['smoke']
    source_hashes = json.loads((path/'source_hashes.json').read_text())
    for p,h in source_hashes.items():
        assert sha256(p) == h and sha256(path/'source_snapshot'/p) == h
    summary = json.loads((path/'retrain/summary.json').read_text())
    assert summary['from_scratch'] and not summary['test_accessed']
    assert summary['epochs'] == protocol['training']['epochs']
    history = [json.loads(line) for line in (path/'retrain/history.jsonl').read_text().splitlines()]
    best = min(history, key=lambda row:row['selection_validation']['mean_sinr_gap_db'])
    assert best['epoch'] == summary['best_epoch'] and best['selection_validation'] == summary['validation']
    checkpoint = path/'retrain/best.pt'
    assert sha256(checkpoint) == summary['checkpoint_sha256']
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['validation'] == summary['validation'] and saved['genotype'] is not None
    item = {'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'run':str(path),
        'role':role,'genotype':saved['genotype'],'family':saved['family'],'space':saved['space'],
        'parameters':summary['parameters'],'training_summary':summary,'validation':summary['validation']}
    if role == 'candidate':
        search = json.loads((path/'search/summary.json').read_text())
        assert search['epochs'] == protocol['search']['epochs']
        assert search['architecture_steps'] > 0 and search['architecture_l2_change'] > 0
        assert saved['genotype'] == json.loads((path/'search/genotype.json').read_text())
        assert sha256(path/'search/best.pt') == search['checkpoint_sha256']
        item['search_summary'] = search
    expected = 'population_component_mse' if role == 'mse' else 'restored_physical_objective'
    assert summary['loss'] == cfg['loss'] == expected
    return item


def verify(freeze):
    for p,h in freeze['source_hashes'].items():
        assert sha256(p) == h and sha256(Path(freeze['source_snapshot_root'])/p) == h
    for item in freeze['models'].values():
        assert sha256(item['checkpoint']) == item['checkpoint_sha256']
        for checkpoint in item.get('checkpoints',{}).values():
            assert sha256(checkpoint['path']) == checkpoint['sha256']
    assert sha256(Path(freeze['data'])/'manifest.json') == freeze['data_manifest_sha256']


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--data', default='data/ch3_grid_good_v11')
    p.add_argument('--runs', required=True, help='Root with candidate, fixed, mse, random run directories')
    p.add_argument('--split', choices=['selection_validation','test'], required=True)
    p.add_argument('--development-audit', help='Required before a test; freezes the candidate without test feedback')
    p.add_argument('--strict-run',help='Optional original-form DNNABF control trained before test')
    args = p.parse_args()
    out, data, runs = Path(args.out), Path(args.data), Path(args.runs)
    assert not out.exists()
    manifest = json.loads((data/'manifest.json').read_text())
    assert manifest['complete'] and manifest['group_overlap'] == 0
    protocol = manifest['protocol']
    assert protocol['id'] == 'chapter3_structured_grid_good_only_v11'
    setup(protocol['training']['seed'])
    models = {role:inspect_run(runs/role, role, protocol, sha256(data/'manifest.json'))
              for role in ['candidate','fixed','mse','random']}
    assert models['candidate']['genotype'] == models['mse']['genotype']
    if args.strict_run:
        models['strict_DNNABF']=inspect_strict(args.strict_run,sha256(data/'manifest.json'))
    if args.split == 'test':
        assert args.development_audit
        development = json.loads((Path(args.development_audit)/'report.json').read_text())
        assert development['split'] == 'selection_validation' and development['complete']
        for name,item in models.items():
            assert development['models'][name]['checkpoint_sha256'] == item['checkpoint_sha256']
        # Testing a predeclared candidate does not become selection: record any
        # known validation shortfall rather than concealing it or changing gates.
    out.mkdir(parents=True)
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path('original_array_evaluation.py'),
        Path('evaluate_restored_holdout.py'),Path('freeze_restored_candidates.py'),
        Path('benchmark_restored_latency.py'),Path('benchmark_ch3_latency.py'),
        Path('ch3_strict_baseline.py'),*Path('beamnas').glob('*.py')]
    hashes = {str(p):sha256(p) for p in sources}
    for source in sources:
        dest = out/'source_snapshot'/source
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    freeze = {'models':models,'data':str(data),'data_manifest_sha256':sha256(data/'manifest.json'),
        'source_hashes':hashes,'source_snapshot_root':str(out/'source_snapshot'),
        'time':datetime.now(timezone.utc).isoformat(),'split':args.split,
        'frozen_before_loading_evaluation_split':True,'protocol':protocol}
    save_json(out/'freeze.json',freeze)
    verify(freeze)
    path = data/(args.split+'.npz')
    assert sha256(path) == manifest['splits'][args.split]['sha256']
    with np.load(path) as archive:
        angles, mask = archive['angles'].astype(float), archive['mask']
        raw = archive['weights'].astype(float)
        sample = raw[:,:12]+1j*raw[:,12:]
        stored_oracle = archive['population_weights']
    assert len(canonical_keys(angles,mask)) == len(angles)
    oracle, _, _ = numpy_reference(angles,mask)
    np.testing.assert_allclose(oracle,stored_oracle,rtol=1e-9,atol=1e-9)
    counts = mask.sum(1)-1
    result = {'complete':False,'split':args.split,'scenes':len(angles),'models':models,
        'protocol':protocol,'metrics':{},'test_used_for_training_or_selection':False,
        'data_manifest_sha256':sha256(data/'manifest.json')}
    start = time.perf_counter()
    oracle_peaks = None
    for name in ['population_MVDR','sample_MVDR',*models]:
        save_json(out/'status.json',{'state':'running','method':name,'split':args.split})
        weights = oracle if name=='population_MVDR' else sample if name=='sample_MVDR' else predict(models[name],angles,mask)
        values = numpy_metrics(weights,angles,mask,oracle,sample)
        mismatch = independent_torch_check(weights,angles,mask,values['sinr_db'])
        peaks = nearest_peak_directions(weights,angles[:,0])
        nulls = directions(weights,angles,mask,device='cuda',step_deg=.001)
        if name == 'population_MVDR':
            oracle_peaks = peaks['peak_deg'].copy()
        item = summarize(values,mask)
        item.update(main_nearest_peak_MAE_deg=float(peaks['main_error_deg'].mean()),
            main_nearest_peak_p95_deg=float(np.quantile(peaks['main_error_deg'],.95)),
            main_nearest_peak_MAE_by_k={str(k):float(peaks['main_error_deg'][counts==k].mean()) for k in range(3,9)},
            paired_peak_difference_mean_deg=float(abs(peaks['peak_deg']-oracle_peaks).mean()),
            paired_peak_difference_p95_deg=float(np.quantile(abs(peaks['peak_deg']-oracle_peaks),.95)),
            endpoint_selected_count=int(peaks['endpoint_selected'].sum()),
            no_local_peak_fallback_count=int(peaks['no_local_peak_fallback'].sum()),
            fine_global_and_null=summarize_directions(nulls,mask),
            all_scenes_numpy_verified=True,max_torch_numpy_SINR_difference_db=mismatch)
        item['by_k_direction'] = {str(k):summarize_directions({key:value[counts==k] for key,value in nulls.items()},mask[counts==k]) for k in range(3,9)}
        if name in models:
            ii = np.linspace(0,len(angles)-1,min(64,len(angles)),dtype=int)
            cpu = predict(models[name],angles[ii],mask[ii],device='cpu')
            np.testing.assert_allclose(cpu,weights[ii],rtol=2e-4,atol=2e-5)
            item['cpu_GPU_check_scenes'] = len(ii)
        item['sinr_gates_passed'] = passes(item,protocol['acceptance'])
        oracle_mae = float(abs(oracle_peaks-angles[:,0]).mean())
        item['main_MAE_difference_vs_Oracle_deg'] = abs(item['main_nearest_peak_MAE_deg']-oracle_mae)
        item['all_proximity_gates_passed'] = item['sinr_gates_passed'] and item['main_MAE_difference_vs_Oracle_deg'] <= protocol['acceptance']['main_MAE_difference_vs_population_MVDR_max_deg']
        destination = out/(name+'.npz')
        np.savez_compressed(destination,weights=weights,**values,**peaks,
            **{'fine_'+key:value for key,value in nulls.items()})
        item['predictions_sha256'] = sha256(destination)
        result['metrics'][name] = item
        save_json(out/'report.json',result)
        print(json.dumps({'method':name,'SINR':item['mean_sinr_db'],'gap':item['mean_sinr_gap_db'],
            'main_MAE':item['main_nearest_peak_MAE_deg'],'null_MAE':item['fine_global_and_null']['null_mean_abs_error_deg_assigned_only'],
            'gates':item['all_proximity_gates_passed']}),flush=True)
    if args.split == 'test':
        result['latency'] = measure_latency(models,angles,mask,protocol['latency'],out)
    verify(freeze)
    result.update(complete=True,seconds=time.perf_counter()-start,hashes_verified=True,
        candidate_meets_proximity=result['metrics']['candidate']['all_proximity_gates_passed'])
    save_json(out/'report.json',result)
    save_json(out/'status.json',{'state':'complete','candidate_meets_proximity':result['candidate_meets_proximity'],
        'split':args.split,'test_used_for_training_or_selection':False})


if __name__ == '__main__':
    main()
