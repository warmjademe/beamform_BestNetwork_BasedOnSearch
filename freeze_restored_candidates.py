"""Freeze a completed candidate and its matched controls before any new test.

This script inventories old scenes but never generates or scores a new scene.
Incomplete candidates are rejected before inventory. The sole retained model's
known validation shortfall is disclosed before its one independent test.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from beamnas.common import save_json, sha256


def passes(metrics, thresholds):
    values = [metrics['mean_sinr_gap_db'],metrics['p95_sinr_gap_db'],*metrics['by_k'].values()]
    return bool(np.isfinite(values).all() and min(values) >= -1e-7
        and set(metrics['by_k']) == {str(k) for k in range(3,9)}
        and metrics['mean_sinr_gap_db'] <= thresholds['mean_gap_db_max']
        and metrics['p95_sinr_gap_db'] <= thresholds['p95_gap_db_max']
        and all(value <= thresholds['each_k_mean_gap_db_max'] for value in metrics['by_k'].values()))


def canonical_keys(angles, mask):
    assert angles.ndim == 2 and mask.shape == angles.shape
    assert bool(mask[:,0].all()) and np.isfinite(angles[mask]).all()
    return {(float(row[0]),*sorted(float(value) for value in row[1:][valid[1:]]))
            for row,valid in zip(angles,mask)}


def explicit_scene_keys(scenes):
    keys = set()
    for scene in scenes:
        angles = np.asarray([scene],dtype=float)
        keys.update(canonical_keys(angles,np.ones_like(angles,dtype=bool)))
    return keys


def inventory_old_scenes(roots):
    keys, inventory = set(), []
    for path in sorted({p for root in roots for p in Path(root).rglob('*.npz')}):
        with np.load(path,allow_pickle=False) as data:
            if 'angles' not in data.files:
                continue
            angles = data['angles']
            assert angles.ndim == 2 and 2 <= angles.shape[1] <= 9,(str(path),angles.shape)
            mask = data['mask'].astype(bool) if 'mask' in data.files else angles != 999
            before = len(keys)
            keys.update(canonical_keys(angles,mask))
            inventory.append({'path':str(path),'sha256':sha256(path),'scenes':len(angles),
                              'new_unique_keys':len(keys)-before,'columns':angles.shape[1]})
    assert inventory and keys
    return keys, inventory


def inspect_run(path, audit_path, role, thresholds):
    path, audit_path = Path(path), Path(audit_path)
    status = json.loads((path/'status.json').read_text())
    config = json.loads((path/'config.json').read_text())
    extended = config['arguments']['task'] == 'retrain_selected'
    expected = 'search_and_retraining_complete' if role == 'candidate' and not extended else 'control_retraining_complete'
    assert status['state'] == expected,(str(path),status)
    assert status['source_hashes_verified'] and not status['test_accessed']
    recorded_sources = json.loads((path/'source_hashes.json').read_text())
    for source,digest in recorded_sources.items():
        assert sha256(path/'source_snapshot'/source) == digest
        if source.startswith('beamnas/'):
            assert sha256(source) == digest,'Inference-source drift: '+source
    assert config['protocol_v9']['id'] == 'broad_DARTS_restored_physical_objective_v9'
    summary = json.loads((path/'retrain/summary.json').read_text())
    assert summary['from_scratch'] and not summary['test_accessed']
    scope = json.loads(Path('configs/restored_single_candidate_v9.json').read_text())
    assert summary['epochs'] == config['arguments']['retrain_epochs'] == scope['control_budgets'][role]
    history = [json.loads(line) for line in (path/'retrain/history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history] == list(range(1,summary['epochs']+1))
    best_row = min(history,key=lambda row:row['selection_validation']['mean_sinr_gap_db'])
    assert best_row['epoch'] == summary['best_epoch']
    assert best_row['selection_validation'] == summary['validation']
    assert history[-1]['weight_steps'] == summary['weight_steps']
    assert all(row['architecture_steps'] == 0 for row in history)
    checkpoint = path/'retrain/best.pt'
    checkpoint_digest = sha256(checkpoint)
    assert checkpoint_digest == summary['checkpoint_sha256']
    audit = json.loads((audit_path/'summary.json').read_text())
    assert audit['all_scenes_numpy_verified'] and audit['online_solve_inv_eigh_disabled']
    assert not audit['test_accessed'] and audit['checkpoint_sha256'] == checkpoint_digest
    assert audit['metrics']['scenes'] == 12000
    assert sha256(audit_path/'predictions.npz') == audit['prediction_sha256']
    assert abs(audit['metrics']['mean_sinr_gap_db']-summary['validation']['mean_sinr_gap_db']) < .001
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['validation'] == summary['validation']
    if role == 'candidate':
        assert config['arguments']['task'] == 'retrain_selected'
        assert str(path) == scope['final_candidate']
        disclosure = json.loads(Path('configs/restored_holdout_v9.json').read_text())['pretest_validation_disclosure']
        assert summary['validation']['mean_sinr_gap_db'] == disclosure['mean_SINR_gap_db']
        assert summary['validation']['by_k']['8'] == disclosure['K8_gap_db']
        assert passes(summary['validation'],thresholds) == disclosure['all_original_validation_gates_passed']
        search_path = Path(config['arguments']['reference_search'])
        original_config = json.loads((search_path/'config.json').read_text())
        assert sha256(search_path/'config.json') == config['reference_config_sha256']
        assert original_config['arguments']['task'] == 'search'
        assert json.loads((search_path/'status.json').read_text())['source_hashes_verified']
        assert json.loads((search_path/'status.json').read_text())['state'] == 'search_and_retraining_complete'
        search = json.loads((search_path/'search/summary.json').read_text())
        assert search['architecture_steps'] > 0 and search['architecture_l2_change'] > 0
        assert search['epochs'] == config['arguments']['search_epochs'] == 60
        assert saved['genotype'] == json.loads((search_path/'search/genotype.json').read_text())
        assert sha256(search_path/'search/best.pt') == search['checkpoint_sha256']
        for source,source_digest in json.loads((search_path/'source_hashes.json').read_text()).items():
            assert sha256(search_path/'source_snapshot'/source) == source_digest
    else:
        expected_task = {'fixed':'fixed_dnnabf_modules','mse':'matched_mse','random':'random'}[role]
        assert config['arguments']['task'] == expected_task
    loss = 'population_component_mse' if role == 'mse' else 'restored_physical_objective'
    assert config['loss'] == summary['loss'] == loss
    assert sha256(checkpoint) == checkpoint_digest
    return {'role':role,'run':str(path),'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_digest,
        'config_sha256':sha256(path/'config.json'),'audit_sha256':sha256(audit_path/'summary.json'),
        'data_manifest_sha256':config['data_manifest_sha256'],'genotype':saved['genotype'],
        'family':saved['family'],'space':saved['space'],'parameters':summary['parameters'],
        'validation':summary['validation'],'training_summary':summary,'config':config}


def review_candidate_selection(selected, thresholds):
    reviewed = []
    for config_path in sorted(Path('runs').glob('*/config.json')):
        config = json.loads(config_path.read_text())
        if not isinstance(config,dict):
            continue
        if config.get('protocol_v9',{}).get('id') != 'broad_DARTS_restored_physical_objective_v9':
            continue
        if config['arguments']['task'] not in ('search','retrain_selected') or config['arguments']['smoke']:
            continue
        run = config_path.parent
        status = json.loads((run/'status.json').read_text())
        if status['state'] in ('failed','cancelled_by_user_scope_change'):
            reviewed.append({'run':str(run),'passes':False,'terminal_state':status['state'],'reason':status.get('reason')})
            continue
        expected = 'control_retraining_complete' if config['arguments']['task'] == 'retrain_selected' else 'search_and_retraining_complete'
        assert status['state'] == expected,('Search not terminal',str(run),status['state'])
        summary = json.loads((run/'retrain/summary.json').read_text())
        audit = json.loads(Path(str(run)+'_audit','summary.json').read_text())
        assert audit['checkpoint_sha256'] == summary['checkpoint_sha256']
        assert audit['all_scenes_numpy_verified'] and not audit['test_accessed']
        reviewed.append({'run':str(run),'validation':summary['validation'],
                         'passes':passes(summary['validation'],thresholds)})
    eligible = [row for row in reviewed if 'validation' in row]
    assert eligible,'No completed candidate is available'
    best = min(eligible,key=lambda row:(row['validation']['mean_sinr_gap_db'],row['run']))
    assert Path(best['run']).resolve() == Path(selected['run']).resolve(),'Selected candidate is not validation-best'
    return reviewed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--fixed',default='runs/restored_v9_fixed_s117')
    parser.add_argument('--mse',required=True)
    parser.add_argument('--random',required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args()
    protocol_path = Path('configs/restored_holdout_v9.json')
    protocol = json.loads(protocol_path.read_text())
    out = Path(args.out)
    assert not out.exists()
    models = {}
    for role in ('candidate','fixed','mse','random'):
        run = getattr(args,role)
        models[role] = inspect_run(run,run+'_audit',role,protocol['acceptance'])
    candidate = models['candidate']
    reviewed = review_candidate_selection(candidate,protocol['acceptance'])
    assert models['mse']['genotype'] == candidate['genotype']
    for role,item in models.items():
        assert item['data_manifest_sha256'] == candidate['data_manifest_sha256']
        assert item['config']['arguments']['seed'] == candidate['config']['arguments']['seed']
        assert item['config']['arguments']['batch_size'] == candidate['config']['arguments']['batch_size']
        assert item['config']['protocol']['optimizer'] == candidate['config']['protocol']['optimizer']
        for key,value in protocol['physics'].items():
            if key != 'noise_variance':
                assert item['config']['protocol']['data'][key] == value
        assert protocol['physics']['noise_variance'] == 1
        if role in ('mse','random'):
            assert item['config']['reference_config_sha256'] == candidate['config']['reference_config_sha256']
            assert item['space'] == candidate['space']
    # Snapshot an inventory only after every completed-model gate has passed.
    keys, inventory = inventory_old_scenes(['data','runs'])
    keys.update(explicit_scene_keys(protocol['known_diagnostic_scenes']))
    sources = [Path(__file__).resolve().relative_to(Path.cwd()),protocol_path,Path('configs/restored_single_candidate_v9.json'),Path('original_array_evaluation.py'),
               Path('evaluate_restored_holdout.py'),Path('evaluate_restored_details.py'),
               Path('benchmark_restored_latency.py'),Path('predict_restored.py'),
               Path('summarize_restored_final.py'),*Path('beamnas').glob('*.py')]
    hashes = {str(path):sha256(path) for path in sources}
    assert all(not path.is_absolute() and '..' not in path.parts for path in sources)
    out.mkdir(parents=True)
    for path in sources:
        dest = out/'source_snapshot'/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
        assert sha256(dest) == hashes[str(path)]
    freeze = {'time_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,
        'protocol_sha256':sha256(protocol_path),'models':models,'source_hashes':hashes,
        'source_snapshot_root':str(out/'source_snapshot'),
        'candidate_selection_review':reviewed,
        'candidate_meets_all_original_validation_gates':passes(candidate['validation'],protocol['acceptance']),
        'exclusion_files':inventory,'excluded_unique_scenes':len(keys),'test_generated':False}
    save_json(out/'freeze.json',freeze)
    save_json(out/'status.json',{'state':'frozen_before_test','test_generated':False})
    print(json.dumps({'state':'frozen_before_test','excluded_unique_scenes':len(keys),
                      'models':{role:item['checkpoint_sha256'] for role,item in models.items()}}))


if __name__ == '__main__':
    main()
