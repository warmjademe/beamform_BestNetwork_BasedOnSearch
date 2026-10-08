"""One-shot evaluation of already frozen original-physics DARTS candidates.

All scenes and all frozen controls are retained. No training or model selection
occurs here. Sources and checkpoints must match the pre-test freeze exactly.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

from beamnas.common import save_json, setup, sha256
from beamnas.data import sample_angles
from beamnas.original_modules import ModuleNetwork, features, physical_weights, covariance, sinr_db
from beamnas.pointing import nearest_peak_directions
from beamnas.strict_dnnabf import steering, mvdr
from freeze_restored_candidates import canonical_keys, explicit_scene_keys, passes
from original_array_evaluation import numpy_reference, numpy_metrics, directions, summarize_directions


def verify_freeze(freeze):
    assert all(sha256(path)==value for path,value in freeze['source_hashes'].items())
    assert all(sha256(Path(freeze['source_snapshot_root'])/path)==value
               for path,value in freeze['source_hashes'].items())
    assert all(sha256(item['checkpoint'])==item['checkpoint_sha256'] for item in freeze['models'].values())
    assert all(sha256(item['path'])==item['sha256'] for item in freeze['exclusion_files'])


@torch.no_grad()
def sample_reference(angles, mask, seed, snapshots=1024):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    weights, checks = [], []
    chosen = set(np.linspace(0,len(angles)-1,min(64,len(angles)),dtype=int).tolist())
    for begin in range(0,len(angles),128):
        aa = torch.tensor(angles[begin:begin+128],device='cuda',dtype=torch.float64)
        mm = torch.tensor(mask[begin:begin+128],device='cuda')
        a = steering(aa)
        def cn(shape):
            raw = torch.randn(*shape,2,device='cuda',dtype=torch.float64,generator=generator)
            return torch.view_as_complex(raw)/math.sqrt(2)
        x = a[:,1:].transpose(1,2) @ (cn((len(a),8,snapshots))*mm[:,1:,None]*math.sqrt(1000))
        x += cn((len(a),12,snapshots))
        w = mvdr(x@x.mH/snapshots,a[:,0])
        weights.append(w.cpu().numpy())
        for i in range(len(a)):
            if begin+i not in chosen:
                continue
            xn, an = x[i].cpu().numpy(), a[i,0].cpu().numpy()
            r = xn@xn.conj().T/snapshots
            v = np.linalg.solve(r,an)
            wn = v/(an.conj()@v)
            np.testing.assert_allclose(wn,w[i].cpu().numpy(),rtol=1e-8,atol=1e-8)
            checks.append((begin+i,xn,an,wn))
    return np.concatenate(weights), checks


@torch.no_grad()
def predict(item, angles, mask, device='cuda'):
    saved = torch.load(item['checkpoint'],map_location=device,weights_only=False)
    assert saved.get('model_class','ModuleNetwork') == 'ModuleNetwork'
    model = ModuleNetwork(saved['family'],saved['space'],saved['genotype']).to(device).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    outputs = []
    with patch('torch.linalg.solve',side_effect=AssertionError('solve in neural inference')), \
         patch('torch.linalg.inv',side_effect=AssertionError('inverse in neural inference')), \
         patch('torch.linalg.eigh',side_effect=AssertionError('eigh in neural inference')):
        for begin in range(0,len(angles),1024):
            aa = torch.tensor(angles[begin:begin+1024],device=device,dtype=torch.float32)
            mm = torch.tensor(mask[begin:begin+1024],device=device)
            canonical = model(features(aa,mm),mm)
            outputs.append(physical_weights(canonical.double(),aa.double()).cpu().numpy())
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(outputs)


def summarize(values, mask):
    counts = mask.sum(-1)-1
    def group(selected):
        gap = values['gap_db'][selected]
        return {'scenes':int(selected.sum()),'mean_sinr_gap_db':float(gap.mean()),
            'p95_sinr_gap_db':float(np.quantile(gap,.95)),
            'mean_sinr_db':float(values['sinr_db'][selected].mean()),
            'component_MSE_population':float(values['component_mse_vs_population'][selected].mean()),
            'component_MSE_sample':float(values['component_mse_vs_sample'][selected].mean()),
            'mean_NMSE_population':float(values['nmse_vs_population'][selected].mean()),
            'max_distortionless_residual':float(values['distortionless_residual'][selected].max())}
    result = group(np.ones(len(mask),dtype=bool))
    result['groups'] = {str(k):group(counts==k) for k in range(3,9)}
    result['by_k'] = {key:value['mean_sinr_gap_db'] for key,value in result['groups'].items()}
    return result


@torch.no_grad()
def independent_torch_check(weights, angles, mask, expected):
    measured = []
    for begin in range(0,len(angles),1024):
        w = torch.tensor(weights[begin:begin+1024],device='cuda',dtype=torch.complex128)
        aa = torch.tensor(angles[begin:begin+1024],device='cuda',dtype=torch.float64)
        mm = torch.tensor(mask[begin:begin+1024],device='cuda')
        measured.append(sinr_db(w,aa,mm).cpu().numpy())
    measured = np.concatenate(measured)
    np.testing.assert_allclose(measured,expected,rtol=1e-8,atol=1e-7)
    return float(abs(measured-expected).max())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',required=True)
    args = parser.parse_args()
    root = Path(args.freeze)
    freeze = json.loads((root/'freeze.json').read_text())
    assert json.loads((root/'status.json').read_text())['state'] == 'frozen_before_test'
    verify_freeze(freeze)
    cfg = freeze['protocol']
    setup(cfg['angle_seed'])
    assert torch.cuda.is_available()
    output = root/'fresh_test'
    assert not output.exists(),'Preserve existing test and failures; do not regenerate'
    output.mkdir()
    try:
        used = explicit_scene_keys(cfg['known_diagnostic_scenes'])
        for item in freeze['exclusion_files']:
            with np.load(item['path'],allow_pickle=False) as archive:
                a = archive['angles']
                m = archive['mask'].astype(bool) if 'mask' in archive.files else a != 999
                used.update(canonical_keys(a,m))
        assert len(used) == freeze['excluded_unique_scenes']
        before = len(used)
        raw = sample_angles(cfg['scenes'],cfg['physics'],np.random.default_rng(cfg['angle_seed']),used)
        mask = raw != 999
        angles = np.where(mask,raw,0).astype(np.float64)
        assert len(used) == before+cfg['scenes']
        save_json(root/'status.json',{'state':'running','stage':'reference_weights','pid':os.getpid(),'test_generated':True})
        np.savez(output/'angles.npz',angles=angles,mask=mask)
        save_json(output/'manifest.json',{'scenes':len(angles),'sha256':sha256(output/'angles.npz'),
            'excluded_unique_scenes':before,'overlap_scenes':0,'frozen_models_before_generation':True,
            'angle_seed':cfg['angle_seed'],'snapshot_seed':cfg['snapshot_seed'],
            'counts_by_k':{str(k):int((mask.sum(-1)-1==k).sum()) for k in range(3,9)}})
        oracle, _, _ = numpy_reference(angles,mask)
        sample, checks = sample_reference(angles,mask,cfg['snapshot_seed'],cfg['physics']['snapshots'])
        np.savez_compressed(output/'snapshot_checks.npz',indices=np.array([c[0] for c in checks]),
            snapshots=np.array([c[1] for c in checks]),steering=np.array([c[2] for c in checks]),
            numpy_weights=np.array([c[3] for c in checks]))
        report = {}
        for name in ['population_MVDR','sample_MVDR',*freeze['models']]:
            save_json(root/'status.json',{'state':'running','stage':'metrics','method':name,'pid':os.getpid(),'test_generated':True})
            if name == 'population_MVDR':
                weights = oracle
            elif name == 'sample_MVDR':
                weights = sample
            else:
                weights = predict(freeze['models'][name],angles,mask)
            values = numpy_metrics(weights,angles,mask,oracle,sample)
            disagreement = independent_torch_check(weights,angles,mask,values['sinr_db'])
            measured = nearest_peak_directions(weights,angles[:,0],cfg['pointing']['grid_step_deg'])
            item = summarize(values,mask)
            item.update({'meets_proximity_criteria':passes(item,cfg['acceptance']),
                'all_scenes_numpy_verified':True,'max_torch_numpy_SINR_difference_db':disagreement,
                'main_nearest_peak_MAE_deg':float(measured['main_error_deg'].mean()),
                'main_global_peak_MAE_deg':float(measured['global_error_deg'].mean()),
                'no_local_peak_fallback_count':int(measured['no_local_peak_fallback'].sum()),
                'main_nearest_peak_MAE_by_k':{str(k):float(measured['main_error_deg'][mask.sum(-1)-1==k].mean()) for k in range(3,9)}})
            if name in freeze['models']:
                cpu_indices = np.linspace(0,len(angles)-1,64,dtype=int)
                cpu = predict(freeze['models'][name],angles[cpu_indices],mask[cpu_indices],device='cpu')
                np.testing.assert_allclose(cpu,weights[cpu_indices],rtol=2e-4,atol=2e-5)
                item['cpu_GPU_prediction_check_scenes'] = len(cpu_indices)
            path = output/(name+'.npz')
            np.savez_compressed(path,weights=weights,**values,**measured)
            item['predictions_sha256'] = sha256(path)
            report[name] = item
            save_json(output/'metrics.json',report)
            print(json.dumps({'method':name,**item}),flush=True)
        verify_freeze(freeze)
        save_json(root/'status.json',{'state':'physical_and_main_metrics_complete_requires_null_and_latency',
            'test_generated':True,'test_used_for_training_or_selection':False,
            'candidate_passes':report['candidate']['meets_proximity_criteria'],
            'source_and_checkpoint_hashes_verified':True,'overall_goal_complete':False})
    except Exception:
        save_json(root/'status.json',{'state':'failed_preserve_test','traceback':traceback.format_exc(),
            'test_regeneration_forbidden':True})
        raise


if __name__ == '__main__':
    main()
