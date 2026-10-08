"""Deterministic separated integer-angle scenes; fixed eligibility before training."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from beamnas.common import save_json, setup, sha256
from beamnas.pointing_boundary import nearest_peak_directions
from evaluate_restored_holdout import sample_reference
from freeze_restored_candidates import inventory_old_scenes, canonical_keys
from original_array_evaluation import numpy_reference, numpy_metrics, directions


def unrank(n, k, rank):
    result, start = [], 0
    for remaining in range(k, 0, -1):
        for value in range(start, n):
            block = math.comb(n-value-1, remaining-1)
            if rank < block:
                result.append(value)
                start = value+1
                break
            rank -= block
    assert len(result) == k and rank == 0
    return result


def scene_stream(cfg):
    g = cfg['generation']
    low, high = g['angle_min_deg'], g['angle_max_deg']
    spacing = g['minimum_any_source_separation_deg']
    desired_gap = g['minimum_desired_interference_separation_deg']
    specs = {}
    for k in g['interferers']:
        n = high-low+1-(spacing-1)*(k-1)
        total = math.comb(n, k)
        stride = total*618033988749894848//10**18
        while math.gcd(stride, total) != 1:
            stride += 1
        specs[k] = n, total, stride
    for turn in itertools.count():
        for desired in range(low, high+1, g['step_deg']):
            for k in g['interferers']:
                n, total, stride = specs[k]
                # A deterministic desired-angle-dependent offset avoids identical
                # sets being proposed together for every desired angle.
                rank = ((turn+(desired-low)*104729+k*15485863)*stride) % total
                values = np.array(unrank(n, k, rank))+(spacing-1)*np.arange(k)+low
                if np.min(abs(values-desired)) < desired_gap:
                    continue
                row, mask = np.zeros(9, float), np.zeros(9, bool)
                row[:k+1], mask[:k+1] = [desired, *values], True
                yield row, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--protocol', default='configs/ch3_grid_v11.json')
    args = p.parse_args()
    out = Path(args.out)
    assert not out.exists()
    cfg = json.loads(Path(args.protocol).read_text())
    g, q = cfg['generation'], cfg['quality']
    setup(g['ordering_seed'])
    assert torch.cuda.is_available()
    used, inventory = inventory_old_scenes(['data', 'runs'])
    out.mkdir(parents=True)
    (out/'audit').mkdir()
    sources = [Path(__file__).resolve().relative_to(Path.cwd()), Path(args.protocol),
        Path('evaluate_restored_holdout.py'), Path('freeze_restored_candidates.py'),
        Path('original_array_evaluation.py'), *Path('beamnas').glob('*.py')]
    hashes = {str(p): sha256(p) for p in sources}
    for source in sources:
        dest = out/'source_snapshot'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    split_by_residue = {r:s for s,rs in g['split_residues_mod_10'].items() for r in rs}
    assert set(split_by_residue) == set(range(10))
    quota = {(s,k): n//6 for s,n in g['counts'].items() for k in g['interferers']}
    assert all(n%6 == 0 for n in g['counts'].values())
    retained = {s:[] for s in g['counts']}
    accepted = Counter()
    statistics = Counter()
    audit_files = []
    generator = scene_stream(cfg)
    start = time.perf_counter()
    manifest = {'protocol':cfg, 'config':cfg['physics'], 'source_hashes':hashes,
        'exclusions':inventory, 'complete':False, 'splits':{}, 'test_used_for_training_or_selection':False}
    save_json(out/'manifest.json', manifest)
    while any(accepted[key] < value for key,value in quota.items()):
        aa, mm, split_names, keys = [], [], [], []
        pending = set()
        while len(aa) < g['candidate_batch_size']:
            row, mask = next(generator)
            key = next(iter(canonical_keys(row[None], mask[None])))
            split, k = split_by_residue[int(row[0])%10], int(mask.sum()-1)
            if accepted[split,k] >= quota[split,k]:
                statistics['skipped_filled_quota'] += 1
                continue
            if key in used or key in pending:
                statistics['excluded_previous_or_duplicate'] += 1
                continue
            aa.append(row); mm.append(mask); split_names.append(split); keys.append(key)
            pending.add(key)
        angles, mask = np.array(aa), np.array(mm)
        oracle, _, _ = numpy_reference(angles, mask)
        physical = numpy_metrics(oracle, angles, mask, oracle)
        peaks = nearest_peak_directions(oracle, angles[:,0])
        preliminary = (peaks['main_error_deg'] <= q['main_error_deg_max']+1e-9) & (physical['sinr_db'] >= q['sinr_db_min']-1e-9)
        null_error = np.full(len(angles), np.nan)
        ix = np.flatnonzero(preliminary)
        if len(ix):
            nulls = directions(oracle[ix], angles[ix], mask[ix], device='cuda', step_deg=.001)
            valid = mask[ix,1:]
            missing = np.any(valid & ~np.isfinite(nulls['null_error_deg']), axis=1)
            errors = np.where(valid, np.nan_to_num(nulls['null_error_deg'],nan=1e6), 0).sum(1)/valid.sum(1)
            errors[missing] = np.inf
            null_error[ix] = errors
        good = preliminary & (null_error <= q['null_mean_error_deg_max']+1e-9)
        selected = np.zeros(len(angles), bool)
        counts = mask.sum(1)-1
        for i in np.flatnonzero(good):
            key = (split_names[i], int(counts[i]))
            if accepted[key] >= quota[key]:
                continue
            selected[i] = True
            accepted[key] += 1
            retained[key[0]].append((angles[i], mask[i], oracle[i], peaks['main_error_deg'][i], null_error[i], physical['sinr_db'][i]))
        used.update(keys)
        path = out/'audit'/f'batch_{len(audit_files):04d}.npz'
        np.savez_compressed(path, angles=angles, mask=mask, oracle=oracle,
            main_error_deg=peaks['main_error_deg'], sinr_db=physical['sinr_db'],
            null_mean_error_deg=null_error, null_evaluated=preliminary, good=good, retained=selected)
        audit_files.append({'path':str(path),'sha256':sha256(path)})
        statistics['candidates'] += len(angles)
        statistics['preliminary_pass'] += int(preliminary.sum())
        statistics['good'] += int(good.sum())
        statistics['retained'] += int(selected.sum())
        for k in g['interferers']:
            statistics[f'K{k}_candidates'] += int((counts==k).sum())
            statistics[f'K{k}_good'] += int(((counts==k)&good).sum())
        progress = {'statistics':dict(statistics),'accepted':{f'{s}/K{k}':n for (s,k),n in accepted.items()},
                    'seconds':time.perf_counter()-start}
        save_json(out/'progress.json',progress)
        print(json.dumps(progress),flush=True)
        assert statistics['candidates'] <= g['maximum_candidates'], 'Eligibility budget exhausted; preserve audit and revisit design.'
    rng = np.random.default_rng(g['ordering_seed'])
    groups_seen = set()
    for split, rows in retained.items():
        rng.shuffle(rows)
        angles = np.array([r[0] for r in rows])
        mask = np.array([r[1] for r in rows])
        groups = set(angles[:,0])
        assert not groups_seen.intersection(groups)
        groups_seen.update(groups)
        assert len(rows) == g['counts'][split]
        sample, checks = sample_reference(angles, mask, g['snapshot_seed']+len(manifest['splits']))
        weights = np.concatenate([sample.real,sample.imag],axis=1).astype(np.float32)
        path = out/(split+'.npz')
        np.savez(path,angles=angles.astype(np.float32),mask=mask,weights=weights,
            population_weights=np.array([r[2] for r in rows]),quality_main_error_deg=np.array([r[3] for r in rows]),
            quality_null_mean_error_deg=np.array([r[4] for r in rows]),quality_sinr_db=np.array([r[5] for r in rows]))
        np.savez_compressed(out/(split+'_snapshot_checks.npz'),indices=np.array([r[0] for r in checks]),
            snapshots=np.array([r[1] for r in checks]),steering=np.array([r[2] for r in checks]),numpy_weights=np.array([r[3] for r in checks]))
        manifest['splits'][split] = {'n':len(rows),'sha256':sha256(path),'counts_by_k':{str(k):int((mask.sum(1)-1==k).sum()) for k in g['interferers']},
            'desired_angle_groups':sorted(groups),'bad_scenes':0}
        save_json(out/'manifest.json',manifest)
        print(json.dumps({'split':split,**manifest['splits'][split]}),flush=True)
    assert all(sha256(p)==value for p,value in hashes.items())
    manifest.update(complete=True,statistics=dict(statistics),audit_files=audit_files,group_overlap=0,
        unique_scene_count=sum(g['counts'].values()),generation_seconds=time.perf_counter()-start)
    save_json(out/'manifest.json',manifest)


if __name__ == '__main__':
    main()
