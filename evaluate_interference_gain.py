"""Evaluate response at actual interference AOAs using frozen full-test weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from beamnas.common import sha256


def summarize(values):
    db = 10 * np.log10(np.maximum(values, 1e-300))
    return {
        'directions': len(values),
        'mean_db': float(db.mean()),
        'db_of_mean_power': float(10 * np.log10(values.mean())),
        'median_db': float(np.median(db)),
        'p95_db': float(np.quantile(db, .95)),
        'p99_db': float(np.quantile(db, .99)),
        'worst_db': float(db.max()),
        'at_most_minus40_fraction': float(np.mean(db <= -40)),
        'at_most_minus50_fraction': float(np.mean(db <= -50)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ch3_grid_good_v11/test.npz')
    parser.add_argument('--predictions', default='runs/ch3_grid_v11/test')
    parser.add_argument('--out', default='results/interference_gain')
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    with np.load(args.data) as data:
        angles = data['angles'].astype(float)
        mask = data['mask'].astype(bool)
        oracle = data['population_weights']
    valid = mask[:, 1:]
    counts = mask.sum(1) - 1
    steering = np.exp(1j * np.pi * np.sin(np.deg2rad(angles))[..., None] * np.arange(12))
    methods = {'Oracle_MVDR': oracle}
    prediction_hashes = {}
    for role in ('candidate', 'strict_DNNABF'):
        path = Path(args.predictions) / (role + '.npz')
        prediction_hashes[role] = sha256(path)
        with np.load(path) as data:
            methods[role] = data['weights']
    scene, direction = np.nonzero(valid)
    arrays = {'scene_index': scene, 'interference_index': direction + 1,
              'desired_angle_deg': angles[scene, 0],
              'interference_angle_deg': angles[scene, direction + 1],
              'interferer_count': counts[scene]}
    report = {
        'scenes': len(angles), 'interference_directions': int(valid.sum()),
        'definition': '10 log10(|w^H a(theta_k)|^2 / |w^H a(theta_0)|^2)',
        'scope': 'All actual interference AOAs; frozen full test; no minimum search or model selection',
        'data_sha256': sha256(args.data), 'prediction_sha256': prediction_hashes,
    }
    for role, weights in methods.items():
        response = np.abs(np.einsum('bm,bsm->bs', weights.conj(), steering)) ** 2
        ratio = response[:, 1:] / response[:, 0, None]
        assert np.isfinite(ratio).all()
        values = ratio[valid]
        arrays[role + '_power_ratio'] = values
        result = summarize(values)
        result['all_interferers_at_most_minus40_scene_fraction'] = float(
            np.mean(np.max(np.where(valid, ratio, 0), axis=1) <= 1e-4))
        result['by_k'] = {str(k): summarize(ratio[counts == k][valid[counts == k]])
                          for k in range(3, 9)}
        # Independent physical consistency check for the same response array.
        signal = 10 * response[:, 0]
        noise = (np.abs(weights) ** 2).sum(1)
        sinr = 10 * np.log10(signal / (1000 * np.where(valid, response[:, 1:], 0).sum(1) + noise))
        result['mean_sinr_db_crosscheck'] = float(sinr.mean())
        report[role] = result
    np.savez_compressed(root / 'per_direction.npz', **arrays)
    report['per_direction_sha256'] = sha256(root / 'per_direction.npz')
    (root / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
