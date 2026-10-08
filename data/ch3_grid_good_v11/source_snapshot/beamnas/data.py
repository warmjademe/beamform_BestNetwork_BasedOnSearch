import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .common import packed, save_json, sha256, sync


def steering(angles, m=12, spacing=0.5):
    idx = torch.arange(m, device=angles.device, dtype=angles.dtype)
    phase = -2 * math.pi * spacing * angles.deg2rad().sin()[..., None] * idx
    return torch.polar(torch.ones_like(phase) / math.sqrt(m), phase)


def solve_mvdr(r, a):
    v = torch.linalg.solve(r, a[..., None]).squeeze(-1)
    return v / (a.conj() * v).sum(-1, keepdim=True)


def covariance(angles, mask, cfg, include_signal=True):
    a = steering(angles, cfg['antennas'], cfg['spacing_wavelengths'])
    p = mask.to(angles.dtype) * 10 ** (cfg['inr_db'] / 10)
    p[:, 0] = 10 ** (cfg['snr_db'] / 10) if include_signal else 0
    eye = torch.eye(cfg['antennas'], device=angles.device, dtype=a.dtype)
    return torch.einsum('bsm,bs,bsn->bmn', a, p.to(a.dtype), a.conj()) + eye


def snapshots(angles, mask, cfg, generator=None):
    b, s = angles.shape
    m, l = cfg['antennas'], cfg['snapshots']
    a = steering(angles, m, cfg['spacing_wavelengths'])
    def cn(shape):
        z = torch.randn(*shape, 2, device=angles.device, dtype=angles.dtype, generator=generator)
        return torch.view_as_complex(z) / math.sqrt(2)
    p = mask.to(angles.dtype) * 10 ** (cfg['inr_db'] / 10)
    p[:, 0] = 10 ** (cfg['snr_db'] / 10)
    signals = cn((b, s, l)) * p.sqrt()[..., None]
    return a.transpose(1, 2) @ signals + cn((b, m, l))


def sample_angles(n, cfg, rng, used):
    """Uniform K; canonical angle scenes unique across all splits."""
    counts = np.resize(np.array(cfg['interferers']), n)
    rng.shuffle(counts)
    out = np.full((n, 9), 999, dtype=np.int16)
    for i, k in enumerate(counts):
        while True:
            v = rng.choice(np.arange(cfg['angle_min'], cfg['angle_max'] + 1,
                                     cfg['angle_step']), int(k) + 1, replace=False)
            v[1:] = np.sort(v[1:])
            key = tuple(int(x) for x in v)
            if key not in used:
                used.add(key)
                out[i, :k + 1] = v
                break
    return out


def generate(config, root, device):
    root = Path(root)
    manifest_path = root / 'manifest.json'
    cfg = config['data']
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        assert manifest['config'] == cfg, 'Dataset config mismatch'
        for name, digest in manifest['sha256'].items():
            assert sha256(root / name) == digest, 'Dataset checksum mismatch: ' + name
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rng, used = np.random.default_rng(cfg['seed']), set()
    gen = torch.Generator(device=device).manual_seed(cfg['seed'] + 100)
    manifest = {'config': cfg, 'sha256': {}, 'splits': {},
                'label': 'sample_covariance_mvdr_1024_received_snapshots',
                'split_overlap_scenes': 0, 'noise_variance': 1.0}
    for split in ['train', 'validation', 'test']:
        n = cfg[split]
        angles = sample_angles(n, cfg, rng, used)
        mask = angles != 999
        y = np.empty((n, 2 * cfg['antennas']), np.float32)
        residual_max = 0.0
        for i in range(0, n, cfg['generation_batch']):
            end = min(i + cfg['generation_batch'], n)
            a = torch.tensor(np.where(mask[i:end], angles[i:end], 0), device=device, dtype=torch.float64)
            valid = torch.tensor(mask[i:end], device=device)
            x = snapshots(a, valid, cfg, gen)
            r = x @ x.mH / cfg['snapshots']
            ad = steering(a[:, 0], cfg['antennas'], cfg['spacing_wavelengths'])
            w = solve_mvdr(r, ad)
            assert torch.isfinite(w).all()
            residual_max = max(residual_max, float(((w.conj() * ad).sum(-1) - 1).abs().max()))
            y[i:end] = packed(w).cpu().numpy()
        name = split + '.npz'
        np.savez(root / name, angles=np.where(mask, angles, 0).astype(np.float32), mask=mask, weights=y)
        manifest['sha256'][name] = sha256(root / name)
        manifest['splits'][split] = {'count': n, 'counts_by_k': {str(k): int((mask.sum(1) == k + 1).sum()) for k in cfg['interferers']}, 'max_distortionless_residual_float64': residual_max}
        print(json.dumps({'stage': 'data', 'split': split, 'count': n, 'elapsed_s': time.perf_counter() - start}), flush=True)
    sync(device)
    manifest['generation_seconds'] = time.perf_counter() - start
    save_json(manifest_path, manifest)
    return manifest


def load_split(root, split, device):
    with np.load(Path(root) / (split + '.npz')) as d:
        return tuple(torch.as_tensor(d[k], device=device) for k in ['angles', 'mask', 'weights'])
