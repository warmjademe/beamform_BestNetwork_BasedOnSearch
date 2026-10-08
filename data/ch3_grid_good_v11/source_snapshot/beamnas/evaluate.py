import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import packed, save_json, sync, unpacked
from .data import covariance, snapshots, solve_mvdr, steering
from .model import Beamformer, MLP


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    return {'mean': float(a.mean()), 'std': float(a.std(ddof=1)) if a.size > 1 else 0.0,
            'median': float(np.median(a)), 'count': int(a.size)}


def timing(function, device, warmup, repeats):
    for _ in range(warmup):
        function()
    sync(device)
    durations = []
    for _ in range(repeats):
        sync(device)
        t = time.perf_counter_ns()
        function()
        sync(device)
        durations.append((time.perf_counter_ns() - t) / 1e6)
    return {'unit': 'ms', 'batch_size': 1, 'warmup': warmup, 'repeats': repeats,
            **summarize(durations), 'p95': float(np.percentile(durations, 95))}


@torch.no_grad()
def latency(model, data, config):
    a, mask, _ = [x[:1] for x in data]
    cfg = config['evaluation']
    args = cfg['latency_warmup'], cfg['latency_repeats']
    result = {}
    if model is not None:
        model.eval()
        result['network_gpu'] = timing(lambda: model(a, mask), a.device, *args)
        cpu = copy.deepcopy(model).cpu().eval()
        ac, mc = a.cpu(), mask.cpu()
        result['network_cpu'] = timing(lambda: cpu(ac, mc), 'cpu', *args)
    else:
        ac, mc = a.cpu().double(), mask.cpu()
        dc = config['data']
        x = snapshots(ac, mc, dc)
        r = x @ x.mH / dc['snapshots']
        ad = steering(ac[:, 0], dc['antennas'], dc['spacing_wavelengths'])
        result['mvdr_cpu_covariance_given'] = timing(lambda: solve_mvdr(r, ad), 'cpu', *args)
        result['mvdr_cpu_snapshots_given'] = timing(lambda: solve_mvdr(x @ x.mH / dc['snapshots'], ad), 'cpu', *args)
    result['cpu_threads'] = torch.get_num_threads()
    return result


@torch.no_grad()
def physical_metrics(weights, angles, mask, config):
    cfg = config['data']
    w = unpacked(weights).to(torch.complex128)
    angles = angles.double()
    a = steering(angles, cfg['antennas'], cfg['spacing_wavelengths'])
    response = (w.conj()[:, None] * a).sum(-1).abs().square()
    noise = w.abs().square().sum(-1)
    desired = 10 ** (cfg['snr_db'] / 10) * response[:, 0]
    interference = 10 ** (cfg['inr_db'] / 10) * (response[:, 1:] * mask[:, 1:]).sum(-1)
    sinr = 10 * (desired.clamp_min(1e-30) / (noise + interference).clamp_min(1e-30)).log10()
    null_db = 10 * (response[:, 1:].clamp_min(1e-30) / response[:, :1].clamp_min(1e-30)).log10()
    return sinr, null_db


@torch.no_grad()
def directional_metrics(weights, angles, mask, config):
    cfg, ev = config['data'], config['evaluation']
    m = cfg['antennas']
    w = unpacked(weights).to(torch.complex128)
    angles = angles.double()
    grid = torch.linspace(-90, 90, round(180 / ev['scan_step_deg']) + 1, device=w.device, dtype=torch.float64)
    av = steering(grid, m, cfg['spacing_wavelengths'])
    power = (w.conj() @ av.T).abs().square()
    main_error = (grid[power.argmax(-1)] - angles[:, 0]).abs()
    distance = (angles[:, :, None] - angles[:, None, :]).abs()
    diagonal = torch.eye(9, device=w.device, dtype=torch.bool)[None]
    distance = distance.masked_fill(diagonal | ~mask[:, None, :], float('inf'))
    radius = (distance[:, 1:].amin(-1) / 2).clamp(max=ev['null_radius_deg'])
    count = round(2 * ev['null_radius_deg'] / ev['null_step_deg']) + 1
    u = torch.linspace(0, 1, count, device=w.device, dtype=torch.float64)
    lo = (angles[:, 1:] - radius).clamp(min=-90)
    hi = (angles[:, 1:] + radius).clamp(max=90)
    queries = lo[..., None] + (hi - lo)[..., None] * u
    aq = steering(queries, m, cfg['spacing_wavelengths'])
    p = (w.conj()[:, None, None] * aq).sum(-1).abs().square()
    idx = p.argmin(-1)
    null_angles = queries.gather(-1, idx[..., None]).squeeze(-1)
    error = (null_angles - angles[:, 1:]).abs()
    boundary = (idx == 0) | (idx == count - 1)
    return main_error.cpu().numpy(), error.cpu().numpy(), boundary.cpu().numpy()


def fixed_directional_indices(data, config):
    k = data[1].sum(1).cpu().numpy() - 1
    n = config['evaluation']['directional_scenes_per_k']
    return np.concatenate([np.flatnonzero(k == count)[:n] for count in config['data']['interferers']])


@torch.no_grad()
def evaluate_predictions(pred, data, config, out, latency_result):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    angles, mask, truth = data
    k = mask.sum(1).cpu().numpy() - 1
    difference = (pred - truth).square().sum(1)
    mse = (difference / config['data']['antennas']).cpu().numpy()
    nmse = (difference / truth.square().sum(1).clamp_min(1e-30)).cpu().numpy()
    sinr, null_depth = [], []
    for i in range(0, len(pred), 2048):
        s, nd = physical_metrics(pred[i:i + 2048], angles[i:i + 2048], mask[i:i + 2048], config)
        sinr.append(s.cpu().numpy())
        null_depth.append(nd.cpu().numpy())
    sinr, null_depth = np.concatenate(sinr), np.concatenate(null_depth)
    subset = fixed_directional_indices(data, config)
    main, null, boundary = [], [], []
    for batch in np.array_split(subset, max(1, (len(subset) + 127) // 128)):
        idx = torch.as_tensor(batch, device=angles.device)
        me, ne, edge = directional_metrics(pred[idx], angles[idx], mask[idx], config)
        main.append(me)
        null.append(ne)
        boundary.append(edge)
    main, null, boundary = np.concatenate(main), np.concatenate(null), np.concatenate(boundary)
    np.savez_compressed(out / 'per_scene_metrics.npz', mse=mse, nmse=nmse, sinr_db=sinr,
                        interference_response_db=null_depth, k=k, directional_indices=subset,
                        main_error_deg=main, null_error_deg=null, null_boundary=boundary)
    np.save(out / 'predictions.npy', pred.cpu().numpy())
    result = {'overall': {'complex_mse': summarize(mse), 'relative_weight_squared_error': summarize(nmse),
                          'sinr_db': summarize(sinr)}, 'by_k': {}, 'latency': latency_result,
              'directional_grid_deg': config['evaluation']['scan_step_deg'],
              'null_grid_max_step_deg': config['evaluation']['null_step_deg']}
    subset_mask = mask[torch.as_tensor(subset, device=mask.device), 1:].cpu().numpy()
    for count in config['data']['interferers']:
        full, select = k == count, k[subset] == count
        valid = subset_mask[select]
        result['by_k'][str(count)] = {'test_scenes': int(full.sum()), 'complex_mse': summarize(mse[full]),
            'sinr_db': summarize(sinr[full]), 'directional_scenes': int(select.sum()),
            'main_error_deg': summarize(main[select]), 'null_neighborhood_min_error_deg': summarize(null[select][valid]),
            'null_boundary_minimum_fraction': float(boundary[select][valid].mean()),
            'interference_response_db': summarize(null_depth[full][mask[torch.as_tensor(full, device=mask.device), 1:].cpu().numpy()])}
    save_json(out / 'metrics.json', result)
    return result


@torch.no_grad()
def evaluate_model(config, data, model_dir, out):
    out = Path(out)
    if (out / 'metrics.json').exists():
        return json.loads((out / 'metrics.json').read_text())
    checkpoint = torch.load(Path(model_dir) / 'model_best.pt', map_location=data[0].device, weights_only=False)
    model = MLP(config['data']['antennas']) if checkpoint['kind'] == 'mlp' else Beamformer(config['search_space'], config['data']['antennas'], checkpoint['genotype'])
    model = model.to(data[0].device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    pred = torch.cat([model(data[0][i:i + 2048], data[1][i:i + 2048]) for i in range(0, len(data[0]), 2048)])
    return evaluate_predictions(pred, data, config, out, latency(model, data, config))


@torch.no_grad()
def evaluate_references(config, data, root):
    root = Path(root)
    if not (root / 'sample_mvdr' / 'metrics.json').exists():
        evaluate_predictions(data[2], data, config, root / 'sample_mvdr', latency(None, data, config))
    if not (root / 'population_mvdr' / 'metrics.json').exists():
        pred = []
        for i in range(0, len(data[0]), 1024):
            a, mask = data[0][i:i + 1024].double(), data[1][i:i + 1024]
            r = covariance(a, mask, config['data'], False)
            ad = steering(a[:, 0], config['data']['antennas'], config['data']['spacing_wavelengths'])
            pred.append(packed(solve_mvdr(r, ad)))
        evaluate_predictions(torch.cat(pred), data, config, root / 'population_mvdr', {})
