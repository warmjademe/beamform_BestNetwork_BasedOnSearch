"""Develop strict AOA/tanh/MSE DARTS on GPU without opening a new test set."""
import argparse
import copy
import json
import math
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from beamnas.common import save_json, setup, sha256
from beamnas.strict_dnnabf import (StrictDense, StrictDenseSearch, mse, mvdr, pack,
                                  population_covariance, steering, unpack)


def read(path):
    return json.loads(Path(path).read_text())


def emit(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in value.items() if k not in ['probabilities', 'genotype']}), flush=True)


def generate(cfg, out, splits=('train', 'architecture_validation', 'selection_validation')):
    dc = cfg['data']
    rng, used, excluded = np.random.default_rng(dc['seed']), set(), []
    for name in cfg['exclude_angle_files']:
        with np.load(name) as old:
            angles = old['angles']
            mask = old['mask'] if 'mask' in old else np.ones_like(angles, dtype=bool)
            for a, m in zip(angles, mask):
                used.add(tuple(int(v) for v in a[m]))
        excluded.append({'path': name, 'sha256': sha256(name), 'scenes': len(angles)})
    old_unique = len(used)
    generator = torch.Generator(device='cuda').manual_seed(dc['seed']+100)
    result, manifest = {}, {'excluded': excluded, 'old_unique_scenes': old_unique,
                            'config': dc, 'splits': {}}
    for split in splits:
        n = dc[split]
        angles = np.empty((n, dc['interferers']+1), np.float32)
        grid = np.arange(dc['angle_min'], dc['angle_max']+1, dc['angle_step'])
        for i in range(n):
            while True:
                candidate = rng.choice(grid, dc['interferers']+1, replace=False)
                candidate[1:] = np.sort(candidate[1:])
                key = tuple(int(x) for x in candidate)
                if key not in used:
                    used.add(key)
                    angles[i] = candidate
                    break
        inputs = torch.tensor(angles, device='cuda')
        targets, residuals = [], []
        start = time.perf_counter()
        for begin in range(0, n, dc['generation_batch']):
            a = steering(inputs[begin:begin+dc['generation_batch']].double())
            b, s, m = a.shape
            def noise(shape):
                return torch.view_as_complex(torch.randn(*shape, 2, device='cuda',
                    dtype=torch.float64, generator=generator))/math.sqrt(2)
            powers = torch.full((s,), 10**(dc['inr_db']/10), device='cuda', dtype=torch.float64)
            powers[0] = 10**(dc['snr_db']/10)
            covariance_content = dc.get('covariance_content', 'received_signal_interference_noise')
            assert covariance_content in ['received_signal_interference_noise', 'interference_noise_only']
            first_source = 1 if covariance_content == 'interference_noise_only' else 0
            x = a[:, first_source:].transpose(1, 2) @ (noise((b, s-first_source, dc['snapshots']))*
                                                     powers[first_source:].sqrt()[None, :, None])
            x += noise((b, m, dc['snapshots']))
            r = x @ x.mH / dc['snapshots']
            w = mvdr(r, a[:, 0])
            assert torch.isfinite(w).all()
            targets.append(pack(w))
            residuals.append(float(((w.conj()*a[:, 0]).sum(-1)-1).abs().max()))
            if begin == 0:
                # Independent NumPy construction from actual received snapshots.
                xn, an = x[:8].cpu().numpy(), a[:8, 0].cpu().numpy()
                rn = xn @ xn.conj().transpose(0, 2, 1)/dc['snapshots']
                vn = np.linalg.solve(rn, an[..., None])[..., 0]
                wn = vn/(an.conj()*vn).sum(-1, keepdims=True)
                np.testing.assert_allclose(wn, w[:8].cpu().numpy(), rtol=1e-8, atol=1e-9)
                np.savez_compressed(out/(split+'_independent_label_audit.npz'),
                                    snapshots=xn, desired_steering=an, weights=wn)
        y = torch.cat(targets)
        assert len(y) == n
        path = out/(split+'.npz')
        np.savez(path, angles=angles, weights=y.cpu().numpy())
        info = {'n': n, 'sha256': sha256(path), 'generation_seconds': time.perf_counter()-start,
                'max_constraint_residual_float64': max(residuals),
                'max_abs_label_component': float(y.abs().max()),
                'scenes_any_label_component_above_one': int((y.abs()>1).any(-1).sum()),
                'tanh_mse_infimum': float((y-y.clamp(-1, 1)).square().mean()),
                'independent_numpy_snapshot_checks': min(8, n)}
        manifest['splits'][split] = info
        result[split] = (inputs, y)
        print(json.dumps({'data_split': split, **info}), flush=True)
    assert len(used)-old_unique == sum(dc[k] for k in result)
    manifest['overlap_scenes'] = 0
    manifest['all_unique_scenes'] = len(used)
    save_json(out/'data_manifest.json', manifest)
    return result


def physics_context(data, dc):
    angles, _ = data
    a = steering(angles.double())[:, 0]
    r = population_covariance(angles.double(), dc['snr_db'], dc['inr_db'])
    oracle = mvdr(r, a)
    return a, r, sinr_db(oracle, a, r, dc['snr_db'])


def sinr_db(w, a, r, snr_db):
    desired = (w.conj()*a).sum(-1).abs().square()*10**(snr_db/10)
    interference = (w.conj()*(r @ w[..., None]).squeeze(-1)).sum(-1).real
    return 10*torch.log10((desired/interference.clamp_min(1e-30)).clamp_min(1e-30))


@torch.no_grad()
def validate(model, data, context, dc):
    model.eval()
    prediction = torch.cat([model(x) for x in data[0].split(1024)])
    a, r, oracle_sinr = context
    gap = oracle_sinr-sinr_db(unpack(prediction.double()), a, r, dc['snr_db'])
    assert float(gap.min()) > -1e-7, 'Prediction exceeded theoretical oracle; investigate physics'
    return {'mse': float(mse(prediction, data[1])), 'mean_sinr_gap_db': float(gap.mean()),
            'p95_sinr_gap_db': float(gap.quantile(.95)),
            'max_abs_output_component': float(prediction.abs().max()),
            'zero_weight_rows': int((prediction.abs().sum(-1)==0).sum())}


def gradient_step(loss, optimizer, parameters, clip):
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()


def make_optimizer(model, parameters, lr, weight_decay, input_scale):
    """Condition first-layer parameters for raw degree units, not input features.

    Both model families receive identical parameterization and optimization.
    The network forward pass still receives the original degree-valued AOA.
    """
    assert input_scale > 0
    first = model.layers[0].weight
    with torch.no_grad():
        first.mul_(input_scale)
    others = [p for p in parameters if p is not first]
    return torch.optim.Adam([{'params': others, 'lr': lr},
                             {'params': [first], 'lr': lr*input_scale}],
                            weight_decay=weight_decay)


def schedule(optimizer, epochs, minimum_ratio):
    # A multiplier preserves the different parameter-group learning-rate scales.
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch:
        minimum_ratio+(1-minimum_ratio)*(1+math.cos(math.pi*epoch/epochs))/2)


def fit(cfg, data, context, genotype, name, root):
    dest = root/name
    dest.mkdir()
    tc = cfg['training']
    setup(cfg['seed']+10000)
    model = StrictDense(cfg['data']['interferers']+1, genotype['widths'], genotype['skips']).cuda()
    optimizer = make_optimizer(model, list(model.parameters()), tc['learning_rate'],
                               tc['weight_decay'], cfg.get('input_initial_weight_scale', 1.))
    scheduler = schedule(optimizer, tc['epochs'], tc['cosine_min_lr']/tc['learning_rate'])
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed']+3000)
    train, selection = data['train'], data['selection_validation']
    save_json(dest/'genotype.json', genotype)
    best, best_epoch, steps = float('inf'), 0, 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, tc['epochs']+1):
        model.train()
        total = 0.
        for idx in torch.randperm(len(train[0]), device='cuda', generator=generator).split(tc['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = mse(model(train[0][idx]), train[1][idx])
            gradient_step(loss, optimizer, model.parameters(), tc['grad_clip'])
            total += float(loss.detach())*len(idx)
            steps += 1
        metrics = validate(model, selection, context, cfg['data'])
        if metrics['mse'] < best:
            best, best_epoch = metrics['mse'], epoch
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'genotype': genotype,
                        'input_dim': cfg['data']['interferers']+1, 'validation': metrics,
                        'optimizer': optimizer.state_dict(), 'config': cfg}, dest/'best.pt')
        torch.cuda.synchronize()
        emit(dest/'history.jsonl', {'stage': name, 'epoch': epoch, 'train_mse': total/len(train[0]),
             'selection_validation': metrics, 'lr': optimizer.param_groups[0]['lr'],
             'first_layer_weight_lr': optimizer.param_groups[1]['lr'],
             'steps': steps, 'elapsed_seconds': time.perf_counter()-start})
        scheduler.step()
    result = {'scope': 'development_selection_validation_only', 'best_mse': best, 'best_epoch': best_epoch,
              'epochs': tc['epochs'], 'steps': steps, 'parameters': sum(p.numel() for p in model.parameters()),
              'training_seconds': time.perf_counter()-start, 'checkpoint_sha256': sha256(dest/'best.pt'),
              'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(), 'from_scratch': True}
    save_json(dest/'summary.json', result)
    return result


def search(cfg, data, context, root):
    dest = root/'search'
    dest.mkdir()
    sc = cfg['search']
    setup(cfg['seed'])
    model = StrictDenseSearch(cfg['data']['interferers']+1, cfg['search_space']).cuda()
    weights, alphas = model.weight_parameters(), model.architecture_parameters()
    wo = make_optimizer(model, weights, sc['weight_lr'], sc['weight_decay'],
                        cfg.get('input_initial_weight_scale', 1.))
    ao = torch.optim.Adam(alphas, lr=sc['architecture_lr'], betas=tuple(sc['alpha_betas']))
    scheduler = schedule(wo, sc['epochs'], 1e-5/sc['weight_lr'])
    wg = torch.Generator(device='cuda').manual_seed(cfg['seed']+1000)
    ag = torch.Generator(device='cuda').manual_seed(cfg['seed']+2000)
    train, architecture, selection = [data[k] for k in ['train', 'architecture_validation', 'selection_validation']]
    best, weight_steps, architecture_steps = float('inf'), 0, 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(1, sc['epochs']+1):
        model.train()
        order = torch.randperm(len(train[0]), device='cuda', generator=wg)
        vorder, cursor = torch.randperm(len(architecture[0]), device='cuda', generator=ag), 0
        for idx in order.split(sc['batch_size']):
            for p in weights: p.requires_grad_(True)
            for p in alphas: p.requires_grad_(False)
            wo.zero_grad(set_to_none=True)
            gradient_step(mse(model(train[0][idx]), train[1][idx]), wo, weights, sc['grad_clip'])
            weight_steps += 1
            if epoch > sc['warmup_epochs']:
                if cursor >= len(vorder):
                    vorder, cursor = torch.randperm(len(architecture[0]), device='cuda', generator=ag), 0
                vi = vorder[cursor:cursor+sc['batch_size']]
                cursor += sc['batch_size']
                for p in weights: p.requires_grad_(False)
                for p in alphas: p.requires_grad_(True)
                ao.zero_grad(set_to_none=True)
                gradient_step(mse(model(architecture[0][vi]), architecture[1][vi]), ao, alphas, sc['grad_clip'])
                architecture_steps += 1
        for p in model.parameters(): p.requires_grad_(True)
        metrics = validate(model, selection, context, cfg['data'])
        # Warmup checkpoints are not eligible architecture selections.
        if epoch > sc['warmup_epochs'] and metrics['mse'] < best:
            best = metrics['mse']
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'validation': metrics,
                        'weight_optimizer': wo.state_dict(), 'architecture_optimizer': ao.state_dict(),
                        'config': cfg}, dest/'best.pt')
        torch.cuda.synchronize()
        emit(dest/'history.jsonl', {'stage': 'search', 'epoch': epoch, 'selection_validation': metrics,
             'elapsed_seconds': time.perf_counter()-start, 'weight_steps': weight_steps,
             'architecture_steps': architecture_steps, 'weight_lr': wo.param_groups[0]['lr'],
             'architecture_lr': sc['architecture_lr'], 'genotype': model.genotype(),
             'first_layer_weight_lr': wo.param_groups[1]['lr'],
             'probabilities': model.probabilities()})
        scheduler.step()
    checkpoint = torch.load(dest/'best.pt', weights_only=False, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])
    genotype = model.genotype()
    save_json(dest/'genotype.json', genotype)
    save_json(dest/'architecture_probabilities.json', model.probabilities())
    save_json(dest/'summary.json', {'best_epoch': checkpoint['epoch'], 'best_soft_validation_mse': best,
        'search_seconds': time.perf_counter()-start, 'weight_steps': weight_steps,
        'architecture_steps': architecture_steps, 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(),
        'supernet_weight_parameters': sum(p.numel() for p in weights),
        'architecture_parameters': sum(p.numel() for p in alphas),
        'algorithm': 'first_order_DARTS', 'width_weight_sharing': 'prefix_masks',
        'alternation': 'one_train_weight_step_one_architecture_validation_step_after_warmup',
        'checkpoint_sha256': sha256(dest/'best.pt')})
    return genotype


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/strict_dnnabf_development.json')
    p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--input-weight-scale', type=float, default=1.,
                   help='Initial first-layer weight and its learning-rate scale; raw inputs unchanged')
    args = p.parse_args()
    cfg = read(args.config)
    cfg['input_initial_weight_scale'] = args.input_weight_scale
    cfg['first_layer_weight_lr_scale'] = args.input_weight_scale
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg['id'] += '_smoke'
        cfg['data'].update(train=64, architecture_validation=16, selection_validation=16)
        cfg['search_space'].update(widths=[16, 32], depths=[2, 4])
        cfg['search'].update(epochs=2, warmup_epochs=0, batch_size=16)
        cfg['training'].update(epochs=2, batch_size=16)
        cfg['fixed_widths'] = [64, 32, 32, 32]
    root = Path(args.out)
    assert not root.exists(), 'Preserve prior runs; choose a new output directory'
    root.mkdir(parents=True)
    setup(cfg['seed'])
    assert torch.cuda.is_available(), 'This workflow requires the authorized CUDA GPU'
    files = [__file__, 'beamnas/strict_dnnabf.py', 'beamnas/common.py', args.config]
    hashes = {str(f): sha256(f) for f in files}
    for source in files:
        source = Path(source)
        destination = root/'source_snapshot'/source.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    save_json(root/'config.json', cfg)
    save_json(root/'source_hashes.json', hashes)
    save_json(root/'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'pid': os.getpid(),
        'tf32': False, 'training_dtype': 'float32', 'label_generation_dtype': 'float64'})
    try:
        save_json(root/'status.json', {'state': 'generating_development_data', 'pid': os.getpid()})
        data = generate(cfg, root)
        context = physics_context(data['selection_validation'], cfg['data'])
        save_json(root/'status.json', {'state': 'training_fixed_baseline', 'pid': os.getpid()})
        fit(cfg, data, context, {'family': 'DNNABF_fixed', 'widths': cfg['fixed_widths'],
            'skips': ['none']*len(cfg['fixed_widths']), 'activation': 'PReLU_per_neuron',
            'input': 'raw_degrees_K_plus_1', 'output': '24_real_components_tanh'}, 'fixed', root)
        save_json(root/'status.json', {'state': 'searching', 'pid': os.getpid()})
        genotype = search(cfg, data, context, root)
        torch.cuda.empty_cache()
        save_json(root/'status.json', {'state': 'retraining_searched_architecture', 'pid': os.getpid()})
        fit(cfg, data, context, genotype, 'searched', root)
        assert hashes == {f: sha256(f) for f in files}, 'Source changed during run'
        save_json(root/'status.json', {'state': 'development_complete_requires_review',
            'scope': 'No independent test generated or evaluated; no overall goal completion',
            'source_hashes_verified': True})
    except Exception:
        save_json(root/'status.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
