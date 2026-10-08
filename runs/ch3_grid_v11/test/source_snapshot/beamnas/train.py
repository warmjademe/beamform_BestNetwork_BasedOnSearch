import json
import time
from pathlib import Path

import torch

from .common import complex_mse, save_json, setup, sync
from .model import Beamformer, MLP


def set_grad(parameters, value):
    for p in parameters:
        p.requires_grad_(value)


@torch.no_grad()
def validate(model, data, batch_size=2048):
    model.eval()
    total = 0.0
    for i in range(0, len(data[0]), batch_size):
        a, mask, y = [v[i:i + batch_size] for v in data]
        total += float(complex_mse(model(a, mask), y)) * len(a)
    return total / len(data[0])


def finite_step(loss, optimizer, parameters, clip):
    assert torch.isfinite(loss), 'Nonfinite loss; preserve failed run'
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def search(config, train_data, validation_data, seed, out):
    out = Path(out)
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    setup(seed, config['runtime']['cpu_threads'])
    device = train_data[0].device
    cfg = config['search']
    model = Beamformer(config['search_space'], config['data']['antennas']).to(device)
    weights, alphas = model.weight_parameters(), model.architecture_parameters()
    assert not {id(p) for p in weights} & {id(p) for p in alphas}
    weight_opt = torch.optim.AdamW(weights, lr=cfg['weight_lr'], weight_decay=cfg['weight_decay'])
    alpha_opt = torch.optim.Adam(alphas, lr=cfg['architecture_lr'], betas=(0.5, 0.999),
                                 weight_decay=cfg['architecture_weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(weight_opt, cfg['epochs'])
    n, nv, bs = len(train_data[0]), len(validation_data[0]), cfg['batch_size']
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    alpha_generator = torch.Generator(device=device).manual_seed(seed + 2000)
    best, best_epoch, weight_steps, alpha_steps = float('inf'), -1, 0, 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        order = torch.randperm(n, device=device, generator=generator)
        vorder = torch.randperm(nv, device=device, generator=alpha_generator)
        cursor, loss_sum = 0, 0.0
        for idx in order.split(bs):
            set_grad(weights, True)
            set_grad(alphas, False)
            weight_opt.zero_grad(set_to_none=True)
            a, mask, target = [v[idx] for v in train_data]
            loss = complex_mse(model(a, mask), target)
            finite_step(loss, weight_opt, weights, cfg['grad_clip'])
            weight_steps += 1
            loss_sum += float(loss.detach()) * len(idx)
            if epoch >= cfg['warmup_epochs']:
                if cursor >= nv:
                    vorder = torch.randperm(nv, device=device, generator=alpha_generator)
                    cursor = 0
                vidx = vorder[cursor:cursor + bs]
                cursor += bs
                set_grad(weights, False)
                set_grad(alphas, True)
                alpha_opt.zero_grad(set_to_none=True)
                va, vm, vy = [v[vidx] for v in validation_data]
                vloss = complex_mse(model(va, vm), vy)
                finite_step(vloss, alpha_opt, alphas, cfg['grad_clip'])
                alpha_steps += 1
        set_grad(weights, True)
        set_grad(alphas, True)
        val = validate(model, validation_data, bs)
        if val < best:
            best, best_epoch = val, epoch + 1
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch + 1, 'validation_mse': val,
                        'weight_optimizer': weight_opt.state_dict(), 'architecture_optimizer': alpha_opt.state_dict()}, out / 'search_best.pt')
        sync(device)
        row = {'stage': 'search', 'seed': seed, 'epoch': epoch + 1,
               'train_mse': loss_sum / n, 'validation_mse': val,
               'elapsed_seconds': time.perf_counter() - start,
               'weight_lr': weight_opt.param_groups[0]['lr'], 'architecture_lr': cfg['architecture_lr'],
               'weight_steps': weight_steps, 'architecture_steps': alpha_steps,
               'genotype': model.genotype(), 'probabilities': model.probabilities()}
        with open(out / 'history.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps({k: v for k, v in row.items() if k not in ['genotype', 'probabilities']}), flush=True)
        scheduler.step()
    sync(device)
    seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(out / 'search_best.pt', map_location=device, weights_only=False)['state_dict'])
    genotype = model.genotype()
    save_json(out / 'genotype.json', genotype)
    save_json(out / 'architecture_probabilities.json', model.probabilities())
    result = {'seed': seed, 'genotype': genotype, 'best_epoch': best_epoch,
              'best_soft_validation_mse': best, 'search_seconds': seconds, 'gpu_hours': seconds / 3600,
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
              'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0,
              'supernet_weight_parameters': sum(p.numel() for p in weights),
              'architecture_parameters': sum(p.numel() for p in alphas),
              'weight_steps': weight_steps, 'architecture_steps': alpha_steps,
              'algorithm': 'first_order_DARTS', 'weight_optimizer': 'AdamW',
              'architecture_optimizer': 'Adam(beta1=0.5,beta2=0.999)',
              'alternation': 'one_training_weight_step_then_one_validation_architecture_step_after_warmup'}
    save_json(out / 'summary.json', result)
    return result


def fit(config, train_data, validation_data, seed, kind, genotype, out):
    out = Path(out)
    if (out / 'summary.json').exists():
        return json.loads((out / 'summary.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    setup(seed + 10000, config['runtime']['cpu_threads'])
    cfg = config['training']
    device = train_data[0].device
    model = MLP(config['data']['antennas']) if kind == 'mlp' else Beamformer(config['search_space'], config['data']['antennas'], genotype)
    model = model.to(device)
    assert not any('alpha' in n for n, _ in model.named_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['epochs'])
    generator = torch.Generator(device=device).manual_seed(seed + 3000)
    n, bs, best, best_epoch = len(train_data[0]), cfg['batch_size'], float('inf'), -1
    save_json(out / 'genotype.json', genotype if genotype is not None else {
        'model': 'DNNABF_style_MLP_reconstruction', 'hidden_widths': [128] * 4,
        'activation': 'PReLU', 'output_activation': 'tanh', 'input': '9_normalized_angles_plus_9_validity_bits'})
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    start = time.perf_counter()
    for epoch in range(cfg['epochs']):
        model.train()
        order = torch.randperm(n, device=device, generator=generator)
        total = 0.0
        for idx in order.split(bs):
            a, mask, y = [v[idx] for v in train_data]
            optimizer.zero_grad(set_to_none=True)
            loss = complex_mse(model(a, mask), y)
            finite_step(loss, optimizer, list(model.parameters()), cfg['grad_clip'])
            total += float(loss.detach()) * len(idx)
        val = validate(model, validation_data, bs)
        if val < best:
            best, best_epoch = val, epoch + 1
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch + 1,
                        'validation_mse': val, 'genotype': genotype, 'kind': kind}, out / 'model_best.pt')
        sync(device)
        row = {'stage': 'retrain', 'kind': kind, 'seed': seed, 'epoch': epoch + 1,
               'train_mse': total / n, 'validation_mse': val,
               'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': time.perf_counter() - start}
        with open(out / 'history.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        scheduler.step()
    result = {'kind': kind, 'seed': seed, 'reinitialized_from_scratch': True,
              'best_epoch': best_epoch, 'best_validation_mse': best,
              'training_seconds': time.perf_counter() - start,
              'parameters': sum(p.numel() for p in model.parameters()),
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
              'epochs': cfg['epochs'], 'training_steps': cfg['epochs'] * math_ceil_div(n, bs)}
    save_json(out / 'summary.json', result)
    return result


def math_ceil_div(n, d):
    return (n + d - 1) // d
