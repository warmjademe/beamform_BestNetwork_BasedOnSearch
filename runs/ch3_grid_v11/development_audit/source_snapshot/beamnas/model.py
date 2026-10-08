import math

import numpy as np
import torch
from torch import nn


POSITIONS = ['sinusoidal', 'learned', 'relative', 'rope', 'alibi', 'fourier']
SKIPS = ['identity', 'linear', 'depthwise_conv']


def additive_encoding(length, width, fourier=False):
    p = torch.arange(length).float()[:, None]
    if fourier:
        phase = 2 * math.pi * p * torch.arange(1, width // 2 + 1)[None, :] / length
    else:
        phase = p * torch.exp(-math.log(10000) * torch.arange(0, width, 2) / width)[None, :]
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)


def rotate(x):
    length, width = x.shape[-2:]
    phase = torch.arange(length, device=x.device)[:, None] * torch.exp(
        -math.log(10000) * torch.arange(0, width, 2, device=x.device) / width)[None, :]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * phase.cos() - odd * phase.sin(),
                        even * phase.sin() + odd * phase.cos()), -1).flatten(-2)


class Attention(nn.Module):
    """Position candidates share QKV within one head-count candidate."""
    def __init__(self, width, heads, position=None, length=9):
        super().__init__()
        self.width, self.heads, self.position = width, heads, position
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        modes = POSITIONS if position is None else [position]
        if 'learned' in modes:
            self.learned = nn.Parameter(torch.zeros(length, width))
            nn.init.normal_(self.learned, std=0.02)
        if 'relative' in modes:
            self.relative = nn.Parameter(torch.zeros(heads, 2 * length - 1))
        self.register_buffer('sinusoidal', additive_encoding(length, width))
        self.register_buffer('fourier', additive_encoding(length, width, True))
        offsets = torch.arange(length)[:, None] - torch.arange(length)[None, :]
        self.register_buffer('relative_index', offsets + length - 1)
        slopes = 2.0 ** (-8.0 * torch.arange(1, heads + 1) / heads)
        self.register_buffer('alibi_bias', -slopes[:, None, None] * offsets.abs()[None])

    def forward(self, x, mask, position_weights=None):
        modes = POSITIONS if self.position is None else [self.position]
        inp = []
        for mode in modes:
            z = x + getattr(self, mode)[None] if mode in ['sinusoidal', 'learned', 'fourier'] else x
            inp.append(z * mask[..., None])
        z = torch.stack(inp, 1)
        b, p, t, d = z.shape
        q, k, v = self.qkv(z).view(b, p, t, 3, self.heads, d // self.heads).permute(3, 0, 1, 4, 2, 5)
        qs, ks, biases = [], [], []
        for i, mode in enumerate(modes):
            qs.append(rotate(q[:, i]) if mode == 'rope' else q[:, i])
            ks.append(rotate(k[:, i]) if mode == 'rope' else k[:, i])
            if mode == 'relative':
                biases.append(self.relative[:, self.relative_index])
            elif mode == 'alibi':
                biases.append(self.alibi_bias)
            else:
                biases.append(torch.zeros_like(self.alibi_bias))
        scores = torch.stack(qs, 1) @ torch.stack(ks, 1).transpose(-1, -2) / math.sqrt(d // self.heads)
        scores = scores + torch.stack(biases, 0)[None]
        scores = scores.masked_fill(~mask[:, None, None, None, :], float('-inf'))
        z = (scores.softmax(-1) @ v).transpose(-2, -3).reshape(b, p, t, d)
        if self.position is None:
            z = (z * position_weights[None, :, None, None]).sum(1)
        else:
            z = z[:, 0]
        return self.out(z) * mask[..., None]


class Skip(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        if kind == 'linear':
            self.op = nn.Linear(width, width, bias=False)
        elif kind == 'depthwise_conv':
            self.op = nn.Conv1d(width, width, 3, padding=1, groups=width, bias=False)
        else:
            self.op = nn.Identity()

    def forward(self, x, mask):
        x = x * mask[..., None]
        y = self.op(x.transpose(1, 2)).transpose(1, 2) if self.kind == 'depthwise_conv' else self.op(x)
        return y * mask[..., None]


def ffn(width, multiplier):
    return nn.Sequential(nn.Linear(width, width * multiplier), nn.GELU(),
                         nn.Linear(width * multiplier, width))


class SearchBlock(nn.Module):
    def __init__(self, space, index):
        super().__init__()
        self.space, self.index = space, index
        d = space['width']
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attentions = nn.ModuleList(Attention(d, h) for h in space['heads'])
        self.ffns = nn.ModuleList(ffn(d, r) for r in space['ff_multipliers'])
        self.attention_skips = nn.ModuleList(Skip(d, kind) for kind in SKIPS)
        self.ff_skips = nn.ModuleList(Skip(d, kind) for kind in SKIPS)
        self.long_options = ['none'] if index == 0 else ['none', 'input']
        if index >= 2:
            self.long_options.append('two_back')
        self.alpha = nn.ParameterDict({
            'heads': nn.Parameter(1e-3 * torch.randn(len(space['heads']))),
            'position': nn.Parameter(1e-3 * torch.randn(len(POSITIONS))),
            'ff': nn.Parameter(1e-3 * torch.randn(len(space['ff_multipliers']))),
            'attention_skip': nn.Parameter(1e-3 * torch.randn(3)),
            'ff_skip': nn.Parameter(1e-3 * torch.randn(3))})
        if len(self.long_options) > 1:
            self.alpha['long_skip'] = nn.Parameter(1e-3 * torch.randn(len(self.long_options)))

    def forward(self, x, mask, states):
        p = {k: v.softmax(0) for k, v in self.alpha.items()}
        z = self.norm1(x)
        att = sum(w * op(z, mask, p['position']) for w, op in zip(p['heads'], self.attentions))
        z = att + sum(w * op(x, mask) for w, op in zip(p['attention_skip'], self.attention_skips))
        norm = self.norm2(z)
        out = sum(w * op(norm) for w, op in zip(p['ff'], self.ffns))
        out = out + sum(w * op(z, mask) for w, op in zip(p['ff_skip'], self.ff_skips))
        for w, kind in zip(p.get('long_skip', []), self.long_options):
            if kind != 'none':
                out = out + w * (states[0] if kind == 'input' else states[-2])
        return out * mask[..., None]

    def genotype(self):
        options = {'heads': self.space['heads'], 'position': POSITIONS,
                   'ff': self.space['ff_multipliers'], 'attention_skip': SKIPS,
                   'ff_skip': SKIPS, 'long_skip': self.long_options}
        return {'long_skip': 'none', **{k: options[k][int(v.argmax())] for k, v in self.alpha.items()}}


class DiscreteBlock(nn.Module):
    def __init__(self, width, genotype):
        super().__init__()
        self.genotype = genotype
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, genotype['heads'], genotype['position'])
        self.ffn = ffn(width, genotype['ff'])
        self.attention_skip = Skip(width, genotype['attention_skip'])
        self.ff_skip = Skip(width, genotype['ff_skip'])

    def forward(self, x, mask, states):
        z = self.attention(self.norm1(x), mask) + self.attention_skip(x, mask)
        out = self.ffn(self.norm2(z)) + self.ff_skip(z, mask)
        kind = self.genotype['long_skip']
        if kind != 'none':
            out = out + (states[0] if kind == 'input' else states[-2])
        return out * mask[..., None]


class Beamformer(nn.Module):
    def __init__(self, space, antennas=12, genotype=None):
        super().__init__()
        self.space, self.architecture = space, genotype
        d = space['width']
        self.angle_projection = nn.Linear(1, d)
        self.type_embedding = nn.Embedding(2, d)
        self.final_norm = nn.LayerNorm(d)
        self.output = nn.Sequential(nn.Linear(9 * d, 128), nn.GELU(), nn.Linear(128, 2 * antennas))
        if genotype is None:
            self.blocks = nn.ModuleList(SearchBlock(space, i) for i in range(space['max_depth']))
            self.alpha_depth = nn.Parameter(1e-3 * torch.randn(len(space['depths'])))
        else:
            self.blocks = nn.ModuleList(DiscreteBlock(d, g) for g in genotype['layers'])

    def forward(self, angles, mask):
        roles = torch.ones(9, device=angles.device, dtype=torch.long)
        roles[0] = 0
        x = self.angle_projection((angles / 90)[..., None]) + self.type_embedding(roles)[None]
        x = x * mask[..., None]
        states = [x]
        for block in self.blocks:
            states.append(block(states[-1], mask, states))
        if self.architecture is None:
            x = sum(p * self.final_norm(states[depth]) for p, depth in zip(self.alpha_depth.softmax(0), self.space['depths']))
        else:
            x = self.final_norm(states[-1])
        return self.output((x * mask[..., None]).flatten(1))

    def architecture_parameters(self):
        if self.architecture is not None:
            return []
        return [self.alpha_depth] + [p for block in self.blocks for p in block.alpha.values()]

    def weight_parameters(self):
        ids = {id(p) for p in self.architecture_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    def genotype(self):
        depth = self.space['depths'][int(self.alpha_depth.argmax())]
        return {'width': self.space['width'], 'depth': depth,
                'layers': [block.genotype() for block in self.blocks[:depth]],
                'readout': 'masked_flatten_432_linear128_GELU_linear24',
                'normalization': 'pre_layernorm', 'selection': 'per_choice_argmax'}

    def probabilities(self):
        return {name: p.detach().softmax(0).cpu().tolist() for name, p in self.named_parameters()
                if name == 'alpha_depth' or '.alpha.' in name}


class MLP(nn.Module):
    def __init__(self, antennas=12):
        super().__init__()
        layers = []
        for inp in [18, 128, 128, 128]:
            layers.extend([nn.Linear(inp, 128), nn.PReLU(128)])
        self.net = nn.Sequential(*layers, nn.Linear(128, 2 * antennas), nn.Tanh())

    def forward(self, angles, mask):
        return self.net(torch.cat((angles / 90 * mask, mask.float()), -1))


def fixed_genotype(space):
    layer = {'heads': 4, 'position': 'sinusoidal', 'ff': 2,
             'attention_skip': 'identity', 'ff_skip': 'identity', 'long_skip': 'none'}
    return {'width': space['width'], 'depth': 2, 'layers': [dict(layer), dict(layer)],
            'selection': 'predeclared_fixed_architecture'}


def random_genotype(space, seed):
    rng = np.random.default_rng(seed)
    depth = int(rng.choice(space['depths']))
    layers = []
    for i in range(depth):
        options = ['none'] if i == 0 else ['none', 'input']
        if i >= 2:
            options.append('two_back')
        layers.append({'heads': int(rng.choice(space['heads'])), 'position': str(rng.choice(POSITIONS)),
                       'ff': int(rng.choice(space['ff_multipliers'])),
                       'attention_skip': str(rng.choice(SKIPS)), 'ff_skip': str(rng.choice(SKIPS)),
                       'long_skip': str(rng.choice(options))})
    return {'width': space['width'], 'depth': depth, 'layers': layers,
            'selection': 'uniform_random_no_validation_selection', 'sampling_seed': seed}
