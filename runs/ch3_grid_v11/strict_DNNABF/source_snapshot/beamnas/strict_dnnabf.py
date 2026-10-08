"""Raw-AOA, tanh, componentwise-MSE models; no physical inference correction.

The dense supernet contains the published DNNABF hidden widths as one choice.
Width candidates share prefix weights. Discretization removes masked neurons.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def steering(angles, antennas=12):
    # Original paper convention: unit element magnitude and positive phase.
    phase = math.pi * angles.deg2rad().sin()[..., None] * torch.arange(
        antennas, device=angles.device, dtype=angles.dtype)
    return torch.polar(torch.ones_like(phase), phase)


def population_covariance(angles, snr_db=10., inr_db=30., include_signal=False):
    a = steering(angles)
    p = torch.full(angles.shape, 10**(inr_db/10), device=angles.device, dtype=angles.dtype)
    p[:, 0] = 10**(snr_db/10) if include_signal else 0
    return torch.einsum('bsm,bs,bsn->bmn', a, p.to(a.dtype), a.conj()) + torch.eye(
        12, device=angles.device, dtype=a.dtype)


def mvdr(r, a):
    v = torch.linalg.solve(r, a[..., None]).squeeze(-1)
    return v/(a.conj()*v).sum(-1, keepdim=True)


def pack(w):
    return torch.cat((w.real, w.imag), -1).float()


def unpack(w):
    return torch.complex(w[..., :12], w[..., 12:])


def mse(prediction, target):
    return F.mse_loss(prediction, target)


class ResizeIdentity(nn.Module):
    """Identity on shared channels, crop or append zeros when widths differ."""
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        return F.pad(x, (0, max(0, self.width-x.shape[-1])))[..., :self.width]


class StrictDense(nn.Module):
    def __init__(self, input_dim, widths, skips=None):
        super().__init__()
        skips = ['none']*len(widths) if skips is None else skips
        assert len(widths) == len(skips) and skips[0] == 'none'
        self.architecture = {'family': 'raw_aoa_dense', 'widths': list(widths),
                             'skips': list(skips), 'depth': len(widths)}
        self.layers, self.activations = nn.ModuleList(), nn.ModuleList()
        for previous, current in zip([input_dim]+list(widths[:-1]), widths):
            self.layers.append(nn.Linear(previous, current))
            self.activations.append(nn.PReLU(current))
        self.output = nn.Linear(widths[-1], 24)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, angles):
        x = angles
        for layer, activation, kind in zip(self.layers, self.activations, self.architecture['skips']):
            z = activation(layer(x))
            x = z if kind == 'none' else z+ResizeIdentity(z.shape[-1])(x)
        return self.output(x).tanh()


class StrictDenseSearch(nn.Module):
    def __init__(self, input_dim, space):
        super().__init__()
        self.space = copy.deepcopy(space)
        self.input_dim = input_dim
        width = max(space['widths'])
        self.layers = nn.ModuleList(nn.Linear(input_dim if i == 0 else width, width)
                                    for i in range(max(space['depths'])))
        self.activations = nn.ModuleList(nn.PReLU(width) for _ in self.layers)
        self.output = nn.Linear(width, 24)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)
        self.alpha_width = nn.Parameter(torch.zeros(len(self.layers), len(space['widths'])))
        self.alpha_skip = nn.Parameter(torch.zeros(len(self.layers)-1, 2))
        self.alpha_depth = nn.Parameter(torch.zeros(len(space['depths'])))
        self.register_buffer('width_masks', torch.stack([
            (torch.arange(width) < w).float() for w in space['widths']]))

    def forward(self, angles):
        width_gates = self.alpha_width.softmax(-1) @ self.width_masks
        skip_probability = self.alpha_skip.softmax(-1)
        x, states = angles, []
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            z = activation(layer(x))
            if i:
                z = z+skip_probability[i-1, 1]*x
            x = z*width_gates[i]
            states.append(x)
        final = sum(p*states[d-1] for p, d in zip(self.alpha_depth.softmax(-1), self.space['depths']))
        return self.output(final).tanh()

    def architecture_parameters(self):
        return [self.alpha_width, self.alpha_skip, self.alpha_depth]

    def weight_parameters(self):
        excluded = set(map(id, self.architecture_parameters()))
        return [p for p in self.parameters() if id(p) not in excluded]

    def genotype(self):
        depth = self.space['depths'][int(self.alpha_depth.argmax())]
        return {'family': 'raw_aoa_dense', 'depth': depth,
                'widths': [self.space['widths'][int(p.argmax())] for p in self.alpha_width[:depth]],
                'skips': ['none']+[['none', 'resize_identity'][int(p.argmax())]
                                   for p in self.alpha_skip[:depth-1]],
                'activation': 'PReLU_per_neuron', 'input': 'raw_degrees_K_plus_1',
                'output': '24_real_components_tanh', 'selection': 'per_choice_argmax'}

    def probabilities(self):
        return {k: p.detach().softmax(-1).cpu().tolist() for k, p in self.named_parameters()
                if k.startswith('alpha_')}

    def export_inherited(self):
        """Only used to check relaxation/export; formal fit starts from scratch."""
        g = self.genotype()
        result = StrictDense(self.input_dim, g['widths'], g['skips']).to(self.output.weight)
        previous = self.input_dim
        with torch.no_grad():
            for i, width in enumerate(g['widths']):
                result.layers[i].weight.copy_(self.layers[i].weight[:width, :previous])
                result.layers[i].bias.copy_(self.layers[i].bias[:width])
                result.activations[i].weight.copy_(self.activations[i].weight[:width])
                previous = width
            result.output.weight.copy_(self.output.weight[:, :previous])
            result.output.bias.copy_(self.output.bias)
        return result
