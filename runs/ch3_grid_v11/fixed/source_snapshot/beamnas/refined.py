"""AOA-only networks with array features and exact distortionless output.

No solve, inverse, eigenproblem or MVDR weights are used in network inference.
Offline targets use the population covariance, which is determined by the AOA.
"""
import math

import torch
from torch import nn

from .common import packed, unpacked
from .data import covariance, solve_mvdr, steering
from .model import SearchBlock, DiscreteBlock


def canonical_features(angles, mask):
    """Rotate the desired spatial frequency to zero, preserving all relative phases."""
    u = angles.deg2rad().sin()
    delta = (u-u[:, :1])*mask
    phase = -math.pi*delta[..., None]*torch.arange(12, device=u.device, dtype=u.dtype)
    # Desired and interference roles remain distinguishable, even at array ambiguities.
    roles = torch.zeros_like(delta)
    roles[:, 0] = 1
    return torch.cat((phase.cos(), phase.sin(), delta[..., None]/2,
                      roles[..., None], mask[..., None].to(u.dtype)), -1)*mask[..., None]


def constrained_canonical(raw):
    """Project onto centrosymmetric-conjugate, unit-response canonical weights."""
    w = unpacked(raw)
    w = (w+w.flip(-1).conj())/2
    w = w-w.mean(-1, keepdim=True)+1/math.sqrt(12)
    return packed(w)


def restore_weights(canonical, angles):
    phase = steering(angles[:, 0], 12)*math.sqrt(12)
    return packed(unpacked(canonical)*phase)


@torch.no_grad()
def prepare(angles, mask, cfg):
    """Offline deterministic labels; no sample-covariance noise is fitted."""
    angles64 = angles.double()
    a = steering(angles64[:, 0], 12)
    r = covariance(angles64, mask, cfg, False)
    w = solve_mvdr(r, a)
    phase = a*math.sqrt(12)
    target = w*phase.conj()
    canonical_r = phase.conj()[..., :, None]*r*phase[..., None, :]
    q = (w.conj()*(r@w[..., None]).squeeze(-1)).sum(-1).real
    return (canonical_features(angles, mask), mask, packed(target),
            canonical_r.to(torch.complex64), q.float())


def error_terms(pred, target, r, q):
    difference = unpacked(pred-target)
    # For feasible weights this is exactly q(pred)/q(opt)-1.
    excess = (difference.conj()*(r@difference[..., None]).squeeze(-1)).sum(-1).real
    ratio = excess.clamp_min(0)/q
    nmse = difference.abs().square().sum(-1)/target.square().sum(-1).clamp_min(1e-8)
    gap_db = (10/math.log(10))*torch.log1p(ratio)
    return gap_db, nmse


def objective(pred, target, r, q):
    gap, nmse = error_terms(pred, target, r, q)
    return (gap*(math.log(10)/10)+0.05*torch.log1p(nmse)).mean()


class ResidualMLP(nn.Module):
    def __init__(self, width=384, depth=6):
        super().__init__()
        self.input = nn.Linear(9*27, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width),
            nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width)) for _ in range(depth))
        self.output = nn.Linear(width, 24)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, mask):
        h = self.input(features.flatten(1))
        for block in self.blocks:
            h = h+block(h)/math.sqrt(len(self.blocks))
        return constrained_canonical(self.output(h))


class RefinedTransformer(nn.Module):
    def __init__(self, space, genotype=None):
        super().__init__()
        self.space, self.architecture = space, genotype
        width = space['width']
        self.input = nn.Linear(27, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Sequential(nn.Linear(9*width, width*2), nn.GELU(), nn.Linear(width*2, 24))
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.zeros_(self.output[-1].bias)
        if genotype is None:
            self.blocks = nn.ModuleList(SearchBlock(space, i) for i in range(space['max_depth']))
            self.alpha_depth = nn.Parameter(torch.zeros(len(space['depths'])))
        else:
            self.blocks = nn.ModuleList(DiscreteBlock(width, g) for g in genotype['layers'])

    def forward(self, features, mask):
        h = self.input(features)*mask[..., None]
        states = [h]
        for block in self.blocks:
            states.append(block(states[-1], mask, states))
        if self.architecture is None:
            h = sum(p*self.norm(states[d]) for p, d in zip(self.alpha_depth.softmax(0), self.space['depths']))
        else:
            h = self.norm(states[-1])
        return constrained_canonical(self.output((h*mask[..., None]).flatten(1)))

    def architecture_parameters(self):
        return [] if self.architecture is not None else [self.alpha_depth]+[p for b in self.blocks for p in b.alpha.values()]

    def weight_parameters(self):
        excluded = set(map(id, self.architecture_parameters()))
        return [p for p in self.parameters() if id(p) not in excluded]

    def genotype(self):
        depth = self.space['depths'][int(self.alpha_depth.argmax())]
        return {'width':self.space['width'], 'depth':depth,
                'layers':[b.genotype() for b in self.blocks[:depth]],
                'input':'relative_spatial_phase_27_features',
                'output':'centrohermitian_distortionless_projection',
                'selection':'per_choice_argmax'}

    def probabilities(self):
        return {name:p.detach().softmax(0).cpu().tolist() for name,p in self.named_parameters()
                if name=='alpha_depth' or '.alpha.' in name}
