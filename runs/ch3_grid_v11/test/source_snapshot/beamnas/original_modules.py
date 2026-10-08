"""Explicit module extensions under the original unit-element array convention.

Input remains AOA only. Phase features, an unbounded linear weight head and a
distortionless/centrohermitian decoder are NEW modules, not recovered DNNABF.
No label clipping, per-vector scaling, or covariance solver enters inference.
"""
import math

import torch
from torch import nn

from .common import packed, unpacked
from .data import sample_angles
from .dense_search import DenseSearch
from .refined import ResidualMLP, RefinedTransformer, constrained_canonical
from .strict_dnnabf import steering, mvdr


def features(angles, mask):
    u = angles.deg2rad().sin()
    delta = (u-u[:, :1])*mask
    phase = math.pi*delta[..., None]*torch.arange(12, device=u.device, dtype=u.dtype)
    roles = torch.zeros_like(delta)
    roles[:, 0] = 1
    return torch.cat((phase.cos(), phase.sin(), delta[..., None]/2,
        roles[..., None], mask[..., None].to(u.dtype)), -1)*mask[..., None]


def covariance(angles, mask, inr_db=30.):
    av = steering(angles)
    powers = mask.to(angles.dtype)*10**(inr_db/10)
    powers = powers.clone()
    powers[:, 0] = 0
    return torch.einsum('bsm,bs,bsn->bmn', av, powers.to(av.dtype), av.conj())+torch.eye(12, device=av.device, dtype=av.dtype)


def canonical_target(weights, angles):
    # A unit-modulus rotation: component MSE is identical in physical coordinates.
    return packed(unpacked(weights)*steering(angles[:, 0]).conj())


def physical_weights(prediction, angles):
    return unpacked(prediction)*steering(angles[:, 0])


def sinr_db(weights, angles, mask, snr_db=10., inr_db=30.):
    av = steering(angles)
    response = torch.einsum('bm,bsm->bs', weights.conj(), av).abs().square()
    signal = 10**(snr_db/10)*response[:, 0]
    interference = 10**(inr_db/10)*(response[:, 1:]*mask[:, 1:]).sum(-1)
    noise = weights.abs().square().sum(-1)
    return 10*torch.log10((signal/(interference+noise).clamp_min(1e-30)).clamp_min(1e-30))


class FixedDnnabfModules(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        widths = [243, 2048, 1024, 1024, 1024]
        for left, right in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(left, right), nn.PReLU(right)]
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(1024, 24)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, f, mask):
        return constrained_canonical(self.output(self.hidden((f*mask[..., None]).flatten(1))))


class ModuleNetwork(nn.Module):
    def __init__(self, family, space=None, genotype=None):
        super().__init__()
        self.family, self.space, self.architecture = family, space, genotype
        if family == 'fixed_dnnabf_modules':self.core = FixedDnnabfModules()
        elif family == 'residual_pilot':self.core = ResidualMLP(space['width'], space['depth'])
        elif family == 'dense':self.core = DenseSearch(space, genotype)
        elif family == 'transformer':self.core = RefinedTransformer(space, genotype)
        else:raise ValueError(family)

    def forward(self, f, mask):
        # Legacy cores have a canonical sum sqrt(M); convert to unit-element sum 1.
        return self.core(f, mask)/math.sqrt(12)

    def predict(self, angles, mask):
        return physical_weights(self(features(angles, mask), mask), angles)

    def architecture_parameters(self):
        return self.core.architecture_parameters() if hasattr(self.core, 'architecture_parameters') else []

    def weight_parameters(self):
        excluded = set(map(id, self.architecture_parameters()))
        return [p for p in self.parameters() if id(p) not in excluded]

    def genotype(self):
        return self.core.genotype()

    def probabilities(self):
        return self.core.probabilities()
