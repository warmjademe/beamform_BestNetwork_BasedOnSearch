"""The former successful objective under the corrected unit-element physics.

Population MVDR references are prepared offline for training only. No matrix
solver or reference weights enter the neural forward pass.
"""
import torch

from .original_modules import covariance
from .strict_dnnabf import steering, mvdr
from .refined import objective


@torch.no_grad()
def population_context(angles, mask):
    angles = angles.double()
    phase = steering(angles[:, 0])
    r = covariance(angles, mask)
    w = mvdr(r, phase)
    canonical = w * phase.conj()
    target = torch.cat((canonical.real, canonical.imag), -1)
    canonical_r = phase.conj()[..., :, None] * r * phase[..., None, :]
    q = (w.conj() * (r @ w[..., None]).squeeze(-1)).sum(-1).real
    assert bool((q > 0).all())
    return target, canonical_r, q


@torch.no_grad()
def prepare_population(data, batch_size=4096):
    parts = [population_context(data[3][i:i+batch_size], data[1][i:i+batch_size])
             for i in range(0, len(data[3]), batch_size)]
    return data + tuple(torch.cat([part[j] for part in parts]) for j in range(3))


def restored_physical_objective(prediction, target, canonical_r, q):
    # Identical expression to the four old networks: log(1+excess/q)
    # plus 0.05*log(1+weight NMSE). Use double precision for deep nulls.
    return objective(prediction.double(), target, canonical_r, q)


def population_component_mse(prediction, target, canonical_r, q):
    return torch.nn.functional.mse_loss(prediction.double(), target)


restored_physical_objective.uses_population_context = True
population_component_mse.uses_population_context = True
