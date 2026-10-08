"""Batch-one GPU deployment of the frozen v11 network and matched MVDR.

Capture records operations, not answers: each call copies fresh AOA/mask values
and recomputes weights. Returned buffers are reused at the next call. Callers
must synchronize before reading results on the CPU. This runtime is serial.
"""
import math
import time

import torch
from torch import nn

from .original_modules import covariance, features, physical_weights
from .strict_dnnabf import steering


class FrozenNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, angles, mask):
        return physical_weights(self.model(features(angles.float(), mask), mask).double(), angles)


class RealFrozenNetwork(nn.Module):
    """Algebraically equivalent real-valued export for compiler fusion.

Weights, architecture, FP32 neural operations and FP64 output rotation remain
unchanged. The complex projection is expressed using its real/imaginary parts.
Numerical parity must be checked because fusion changes rounding.
"""
    def __init__(self, model):
        super().__init__()
        if model.family != 'dense':
            raise ValueError('This export supports only the frozen dense architecture')
        self.core = model.core.eval().requires_grad_(False)

    def forward(self, angles, mask):
        f = features(angles.float(), mask)
        states = [self.core.input((f*mask[..., None]).flatten(1))]
        for block in self.core.blocks:
            states.append(block(states[-1], states))
        raw = self.core.output(states[-1])
        re, im = raw[..., :12], raw[..., 12:]
        re = (re+re.flip(-1))/2
        im = (im-im.flip(-1))/2
        re = (re-re.mean(-1, keepdim=True)+1/math.sqrt(12))/math.sqrt(12)
        im = (im-im.mean(-1, keepdim=True))/math.sqrt(12)
        phase = math.pi*angles[:, :1].deg2rad().sin()*torch.arange(12, device=angles.device)
        c, s = phase.cos(), phase.sin()
        re, im = re.double(), im.double()
        return torch.stack((re*c-im*s, re*s+im*c), -1).contiguous()


class MVDRRuntime(nn.Module):
    def __init__(self, solver='solve'):
        super().__init__()
        if solver not in ('solve', 'solve_ex', 'cholesky', 'inverse'):
            raise ValueError(solver)
        self.solver = solver

    def forward(self, angles, mask):
        r, a = covariance(angles, mask), steering(angles[:, 0])
        if self.solver == 'solve':
            v = torch.linalg.solve(r, a[..., None]).squeeze(-1)
            return v/(a.conj()*v).sum(-1, keepdim=True)
        elif self.solver == 'solve_ex':
            v, info = torch.linalg.solve_ex(r, a[..., None], check_errors=False)
            v = v.squeeze(-1)
        elif self.solver == 'cholesky':
            factor, info = torch.linalg.cholesky_ex(r, check_errors=False)
            v = torch.cholesky_solve(a[..., None], factor).squeeze(-1)
        else:
            inverse, info = torch.linalg.inv_ex(r, check_errors=False)
            v = (inverse@a[..., None]).squeeze(-1)
        return v/(a.conj()*v).sum(-1, keepdim=True), info


class BatchOneGraph:
    """Fixed shape [1,9]; mask supports 3--8 interferers without recapture."""
    def __init__(self, function, angles, mask, warmups=10):
        if angles.shape != (1, 9) or mask.shape != (1, 9) or not angles.is_cuda:
            raise ValueError('Expected CUDA angles/mask with shape [1,9]')
        self.function = function
        self.angles, self.mask = angles.clone(), mask.clone()
        self.setup_started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(warmups):
                function(self.angles, self.mask)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.output = function(self.angles, self.mask)
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter()-self.setup_started

    def __call__(self, angles, mask):
        self.angles.copy_(angles, non_blocking=True)
        self.mask.copy_(mask, non_blocking=True)
        self.graph.replay()
        return self.output


def complex_output(result):
    weights = result[0] if isinstance(result, tuple) else result
    return weights if weights.is_complex() else torch.view_as_complex(weights)


def load_frozen_runtime(checkpoint, angles, mask):
    """Create the selected fused/graph runtime once, then serve fresh AOAs."""
    from .original_modules import ModuleNetwork
    saved = torch.load(checkpoint, map_location='cuda', weights_only=False)
    model = ModuleNetwork(saved['family'], saved['space'], saved['genotype']).cuda().eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    function = torch.compile(RealFrozenNetwork(model), fullgraph=True, dynamic=False)
    return BatchOneGraph(function, angles, mask)
