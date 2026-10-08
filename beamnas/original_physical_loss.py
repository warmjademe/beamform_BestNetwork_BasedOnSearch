"""Prepared physical-loss option; not enabled by any current training entrypoint.

The current original-array experiments train with ordinary MSE. This module is
only numerical preparation for the separately proposed change of training loss.
It requires an AOA-to-canonical-weight network, and never invokes an MVDR solver.
"""
import math

import torch

from .common import unpacked


def physical_power_terms(prediction, angles, mask, snr_db=10., inr_db=30.):
    """Original unit-element ULA powers, evaluated in float64 for cancellation.

    prediction holds 12 real and 12 imaginary canonical weight components.
    The desired spatial phase is zero in these coordinates. A network's
    distortionless decoder prevents the all-zero weight degeneracy.
    """
    assert prediction.shape[-1]==24
    a=angles.double()
    u=a.deg2rad().sin()
    delta=u-u[:,:1]
    phase=math.pi*delta[...,None]*torch.arange(12,device=a.device,dtype=a.dtype)
    phi=torch.polar(torch.ones_like(phase),phase)
    w=unpacked(prediction.double())
    response=torch.einsum('bm,bsm->bs',w.conj(),phi).abs().square()
    signal=10**(snr_db/10)*response[:,0]
    interference=10**(inr_db/10)*(response[:,1:]*mask[:,1:]).sum(-1)
    noise=w.abs().square().sum(-1)
    return signal,interference,noise


def physical_negative_log_sinr(prediction, angles, mask, reduction='mean'):
    """Negative natural log SINR, with the fixed original-paper powers.

    This is a different objective from weight MSE. No reference weights enter
    this loss; the population MVDR remains an independent evaluation reference.
    """
    signal,interference,noise=physical_power_terms(prediction,angles,mask)
    result=(interference+noise).clamp_min(1e-30).log()-signal.clamp_min(1e-30).log()
    if reduction=='none':return result
    if reduction=='mean':return result.mean()
    raise ValueError(reduction)
