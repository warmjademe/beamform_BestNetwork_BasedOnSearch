"""Searchable input aggregation for AOA-only, ordinary-MSE networks.

Summed relative spatial phases are permutation invariant. With the fixed equal
interference powers and ULA they encode the population Toeplitz covariance, but
inference forms neither that matrix nor its inverse: the network predicts weights.
"""
import torch
from torch import nn

from .original_modules import ModuleNetwork


def spatial_moments(flat, kind):
    f=flat.reshape(-1,9,27)
    interferer=f[...,26]*(1-f[...,25])
    count=interferer.sum(-1,keepdim=True)
    # Lag zero is the count; its imaginary part is identically zero.
    harmonics=torch.cat((f[...,1:12],f[...,13:24]),-1)
    summed=(harmonics*interferer[...,None]).sum(1)
    scale=8. if kind=='sum' else count.clamp_min(1.)
    return torch.cat((summed/scale,count/8.),-1)


class InputAggregation(nn.Module):
    choices=['flatten','sum','mean']

    def __init__(self,width,kind=None):
        super().__init__()
        self.kind=kind
        selected=self.choices if kind is None else [kind]
        self.projections=nn.ModuleDict({key:nn.Linear(243 if key=='flatten' else 23,width) for key in selected})
        if kind is None:self.alpha=nn.Parameter(torch.zeros(3))

    def operation(self,flat,kind):
        return self.projections[kind](flat if kind=='flatten' else spatial_moments(flat,kind))

    def forward(self,flat):
        if self.kind is not None:return self.operation(flat,self.kind)
        return sum(p*self.operation(flat,k) for p,k in zip(self.alpha.softmax(0),self.choices))

    def genotype(self):
        return self.kind if self.kind is not None else self.choices[int(self.alpha.argmax())]


class PooledModuleNetwork(ModuleNetwork):
    def __init__(self,family,space,genotype=None):
        assert family in ['dense','fixed_dnnabf_modules','residual_pilot']
        super().__init__(family,space,genotype)
        kind=genotype['input_pooling'] if genotype is not None else None
        if family=='fixed_dnnabf_modules':self.core.hidden[0]=InputAggregation(2048,kind)
        else:self.core.input=InputAggregation(space['width'],kind)

    @property
    def aggregation(self):
        return self.core.hidden[0] if self.family=='fixed_dnnabf_modules' else self.core.input

    def architecture_parameters(self):
        params=super().architecture_parameters()
        return params+([self.aggregation.alpha] if self.aggregation.kind is None else [])

    def genotype(self):
        result=super().genotype()
        return {**result,'input_pooling':self.aggregation.genotype()}

    def probabilities(self):
        return {**super().probabilities(),
            'input_pooling':self.aggregation.alpha.detach().softmax(0).cpu().tolist()}


def export_pooled(source):
    g=source.genotype()
    assert source.family=='dense'
    result=PooledModuleNetwork(source.family,source.space,g).to(next(source.parameters()).device)
    old,new=source.core,result.core
    result.aggregation.projections[g['input_pooling']].load_state_dict(
        source.aggregation.projections[g['input_pooling']].state_dict())
    new.output.load_state_dict(old.output.state_dict())
    for original,dest,layer in zip(old.blocks,new.blocks,g['layers']):
        dest.norm.load_state_dict(original.norm.state_dict())
        dest.ffn.load_state_dict(original.ffns[source.space['ff_multipliers'].index(layer['ff'])].state_dict())
        if layer['skip']=='linear':dest.linear_skip.load_state_dict(original.linear_skip.state_dict())
    return result
