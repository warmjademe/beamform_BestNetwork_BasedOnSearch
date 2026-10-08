"""First-order DARTS search space over residual dense networks."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .refined import constrained_canonical


ACTIVATIONS = ['gelu', 'silu', 'tanh']


def activate(x, kind):
    return F.gelu(x) if kind=='gelu' else F.silu(x) if kind=='silu' else torch.tanh(x)


class FeedForward(nn.Module):
    def __init__(self, width, ratio, activation=None):
        super().__init__()
        self.activation=activation
        self.first=nn.Linear(width,width*ratio)
        self.second=nn.Linear(width*ratio,width)

    def forward(self,x,probability=None):
        z=self.first(x)
        z=activate(z,self.activation) if self.activation else sum(
            p*activate(z,kind) for p,kind in zip(probability,ACTIVATIONS))
        return self.second(z)


class DenseBlock(nn.Module):
    def __init__(self,space,index,genotype=None):
        super().__init__()
        self.space,self.index,self.architecture=space,index,genotype
        width=space['width']
        self.norm=nn.LayerNorm(width)
        self.long_options=['none'] if index==0 else ['none','input']
        if index>=2:self.long_options.append('two_back')
        self.linear_skip=nn.Linear(width,width,bias=False)
        nn.init.eye_(self.linear_skip.weight)
        if genotype is None:
            self.ffns=nn.ModuleList(FeedForward(width,r) for r in space['ff_multipliers'])
            self.alpha=nn.ParameterDict({key:nn.Parameter(torch.zeros(n)) for key,n in
                [('ff',len(space['ff_multipliers'])),('activation',3),('skip',2)]})
            if len(self.long_options)>1:
                self.alpha['long_skip']=nn.Parameter(torch.zeros(len(self.long_options)))
        else:
            self.ffn=FeedForward(width,genotype['ff'],genotype['activation'])
            if genotype['skip']=='identity':
                self.linear_skip=nn.Identity()

    def forward(self,x,states):
        z=self.norm(x)
        if self.architecture is None:
            p={key:value.softmax(0) for key,value in self.alpha.items()}
            y=sum(v*op(z,p['activation']) for v,op in zip(p['ff'],self.ffns))
            result=p['skip'][0]*x+p['skip'][1]*self.linear_skip(x)
            if 'long_skip' in p:
                for value,kind in zip(p['long_skip'],self.long_options):
                    if kind!='none': result=result+.25*value*(states[0] if kind=='input' else states[-2])
        else:
            y=self.ffn(z)
            result=self.linear_skip(x)
            kind=self.architecture['long_skip']
            if kind!='none':result=result+.25*(states[0] if kind=='input' else states[-2])
        return result+y/math.sqrt(self.space['max_depth'])

    def genotype(self):
        a=self.alpha
        return {'ff':self.space['ff_multipliers'][int(a['ff'].argmax())],
                'activation':ACTIVATIONS[int(a['activation'].argmax())],
                'skip':['identity','linear'][int(a['skip'].argmax())],
                'long_skip':self.long_options[int(a['long_skip'].argmax())] if 'long_skip' in a else 'none'}


class DenseSearch(nn.Module):
    def __init__(self,space,genotype=None):
        super().__init__()
        self.space,self.architecture=space,genotype
        self.input=nn.Linear(9*27,space['width'])
        self.output=nn.Linear(space['width'],24)
        nn.init.normal_(self.output.weight,std=.001)
        nn.init.zeros_(self.output.bias)
        if genotype is None:
            self.blocks=nn.ModuleList(DenseBlock(space,i) for i in range(space['max_depth']))
            self.alpha_depth=nn.Parameter(torch.zeros(len(space['depths'])))
        else:
            self.blocks=nn.ModuleList(DenseBlock(space,i,g) for i,g in enumerate(genotype['layers']))

    def forward(self,features,mask):
        states=[self.input((features*mask[...,None]).flatten(1))]
        for block in self.blocks:states.append(block(states[-1],states))
        if self.architecture is None:
            h=sum(p*states[depth] for p,depth in zip(self.alpha_depth.softmax(0),self.space['depths']))
        else:h=states[-1]
        return constrained_canonical(self.output(h))

    def architecture_parameters(self):
        return [] if self.architecture is not None else [self.alpha_depth]+[p for b in self.blocks for p in b.alpha.values()]

    def weight_parameters(self):
        excluded=set(map(id,self.architecture_parameters()))
        return [p for p in self.parameters() if id(p) not in excluded]

    def genotype(self):
        depth=self.space['depths'][int(self.alpha_depth.argmax())]
        return {'family':'residual_dense','width':self.space['width'],'depth':depth,
                'layers':[b.genotype() for b in self.blocks[:depth]],
                'residual_scale':1/math.sqrt(self.space['max_depth']), 'long_skip_scale':.25,
                'input':'relative_spatial_phase_27_features','output':'centrohermitian_distortionless_projection',
                'selection':'per_choice_argmax'}

    def probabilities(self):
        return {name:p.detach().softmax(0).cpu().tolist() for name,p in self.named_parameters()
                if name=='alpha_depth' or '.alpha.' in name}


def export_inherited(model):
    """For discretization diagnostics only; formal training starts from scratch."""
    genotype=model.genotype()
    discrete=DenseSearch(model.space,genotype).to(next(model.parameters()).device)
    discrete.input.load_state_dict(model.input.state_dict())
    discrete.output.load_state_dict(model.output.state_dict())
    for source,dest,g in zip(model.blocks,discrete.blocks,genotype['layers']):
        dest.norm.load_state_dict(source.norm.state_dict())
        dest.ffn.load_state_dict(source.ffns[model.space['ff_multipliers'].index(g['ff'])].state_dict())
        if g['skip']=='linear':dest.linear_skip.load_state_dict(source.linear_skip.state_dict())
    return discrete
