"""Add the raw-AOA baseline to the unchanged batch-one timing procedure."""
from unittest.mock import patch
import benchmark_restored_latency as legacy
from ch3_strict_baseline import load_models as load_strict
from beamnas.strict_dnnabf import unpack


def measure_latency(frozen,angles,mask,config,destination):
    original_load,original_call=legacy.load_models,legacy.call_method
    def load(items,device):
        modules={k:v for k,v in items.items() if v.get('kind')!='strict_DNNABF_by_K'}
        models=original_load(modules,device)
        for name,item in items.items():
            if item.get('kind')=='strict_DNNABF_by_K':models[name]=load_strict(item,device)
        return models
    def call(name,inputs,models):
        if name in models and isinstance(models[name],dict):
            aa,mm=inputs[:2]
            k=int(mm.sum().item())-1
            return unpack(models[name][k](aa[:,:k+1].float()).double())
        return original_call(name,inputs,models)
    with patch.object(legacy,'load_models',load),patch.object(legacy,'call_method',call):
        return legacy.measure_latency(frozen,angles,mask,config,destination)
