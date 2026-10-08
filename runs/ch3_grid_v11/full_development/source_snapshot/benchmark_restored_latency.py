"""Matched batch-one latency for the declared original-physics pipeline."""
import csv
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from beamnas.common import save_json
from beamnas.original_modules import ModuleNetwork,features,physical_weights,covariance
from beamnas.strict_dnnabf import steering,mvdr


def require_idle_gpu():
    result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                            check=True,capture_output=True,text=True)
    others = [int(line.strip()) for line in result.stdout.splitlines()
              if line.strip().isdigit() and int(line.strip()) != os.getpid()]
    assert not others,('Other GPU jobs are active; timing is not valid',others)


def make_inputs(angles,mask,device,snapshot_seed=2026100823):
    aa = np.asarray(angles,dtype=float)
    mm = np.asarray(mask,dtype=bool)
    a = np.exp(1j*np.pi*np.sin(np.deg2rad(aa))[...,None]*np.arange(12))
    rng = np.random.default_rng(snapshot_seed)
    def cn(shape):
        return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)
    x = a[:,1:].transpose(0,2,1) @ (cn((len(aa),8,1024))*mm[:,1:,None]*np.sqrt(1000))
    x += cn((len(aa),12,1024))
    at = torch.tensor(aa,device=device,dtype=torch.float64)
    mt = torch.tensor(mm,device=device)
    ad = steering(at[:,0])
    r = covariance(at,mt)
    xt = torch.tensor(x,device=device,dtype=torch.complex128)
    return [(at[i:i+1],mt[i:i+1],ad[i:i+1],r[i:i+1],xt[i:i+1]) for i in range(len(aa))]


def load_models(frozen,device):
    models = {}
    for name,item in frozen.items():
        saved = torch.load(item['checkpoint'],map_location=device,weights_only=False)
        model = ModuleNetwork(saved['family'],saved['space'],saved['genotype']).to(device).eval()
        model.load_state_dict(saved['state_dict'],strict=True)
        models[name] = model
    return models


def call_method(name,inputs,models):
    angles,mask,desired,r,x = inputs
    if name == 'population_covariance_given':
        return mvdr(r,desired)
    if name == 'population_AOA_to_weights':
        return mvdr(covariance(angles,mask),steering(angles[:,0]))
    if name == 'sample_snapshots_given':
        return mvdr(x@x.mH/x.shape[-1],desired)
    prediction = models[name](features(angles.float(),mask),mask)
    return physical_weights(prediction.double(),angles)


@torch.no_grad()
def measure_latency(frozen,angles,mask,config,destination):
    require_idle_gpu()
    destination = Path(destination)
    torch.set_num_threads(config['cpu_threads'])
    counts = mask.sum(-1)-1
    chosen = np.concatenate([np.flatnonzero(counts==k)[:config['scenes_per_k']] for k in range(3,9)])
    assert len(chosen) == 6*config['scenes_per_k']
    methods = ['population_covariance_given','population_AOA_to_weights','sample_snapshots_given',*frozen]
    observations,summary = [],{}
    for device in config['devices']:
        require_idle_gpu()
        inputs = make_inputs(angles[chosen],mask[chosen],device)
        models = load_models(frozen,device)
        for name in methods:
            for _ in range(config['warmup_calls_per_method']):
                call_method(name,inputs[0],models)
        order = [(name,i,repeat) for repeat in range(config['passes']) for i in range(len(chosen)) for name in methods]
        np.random.default_rng(config['order_seed']).shuffle(order)
        for name,i,repeat in order:
            if device == 'cuda':torch.cuda.synchronize()
            start = time.perf_counter_ns()
            call_method(name,inputs[i],models)
            if device == 'cuda':torch.cuda.synchronize()
            duration = (time.perf_counter_ns()-start)/1e6
            observations.append({'device':device,'method':name,'scene':int(chosen[i]),
                                 'repeat':repeat,'milliseconds':duration})
        require_idle_gpu()
        summary[device] = {}
        for name in methods:
            values = np.array([row['milliseconds'] for row in observations if row['device']==device and row['method']==name])
            summary[device][name] = {'mean_ms':float(values.mean()),'median_ms':float(np.median(values)),
                'p95_ms':float(np.quantile(values,.95)),'observations':len(values)}
        del inputs,models
        if device == 'cuda':torch.cuda.empty_cache()
    with (destination/'latency_observations.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(observations[0]))
        writer.writeheader()
        writer.writerows(observations)
    result = {'summary':summary,'config':config,'test_scene_indices':chosen.tolist(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),
        'platform':platform.platform(),'snapshot_seed':2026100823,
        'precision':'FP32 neural core; FP64 phase rotation, covariance and MVDR',
        'no_other_GPU_compute_process_at_start_and_end':True}
    save_json(destination/'latency.json',result)
    return result
