"""Development-only comparison of equivalent batch-one execution variants."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from beamnas.common import save_json, sha256
from beamnas.original_modules import ModuleNetwork
from beamnas.gpu_runtime import (BatchOneGraph, FrozenNetwork, RealFrozenNetwork,
                                MVDRRuntime, complex_output)
from benchmark_restored_latency import require_idle_gpu


def summarize(values):
    return {'mean_ms':float(np.mean(values)), 'median_ms':float(np.median(values)),
            'p95_ms':float(np.quantile(values, .95)), 'n':len(values)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path('configs/ch3_gpu_deployment_v1.json').read_text())
    assert sha256(cfg['checkpoint']) == cfg['checkpoint_sha256']
    require_idle_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(cfg['checkpoint'], map_location='cuda', weights_only=False)
    model = ModuleNetwork(saved['family'], saved['space'], saved['genotype']).cuda().eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    with np.load(cfg['development_data']) as data:
        counts = data['mask'].sum(-1)-1
        chosen = np.concatenate([np.flatnonzero(counts == k)[:cfg['development']['scenes_per_k']] for k in range(3,9)])
        angles = torch.tensor(data['angles'][chosen], device='cuda', dtype=torch.float64)
        mask = torch.tensor(data['mask'][chosen], device='cuda', dtype=torch.bool)
    inputs = [(angles[i:i+1], mask[i:i+1]) for i in range(len(angles))]
    nn_fn = FrozenNetwork(model)
    mvdr_fn = MVDRRuntime('solve')
    references = {'NN':torch.cat([nn_fn(*x) for x in inputs]),
                  'MVDR':torch.cat([mvdr_fn(*x)[0] for x in inputs])}
    funcs = {'NN_eager':nn_fn, 'MVDR_eager':mvdr_fn}
    errors, setup = {}, {}
    specs = [('NN_graph', nn_fn), ('MVDR_solve_graph', MVDRRuntime('solve_ex')),
             ('MVDR_cholesky_graph', MVDRRuntime('cholesky')),
             ('MVDR_inverse_graph', MVDRRuntime('inverse'))]
    if args.compile:
        for name, function in [('NN_compiled_graph', RealFrozenNetwork(model)),
                               ('MVDR_solve_compiled_graph', MVDRRuntime('solve_ex')),
                               ('MVDR_cholesky_compiled_graph', MVDRRuntime('cholesky'))]:
            start = time.perf_counter()
            try:
                compiled = torch.compile(function, fullgraph=True, dynamic=False)
                compiled(*inputs[0])
                torch.cuda.synchronize()
                setup[name+'_compilation_seconds'] = time.perf_counter()-start
                specs.append((name, compiled))
            except Exception as exc:
                errors[name] = type(exc).__name__+': '+str(exc)[:2500]
                print('COMPILE_FAILED', name, errors[name], flush=True)
    parity = {}
    for name, function in specs:
        try:
            fn = BatchOneGraph(function, *inputs[0])
            setup[name] = fn.setup_seconds
            predictions = []
            all_info = []
            for x in inputs:
                value = fn(*x)
                predictions.append(complex_output(value).clone())
                if isinstance(value, tuple):all_info.append(value[1].clone())
            result = torch.cat(predictions)
            delta = float((result-references[name.split('_')[0]]).abs().max())
            limit = cfg['parity']['network_max_abs_weight_difference' if name.startswith('NN') else 'mvdr_max_abs_weight_difference']
            assert delta < limit, (delta, limit)
            if all_info:assert int(torch.cat(all_info).abs().max()) == 0
            parity[name] = {'max_abs_weight_difference':delta, 'solver_info_all_zero':True if all_info else None}
            funcs[name] = fn
            print('READY', name, json.dumps(parity[name]), flush=True)
        except Exception as exc:
            errors[name] = type(exc).__name__+': '+str(exc)[:2500]
            print('FAILED', name, errors[name], flush=True)
    for name, fn in funcs.items():
        for _ in range(cfg['development']['warmup']):fn(*inputs[0])
    torch.cuda.synchronize()
    jobs = [(name,i,repeat) for repeat in range(cfg['development']['passes'])
            for i in range(len(inputs)) for name in funcs]
    np.random.default_rng(cfg['development']['seed']).shuffle(jobs)
    observations = []
    for name, i, repeat in jobs:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        funcs[name](*inputs[i])
        torch.cuda.synchronize()
        observations.append((name,int(chosen[i]),repeat,(time.perf_counter_ns()-start)/1e6))
    require_idle_gpu()
    summary = {name:summarize([v[3] for v in observations if v[0] == name]) for name in funcs}
    import csv
    with (out/'observations.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method','development_scene','repeat','milliseconds'])
        writer.writerows(observations)
    report = {'scope':'development only; frozen weights; same precision', 'config':cfg,
              'source_hashes':{p:sha256(p) for p in [__file__,'beamnas/gpu_runtime.py','configs/ch3_gpu_deployment_v1.json']},
              'torch':torch.__version__,'GPU':torch.cuda.get_device_name(),
              'setup':setup,'parity':parity,'errors':errors,'timing':summary}
    save_json(out/'report.json',report)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':main()
