"""Whole-test-set time / N, sequential batch one, frozen v11 checkpoints.

Primary references remain standard implementations, as requested by the user.
An optimized MVDR reference is also reported to separate deployment gains from
algorithmic claims. All complete complex outputs are materialized within timing.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from beamnas.common import save_json, sha256
from beamnas.gpu_runtime import (BatchOneGraph, FrozenNetwork, RealFrozenNetwork,
                                MVDRRuntime, complex_output)
from beamnas.original_modules import ModuleNetwork, sinr_db
from beamnas.pointing_boundary import nearest_peak_directions
from beamnas.strict_dnnabf import unpack
from benchmark_restored_latency import require_idle_gpu
from ch3_strict_baseline import load_models


def statistics(values):
    return {'mean':float(np.mean(values)), 'std':float(np.std(values, ddof=1)),
            'min':float(np.min(values)), 'max':float(np.max(values)), 'rounds':len(values)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path('configs/ch3_gpu_deployment_v1.json').read_text())
    require_idle_gpu()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha256(cfg['checkpoint']) == cfg['checkpoint_sha256']
    prior = json.loads(Path('results/ch3_grid_v11/FINAL.json').read_text())
    strict_item = prior['models']['strict_DNNABF']
    strict = load_models(strict_item, 'cuda')
    for model in strict.values():model.requires_grad_(False)
    saved = torch.load(cfg['checkpoint'], map_location='cuda', weights_only=False)
    model = ModuleNetwork(saved['family'], saved['space'], saved['genotype']).cuda().eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    source_paths = [Path(__file__).name, 'beamnas/gpu_runtime.py', 'configs/ch3_gpu_deployment_v1.json',
                    'beamnas/dense_search.py','beamnas/original_modules.py','beamnas/strict_dnnabf.py',
                    'beamnas/refined.py','beamnas/pointing.py','beamnas/pointing_boundary.py']
    hashes = {p:sha256(p) for p in source_paths}
    for p in source_paths:
        destination = out/'source_snapshot'/p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,destination)
    # Freeze implementations before reading test arrays. Use development input
    # to compile and capture; no test-based model/implementation selection.
    with np.load(cfg['development_data']) as dev:
        capture_a = torch.tensor(dev['angles'][:1], device='cuda', dtype=torch.float64)
        capture_m = torch.tensor(dev['mask'][:1], device='cuda', dtype=torch.bool)
    start = time.perf_counter()
    compiled_nn = torch.compile(RealFrozenNetwork(model), fullgraph=True, dynamic=False)
    nn_graph = BatchOneGraph(compiled_nn, capture_a, capture_m)
    setup = {'NN_compile_and_capture_seconds':time.perf_counter()-start}
    start = time.perf_counter()
    compiled_mvdr = torch.compile(MVDRRuntime('solve_ex'), fullgraph=True, dynamic=False)
    mvdr_graph = BatchOneGraph(compiled_mvdr, capture_a, capture_m)
    setup['MVDR_supplement_compile_and_capture_seconds'] = time.perf_counter()-start
    methods = {
        'ours_standard':FrozenNetwork(model),
        'ours_optimized':nn_graph,
        'MVDR_standard':MVDRRuntime('solve'),
        'MVDR_optimized_supplement':mvdr_graph,
    }
    # DNNABF's per-K model is selected from the already-known input length.
    # Avoid a device-to-host .item() merely to rediscover this input metadata.
    def strict_call(a,m,k):return unpack(strict[k](a[:, :k+1].float()).double())
    frozen = {'checkpoint_sha256':cfg['checkpoint_sha256'],
              'strict_checkpoint_hashes':{k:v['sha256'] for k,v in strict_item['checkpoints'].items()},
              'source_hashes':hashes,'config':cfg,'setup':setup,
              'selection':'NN compiled graph and fastest development MVDR compiled solve; standard references as primary'}
    save_json(out/'freeze.json', frozen)
    with np.load(cfg['test_data']) as data:
        aa, mm = data['angles'].copy(), data['mask'].copy()
    n = len(aa)
    assert n == cfg['confirmation']['all_test_scenes']
    ks = (mm.sum(-1)-1).astype(int)
    assert all(np.sum(ks == k) == 2000 for k in range(3,9))
    angles = torch.tensor(aa, device='cuda', dtype=torch.float64)
    mask = torch.tensor(mm, device='cuda', dtype=torch.bool)
    inputs = [(angles[i:i+1], mask[i:i+1], int(ks[i])) for i in range(n)]
    output = torch.empty((n,12), device='cuda', dtype=torch.complex128)
    output_views = [output[i:i+1] for i in range(n)]
    info_out = torch.zeros(n, device='cuda', dtype=torch.int32)
    info_views = [info_out[i:i+1] for i in range(n)]
    names = [*methods,'DNNABF_standard']
    for name in names:
        for _ in range(cfg['confirmation']['warmup']):
            if name == 'DNNABF_standard':
                for k in range(3,9):strict_call(*inputs[int(np.flatnonzero(ks==k)[0])])
            else:methods[name](*inputs[0][:2])
    torch.cuda.synchronize()
    rows, predictions = [], {}
    rng = np.random.default_rng(cfg['confirmation']['seed'])
    for repeat in range(cfg['confirmation']['rounds']):
        for name in rng.permutation(names):
            require_idle_gpu()
            output.fill_(complex(float('nan'),float('nan')))
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for i,(a,m,k) in enumerate(inputs):
                value = strict_call(a,m,k) if name == 'DNNABF_standard' else methods[name](a,m)
                output_views[i].copy_(complex_output(value))
                if name == 'MVDR_optimized_supplement':info_views[i].copy_(value[1])
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns()-start)/1e9
            if name == 'MVDR_optimized_supplement':assert int(info_out.abs().max()) == 0
            actual = output.cpu().numpy().copy()
            assert np.isfinite(actual).all()
            if name in predictions:np.testing.assert_array_equal(actual,predictions[name])
            else:predictions[name] = actual
            row = {'scope':'GPU_resident_whole_test','method':str(name),'repeat':repeat,
                   'total_seconds':elapsed,'samples':n,'mean_ms':elapsed*1000/n}
            rows.append(row)
            print(json.dumps(row),flush=True)
            save_json(out/'progress.json',{'stage':'whole_test_timing','rows':rows})
    # Supplemental serial request/response timing: each CPU input produces a
    # completed CPU output. All three primary methods include both transfers.
    cpu_a = torch.tensor(aa,dtype=torch.float64).pin_memory()
    cpu_m = torch.tensor(mm,dtype=torch.bool).pin_memory()
    host_out = torch.empty((n,12),dtype=torch.complex128,pin_memory=True)
    input_a = torch.empty((1,9),device='cuda',dtype=torch.float64)
    input_m = torch.empty((1,9),device='cuda',dtype=torch.bool)
    host_inputs = [(cpu_a[i:i+1],cpu_m[i:i+1],int(ks[i])) for i in range(n)]
    host_outputs = [host_out[i:i+1] for i in range(n)]
    for repeat in range(3):
        for name in rng.permutation(['ours_optimized','MVDR_standard','DNNABF_standard']):
            require_idle_gpu()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for i,(a,m,k) in enumerate(host_inputs):
                input_a.copy_(a,non_blocking=True)
                input_m.copy_(m,non_blocking=True)
                value = strict_call(input_a,input_m,k) if name == 'DNNABF_standard' else methods[name](input_a,input_m)
                host_outputs[i].copy_(complex_output(value),non_blocking=False)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns()-start)/1e9
            np.testing.assert_array_equal(host_out.numpy(),predictions[name])
            row = {'scope':'CPU_input_GPU_compute_CPU_output','method':str(name),'repeat':repeat,
                   'total_seconds':elapsed,'samples':n,'mean_ms':elapsed*1000/n}
            rows.append(row)
            print(json.dumps(row),flush=True)
    parity = {}
    for name,reference in [('ours_optimized','ours_standard'),('MVDR_optimized_supplement','MVDR_standard')]:
        diff = float(np.max(abs(predictions[name]-predictions[reference])))
        limit = cfg['parity']['network_max_abs_weight_difference' if name.startswith('ours') else 'mvdr_max_abs_weight_difference']
        assert diff < limit,(name,diff,limit)
        parity[name] = {'max_abs_weight_difference':diff}
    metrics = {}
    for name in names:
        weights = predictions[name]
        ss = []
        for begin in range(0,n,512):
            ss.extend(sinr_db(torch.tensor(weights[begin:begin+512],device='cuda'),
                             angles[begin:begin+512],mask[begin:begin+512]).cpu().numpy())
        ss = np.asarray(ss)
        peaks = nearest_peak_directions(weights,aa[:,0])
        metrics[name] = {'mean_sinr_db':float(ss.mean()),'main_mae_deg':float(peaks['main_error_deg'].mean()),
                         'by_k':{str(k):{'mean_sinr_db':float(ss[ks==k].mean()),
                                        'main_mae_deg':float(peaks['main_error_deg'][ks==k].mean())} for k in range(3,9)}}
        np.savez_compressed(out/(name+'.npz'),weights=weights,sinr_db=ss,**peaks)
        if name in ['ours_optimized','MVDR_optimized_supplement']:
            reference = 'ours_standard' if name.startswith('ours') else 'MVDR_standard'
            with np.load(out/(reference+'.npz')) as prior_values:
                delta_sinr = float(np.max(abs(ss-prior_values['sinr_db'])))
                delta_main = float(np.max(abs(peaks['main_error_deg']-prior_values['main_error_deg'])))
            assert delta_sinr < cfg['parity']['max_abs_sinr_difference_db']
            assert delta_main <= cfg['parity']['max_abs_main_error_difference_deg']
            parity[name].update(max_abs_sinr_difference_db=delta_sinr,
                                max_abs_main_error_difference_deg=delta_main)
        print('METRICS',name,json.dumps(metrics[name]),flush=True)
    summary = {}
    for scope in sorted({row['scope'] for row in rows}):
        summary[scope] = {}
        for name in names:
            values = [row['total_seconds'] for row in rows if row['scope']==scope and row['method']==name]
            if values:
                summary[scope][name] = {'total_seconds':statistics(values),
                                       'per_sample_ms':statistics([v*1000/n for v in values])}
    primary = summary['GPU_resident_whole_test']
    speedups = {name:primary[name]['total_seconds']['mean']/primary['ours_optimized']['total_seconds']['mean']
                for name in primary if name != 'ours_optimized'}
    assert all(sha256(p)==h for p,h in hashes.items())
    report = {'complete':True,'scope':'execution optimization of frozen weights; not new training or architecture search',
              'samples':n,'batch_size':1,'freeze':frozen,'test_sha256':sha256(cfg['test_data']),
              'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(),
                             'platform':platform.platform(),'threads':torch.get_num_threads(),'TF32':False},
              'timing':summary,'speedup_vs_ours_optimized':speedups,'observations':rows,
              'metrics':metrics,'parity':parity,
              'interpretation':'Primary comparison optimizes ours and retains standard reference implementations by user instruction. Optimized MVDR is supplemental. Whole-set time/N is amortized sequential batch-one latency, not the same as isolated-call latency or batch inference.',
              'limitations':['DNNABF checkpoint reconstructs the original network form; its accuracy is not a reproduction of the published numerical results.',
                             'Inputs are known AOA and fixed powers; no signal acquisition or AOA estimation is included.',
                             'Compilation/capture/model loading is a one-time setup excluded from steady-state totals.',
                             'Existing test set used for deployment regression only; no new independent generalization claim.']}
    save_json(out/'report.json',report)
    print('COMPLETE',json.dumps(speedups),flush=True)


if __name__=='__main__':main()
