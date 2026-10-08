"""Serve the frozen v11 network with compiled batch-one CUDA Graph inference.

JSONL mode loads/captures once and accepts multiple changing AOA requests.
Each line is a list [desired, interference_1, ...] or {"angles": [...]}. The
caller supplies known AOA; no signal acquisition or AOA estimation is included.
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext

import torch

from beamnas.common import sha256
from beamnas.gpu_runtime import complex_output, load_frozen_runtime


def parse_request(values):
    values = [float(v) for v in values]
    if not 4 <= len(values) <= 9:
        raise ValueError('Expected one desired and 3--8 interference AOAs')
    if not all(-90 <= v <= 90 and v.is_integer() for v in values):
        raise ValueError('AOAs must lie on the trained integer-degree grid [-90,90]')
    if len(set(values)) != len(values):
        raise ValueError('AOAs must be distinct')
    values = [values[0],*sorted(values[1:])]
    angles = torch.zeros((1,9),device='cuda',dtype=torch.float64)
    mask = torch.zeros((1,9),device='cuda',dtype=torch.bool)
    angles[0,:len(values)] = torch.tensor(values,device='cuda',dtype=torch.float64)
    mask[0,:len(values)] = True
    return values,angles,mask


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='runs/ch3_grid_v11/candidate/retrain/best.pt')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--angles',help='Comma-separated desired AOA then interference AOAs')
    group.add_argument('--jsonl',help='JSONL file, or - to read standard input')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    digest = sha256(args.checkpoint)
    runtime = None
    if args.angles:
        source = nullcontext([json.dumps([float(v) for v in args.angles.split(',')])])
    else:
        source = nullcontext(sys.stdin) if args.jsonl == '-' else open(args.jsonl)
    with source as lines:
        for line in lines:
            if not line.strip():continue
            request = json.loads(line)
            values,angles,mask = parse_request(request['angles'] if isinstance(request,dict) else request)
            setup_seconds = None
            if runtime is None:
                start = time.perf_counter()
                runtime = load_frozen_runtime(args.checkpoint,angles,mask)
                setup_seconds = time.perf_counter()-start
            weights = complex_output(runtime(angles,mask))[0].cpu()
            print(json.dumps({'angles_deg':values,'weights_real':weights.real.tolist(),
                              'weights_imag':weights.imag.tolist(),'checkpoint_sha256':digest,
                              'execution':'FP32 network, FP64 output rotation; compiled CUDA Graph; batch one',
                              'one_time_setup_seconds':setup_seconds}),flush=True)


if __name__=='__main__':main()
