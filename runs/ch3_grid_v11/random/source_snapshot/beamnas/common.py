import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(seed, threads=4):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def complex_mse(prediction, target):
    return 2 * (prediction - target).square().mean()


def packed(w):
    return torch.cat((w.real, w.imag), dim=-1).float()


def unpacked(w):
    return torch.complex(w[..., :w.shape[-1] // 2], w[..., w.shape[-1] // 2:])
