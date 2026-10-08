"""AOA-only inference for the original-physics, restored-loss neural model."""
import argparse
import json
from unittest.mock import patch

import torch

from beamnas.common import sha256
from beamnas.original_modules import ModuleNetwork, features, physical_weights


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--angles', required=True, help='Desired AOA first, followed by 3 to 8 interference AOAs in degrees')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    values = [float(value) for value in args.angles.split(',')]
    assert 4 <= len(values) <= 9
    assert all(-90 <= angle <= 90 and angle.is_integer() for angle in values)
    assert len(set(values)) == len(values),'AOAs must be distinct, as in the training distribution'
    values = [values[0],*sorted(values[1:])]
    angles = torch.zeros((1,9),device=args.device,dtype=torch.float64)
    mask = torch.zeros((1,9),device=args.device,dtype=torch.bool)
    angles[0,:len(values)] = torch.tensor(values,device=args.device,dtype=torch.float64)
    mask[0,:len(values)] = True
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    model = ModuleNetwork(saved['family'],saved['space'],saved['genotype']).to(args.device).eval()
    model.load_state_dict(saved['state_dict'],strict=True)
    with patch('torch.linalg.solve',side_effect=AssertionError('solve in neural inference')), \
         patch('torch.linalg.inv',side_effect=AssertionError('inverse in neural inference')), \
         patch('torch.linalg.eigh',side_effect=AssertionError('eigh in neural inference')):
        canonical = model(features(angles.float(),mask),mask)
        weights = physical_weights(canonical.double(),angles)[0].cpu()
    print(json.dumps({'checkpoint_sha256':sha256(args.checkpoint),'angles_deg':values,
        'device':args.device,'weights_real':weights.real.tolist(),'weights_imag':weights.imag.tolist(),
        'parameters':sum(p.numel() for p in model.parameters()),
        'inference':'AOA features, neural forward and output projection; no covariance or matrix solver'}))


if __name__ == '__main__':
    main()
