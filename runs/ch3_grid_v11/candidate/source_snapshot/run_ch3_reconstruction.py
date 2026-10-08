"""New Chapter 3 structured-angle experiment, with frozen historical runs intact."""
import argparse
import json
from pathlib import Path
import traceback

import torch

from beamnas.common import save_json, setup, sha256
from beamnas.restored_objective import prepare_population, restored_physical_objective, population_component_mse
from pilot_original_modules import load_data, context, snapshot
from search_original_modules import stage
from train_original_modules_controls import choose_control


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--data', default='data/ch3_grid_good_v11')
    p.add_argument('--task', choices=['search', 'matched_mse', 'fixed_dnnabf_modules', 'random'], required=True)
    p.add_argument('--reference-search')
    p.add_argument('--protocol', default='configs/ch3_grid_v11.json')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    manifest = json.loads((Path(args.data)/'manifest.json').read_text())
    assert manifest['complete'] and manifest['protocol'] == protocol
    s, t = protocol['search'], protocol['training']
    args.family, args.seed = s['family'], s['seed']
    args.batch_size, args.search_epochs = s['batch_size'], s['epochs']
    args.warmup_epochs, args.retrain_epochs = s['warmup_epochs'], t['epochs']
    args.selection_metric = 'mean_sinr_gap_db'
    assert s['architecture_lr'] == .0003 and s['batch_size'] == t['batch_size'] and s['seed'] == t['seed']
    if args.smoke:
        args.search_epochs, args.retrain_epochs, args.warmup_epochs, args.batch_size = 2, 2, 1, 32
    out = Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    loss_fn = population_component_mse if args.task == 'matched_mse' else restored_physical_objective
    original = json.loads(Path('configs/original_modules_v4.json').read_text())
    cfg = {'id': protocol['id'], 'optimizer': t['optimizer'], 'data': protocol['physics'],
           'module_changes': original['module_changes'], 'scope': protocol['scope_note']}
    cfg['module_changes'].update(loss=loss_fn.__name__, supervision='offline population MVDR')
    setup(args.seed)
    assert torch.cuda.is_available()
    hashes = snapshot(out, __file__)
    for source in [args.protocol, 'pilot_original_modules.py', 'search_original_modules.py', 'train_original_modules_controls.py']:
        hashes[source] = sha256(source)
        dest = out/'source_snapshot'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(source).read_bytes())
    save_json(out/'source_hashes.json', hashes)
    reference = None
    if args.reference_search:
        reference = json.loads((Path(args.reference_search)/'config.json').read_text())
        assert reference['experiment_protocol'] == protocol
    if args.task == 'search':
        space = {key: s[key] for key in ['width', 'max_depth', 'depths', 'ff_multipliers']}
        if args.smoke:
            space.update(max_depth=4, depths=[2, 4])
        genotype = None
    elif args.task == 'matched_mse':
        assert reference is not None
        args.family, space = reference['arguments']['family'], reference['space']
        genotype = json.loads((Path(args.reference_search)/'search/genotype.json').read_text())
    else:
        args.family, space, genotype = choose_control(args.task, reference, args.seed)
    save_json(out/'config.json', {'arguments': vars(args), 'space': space, 'genotype': genotype,
        'protocol': cfg, 'experiment_protocol': protocol, 'loss': loss_fn.__name__,
        'data_manifest_sha256': sha256(Path(args.data)/'manifest.json'),
        'test_accessed': False})
    try:
        save_json(out/'status.json', {'state': 'preparing', 'test_accessed': False})
        limit = 96 if args.smoke else None
        train = prepare_population(load_data(args.data, 'train', limit))
        architecture = prepare_population(load_data(args.data, 'architecture_validation', limit)) if args.task == 'search' else None
        selection = load_data(args.data, 'selection_validation', limit)
        reference_values, info = context(selection)
        save_json(out/'reference.json', info)
        if args.task == 'search':
            genotype = stage(args, cfg, space, train, architecture, selection, reference_values, out/'search', loss_fn=loss_fn)
        stage(args, cfg, space, train, None, selection, reference_values, out/'retrain', genotype, loss_fn=loss_fn)
        assert all(sha256(source) == value for source, value in hashes.items())
        save_json(out/'status.json', {'state': 'complete', 'source_hashes_verified': True, 'test_accessed': False})
    except Exception:
        save_json(out/'status.json', {'state': 'failed', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
