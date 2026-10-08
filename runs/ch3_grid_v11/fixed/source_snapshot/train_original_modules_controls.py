"""Budget-matched MSE controls for v4; no architecture or test-based resampling."""
import argparse
import json
import random
import traceback
from pathlib import Path

import torch

from beamnas.common import save_json, setup, sha256
from pilot_original_modules import load_data, context, snapshot
from search_original_modules import stage


def choose_control(kind, reference, seed):
    if kind=='fixed_dnnabf_modules':
        return kind,{}, {'family':kind,'hidden_widths':[2048,1024,1024,1024],
            'activation':'PReLU','selection':'fixed_before_training'}
    if kind=='fixed_transformer':
        space={'width':96,'max_depth':12,'depths':[2,4,6,8,10,12],
               'heads':[1,2,4,8],'ff_multipliers':[2,4,8]}
        layers=[{'heads':4,'position':'sinusoidal','ff':4,
            'attention_skip':'identity','ff_skip':'identity','long_skip':'none'} for _ in range(4)]
        return 'transformer',space,{'depth':4,'layers':layers,'selection':'fixed_before_training'}
    assert reference is not None,'Random control requires a declared search space.'
    family,space=reference['arguments']['family'],reference['space']
    rng=random.Random(seed+9000)
    depth=rng.choice(space['depths'])
    layers=[]
    for index in range(depth):
        long_options=['none'] if index==0 else ['none','input'] if index==1 else ['none','input','two_back']
        layer={'ff':rng.choice(space['ff_multipliers']),'long_skip':rng.choice(long_options)}
        if family=='dense':
            layer.update(activation=rng.choice(['gelu','silu','tanh']),skip=rng.choice(['identity','linear']))
        else:
            layer.update(heads=rng.choice(space['heads']),
                position=rng.choice(['sinusoidal','learned','relative','rope','alibi','fourier']),
                attention_skip=rng.choice(['identity','linear','depthwise_conv']),
                ff_skip=rng.choice(['identity','linear','depthwise_conv']))
        layers.append(layer)
    return family,space,{'depth':depth,'layers':layers,'selection':'one_random_draw_before_training',
        'architecture_seed':seed+9000}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--control',choices=['fixed_dnnabf_modules','fixed_transformer','random'],required=True)
    p.add_argument('--reference-search')
    p.add_argument('--seed',type=int,default=117)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--retrain-epochs',type=int,default=160)
    p.add_argument('--selection-metric',choices=['mse','mean_sinr_gap_db'],default='mse')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    out=Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    if args.smoke:args.retrain_epochs,args.batch_size=2,32
    config=json.loads(Path('configs/original_modules_v4.json').read_text())
    reference=json.loads((Path(args.reference_search)/'config.json').read_text()) if args.reference_search else None
    args.family,space,genotype=choose_control(args.control,reference,args.seed)
    setup(args.seed)
    assert torch.cuda.is_available()
    hashes=snapshot(out,__file__)
    extra_sources=['pilot_original_modules.py','search_original_modules.py']
    if args.selection_metric=='mean_sinr_gap_db':extra_sources.append('configs/sinr_selection_v4b.json')
    for source in extra_sources:
        hashes[source]=sha256(source)
        (out/'source_snapshot'/source).write_bytes(Path(source).read_bytes())
    save_json(out/'source_hashes.json',hashes)
    save_json(out/'config.json',{'arguments':vars(args),'space':space,'genotype':genotype,
        'protocol':config,'loss':'plain_component_MSE','test_accessed':False,
        'checkpoint_selection_metric':args.selection_metric,
        'data_manifest_sha256':sha256(Path(args.data)/'manifest.json'),
        'reference_search_config_sha256':sha256(Path(args.reference_search)/'config.json') if reference else None})
    try:
        train=load_data(args.data,'train',96 if args.smoke else None)
        selection=load_data(args.data,'selection_validation')
        context_values,info=context(selection)
        save_json(out/'reference.json',info)
        stage(args,config,space,train,None,selection,context_values,out/'retrain',genotype)
        assert all(sha256(s)==h for s,h in hashes.items())
        save_json(out/'status.json',{'state':'control_retraining_complete','source_hashes_verified':True,
            'test_accessed':False,'overall_goal_complete':False})
    except Exception:
        save_json(out/'status.json',{'state':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
