"""DARTS and from-scratch retraining with raw sample-MVDR supervision and MSE.

New modules and original array conventions are explicit. No independent test is
loaded or generated, and architecture parameters use a separate validation set.
"""
import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

import torch

from beamnas.common import save_json, setup, sha256
from beamnas.original_modules import ModuleNetwork
from beamnas.dense_search import export_inherited
from beamnas.train import set_grad
from pilot_original_modules import load_data, context, validate, snapshot


def export_module_network(source):
    g=source.genotype()
    dest=ModuleNetwork(source.family,source.space,g).to(next(source.parameters()).device)
    if source.family=='dense':
        dest.core=export_inherited(source.core)
    else:
        src, dst=source.core,dest.core
        dst.input.load_state_dict(src.input.state_dict())
        dst.norm.load_state_dict(src.norm.state_dict())
        dst.output.load_state_dict(src.output.state_dict())
        for old,new,choice in zip(src.blocks,dst.blocks,g['layers']):
            new.norm1.load_state_dict(old.norm1.state_dict())
            new.norm2.load_state_dict(old.norm2.state_dict())
            attention=old.attentions[source.space['heads'].index(choice['heads'])]
            new.attention.load_state_dict({k:attention.state_dict()[k] for k in new.attention.state_dict()})
            new.ffn.load_state_dict(old.ffns[source.space['ff_multipliers'].index(choice['ff'])].state_dict())
            options=['identity','linear','depthwise_conv']
            new.attention_skip.load_state_dict(old.attention_skips[options.index(choice['attention_skip'])].state_dict())
            new.ff_skip.load_state_dict(old.ff_skips[options.index(choice['ff_skip'])].state_dict())
    return dest


def checkpoint(path,value):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def batch_loss(prediction, data, indices, loss_fn=None):
    """Keep historical MSE exact; alternative objectives require an explicit callback."""
    if loss_fn is None:
        return torch.nn.functional.mse_loss(prediction, data[2][indices])
    if getattr(loss_fn, 'uses_population_context', False):
        return loss_fn(prediction, *(value[indices] for value in data[5:8]))
    return loss_fn(prediction, data[3][indices], data[1][indices])


def stage(args,cfg,space,train,architecture,selection,reference,out,genotype=None,
          model_factory=ModuleNetwork,exporter=export_module_network,loss_fn=None):
    out.mkdir()
    search=genotype is None
    setup(args.seed if search else args.seed+10000)
    model=model_factory(args.family,space,genotype).cuda()
    weights,alphas=model.weight_parameters(),model.architecture_parameters()
    assert bool(alphas)==search
    assert not set(map(id,weights)) & set(map(id,alphas))
    assert all(p.is_cuda for p in weights+alphas)
    oc=cfg['optimizer']
    opt=torch.optim.AdamW(weights,lr=oc['lr'],weight_decay=oc['weight_decay'])
    arch_opt=torch.optim.Adam(alphas,lr=.0003,betas=(.5,.999),weight_decay=.001) if search else None
    epochs=args.search_epochs if search else args.retrain_epochs
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=oc['cosine_min_lr'])
    wg=torch.Generator(device='cuda').manual_seed(args.seed+(1000 if search else 3000))
    ag=torch.Generator(device='cuda').manual_seed(args.seed+2000)
    best=float('inf')
    selection_metric=getattr(args,'selection_metric','mse')
    assert selection_metric in ['mse','mean_sinr_gap_db']
    loss_name='plain_component_MSE' if loss_fn is None else loss_fn.__name__
    weight_steps,architecture_steps=0,0
    initial_alphas=[p.detach().clone() for p in alphas]
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if not search:save_json(out/'genotype.json',genotype)
    for epoch in range(1,epochs+1):
        model.train()
        order=torch.randperm(len(train[0]),device='cuda',generator=wg)
        aorder=torch.randperm(len(architecture[0]),device='cuda',generator=ag) if search else None
        cursor=0
        total=0.
        arch_gradient_sum=0.
        epoch_arch_steps=0
        for idx in order.split(args.batch_size):
            set_grad(weights,True)
            set_grad(alphas,False)
            opt.zero_grad(set_to_none=True)
            loss=batch_loss(model(train[0][idx],train[1][idx]),train,idx,loss_fn)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weights,oc['gradient_clip'],error_if_nonfinite=True)
            opt.step()
            weight_steps+=1
            total+=float(loss.detach())*len(idx)
            if search and epoch>args.warmup_epochs:
                if cursor>=len(aorder):
                    aorder=torch.randperm(len(architecture[0]),device='cuda',generator=ag)
                    cursor=0
                ai=aorder[cursor:cursor+args.batch_size]
                cursor+=args.batch_size
                set_grad(weights,False)
                set_grad(alphas,True)
                arch_opt.zero_grad(set_to_none=True)
                aloss=batch_loss(model(architecture[0][ai],architecture[1][ai]),architecture,ai,loss_fn)
                assert torch.isfinite(aloss)
                aloss.backward()
                norm=torch.nn.utils.clip_grad_norm_(alphas,5.,error_if_nonfinite=True)
                arch_opt.step()
                arch_gradient_sum+=float(norm)
                epoch_arch_steps+=1
                architecture_steps+=1
        set_grad(weights,True)
        set_grad(alphas,True)
        metrics=validate(model,selection,reference)
        eligible=(not search) or epoch>args.warmup_epochs
        if eligible and metrics[selection_metric]<best:
            best=metrics[selection_metric]
            checkpoint(out/'best.pt',{'state_dict':model.state_dict(),'epoch':epoch,'validation':metrics,
                'family':args.family,'space':space,'genotype':genotype,'config':cfg,'arguments':vars(args),
                'model_class':type(model).__name__})
        train_key='train_mse' if loss_fn is None else 'train_objective'
        row={'epoch':epoch,train_key:total/len(train[0]),'loss':loss_name,'selection_validation':metrics,
            'weight_steps':weight_steps,'architecture_steps':architecture_steps,
            'mean_architecture_gradient_norm':arch_gradient_sum/max(1,epoch_arch_steps),
            'seconds':time.perf_counter()-start}
        if search:row.update(genotype=model.genotype(),probabilities=model.probabilities())
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        save_json(out/'status.json',{'state':'running','pid':os.getpid(),**row})
        if epoch%5==0 or args.smoke:
            print(json.dumps({k:v for k,v in row.items() if k not in ['genotype','probabilities']}),flush=True)
        schedule.step()
    checkpoint(out/'last.pt',{'state_dict':model.state_dict(),'epoch':epochs,'validation':metrics,
        'family':args.family,'space':space,'genotype':genotype,'config':cfg,'arguments':vars(args),
        'model_class':type(model).__name__})
    saved=torch.load(out/'best.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    assert validate(model,selection,reference)==saved['validation']
    summary={'scope':'development_validation_only','best_epoch':saved['epoch'],'validation':saved['validation'],
        'epochs':epochs,'weight_steps':weight_steps,'architecture_steps':architecture_steps,
        'training_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in weights),
        'architecture_parameters':sum(p.numel() for p in alphas),'gpu':torch.cuda.get_device_name(),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'checkpoint_sha256':sha256(out/'best.pt'),
        'from_scratch':not search,'test_accessed':False,'selection_metric':selection_metric,'loss':loss_name,
        'last_checkpoint_sha256':sha256(out/'last.pt')}
    if search:
        g=model.genotype()
        summary['architecture_l2_change']=float(sum((p.detach()-q).square().sum() for p,q in zip(alphas,initial_alphas)).sqrt())
        assert summary['architecture_l2_change']>0 and architecture_steps>0
        summary['inherited_discrete_validation']=validate(exporter(model),selection,reference)
        save_json(out/'genotype.json',g)
        save_json(out/'probabilities.json',model.probabilities())
    else:g=genotype
    save_json(out/'summary.json',summary)
    save_json(out/'status.json',{'state':'complete','checkpoint_reload_verified':True})
    del model,opt,arch_opt
    torch.cuda.empty_cache()
    return g


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--family',choices=['dense','transformer'],required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--seed',type=int,default=117)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--search-scenes',type=int,default=128000)
    p.add_argument('--search-epochs',type=int,default=60)
    p.add_argument('--retrain-epochs',type=int,default=160)
    p.add_argument('--warmup-epochs',type=int,default=5)
    p.add_argument('--selection-metric',choices=['mse','mean_sinr_gap_db'],default='mse')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    out=Path(args.out)
    assert not out.exists()
    out.mkdir(parents=True)
    cfg=json.loads(Path('configs/original_modules_v4.json').read_text())
    if args.smoke:
        args.search_scenes=96
        args.search_epochs=args.retrain_epochs=2
        args.warmup_epochs=1
        args.batch_size=32
    setup(args.seed)
    assert torch.cuda.is_available()
    hashes=snapshot(out,__file__)
    extra_sources=['pilot_original_modules.py']
    if args.selection_metric=='mean_sinr_gap_db':extra_sources.append('configs/sinr_selection_v4b.json')
    for helper in extra_sources:
        hashes[helper]=sha256(helper)
        (out/'source_snapshot'/helper).write_bytes(Path(helper).read_bytes())
    save_json(out/'source_hashes.json',hashes)
    if args.family=='dense':space={'width':args.width,'max_depth':16,'depths':[2,4,6,8,12,16],'ff_multipliers':[1,2,4]}
    else:space={'width':args.width,'max_depth':12,'depths':[2,4,6,8,10,12],'heads':[1,2,4,8],'ff_multipliers':[2,4,8]}
    if args.smoke:space.update(max_depth=4,depths=[2,4])
    metadata={'arguments':vars(args),'space':space,'protocol':cfg,'loss':'plain_component_MSE',
        'checkpoint_selection_metric':args.selection_metric,
        'architecture_optimizer':{'name':'Adam','lr':.0003,'betas':[.5,.999],'weight_decay':.001},
        'data_manifest_sha256':sha256(Path(args.data)/'manifest.json'),'test_accessed':False}
    save_json(out/'config.json',metadata)
    try:
        train=load_data(args.data,'train',96 if args.smoke else None)
        architecture=load_data(args.data,'architecture_validation')
        selection=load_data(args.data,'selection_validation')
        reference,info=context(selection)
        save_json(out/'reference.json',info)
        search_train=tuple(value[:args.search_scenes] for value in train)
        g=stage(args,cfg,space,search_train,architecture,selection,reference,out/'search')
        # Fresh initialization; the supernet checkpoint is not loaded into this model.
        stage(args,cfg,space,train,architecture,selection,reference,out/'retrain',g)
        assert all(sha256(s)==h for s,h in hashes.items())
        save_json(out/'status.json',{'state':'search_and_retraining_complete','source_hashes_verified':True,
            'test_accessed':False,'overall_goal_complete':False})
    except Exception:
        save_json(out/'status.json',{'state':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
