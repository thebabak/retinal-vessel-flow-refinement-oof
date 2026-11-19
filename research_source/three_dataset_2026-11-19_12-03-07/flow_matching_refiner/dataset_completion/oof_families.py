"""Missing HRF strict-OOF families, retaining original validation rules."""
from __future__ import annotations
import copy
from pathlib import Path
from .common import ROOT,SEEDS,load,write_json,sha256_file
from .adapters import module,native_ref,completed_fit,lock,variable_budget,supervision
from .sources import read_strict

def run_family(ctx):
    ref=native_ref(ctx);ref,routes,_=read_strict(ctx,ref);sv=supervision(ref)
    if ctx.phase=='fully_oof':
        r=module(ctx,'run_fully_oof_flow',variable_budget())
        from flow_matching_refiner.run_native_patch_flow_v2 import evaluate
        evaluator=evaluate;configs=r.CONFIGS
    elif ctx.phase=='topology':
        r=module(ctx,'run_topology_oof_flow',variable_budget());evaluator=r.evaluate;configs=r.CONFIGS
        if not (ctx.out/'diagnosis'/f'diagnosis_{ctx.tag}.json').exists():
            diag=module(ctx,'diagnose_topology_oof',[('all10_choices_locked_before_test_','all5_choices_before_test_'),('OOF source fits12images with nested baselines fit8; deployment anchor fits16. Training-size shift remains.','Four-group source models fit fewer training images than the deployed full anchor; group sizes are in the nested lineage manifests. Training-size shift remains.')])
            diag.OLD=ctx.root/'fully_oof'/ctx.dataset;diag.OT=ctx.tag
            diag.diagnose(ref,ctx.out/'diagnosis',ctx.tag,routes,ctx.status)
    elif ctx.phase=='retained':
        r=module(ctx,'run_retained_oof_flow',variable_budget()+[('all10_choices_locked_before_test_','all5_choices_before_test_')]);evaluator=r.evaluate
        r.OLD=ctx.root/'topology'/ctx.dataset;r.OT=ctx.tag
        from flow_matching_refiner.run_dice_guarded_oof_flow import CONFIGS
        warm=copy.deepcopy(r.WARM_CONFIGS);prior=r.prior_decision(ctx.dataset,42)
        for cfg in warm:cfg['cap']=prior['config']['cap']
        configs=[dict(c,family='signed') for c in CONFIGS]+warm
    else:raise ValueError(ctx.phase)
    def train(p,seed,anchor,cfg,epochs,pilot):
        args=[ref,p,ctx.tag,seed,anchor,cfg,epochs,routes]
        if ctx.phase!='fully_oof':args.append(sv)
        return r.train(*args,ctx.status,pilot)
    pilots=[]
    for cfg in configs:
        p=ctx.out/'pilots'/cfg['name'];pilots.append(completed_fit(p,ctx.tag,lambda:train(p,91011,42,cfg,40,True),40))
    if ctx.phase=='fully_oof':key=lambda d:(d['selected']['validation_dice']-d['anchor_validation_dice'],d['raw']['validation_dice']-d['anchor_validation_dice'])
    elif ctx.phase=='topology':key=lambda d:(d['selected']['structural_score'],d['raw']['structural_score'])
    else:key=lambda d:(d['selected']['full_gain_retained'],d['selected']['structural_score'],d['raw']['structural_score'])
    cfg=copy.deepcopy(max(pilots,key=key)['config']);cfg.pop('train_beta',None)
    lock(ctx,'validation_configuration',[cfg],pilots=pilots,HRF_only_validation_extension=True)
    ds=[]
    for seed in SEEDS:
        p=ctx.out/'final'/f'seed_{seed}';ds.append(completed_fit(p,ctx.tag,lambda:train(p,seed,seed,cfg,120,False),120))
    lp=lock(ctx,'all5_choices_before_test',ds);entries=[]
    for d in ds:
        p=Path(d['selected_path']).parent/f'inference_{ctx.tag}.json'
        entries.extend(load(p) if p.exists() else evaluator(ref,d,ctx.out,ctx.tag,ctx.status))
    write_json(ctx.out/f'prediction_manifest_{ctx.tag}.json',dict(records=entries,selection_lock=str(lp),selection_lock_sha256=sha256_file(lp)))
    return ctx.done(dict(pilot_fits=len(configs),final_fits=5,test_prediction_files=len(entries),strict_OOF=True))
