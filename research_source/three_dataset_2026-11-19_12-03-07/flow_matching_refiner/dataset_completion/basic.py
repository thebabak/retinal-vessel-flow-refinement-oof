"""Missing preliminary and native-flow phases, dispatched in isolated processes."""
from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,SEEDS,load,write_json,sha256_file,savecsv,csvrows
from .adapters import module,native_ref,completed_fit,lock,variable_budget,cleanup

def track(ctx,r):
    original=r.write_json
    def write(path,value):
        original(path,value)
        if Path(path).name==f'status_{ctx.tag}.json' and isinstance(value,dict) and 'stage' in value:
            ctx.status(value['stage'],**{k:v for k,v in value.items() if k not in ('stage','updated_at','status','pid','phase','root','output','tag')})
    r.write_json=write

def encoded(ctx):
    r=module(ctx,'run');track(ctx,r)
    args=argparse.Namespace(epochs=100,batch_size=4,steps=[20],solver='euler',sigma=.1,dice_weight=.5,lr=1e-4,weight_decay=.001,device='cuda',sampling_seeds=list(SEEDS))
    ctx.protect(ROOT/'flow_matching_refiner/run.py');ctx.status('encoded_flow_single_training')
    results=r.run_job(args,ctx.out,ctx.tag,ctx.dataset,42)
    return ctx.done(dict(training_fits=1,sampling_seeds=list(SEEDS),results=results))

def pixel(ctx):
    from flow_matching_refiner.data import load_reference
    ref=load_reference(ctx.dataset,42,ctx.out/'loader','cuda');ctx.protect(ref.baseline_checkpoint);ctx.protect(ref.autoencoder_checkpoint)
    b=ref.LNet_RA(in_channels=1).cuda();b.load_state_dict(torch.load(ref.baseline_checkpoint,map_location='cuda',weights_only=False)['state']);b.eval()
    with torch.inference_mode():
        for d in ref.data.values():d['coarse']=b(d['x'][None].cuda()).cpu()[0].clone()
    del b;cleanup()
    # Extend the exact CHASE recipe to valid retinal support; no artificial FOV.
    r=module(ctx,'run_pixel',[
      ('reference.score(p[index], data[name]["g"])','reference.score(p[index], data[name]["g"], data[name]["fov"]>.5)'),
      ('reference.score(data[n]["coarse"], data[n]["g"])','reference.score(data[n]["coarse"], data[n]["g"], data[n]["fov"]>.5)'),
      ('[sample["coarse"], sample["rgb"], sample["g"]]','[sample["coarse"], sample["rgb"], sample["g"], sample["fov"].float()]'),
      ('coarse, rgb, truth = [','coarse, rgb, truth, valid = ['),('for i in range(3)]','for i in range(4)]'),
      ('truth, torch.ones_like(truth))','truth, valid)'),
      ('reference.score(x,sample["g"])','reference.score(x,sample["g"],sample["fov"]>.5)'),
      ('subject=sample["subject"]','subject=sample.get("subject",name)'),
      ('fov=np.ones_like(sample["g"].numpy(),dtype=bool)','fov=sample["fov"].numpy().astype(bool)'),
      ('torch.from_numpy(saved["ground_truth"]))','torch.from_numpy(saved["ground_truth"]),torch.from_numpy(saved["fov"]))')])
    track(ctx,r)
    decisions=[]
    for seed in SEEDS:
        p=ctx.out/f'refiner_seed_{seed}';ctx.status('pixel_training',seed=seed)
        decisions.append(completed_fit(p,ctx.tag,lambda:r.train_one(ref,ctx.out,ctx.tag,seed,150,8,10),150))
    lp=lock(ctx,'all5_choices_before_test',decisions,conditional_on_one_baseline=True);rows=[]
    for d in decisions:
        p=ctx.out/f'refiner_seed_{d["refiner_seed"]}'/f'per_image_test_{ctx.tag}.csv'
        rows.extend(csvrows(p) if p.exists() else r.evaluate_one(ref,ctx.out,ctx.tag,d,8)[0])
    savecsv(ctx.out/f'all_per_image_{ctx.tag}.csv',rows)
    return ctx.done(dict(final_fits=5,epochs=150,test_records=len(rows),selection_lock=str(lp)))

def native_flow(ctx):
    from flow_matching_refiner.run_native_patch_flow import train_one,evaluate_one
    from flow_matching_refiner.native_patch_model import training_center_pools
    ctx.protect(ROOT/'flow_matching_refiner/run_native_patch_flow.py');ref=native_ref(ctx)
    pools={n:training_center_pools(ref.data[n]['n_g'],ref.data[n]['n_fov']) for n in ref.manifest['train']};ds=[]
    def status(stage,**extra):ctx.status(stage,**extra)
    status.state=ctx.state
    for seed in SEEDS:
        p=ctx.out/'native_patch_flow'/f'seed_{seed}';ctx.status('native_flow_training',seed=seed)
        ds.append(completed_fit(p,ctx.tag,lambda:train_one(ref,ctx.out,ctx.tag,seed,pools,status),120))
        ctx.state['completed_fits']=len(ds);ctx.status('training_progress',fits_completed=len(ds),fits_expected=5)
    lp=lock(ctx,'all5_choices_before_test',ds);rows=[]
    for d in ds:
        p=ctx.out/'native_patch_flow'/f'seed_{d["refiner_seed"]}'/f'test_metrics_{ctx.tag}.csv'
        if p.exists():rows.extend(csvrows(p))
        else:
            rr,entries=evaluate_one(ref,ctx.out,ctx.tag,d,1.,status);rows.extend(rr)
    savecsv(ctx.out/f'all_per_image_{ctx.tag}.csv',rows)
    return ctx.done(dict(final_fits=5,epochs=120,test_records=len(rows),selection_lock=str(lp)))

def anchored(ctx):
    ref=native_ref(ctx);r=module(ctx,'run_native_patch_flow_v2',variable_budget()+[("f'{ref.dataset}_{OT}/native_patch_flow/seed_", "f'{ref.dataset}/native_patch_flow/seed_"),('teacher_forced_endpoint_patch_Dice=2*tp/(2*tp+fp+fn)','teacher_forced_endpoint_patch_Dice=(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)')]);pilots=[]
    r.OLD=ctx.root/'native_flow';r.OT=ctx.tag
    if not (ctx.out/f'validation_diagnosis_{ctx.tag}.json').exists():r.diagnose(ref,ctx.out,ctx.tag,ctx.status)
    for cfg in r.CONFIGS:
        p=ctx.out/'pilots'/cfg['name'];pilots.append(completed_fit(p,ctx.tag,lambda:r.train(ref,p,ctx.tag,91011,42,cfg,40,ctx.status,True),40))
    cfg=max(pilots,key=lambda d:(d['selected']['validation_dice']-d['anchor_validation_dice'],d['raw']['validation_dice']-d['anchor_validation_dice']))['config']
    lock(ctx,'validation_configuration',[cfg],pilots=pilots,extension_uses_HRF_validation_only=True)
    ds=[]
    for seed in SEEDS:
        p=ctx.out/'final'/f'seed_{seed}';ds.append(completed_fit(p,ctx.tag,lambda:r.train(ref,p,ctx.tag,seed,seed,cfg,120,ctx.status),120))
    lp=lock(ctx,'all5_choices_before_test',ds);entries=[]
    for d in ds:
        p=Path(d['selected_path']).parent/f'inference_{ctx.tag}.json'
        entries.extend(load(p) if p.exists() else r.evaluate(ref,d,ctx.out,ctx.tag,ctx.status))
    write_json(ctx.out/f'prediction_manifest_{ctx.tag}.json',dict(records=entries,selection_lock=str(lp)))
    return ctx.done(dict(pilot_fits=4,final_fits=5,test_prediction_files=len(entries)))
