"""Complete the missing official-comparator scopes without altering old fits."""
from __future__ import annotations
import importlib.util,gc
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,SEEDS,load,write_json,sha256_file,savecsv
from .adapters import native_ref,cleanup
BASE=ROOT/'outputs/external_methods_audit_2026-11-16_19-40-44'

def script(ctx,path,name):
    ctx.protect(path);spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def frunet(ctx):
    from flow_matching_refiner.data import load_reference
    r=script(ctx,BASE/'run_official_frunet_2026-11-16_20-19-15.py','completion_frunet');official=r.official_components(BASE/'FR-UNet');ref=load_reference(ctx.dataset,42,ctx.out/'loader','cuda');choices=[]
    for lr in r.LRS:
        ctx.status('FR_UNet_validation_pilot',learning_rate=lr,epochs=120)
        choices.append(r.train_fit(official,ref,ctx.out/'pilots'/f'lr_{lr:g}',ctx.tag,91011,lr,120,'cuda',1))
    write_json(ctx.out/f'all3_validation_pilots_{ctx.tag}.json',dict(decisions=choices,scope='Matches CHASE and HRF pilot-only FR-UNet scope. No five-seed final FR-UNet study existed on CHASE.'))
    return ctx.done(dict(pilot_fits=3,epochs=120,new_test_inference=False))

def segflow(ctx):
    from flow_matching_refiner.structure_metrics_all import metrics
    ref=native_ref(ctx);r=script(ctx,BASE/'run_official_segflow_2026-11-16_20-19-15.py','completion_segflow');official=r.official_components(BASE/'FMS2');tag='2026-11-16_20-03-21';src=ROOT/f'outputs/reviewer_completion_2026-11-16_19-37-22/official_segflow_retinal_adaptation_{tag}'
    rows=[];records=[]
    for seed in SEEDS:
        folder=src/ctx.dataset/'final'/f'seed_{seed}';choice=load(folder/f'selection_locked_{tag}.json');cp=folder/choice['checkpoint'];assert ctx.protect(cp)==choice['checkpoint_sha256'];model=r.make_model(official).cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['state']);model.eval().requires_grad_(False);dest=ctx.out/f'seed_{seed}';dest.mkdir(exist_ok=True)
        for n in ref.manifest['test']:
            ctx.status('frozen_SegFlow_test_inference',seed=seed,image=n);d=ref.data[n];path=dest/f'{n}_prediction_{ctx.tag}.npz'
            if path.exists():
                with np.load(path,allow_pickle=False) as z:p=z['raw'].copy()
            else:
                with torch.inference_mode():state=official.generate(model,r.image_inputs(d['rgb'][None]).cuda(),num_steps=10,integration_method='euler');p=r.probability_from_state(state).cpu()[0].numpy().copy()
                np.savez_compressed(path,raw=p,baseline=d['coarse'].numpy(),ground_truth=d['g'].numpy(),fov=d['fov'].numpy())
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['raw'],p) and np.array_equal(z['ground_truth'],d['g'].numpy()) and np.array_equal(z['fov'],d['fov'].numpy());mm=metrics(p[0],z['ground_truth'][0],z['baseline'][0],1.,z['fov'][0])
            rows.append(dict(dataset=ctx.dataset,seed=seed,image=n,method='SegFlow',**mm));records.append(dict(dataset=ctx.dataset,seed=seed,image=n,path=str(path),sha256=sha256_file(path),checkpoint=str(cp),checkpoint_sha256=choice['checkpoint_sha256'],selection=choice))
        del model;cleanup()
    savecsv(ctx.out/f'per_image_metrics_{ctx.tag}.csv',rows);write_json(ctx.out/f'prediction_manifest_{ctx.tag}.json',dict(records=records,training_performed=False))
    return ctx.done(dict(new_training_fits=0,existing_trained_models=5,test_prediction_files=len(records)))
