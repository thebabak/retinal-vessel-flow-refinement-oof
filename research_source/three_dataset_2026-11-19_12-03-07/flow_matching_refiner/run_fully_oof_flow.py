"""Fit bounded Flow using strict nested-cross-fitted O3 patch inputs only."""
from __future__ import annotations
import argparse,csv,gc,json,random,shutil,traceback
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.run_native_patch_flow_v2 import DS,SEEDS,BETAS,CONFIGS,model_for,score,anchor_path,AT,O3

def audit_routes(cross):
    tag=cross.name[-19:];done=json.loads((cross/f'completed_strict_crossfit_{tag}.json').read_text())
    assert done['nested_baseline_fits']==12 and done['O3_fold_fits']==40 and done['strict_indirect_leakage_exclusions_verified']
    m=json.loads((cross/f'OOF_anchor_manifest_{tag}.json').read_text());lock=json.loads(Path(m['global_lock_path']).read_text())
    assert sha256_file(m['global_lock_path'])==m['global_lock_sha256'] and len(m['records'])==160
    routes={}
    for e in m['records']:
        assert e['image'] in e['held_images'] and e['image'] not in e['fit_images']+e['validation_images']
        fp=json.loads(Path(e['fold_protocol']).read_text());held=set(e['held_images'])
        for row in fp['training_input_lineage']:
            assert not held&set(row['baseline_fit_images']) and row['image'] not in row['baseline_fit_images']
        assert sha256_file(e['path'])==e['sha256'] and sha256_file(e['anchor_model_path'])==e['anchor_model_sha256']
        routes[e['dataset'],e['seed'],e['image']]=e
    return m,lock,routes

def train(ref,out,tag,seed,anchor_seed,cfg,epochs,routes,status,pilot=False):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
    from flow_matching_refiner.native_patch_model import sample_training_patch,training_center_pools,project_residual_512,apply_residual,ErrorSelectiveModel
    from flow_matching_refiner.fully_oof_flow import objective,routed_anchor
    out.mkdir(parents=True,exist_ok=False);model=model_for(ref.dataset,anchor_seed,seed,cfg['cap'],ref.device)
    source_models={};routing={}
    for n in ref.manifest['train']:
        e=routes[ref.dataset,anchor_seed,n];routing[n]=e
        with np.load(e['path'],allow_pickle=False) as z:assert np.array_equal(z['coarse512'],ref.data[n]['coarse'].numpy())
        if e['outer_fold'] not in source_models:
            network=ErrorSelectiveModel(False).to(ref.device)
            network.load_state_dict(torch.load(e['anchor_model_path'],map_location=ref.device,weights_only=True)['state']);source_models[e['outer_fold']]=network.eval().requires_grad_(False)
    assert len(source_models)==4
    optimizer=torch.optim.AdamW(model.field.parameters(),lr=1e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=random.Random(seed+141001);trng=torch.Generator(device=ref.device).manual_seed(seed+20261119)
    pools={n:training_center_pools(ref.data[n]['n_g'],ref.data[n]['n_fov']) for n in ref.manifest['train']}
    fitting=[False]
    def reject_full_anchor_during_fitting(module,args):
        if fitting[0]:raise RuntimeError('Full O3 anchor was accessed during Flow fitting')
    hook=model.anchor.register_forward_pre_hook(reject_full_anchor_during_fitting)
    def validation():
        values={b:[] for b in BETAS}
        for n in ref.manifest['validation']:
            d=ref.data[n];a,f,_=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device)
            a=project_residual_512(a,ref.dataset);f=project_residual_512(f,ref.dataset)
            for beta in BETAS:values[beta].append(score(ref,apply_residual(d['coarse'],a+beta*f),n))
        return {b:float(np.mean(v)) for b,v in values.items()}
    initial=validation();assert len(set(initial.values()))==1
    old=json.loads((anchor_path(ref.dataset,anchor_seed).parent/f'selection_locked_{AT}.json').read_text());assert abs(initial[0.]-old['raw']['validation_dice'])<1e-12
    best=dict(epoch=0,beta=0.,validation_dice=initial[0.]);raw=dict(epoch=None,beta=1.,validation_dice=-1.)
    sp=out/f'selected_field_{tag}.pt';rp=out/f'raw_field_{tag}.pt';torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),sp)
    with (out/f'training_{tag}.csv').open('x',newline='',encoding='utf-8') as tf,(out/f'validation_{tag}.csv').open('x',newline='',encoding='utf-8') as vf,(out/f'patch_draws_{tag}.csv').open('x',newline='',encoding='utf-8') as pf:
        tw,vw,pw=csv.writer(tf),csv.writer(vf),csv.writer(pf)
        tw.writerow(['epoch','loss','fm','bce','dice','cldice','gate','preservation','lr','patches','optimizer_updates'])
        vw.writerow(['epoch','beta','validation_dice']);vw.writerow([0,0,initial[0.]])
        pw.writerow(['epoch','patch_index','image','centre_y','centre_x','top','left','used_vessel_centre','rotation90','flip_horizontal','flip_vertical','OOF_fold','OOF_anchor_checkpoint_sha256'])
        for epoch in range(1,epochs+1):
            status('pilot_training' if pilot else 'final_training',dataset=ref.dataset,seed=seed,epoch=epoch,epochs=epochs,config=cfg['name'])
            model.train();names=ref.manifest['train']*4;rng.shuffle(names);aggregate=np.zeros(7);seen=0;updates=0;lr=optimizer.param_groups[0]['lr'];fitting[0]=True
            for start in range(0,len(names),16):
                samples=[];fold_ids=[]
                for idx,n in enumerate(names[start:start+16],start=start):
                    d=ref.data[n];crop,meta=sample_training_patch(d['n_coarse'],d['n_rgb'],d['n_g'],d['n_fov'],rng=rng,center_pools=pools[n]);k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5
                    fields=[torch.rot90(crop[q],k,(-2,-1)) for q in ('coarse','rgb','gt','fov')]
                    if fh:fields=[t.flip(-1) for t in fields]
                    if fv:fields=[t.flip(-2) for t in fields]
                    e=routing[n];samples.append(fields);fold_ids.append(e['outer_fold'])
                    pw.writerow([epoch,idx,n,meta['centre_y'],meta['centre_x'],meta['top'],meta['left'],meta['used_vessel_centre'],k,fh,fv,e['outer_fold'],e['anchor_model_sha256']])
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0,len(samples),8):
                    batch=samples[micro:micro+8];c,rgb,g,fov=[torch.stack([s[j] for s in batch]).to(ref.device) for j in range(4)]
                    a=routed_anchor(source_models,c,rgb,fold_ids[micro:micro+8])
                    time=torch.rand(len(c),device=ref.device,generator=trng);time=torch.where(torch.rand(len(c),device=ref.device,generator=trng)<.25,torch.zeros_like(time),time)
                    loss,parts=objective(model,c,rgb,g,fov,time,cfg['fm_weight'],a);assert bool(torch.isfinite(loss));(loss*len(batch)/len(samples)).backward()
                    aggregate+=np.array([loss.item(),*(parts[k].item() for k in ('fm','bce','dice','cldice','gate','preservation'))])*len(batch);seen+=len(batch)
                torch.nn.utils.clip_grad_norm_(model.field.velocity_net.parameters(),1.);torch.nn.utils.clip_grad_norm_(model.field.error_net.parameters(),1.);optimizer.step();updates+=1
            fitting[0]=False;schedule.step();assert seen==64 and updates==4
            tw.writerow([epoch,*(aggregate/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:
                status('pilot_validation' if pilot else 'final_validation',dataset=ref.dataset,seed=seed,epoch=epoch,epochs=epochs,config=cfg['name']);grid=validation()
                if grid[1.]>raw['validation_dice']+1e-12:raw=dict(epoch=epoch,beta=1.,validation_dice=grid[1.]);torch.save(dict(state=model.field.state_dict(),selection=raw,config=cfg),rp)
                beta=max(BETAS,key=lambda b:grid[b])
                if grid[beta]>best['validation_dice']+1e-12:best=dict(epoch=epoch,beta=beta,validation_dice=grid[beta]);torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),sp)
                for b in BETAS:vw.writerow([epoch,b,grid[b]])
                vf.flush();print(ref.dataset,'PILOT' if pilot else 'FINAL',seed,cfg['name'],epoch,'best',best,'raw',raw,flush=True)
    hook.remove()
    decision=dict(dataset=ref.dataset,seed=seed,anchor_seed=anchor_seed,pilot=pilot,config=cfg,epochs=epochs,selected=best,raw=raw,anchor_validation_dice=initial[0.],selected_path=str(sp),raw_path=str(rp),anchor_path=str(anchor_path(ref.dataset,anchor_seed)),anchor_sha256=sha256_file(anchor_path(ref.dataset,anchor_seed)),selected_sha256=sha256_file(sp),raw_sha256=sha256_file(rp),locked_at=stamp(),strict_OOF_training=True,full_anchor_fitting_access_rejected=True,training_OOF_sources={n:dict(fold=e['outer_fold'],model_path=e['anchor_model_path'],model_sha256=e['anchor_model_sha256'],fit=e['fit_images'],validation=e['validation_images'],fold_protocol=e['fold_protocol']) for n,e in routing.items()})
    write_json(out/f'selection_locked_{tag}.json',decision)
    del model,optimizer,schedule,pools,source_models;gc.collect();torch.cuda.empty_cache();return decision

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--crossfit',type=Path,required=True);ap.add_argument('--output',type=Path);args=ap.parse_args()
    if not args.execute:print('Inert; --execute required. Eightvalidationpilots +tenfinalfields; no HRF.');return
    cross=args.crossfit.resolve();manifest,crosslock,routes=audit_routes(cross);ct=cross.name[-19:]
    out=(args.output or ROOT/'outputs'/f'fully_oof_flow_{stamp()}').resolve();out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,crossfit_source=str(cross),datasets=list(DS),seeds=list(SEEDS),completed_pilots=0,completed_final_fits=0,HRF_started=False)
    def status(stage,**extra):state.update(stage=stage,updated_at=stamp(),**extra);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_fully_oof_flow.json',state)
    status('protecting_inputs');protected=json.loads((cross/f'protected_before_including_loaded_sources_{ct}.json').read_text());assert all(sha256_file(p)==h for p,h in protected.items())
    for p in cross.rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    write_json(out/f'protected_before_{tag}.json',protected)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):
        d=out/'source'/f'{p.stem}_{tag}.py';d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
    write_json(out/f'plan_{tag}.json',dict(**state,configs=CONFIGS,pilot_seed=91011,pilot_epochs=40,final_epochs=120,
       training='For each augmented nativepatch, originalbaselineOOFcoarse +matchingseed/fold O3model excluding wholeheldfold and allindirectnestedbaseline dependencies. GT/FOV supervise train-onlysampling, FMtargets and losses; sourcepredictors and inference receive noGT/FOV. FullO3forward hook rejects accidental fitting access.',
       evaluation='Unchanged original fullO3anchor forvalidation/test as deployable full-data estimator; model architecture/objective/2Eulersteps matchv2. FullO3exactreplay checked.',
       validation='Fresh4configpilots per dataset; globalbest meanselectedgain overO3 on externalvalidation; rawgain tiebreak. All8pilots beforeHP lock and all10finalchoices before newtest.',
       selection='Primaryvalidationselectedbeta0,.25,.5,1; beta0is exactoldO3identity explicitlycounted. Learnedrawbeta1 separate. No testselection.',
       statistics='PrimaryselectedminusO3:2datasets xDice/clDice/thinrecall=6-testHolmfamily, t/W andseed/cohortunitclasses separate. Baseline/priorv2 descriptive. Five trainingseedmeans/sampleSD, t95CI df4, subject4/image20sensitivity, cohortbootstrap256/50000 seed20261117.',
       limits='Previously inspected fixedcohorts; sharedbaselinegenerations/splits, notfiveindependentwholebaselinepipelines. OuterO3modelsfit12images withinnerbaselinefit8; deployedO3fits16 withbaselineOOFfit12. Repeatedexternalvalidationselection and unknownDRIVEpatient/provenance remain limitations.',old_results_unchanged=True))
    try:
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        from flow_matching_refiner.run_native_patch_flow_v2 import evaluate
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available();write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),training='FP32',TF32=False))
        refs={};pilots=[]
        for ds in DS:
            status('loading_audited_data',dataset=ds);ref,audit=load_native_dataset(ds,out/f'{ds}_{tag}/input_audit','cuda');refs[ds]=ref
            for p,h in audit['source_sha256_before'].items():
                if p in protected:assert protected[p]==h
                protected[p]=h
            write_json(out/f'{ds}_{tag}/input_metadata_{tag}.json',dict(dataset=ds,split=ref.manifest,baseline_OOF=True,O3_OOF=True,audit=audit['output_audit']))
            for cfg in CONFIGS:pilots.append(train(ref,out/f'{ds}_{tag}/pilots/{cfg["name"]}',tag,91011,42,cfg,40,routes,status,True));status('pilot_complete',completed_pilots=len(pilots))
        ranking=[]
        for cfg in CONFIGS:
            rr=[p for p in pilots if p['config']==cfg];assert len(rr)==2
            ranking.append(dict(config=cfg,mean_selected_validation_gain=sum(p['selected']['validation_dice']-p['anchor_validation_dice'] for p in rr)/2,mean_raw_validation_gain=sum(p['raw']['validation_dice']-p['anchor_validation_dice'] for p in rr)/2))
        cfg=max(ranking,key=lambda r:(r['mean_selected_validation_gain'],r['mean_raw_validation_gain']))['config']
        hp=out/f'hyperparameters_locked_before_final_{tag}.json';write_json(hp,dict(locked_at=stamp(),selected=cfg,ranking=ranking,pilots=pilots))
        decisions=[]
        for ds in DS:
            for seed in SEEDS:decisions.append(train(refs[ds],out/f'{ds}_{tag}/final/seed_{seed}',tag,seed,seed,cfg,120,routes,status));status('final_fit_complete',completed_final_fits=len(decisions))
        lock=out/f'all10_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=decisions,hyperparameter_lock_sha256=sha256_file(hp),strict_crossfit_lock_path=manifest['global_lock_path'],strict_crossfit_lock_sha256=manifest['global_lock_sha256']));lh=sha256_file(lock)
        entries=[]
        for ds in DS:
            for d in decisions:
                if d['dataset']==ds:entries.extend(evaluate(refs[ds],d,out,tag,status))
        assert len(entries)==140 and sha256_file(lock)==lh
        write_json(out/f'prediction_manifest_{tag}.json',dict(records=entries,global_lock_path=str(lock),global_lock_sha256=lh))
        status('checking_previous_artifacts');after={p:sha256_file(p) for p in protected};assert after==protected
        write_json(out/f'protected_before_including_loaded_sources_{tag}.json',protected);write_json(out/f'protected_after_{tag}.json',after)
        write_json(out/f'completed_fully_oof_flow_{tag}.json',dict(completed_at=stamp(),crossfit_source=str(cross),pilot_fits=8,final_fits=10,prediction_files=140,strict_OOF_training=True,identity_selections={ds:sum(d['dataset']==ds and d['selected']['beta']==0 for d in decisions) for ds in DS},old_files_unchanged=True,HRF_started=False,lock_sha256=lh))
        status('fully_oof_flow_complete');print('FULLY_OOF_FLOW_COMPLETE',str(out),flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
