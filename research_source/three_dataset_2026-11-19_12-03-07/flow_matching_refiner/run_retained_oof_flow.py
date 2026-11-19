"""Validation-selected retained-structure Flow: signed from-scratch or warm field fine-tuning."""
from __future__ import annotations
import argparse,copy,csv,gc,json,random,shutil,traceback
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.run_native_patch_flow_v2 import DS,SEEDS,anchor_path,AT,O3
from flow_matching_refiner.run_fully_oof_flow import audit_routes
OLD=ROOT/'outputs/topology_oof_flow_2026-11-17_18-49-15';OT='2026-11-17_18-49-15'
SCREEN=ROOT/'outputs/dice_guarded_oof_flow_2026-11-17_19-31-34'
BETAS=(0.,.0625,.125,.25,.5,1.)
SIGNED=ROOT/'outputs/dice_guarded_oof_flow_2026-11-17_19-36-12';ST='2026-11-17_19-36-12'
WARM_CONFIGS=[dict(name=f'warm_lr{lr:g}_fp{fp:g}',family='warm',cap=2.,lr=lr,fp_weight=fp) for lr in (1e-5,3e-5) for fp in (.05,.2)]

def prior_decision(dataset,seed):
    return next(q for q in json.loads((OLD/f'all10_choices_locked_before_test_{OT}.json').read_text())['decisions'] if q['dataset']==dataset and q['seed']==seed)

def model_for(dataset,anchor_seed,field_seed,cap,device,cfg=None):
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    from flow_matching_refiner.topology_oof_flow import TopologyFlow
    from flow_matching_refiner.run_dice_guarded_oof_flow import model_for as signed_model
    if cfg and cfg.get('family')=='signed':return signed_model(dataset,anchor_seed,field_seed,cap,device)
    anchor=ErrorSelectiveModel(False).to(device);anchor.load_state_dict(torch.load(anchor_path(dataset,anchor_seed),map_location=device,weights_only=True)['state'])
    torch.manual_seed(field_seed+20261118);torch.cuda.manual_seed_all(field_seed+20261118)
    model=TopologyFlow(anchor,cap).to(device)
    prior=prior_decision(dataset,anchor_seed);kind='selected' if prior['selected']['beta']>0 else 'raw';path=prior[f'{kind}_path']
    assert sha256_file(path)==prior[f'{kind}_sha256']
    model.field.load_state_dict(torch.load(path,map_location=device,weights_only=True)['state'])
    return model

def train(ref,out,tag,seed,anchor_seed,cfg,epochs,routes,supervision,status,pilot=False):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
    from flow_matching_refiner.native_patch_model import sample_training_patch,training_center_pools,project_residual_512,apply_residual,ErrorSelectiveModel
    from flow_matching_refiner.fully_oof_flow import routed_anchor
    from flow_matching_refiner.topology_oof_flow import patch_supervision
    from flow_matching_refiner.dice_guarded_oof_flow import selection_score,feasible,retained,primary_key,raw_key
    from flow_matching_refiner.retained_oof_flow import objective as warm_objective
    from flow_matching_refiner.dice_guarded_oof_flow import objective as signed_objective
    from flow_matching_refiner.structure_metrics_all import metrics
    out.mkdir(parents=True,exist_ok=False);cfg=copy.deepcopy(cfg);warm=cfg['family']=='warm';prior=prior_decision(ref.dataset,anchor_seed)
    if warm:cfg['train_beta']=prior['selected']['beta'] if prior['selected']['beta']>0 else .25
    model=model_for(ref.dataset,anchor_seed,seed,cfg['cap'],ref.device,cfg);reference=copy.deepcopy(model).eval().requires_grad_(False) if warm else None;sources={};routing={}
    for n in ref.manifest['train']:
        e=routes[ref.dataset,anchor_seed,n];routing[n]=e
        with np.load(e['path'],allow_pickle=False) as z:assert np.array_equal(z['coarse512'],ref.data[n]['coarse'].numpy())
        if e['outer_fold'] not in sources:
            net=ErrorSelectiveModel(False).to(ref.device);net.load_state_dict(torch.load(e['anchor_model_path'],map_location=ref.device,weights_only=True)['state']);sources[e['outer_fold']]=net.eval().requires_grad_(False)
    assert len(sources)==4
    optimizer=torch.optim.AdamW(model.field.parameters(),lr=cfg.get('lr',1e-4),weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=random.Random(seed+141001);trng=torch.Generator(device=ref.device).manual_seed(seed+20261119);pools={n:training_center_pools(ref.data[n]['n_g'],ref.data[n]['n_fov']) for n in ref.manifest['train']}
    fitting=[False]
    def reject(module,args):
        if fitting[0]:raise RuntimeError('Full O3 access during fitting rejected')
    hook=model.anchor.register_forward_pre_hook(reject);reference_hook=reference.anchor.register_forward_pre_hook(reject) if warm else None;validation_records=[]
    def validation(epoch):
        values={b:[] for b in BETAS}
        for n in ref.manifest['validation']:
            d=ref.data[n];a,f,_=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device);a=project_residual_512(a,ref.dataset);f=project_residual_512(f,ref.dataset)
            for beta in BETAS:
                p=apply_residual(d['coarse'],a+beta*f);m=metrics(p.numpy()[0],d['g'].numpy()[0],d['coarse'].numpy()[0],1.,d['fov'].numpy()[0])
                q={k:m[k] for k in ('dice','precision','cldice','thin_skeleton_recall')};values[beta].append(q);validation_records.append(dict(epoch=epoch,beta=beta,image=n,**q))
        return {b:{k:float(np.mean([q[k] for q in mm])) for k in mm[0]} for b,mm in values.items()}
    initial=validation(0);base=initial[0.]
    if not warm:assert all(initial[b]==initial[0.] for b in BETAS)
    old=json.loads((anchor_path(ref.dataset,anchor_seed).parent/f'selection_locked_{AT}.json').read_text());assert abs(base['dice']-old['raw']['validation_dice'])<1e-12
    prior_path=OLD/f'all10_choices_locked_before_test_{OT}.json'
    prior=next(q for q in json.loads(prior_path.read_text())['decisions'] if q['dataset']==ref.dataset and q['seed']==anchor_seed)
    assert base==prior['anchor_validation_metrics'];previous_target=prior['selected']['validation_metrics']
    def choice(epoch,beta,m):return dict(epoch=epoch,beta=beta,validation_dice=m['dice'],validation_metrics=m,structural_score=selection_score(m,base),structure_feasible=feasible(m,base),quality_feasible=feasible(m,base,False),full_gain_retained=retained(m,base,previous_target))
    best=choice(0,0.,base);raw=choice(0,1.,initial[1.]) if warm else None;rawkey=raw_key(raw) if raw else None
    if warm:
        for b,m in initial.items():
            q=choice(0,b,m)
            if q['structure_feasible'] and primary_key(q)>primary_key(best):best=q
        if prior['selected']['beta']>0:assert initial[prior['selected']['beta']]==previous_target
    sp=out/f'selected_field_{tag}.pt';rp=out/f'raw_field_{tag}.pt';torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),sp)
    if warm:torch.save(dict(state=model.field.state_dict(),selection=raw,config=cfg),rp)
    parts_keys=('fm','bce','dice','eval_bce','eval_dice','cldice','thin','skeleton','dice_regret','fp_repair','preservation','centerline_drop','reference_structure','gate')
    with (out/f'training_{tag}.csv').open('x',newline='',encoding='utf-8') as tf,(out/f'validation_{tag}.csv').open('x',newline='',encoding='utf-8') as vf,(out/f'patch_draws_{tag}.csv').open('x',newline='',encoding='utf-8') as pf:
        tw,vw,pw=csv.writer(tf),csv.writer(vf),csv.writer(pf);tw.writerow(['epoch','loss',*parts_keys,'lr','patches','optimizer_updates']);vw.writerow(['epoch','beta','dice','precision','cldice','thin_skeleton_recall','structural_score','feasible','quality_feasible','full_gain_retained'])
        for b,m in initial.items():vw.writerow([0,b,*(m[k] for k in ('dice','precision','cldice','thin_skeleton_recall')),selection_score(m,base),feasible(m,base),feasible(m,base,False),retained(m,base,previous_target)])
        pw.writerow(['epoch','patch_index','image','centre_y','centre_x','top','left','used_vessel_centre','rotation90','flip_horizontal','flip_vertical','OOF_fold','OOF_anchor_checkpoint_sha256','eval_height','eval_width'])
        for epoch in range(1,epochs+1):
            status('pilot_training' if pilot else 'final_training',dataset=ref.dataset,seed=seed,epoch=epoch,epochs=epochs,config=cfg['name']);model.train();names=ref.manifest['train']*4;rng.shuffle(names);aggregate=np.zeros(1+len(parts_keys));seen=0;updates=0;lr=optimizer.param_groups[0]['lr'];fitting[0]=True
            for start in range(0,len(names),16):
                samples=[];folds=[];align=[]
                for idx,n in enumerate(names[start:start+16],start=start):
                    d=ref.data[n];crop,meta=sample_training_patch(d['n_coarse'],d['n_rgb'],d['n_g'],d['n_fov'],rng=rng,center_pools=pools[n]);extra,sk=patch_supervision(supervision[n],meta);k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5
                    fields=[torch.rot90(t,k,(-2,-1)) for t in [*(crop[q] for q in ('coarse','rgb','gt','fov')),sk]]
                    if fh:fields=[t.flip(-1) for t in fields]
                    if fv:fields=[t.flip(-2) for t in fields]
                    extra.update(rotation90=k,flip_horizontal=fh,flip_vertical=fv);e=routing[n];samples.append(fields);folds.append(e['outer_fold']);align.append(extra)
                    pw.writerow([epoch,idx,n,meta['centre_y'],meta['centre_x'],meta['top'],meta['left'],meta['used_vessel_centre'],k,fh,fv,e['outer_fold'],e['anchor_model_sha256'],extra['gt'].shape[-2],extra['gt'].shape[-1]])
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0,len(samples),8):
                    batch=samples[micro:micro+8];c,rgb,g,fov,sk=[torch.stack([s[j] for s in batch]).to(ref.device) for j in range(5)];a=routed_anchor(sources,c,rgb,folds[micro:micro+8]);time=torch.rand(len(c),device=ref.device,generator=trng);time=torch.where(torch.rand(len(c),device=ref.device,generator=trng)<.25,torch.zeros_like(time),time)
                    if warm:loss,parts=warm_objective(model,reference,c,rgb,g,fov,time,a,sk,align[micro:micro+8],cfg)
                    else:
                        loss,parts=signed_objective(model,c,rgb,g,fov,time,a,sk,align[micro:micro+8],cfg);parts['reference_structure']=loss.detach()*0
                    assert bool(torch.isfinite(loss));(loss*len(batch)/len(samples)).backward();aggregate+=np.array([loss.item(),*(parts[k].item() for k in parts_keys)])*len(batch);seen+=len(batch)
                torch.nn.utils.clip_grad_norm_(model.field.velocity_net.parameters(),1.);torch.nn.utils.clip_grad_norm_(model.field.error_net.parameters(),1.);optimizer.step();updates+=1
            fitting[0]=False;schedule.step();assert seen==64 and updates==4;tw.writerow([epoch,*(aggregate/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:
                status('pilot_validation' if pilot else 'final_validation',dataset=ref.dataset,seed=seed,epoch=epoch,epochs=epochs,config=cfg['name']);grid=validation(epoch)
                rr=choice(epoch,1.,grid[1.]);key=raw_key(rr)
                if rawkey is None or key>rawkey:raw=rr;rawkey=key;torch.save(dict(state=model.field.state_dict(),selection=raw,config=cfg),rp)
                for b,m in grid.items():
                    q=choice(epoch,b,m)
                    if q['structure_feasible'] and (q['full_gain_retained']>best['full_gain_retained'] or (q['full_gain_retained']==best['full_gain_retained'] and q['structural_score']>best['structural_score']+1e-12)):best=q;torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),sp)
                    vw.writerow([epoch,b,*(m[k] for k in ('dice','precision','cldice','thin_skeleton_recall')),q['structural_score'],q['structure_feasible'],q['quality_feasible'],q['full_gain_retained']])
                vf.flush();print(ref.dataset,'PILOT' if pilot else 'FINAL',seed,cfg['name'],epoch,'best',best['epoch'],best['beta'],best['structural_score'],'retained',best['full_gain_retained'],'raw',raw['structural_score'],flush=True)
    hook.remove()
    if reference_hook:reference_hook.remove()
    with (out/f'validation_per_image_{tag}.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(validation_records[0]));w.writeheader();w.writerows(validation_records)
    dec=dict(dataset=ref.dataset,seed=seed,anchor_seed=anchor_seed,pilot=pilot,config=cfg,epochs=epochs,selected=best,raw=raw,anchor_validation_metrics=base,previous_validation_target=previous_target,previous_target_source=str(prior_path),previous_target_sha256=sha256_file(prior_path),anchor_validation_dice=base['dice'],selected_path=str(sp),raw_path=str(rp),anchor_path=str(anchor_path(ref.dataset,anchor_seed)),anchor_sha256=sha256_file(anchor_path(ref.dataset,anchor_seed)),selected_sha256=sha256_file(sp),raw_sha256=sha256_file(rp),locked_at=stamp(),strict_OOF_training=True,full_anchor_fitting_access_rejected=True,training_OOF_sources={n:dict(fold=e['outer_fold'],model_path=e['anchor_model_path'],model_sha256=e['anchor_model_sha256'],fit=e['fit_images'],validation=e['validation_images'],fold_protocol=e['fold_protocol']) for n,e in routing.items()})
    dec['family']=cfg['family'];dec['loss_only_reference']=warm
    if warm:
        kind='selected' if prior['selected']['beta']>0 else 'raw';dec['warm_initialization']=dict(path=prior[f'{kind}_path'],sha256=prior[f'{kind}_sha256'],kind=kind,prior_selected_beta=prior['selected']['beta']);dec['selected_reuses_initial_field']=best['epoch']==0 and best['beta']>0
    write_json(out/f'selection_locked_{tag}.json',dec);del reference,model,optimizer,schedule,pools,sources;gc.collect();torch.cuda.empty_cache();return dec

def record_previous_targets(ds,out,tag):
    out.mkdir(parents=True,exist_ok=False);p=OLD/f'all10_choices_locked_before_test_{OT}.json'
    dd=[q for q in json.loads(p.read_text())['decisions'] if q['dataset']==ds]
    rr=[dict(seed=q['seed'],base=q['anchor_validation_metrics'],previous_selected=q['selected']['validation_metrics'],previous_Dice_change=q['selected']['validation_metrics']['dice']-q['anchor_validation_metrics']['dice']) for q in dd]
    write_json(out/f'diagnosis_{tag}.json',dict(dataset=ds,validation_only=True,source=str(p),sha256=sha256_file(p),records=rr,observations=['Prior allowed Dice drop0.0005; new guard permits no Dice drop.','Prior had positive thin recovery without an explicit normalized FP-repair term.','Prior auxiliary error head did not gate velocity; new signed growth/removal gates are actually applied.'],causal_scope='Combined candidate; no isolated causal or population guarantee.'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--crossfit',type=Path,default=ROOT/'outputs/strict_o3_crossfit_2026-11-17_03-16-57');args=ap.parse_args()
    if not args.execute:print('Inert. Requires --execute:8warm validation pilots,8retained signed pilots and10final fits; no HRF.');return
    cross=args.crossfit.resolve();manifest,crosslock,routes=audit_routes(cross);out=(args.output or ROOT/'outputs'/f'retained_oof_flow_{stamp()}').resolve();out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,crossfit_source=str(cross),datasets=list(DS),seeds=list(SEEDS),completed_pilots=0,reused_signed_pilots=8,completed_final_fits=0,HRF_started=False)
    def status(stage,**extra):state.update(stage=stage,updated_at=stamp(),**extra);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_retained_oof_flow.json',state)
    status('protecting_previous_files');protected=json.loads((OLD/f'protected_before_including_loaded_sources_{OT}.json').read_text());assert all(sha256_file(p)==h for p,h in protected.items())
    probe=Path(json.loads((ROOT/'tmp/current_calibration_probe.json').read_text())['output'])
    for folder in (OLD,SCREEN,SIGNED,probe):
        for p in folder.rglob('*'):
            if p.is_file():protected[str(p)]=sha256_file(p)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    write_json(out/f'protected_before_{tag}.json',protected)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):
        d=out/'source'/f'{p.stem}_{tag}.py';d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
    write_json(out/f'plan_{tag}.json',dict(**state,warm_configs=WARM_CONFIGS,pilot_seed=91011,pilot_epochs=40,final_epochs=120,
        method='Two validation-selected families. Signed gated field trained from scratch; warm fine-tuning of previous TopologyFlow field with training-loss-only frozen reference. Actual two-Euler Flow rollout, strictly nested OOF O3 source during all fitting, unchanged full O3 at validation/test. No GT inference input.',
        selection='Validation only: Dice>=O3 (tolerance1e-12), precision>=O3-.002, clDice/thin each>=O3. Prefer BOTH structural metrics>=max(O3,previous TopologyFlow selected); then maximize their mean gain. If impossible, flag partial or identity. beta0,.0625,.125,.25,.5,1. Epoch0 warm reuse distinguished from learned improvement. Raw beta1 is secondary only.',
        hyperparameters='Per-dataset ranking of4retained signed pilots (regret2/8 xstructure1/2) and4new warm pilots (lr1e-5/3e-5 xFPweight.05/.2), same pilot seed91011/anchor42/40epochs. Prefer full retention, then selected structural score, then raw score. Choices lock before10final120epoch fits and all10checkpoint/beta choices lock before new test predictions.',
        success_criteria='Audit only after global lock: mean Dice>=unchanged O3, clDice and thin recall each>=previous TopologyFlow selected. Matched-seed pass counts and paired uncertainty reported; no test-based reselection.',
        screening_sources=[str(SCREEN),str(SIGNED),str(probe)],screening_counts='Before current phase:4completed CHASE pilots+4epochs DRIVE rejected;8completed signed pilots+31logged epochs partial CHASE final. Validation calibration24grid points, no test. Reuse8completed signed pilots in current method-family selection. All previous candidates/partial fits preserved and excluded from ten final fits.',
        statistics='Selected-minus-O3 and selected-minus-previousTopology:2contrasts x2datasets x3metrics=12-test Holmfamily; seed/cohort and t/W separate. Five conditional field fits/sampleSD; paired seed t95CI, cohort sensitivity and exploratory bootstrap.',
        limitations='Warm starts are five matched field fine-tunes, not five new full pipelines. Combined family/recipe comparison not isolated Flow ablation. Previously seen tests, validation4images reused, source12/deployment16 shift, DRIVE patient/test-label provenance unverified.',old_files_unchanged=True))
    try:
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        from flow_matching_refiner.topology_oof_flow import prepare_supervision
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
        write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False))
        signed_hp=SIGNED/f'hyperparameters_locked_before_final_{ST}.json';pilots=copy.deepcopy(json.loads(signed_hp.read_text())['pilots']);assert len(pilots)==8
        for d in pilots:
            for k in ('selected','raw','anchor'):assert sha256_file(d[f'{k}_path'])==d[f'{k}_sha256']
            d['config']['family']='signed';d['family']='signed';d['reused_validation_pilot']=True
        refs={};supervisions={}
        for ds in DS:
            status('loading_audited_data',dataset=ds);ref,audit=load_native_dataset(ds,out/f'{ds}_{tag}/input_audit','cuda');refs[ds]=ref
            for p,h in audit['source_sha256_before'].items():
                if p in protected:assert protected[p]==h
                protected[p]=h
            supervisions[ds]=prepare_supervision(ref)
            write_json(out/f'{ds}_{tag}/input_metadata_{tag}.json',dict(dataset=ds,split=ref.manifest,strict_OOF=True,supervision_shape=next(iter(supervisions[ds].values()))['shape'],exact_native_to512_GT_matching=True))
            record_previous_targets(ds,out/f'{ds}_{tag}/diagnosis',tag)
            for cfg in WARM_CONFIGS:
                pilots.append(train(ref,out/f'{ds}_{tag}/pilots/{cfg['name']}',tag,91011,42,cfg,40,routes,supervisions[ds],status,True));status('pilot_complete',completed_pilots=len(pilots)-8)
        rank={};selected_cfg={}
        for ds in DS:
            rr=[q for q in pilots if q['dataset']==ds];assert len(rr)==8
            ranked=sorted(rr,key=lambda q:(q['selected']['full_gain_retained'],q['selected']['structural_score'],q['raw']['structural_score']),reverse=True)
            rank[ds]=[dict(config=q['config'],selected=q['selected'],raw=q['raw'],source_selected_sha256=q['selected_sha256']) for q in ranked];selected_cfg[ds]=copy.deepcopy(ranked[0]['config']);selected_cfg[ds].pop('train_beta',None)
        hp=out/f'hyperparameters_locked_before_final_{tag}.json';write_json(hp,dict(locked_at=stamp(),selected_by_dataset=selected_cfg,ranking_by_dataset=rank,pilots=pilots,reused_signed_source=str(signed_hp),reused_signed_source_sha256=sha256_file(signed_hp)));decisions=[]
        for ds in DS:
            for seed in SEEDS:
                decisions.append(train(refs[ds],out/f'{ds}_{tag}/final/seed_{seed}',tag,seed,seed,selected_cfg[ds],120,routes,supervisions[ds],status));status('final_fit_complete',completed_final_fits=len(decisions))
        lock=out/f'all10_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=decisions,hyperparameter_lock_sha256=sha256_file(hp),strict_crossfit_lock_path=manifest['global_lock_path'],strict_crossfit_lock_sha256=manifest['global_lock_sha256']));lh=sha256_file(lock);entries=[]
        for ds in DS:
            for d in decisions:
                if d['dataset']==ds:entries.extend(evaluate(refs[ds],d,out,tag,status))
        assert len(entries)==140 and sha256_file(lock)==lh;write_json(out/f'prediction_manifest_{tag}.json',dict(records=entries,global_lock_path=str(lock),global_lock_sha256=lh))
        status('checking_previous_artifacts');after={p:sha256_file(p) for p in protected};assert after==protected;write_json(out/f'protected_before_including_loaded_sources_{tag}.json',protected);write_json(out/f'protected_after_{tag}.json',after)
        write_json(out/f'completed_retained_oof_flow_{tag}.json',dict(completed_at=stamp(),crossfit_source=str(cross),new_pilot_fits=8,reused_signed_pilots=8,final_fits=10,prediction_files=140,strict_OOF_training=True,identity_selections={ds:sum(d['dataset']==ds and d['selected']['beta']==0 for d in decisions) for ds in DS},full_retention_selections={ds:sum(d['dataset']==ds and d['selected']['full_gain_retained'] for d in decisions) for ds in DS},selected_families={ds:selected_cfg[ds]['family'] for ds in DS},old_files_unchanged=True,HRF_started=False,lock_sha256=lh))
        status('retained_oof_flow_complete');print('RETAINED_OOF_FLOW_COMPLETE',str(out),flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise


def evaluate(ref,decision,out,tag,status):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
    dest=Path(decision['selected_path']).parent;model=model_for(ref.dataset,decision['anchor_seed'],decision['seed'],decision['config']['cap'],ref.device,decision['config'])
    for k in ('anchor','selected','raw'):assert sha256_file(decision[f'{k}_path'])==decision[f'{k}_sha256']
    ss=torch.load(decision['selected_path'],map_location=ref.device,weights_only=True)['state'];rs=torch.load(decision['raw_path'],map_location=ref.device,weights_only=True)['state']
    entries=[];beta=decision['selected']['beta']
    for name in ref.manifest['test']:
        status('locked_final_test',dataset=ref.dataset,seed=decision['seed'],image=name)
        d=ref.data[name];model.field.load_state_dict(rs)
        a,f,meta=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device)
        a512=project_residual_512(a,ref.dataset);f512=project_residual_512(f,ref.dataset)
        raw=apply_residual(d['coarse'],a512+f512);anchor_p=apply_residual(d['coarse'],a512)
        previous=O3/f'{ref.dataset}_{AT}/native_patch_direct/seed_{decision["seed"]}/{name}_predictions_{AT}.npz'
        with np.load(previous,allow_pickle=False) as old:
            assert np.array_equal(old['raw'],anchor_p.numpy()),'Native frozen anchor fails exact O3 test replay'
            assert all(np.array_equal(old[k],d[field].numpy()) for k,field in (('baseline','coarse'),('ground_truth','g'),('fov','fov')))
        native=dict(anchor_residual=a.numpy(),raw_flow_residual=f.numpy());sm=None
        if beta==0:selected=anchor_p.clone()
        else:
            if decision['selected']['epoch']==decision['raw']['epoch']:sf=f;sm=dict(meta,reused_raw=True)
            else:
                model.field.load_state_dict(ss);aa,sf,sm=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device);assert torch.equal(a,aa)
            selected=apply_residual(d['coarse'],a512+beta*project_residual_512(sf,ref.dataset));native['selected_flow_residual']=sf.numpy()
        path=dest/f'{name}_predictions_{tag}.npz';np.savez_compressed(path,baseline=d['coarse'].numpy(),O3=anchor_p.numpy(),raw=raw.numpy(),selected=selected.numpy(),ground_truth=d['g'].numpy(),fov=d['fov'].numpy().astype(bool))
        nativepath=dest/f'{name}_native_residuals_{tag}.npz';np.savez_compressed(nativepath,**native)
        with np.load(path,allow_pickle=False) as z:assert np.array_equal(z['raw'],raw.numpy()) and np.array_equal(z['selected'],selected.numpy())
        entries.append(dict(dataset=ref.dataset,seed=decision['seed'],image=name,path=str(path),sha256=sha256_file(path),native_path=str(nativepath),native_sha256=sha256_file(nativepath),beta=beta,selected_epoch=decision['selected']['epoch'],raw_epoch=decision['raw']['epoch'],anchor_path=str(previous),anchor_prediction_sha256=sha256_file(previous),raw_calls=meta,selected_calls=sm))
    write_json(dest/f'inference_{tag}.json',entries);del model;gc.collect();torch.cuda.empty_cache();return entries


if __name__=='__main__':main()
