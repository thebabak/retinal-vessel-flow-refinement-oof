"""Strict Dice-safe and structure-retaining validation selection of bidirectional OOF Flow."""
from __future__ import annotations
import argparse,csv,gc,json,random,shutil,traceback
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.run_native_patch_flow_v2 import DS,SEEDS,anchor_path,AT,O3
from flow_matching_refiner.run_fully_oof_flow import audit_routes
OLD=ROOT/'outputs/topology_oof_flow_2026-11-17_18-49-15';OT='2026-11-17_18-49-15'
SCREEN=ROOT/'outputs/dice_guarded_oof_flow_2026-11-17_19-31-34'
BETAS=(0.,.0625,.125,.25,.5,1.)
CONFIGS=[dict(name=f'regret{r:g}_structure{s:g}',cap=2.,regret_weight=r,structure_weight=s,fm_weight=.01) for r in (2.,8.) for s in (1.,2.)]

def model_for(dataset,anchor_seed,field_seed,cap,device):
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    from flow_matching_refiner.dice_guarded_oof_flow import DiceGuardedFlow
    anchor=ErrorSelectiveModel(False).to(device);anchor.load_state_dict(torch.load(anchor_path(dataset,anchor_seed),map_location=device,weights_only=True)['state'])
    torch.manual_seed(field_seed+20261118);torch.cuda.manual_seed_all(field_seed+20261118)
    return DiceGuardedFlow(anchor,cap).to(device)

def train(ref,out,tag,seed,anchor_seed,cfg,epochs,routes,supervision,status,pilot=False):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
    from flow_matching_refiner.native_patch_model import sample_training_patch,training_center_pools,project_residual_512,apply_residual,ErrorSelectiveModel
    from flow_matching_refiner.fully_oof_flow import routed_anchor
    from flow_matching_refiner.topology_oof_flow import patch_supervision
    from flow_matching_refiner.dice_guarded_oof_flow import objective,selection_score,feasible,retained,primary_key,raw_key
    from flow_matching_refiner.structure_metrics_all import metrics
    out.mkdir(parents=True,exist_ok=False);model=model_for(ref.dataset,anchor_seed,seed,cfg['cap'],ref.device);sources={};routing={}
    for n in ref.manifest['train']:
        e=routes[ref.dataset,anchor_seed,n];routing[n]=e
        with np.load(e['path'],allow_pickle=False) as z:assert np.array_equal(z['coarse512'],ref.data[n]['coarse'].numpy())
        if e['outer_fold'] not in sources:
            net=ErrorSelectiveModel(False).to(ref.device);net.load_state_dict(torch.load(e['anchor_model_path'],map_location=ref.device,weights_only=True)['state']);sources[e['outer_fold']]=net.eval().requires_grad_(False)
    assert len(sources)==4
    optimizer=torch.optim.AdamW(model.field.parameters(),lr=1e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=random.Random(seed+141001);trng=torch.Generator(device=ref.device).manual_seed(seed+20261119);pools={n:training_center_pools(ref.data[n]['n_g'],ref.data[n]['n_fov']) for n in ref.manifest['train']}
    fitting=[False]
    def reject(module,args):
        if fitting[0]:raise RuntimeError('Full O3 access during fitting rejected')
    hook=model.anchor.register_forward_pre_hook(reject);validation_records=[]
    def validation(epoch):
        values={b:[] for b in BETAS}
        for n in ref.manifest['validation']:
            d=ref.data[n];a,f,_=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device);a=project_residual_512(a,ref.dataset);f=project_residual_512(f,ref.dataset)
            for beta in BETAS:
                p=apply_residual(d['coarse'],a+beta*f);m=metrics(p.numpy()[0],d['g'].numpy()[0],d['coarse'].numpy()[0],1.,d['fov'].numpy()[0])
                q={k:m[k] for k in ('dice','precision','cldice','thin_skeleton_recall')};values[beta].append(q);validation_records.append(dict(epoch=epoch,beta=beta,image=n,**q))
        return {b:{k:float(np.mean([q[k] for q in mm])) for k in mm[0]} for b,mm in values.items()}
    initial=validation(0);assert all(initial[b]==initial[0.] for b in BETAS);base=initial[0.]
    old=json.loads((anchor_path(ref.dataset,anchor_seed).parent/f'selection_locked_{AT}.json').read_text());assert abs(base['dice']-old['raw']['validation_dice'])<1e-12
    prior_path=OLD/f'all10_choices_locked_before_test_{OT}.json'
    prior=next(q for q in json.loads(prior_path.read_text())['decisions'] if q['dataset']==ref.dataset and q['seed']==anchor_seed)
    assert base==prior['anchor_validation_metrics'];previous_target=prior['selected']['validation_metrics']
    def choice(epoch,beta,m):return dict(epoch=epoch,beta=beta,validation_dice=m['dice'],validation_metrics=m,structural_score=selection_score(m,base),structure_feasible=feasible(m,base),quality_feasible=feasible(m,base,False),full_gain_retained=retained(m,base,previous_target))
    best=choice(0,0.,base);raw=None;rawkey=None;sp=out/f'selected_field_{tag}.pt';rp=out/f'raw_field_{tag}.pt';torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),sp)
    parts_keys=('fm','bce','dice','eval_bce','eval_dice','cldice','thin','skeleton','dice_regret','fp_repair','preservation','centerline_drop','gate')
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
                    loss,parts=objective(model,c,rgb,g,fov,time,a,sk,align[micro:micro+8],cfg);assert bool(torch.isfinite(loss));(loss*len(batch)/len(samples)).backward();aggregate+=np.array([loss.item(),*(parts[k].item() for k in parts_keys)])*len(batch);seen+=len(batch)
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
    with (out/f'validation_per_image_{tag}.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(validation_records[0]));w.writeheader();w.writerows(validation_records)
    dec=dict(dataset=ref.dataset,seed=seed,anchor_seed=anchor_seed,pilot=pilot,config=cfg,epochs=epochs,selected=best,raw=raw,anchor_validation_metrics=base,previous_validation_target=previous_target,previous_target_source=str(prior_path),previous_target_sha256=sha256_file(prior_path),anchor_validation_dice=base['dice'],selected_path=str(sp),raw_path=str(rp),anchor_path=str(anchor_path(ref.dataset,anchor_seed)),anchor_sha256=sha256_file(anchor_path(ref.dataset,anchor_seed)),selected_sha256=sha256_file(sp),raw_sha256=sha256_file(rp),locked_at=stamp(),strict_OOF_training=True,full_anchor_fitting_access_rejected=True,training_OOF_sources={n:dict(fold=e['outer_fold'],model_path=e['anchor_model_path'],model_sha256=e['anchor_model_sha256'],fit=e['fit_images'],validation=e['validation_images'],fold_protocol=e['fold_protocol']) for n,e in routing.items()})
    write_json(out/f'selection_locked_{tag}.json',dec);del model,optimizer,schedule,pools,sources;gc.collect();torch.cuda.empty_cache();return dec

def record_previous_targets(ds,out,tag):
    out.mkdir(parents=True,exist_ok=False);p=OLD/f'all10_choices_locked_before_test_{OT}.json'
    dd=[q for q in json.loads(p.read_text())['decisions'] if q['dataset']==ds]
    rr=[dict(seed=q['seed'],base=q['anchor_validation_metrics'],previous_selected=q['selected']['validation_metrics'],previous_Dice_change=q['selected']['validation_metrics']['dice']-q['anchor_validation_metrics']['dice']) for q in dd]
    write_json(out/f'diagnosis_{tag}.json',dict(dataset=ds,validation_only=True,source=str(p),sha256=sha256_file(p),records=rr,observations=['Prior allowed Dice drop0.0005; new guard permits no Dice drop.','Prior had positive thin recovery without an explicit normalized FP-repair term.','Prior auxiliary error head did not gate velocity; new signed growth/removal gates are actually applied.'],causal_scope='Combined candidate; no isolated causal or population guarantee.'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--crossfit',type=Path,default=ROOT/'outputs/strict_o3_crossfit_2026-11-17_03-16-57');args=ap.parse_args()
    if not args.execute:print('Inert. Requires --execute for8validationpilots +10finalfits; no HRF.');return
    cross=args.crossfit.resolve();manifest,crosslock,routes=audit_routes(cross);out=(args.output or ROOT/'outputs'/f'dice_guarded_oof_flow_{stamp()}').resolve();out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,crossfit_source=str(cross),datasets=list(DS),seeds=list(SEEDS),completed_pilots=0,completed_final_fits=0,HRF_started=False)
    def status(stage,**extra):state.update(stage=stage,updated_at=stamp(),**extra);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_dice_guarded_oof_flow.json',state)
    status('protecting_previous_files');protected=json.loads((OLD/f'protected_before_including_loaded_sources_{OT}.json').read_text());assert all(sha256_file(p)==h for p,h in protected.items())
    for p in OLD.rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in SCREEN.rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    write_json(out/f'protected_before_{tag}.json',protected)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):
        d=out/'source'/f'{p.stem}_{tag}.py';d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
    write_json(out/f'plan_{tag}.json',dict(**state,configs=CONFIGS,pilot_seed=91011,pilot_epochs=40,final_epochs=120,
       method='Same strict nested OOF sources/unchanged O3. Bidirectional GT-free growth/removal gates, actual two-Euler rollout; native and exact-nearest512 BCE/Dice each0.25; centerline/thin terms retained, soft-Dice regret, FP-repair0.05, preservation0.25 and centerline-logit-decrease hinge0.25; no GT inference input.',
       selection='Validation only: strict Dice>=O3 (numerical tolerance1e-12), precision>=O3-.002, clDice/thin each>=O3. Prefer candidates also retaining100percent of BOTH previous TopologyFlow selected validation structural metrics; then maximize mean structural gain. If full retention unavailable, flag partial retention; O3identity last fallback. beta0,.0625,.125,.25,.5,1. Rawbeta1 separately ranked by fullretention/pareto/quality/structural score; infeasible raw is diagnostic only.',
       hyperparameters='4configs regretweight2/8 xstructureweight1/2, cap2 fixed;8pilots40epochs seed91011. Prefer number of full-retained dataset targets, then mean selected structural gain, then raw gain. Global lock before10finalfits120epochs; all10choices locked before newtest predictions.',
       statistics='Selected minus unchanged O3 and preceding TopologyFlow selected:2contrasts x2datasets x3metrics=12-test Holmfamily; seed/cohort and t/W separate. Five conditional refinerseed means/sampleSD; seed t95CI, CHASE4subject/DRIVE20image sensitivity and exploratory cohort bootstrap256/50000 RNG20261117.',
       success_criteria='Post-lock audit only: mean Dice must be >=unchanged O3; mean clDice and thin recall must each be >=previous TopologyFlow selected. Also report matched-seed pass counts and confidence intervals. These test checks never select/tune a model; failure is reported explicitly.',
       rejected_screening=str(SCREEN),screening_scope='Four completed40epoch CHASE pilots and four epochs of one DRIVE pilot, rejected using validation only; no finalseed fits/test predictions. These screening records are preserved and excluded from the five-seed outcome.',
       limitations='Combined changes, not causal single-factor ablation. Previously examined tests, reused external validation4images, shared baseline generations, source12/deployment16 training-size shift. DRIVE patient/test-label provenance unverified. Identity not learned Flow gain.',old_files_unchanged=True))
    try:
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        from flow_matching_refiner.topology_oof_flow import prepare_supervision
        
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
        write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False))
        refs={};supervisions={};pilots=[]
        for ds in DS:
            status('loading_audited_data',dataset=ds);ref,audit=load_native_dataset(ds,out/f'{ds}_{tag}/input_audit','cuda');refs[ds]=ref
            for p,h in audit['source_sha256_before'].items():
                if p in protected:assert protected[p]==h
                protected[p]=h
            supervisions[ds]=prepare_supervision(ref)
            write_json(out/f'{ds}_{tag}/input_metadata_{tag}.json',dict(dataset=ds,split=ref.manifest,strict_OOF=True,supervision_shape=next(iter(supervisions[ds].values()))['shape'],exact_native_to512_GT_matching=True))
            record_previous_targets(ds,out/f'{ds}_{tag}/diagnosis',tag)
            for cfg in CONFIGS:pilots.append(train(ref,out/f'{ds}_{tag}/pilots/{cfg["name"]}',tag,91011,42,cfg,40,routes,supervisions[ds],status,True));status('pilot_complete',completed_pilots=len(pilots))
        rank=[]
        for cfg in CONFIGS:
            rr=[p for p in pilots if p['config']==cfg];assert len(rr)==2
            rank.append(dict(config=cfg,retained_targets=sum(q['selected']['full_gain_retained'] for q in rr),mean_selected_structural_gain=sum(q['selected']['structural_score'] for q in rr)/2,mean_raw_structural_gain=sum(q['raw']['structural_score'] for q in rr)/2))
        assert max(r['mean_selected_structural_gain'] for r in rank)>0,'No validation Pareto improvement; do not launch final/test runs'
        cfg=max(rank,key=lambda r:(r['retained_targets'],r['mean_selected_structural_gain'],r['mean_raw_structural_gain']))['config'];hp=out/f'hyperparameters_locked_before_final_{tag}.json';write_json(hp,dict(locked_at=stamp(),selected=cfg,ranking=rank,pilots=pilots));decisions=[]
        for ds in DS:
            for seed in SEEDS:decisions.append(train(refs[ds],out/f'{ds}_{tag}/final/seed_{seed}',tag,seed,seed,cfg,120,routes,supervisions[ds],status));status('final_fit_complete',completed_final_fits=len(decisions))
        lock=out/f'all10_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=decisions,hyperparameter_lock_sha256=sha256_file(hp),strict_crossfit_lock_path=manifest['global_lock_path'],strict_crossfit_lock_sha256=manifest['global_lock_sha256']));lh=sha256_file(lock);entries=[]
        for ds in DS:
            for d in decisions:
                if d['dataset']==ds:entries.extend(evaluate(refs[ds],d,out,tag,status))
        assert len(entries)==140 and sha256_file(lock)==lh;write_json(out/f'prediction_manifest_{tag}.json',dict(records=entries,global_lock_path=str(lock),global_lock_sha256=lh))
        status('checking_previous_artifacts');after={p:sha256_file(p) for p in protected};assert after==protected;write_json(out/f'protected_before_including_loaded_sources_{tag}.json',protected);write_json(out/f'protected_after_{tag}.json',after)
        write_json(out/f'completed_dice_guarded_oof_flow_{tag}.json',dict(completed_at=stamp(),crossfit_source=str(cross),pilot_fits=8,final_fits=10,prediction_files=140,strict_OOF_training=True,identity_selections={ds:sum(d['dataset']==ds and d['selected']['beta']==0 for d in decisions) for ds in DS},full_retention_selections={ds:sum(d['dataset']==ds and d['selected']['full_gain_retained'] for d in decisions) for ds in DS},old_files_unchanged=True,HRF_started=False,lock_sha256=lh))
        status('dice_guarded_oof_flow_complete');print('DICE_GUARDED_OOF_FLOW_COMPLETE',str(out),flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise


def evaluate(ref,decision,out,tag,status):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
    dest=Path(decision['selected_path']).parent;model=model_for(ref.dataset,decision['anchor_seed'],decision['seed'],decision['config']['cap'],ref.device)
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
