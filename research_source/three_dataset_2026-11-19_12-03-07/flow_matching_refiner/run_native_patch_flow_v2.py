"""Validation-only diagnosis/tuning then ten locked O3-anchored flow fits."""
from __future__ import annotations
import argparse,csv,gc,json,random,shutil,traceback
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
DS=('CHASE','DRIVE');SEEDS=(42,123,2026,3000,3001);BETAS=(0.,.25,.5,1.)
OLD=ROOT/'outputs/native_patch_flow_study_2026-11-17_01-36-08';OT='2026-11-17_01-36-08'
O3=ROOT/'outputs/native_patch_study_2026-11-16_16-53-08';AT='2026-11-16_16-53-08'
CONFIGS=[dict(name=f'cap{cap:g}_fm{weight:g}',cap=cap,fm_weight=weight) for cap in (1.,2.) for weight in (.01,.1)]
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def savecsv(path,items):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
def anchor_path(dataset,seed):return O3/f'{dataset}_{AT}/native_patch_direct/seed_{seed}/best_raw_model_{AT}.pt'
def score(ref,p,n):
    d=ref.data[n]
    return ref.score(p,d['g'])['dice'] if ref.dataset=='CHASE' else ref.score(p,d['g'],d['fov']>.5)['dice']
def model_for(dataset,anchor_seed,field_seed,cap,device):
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    from flow_matching_refiner.native_patch_flow_v2 import AnchoredFlow
    anchor=ErrorSelectiveModel(False).to(device)
    anchor.load_state_dict(torch.load(anchor_path(dataset,anchor_seed),map_location=device,weights_only=True)['state'])
    torch.manual_seed(field_seed+20261118);torch.cuda.manual_seed_all(field_seed+20261118)
    return AnchoredFlow(anchor,cap).to(device)

def diagnose(ref,out,tag,status):
    """Existing logs + validation-only solver/path probes; no weight update."""
    import numpy as np
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel,source_logits,MARGIN
    from flow_matching_refiner.native_patch_flow import predict_flow_tiled
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
    from flow_matching_refiner.model import linear_path
    history=[]
    for seed in SEEDS:
        folder=OLD/f'{ref.dataset}_{OT}/native_patch_flow/seed_{seed}'
        val=rows(folder/f'validation_{OT}.csv');train=rows(folder/f'training_{OT}.csv')
        v=[r for r in val if float(r['alpha'])==1.]
        best=max(v,key=lambda r:float(r['validation_dice']))
        history.append(dict(dataset=ref.dataset,seed=seed,best_epoch=int(best['epoch']),best_validation_dice=float(best['validation_dice']),final_validation_dice=float(v[-1]['validation_dice']),first_FM=float(train[0]['fm']),final_FM=float(train[-1]['fm'])))
    selected=json.loads((OLD/f'{ref.dataset}_{OT}/native_patch_flow/seed_42/selection_locked_{OT}.json').read_text())
    model=ErrorSelectiveModel(False).to(ref.device)
    model.load_state_dict(torch.load(selected['raw_path'],map_location=ref.device,weights_only=True)['state']);model.eval()
    solver=[]
    for steps in (1,2,4,8):
        values=[]
        for name in ref.manifest['validation']:
            status('diagnosing_validation_solver',dataset=ref.dataset,steps=steps,image=name)
            d=ref.data[name];_,delta,_=predict_flow_tiled(model,d['n_coarse'],d['n_rgb'],ref.device,steps=steps,return_residual=True)
            p=apply_residual(d['coarse'],project_residual_512(delta,ref.dataset));values.append(score(ref,p,name))
        solver.append(dict(dataset=ref.dataset,seed=42,steps=steps,validation_mean_dice=float(np.mean(values))))
    # Fixed center crops are specified without looking at prediction quality.
    probes=[]
    with torch.no_grad():
        for name in ref.manifest['validation']:
            d=ref.data[name];h,w=d['n_coarse'].shape[-2:];y=(h-256)//2;x=(w-256)//2
            c,rgb,g,fov=[d[k][:,y:y+256,x:x+256][None].to(ref.device) for k in ('n_coarse','n_rgb','n_g','n_fov')]
            origin=source_logits(c);target=MARGIN*(2*g-1);dv=target-origin
            correct=((c>=.5)==(g>=.5))&(fov>.5)
            err=((c>=.5)!=(g>=.5))&(fov>.5)
            for t in (0.,.5,.9):
                time=torch.tensor([t],device=ref.device);state,_=linear_path(origin,target,time)
                endpoint=state+(1-t)*model(state,time,c,rgb,None)
                pred=endpoint>=0;truth=g>=.5;valid=fov>.5
                tp=int((pred&truth&valid).sum());fp=int((pred&~truth&valid).sum());fn=int((~pred&truth&valid).sum())
                probes.append(dict(dataset=ref.dataset,image=name,crop='fixed native center256',time=t,teacher_forced_endpoint_patch_Dice=2*tp/(2*tp+fp+fn),correct_pixel_count=int(correct.sum()),wrong_pixel_count=int(err.sum()),mean_abs_full_target_velocity_on_correct=float(dv[correct].abs().mean()),fraction_correct_with_nonzero_full_target_velocity=float((dv[correct].abs()>1e-6).float().mean())))
    write_json(out/f'validation_diagnosis_{tag}.json',dict(history=history,solver_probe=solver,path_probes=probes,validation_only=True,weights_updated=False,causal_scope='Evidence for hypotheses, not an isolated causal ablation'))
    del model;gc.collect();torch.cuda.empty_cache()

def train(ref,out,tag,seed,anchor_seed,cfg,epochs,status,pilot=False):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow_v2 import objective,predict_tiled
    from flow_matching_refiner.native_patch_model import sample_training_patch,training_center_pools,project_residual_512,apply_residual
    out.mkdir(parents=True,exist_ok=False)
    model=model_for(ref.dataset,anchor_seed,seed,cfg['cap'],ref.device)
    optimizer=torch.optim.AdamW(model.field.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    rng=random.Random(seed+141001);trng=torch.Generator(device=ref.device).manual_seed(seed+20261119)
    pools={n:training_center_pools(ref.data[n]['n_g'],ref.data[n]['n_fov']) for n in ref.manifest['train']}
    def validation():
        scores={b:[] for b in BETAS}
        for n in ref.manifest['validation']:
            d=ref.data[n];a,f,meta=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device)
            a=project_residual_512(a,ref.dataset);f=project_residual_512(f,ref.dataset)
            for beta in BETAS:scores[beta].append(score(ref,apply_residual(d['coarse'],a+beta*f),n))
        return {b:float(np.mean(v)) for b,v in scores.items()}
    initial=validation();assert len(set(initial.values()))==1,'Zero-init flow must be exactly O3'
    recorded=json.loads((anchor_path(ref.dataset,anchor_seed).parent/f'selection_locked_{AT}.json').read_text())
    assert abs(initial[0.]-recorded['raw']['validation_dice'])<1e-12,'Frozen anchor does not reproduce O3 validation'
    best=dict(epoch=0,beta=0.,validation_dice=initial[0.]);raw=dict(epoch=None,beta=1.,validation_dice=-1.)
    selected_path=out/f'selected_field_{tag}.pt';raw_path=out/f'raw_field_{tag}.pt'
    torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),selected_path)
    with (out/f'training_{tag}.csv').open('x',newline='',encoding='utf-8') as tf,(out/f'validation_{tag}.csv').open('x',newline='',encoding='utf-8') as vf,(out/f'patch_draws_{tag}.csv').open('x',newline='',encoding='utf-8') as pf:
        tw,vw,pw=csv.writer(tf),csv.writer(vf),csv.writer(pf)
        tw.writerow(['epoch','loss','fm','bce','dice','cldice','gate','preservation','lr','patches','optimizer_updates'])
        vw.writerow(['epoch','beta','validation_dice']);vw.writerow([0,0,initial[0.]])
        pw.writerow(['epoch','patch_index','image','centre_y','centre_x','top','left','used_vessel_centre','rotation90','flip_horizontal','flip_vertical'])
        for epoch in range(1,epochs+1):
            status('pilot_training' if pilot else 'final_training',dataset=ref.dataset,seed=seed,epoch=epoch,epochs=epochs,config=cfg['name'])
            model.train();names=ref.manifest['train']*4;rng.shuffle(names);aggregate=np.zeros(7);seen=0;updates=0;lr=optimizer.param_groups[0]['lr']
            for start in range(0,len(names),16):
                samples=[]
                for idx,n in enumerate(names[start:start+16],start=start):
                    d=ref.data[n];crop,meta=sample_training_patch(d['n_coarse'],d['n_rgb'],d['n_g'],d['n_fov'],rng=rng,center_pools=pools[n])
                    k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5
                    fields=[torch.rot90(crop[q],k,(-2,-1)) for q in ('coarse','rgb','gt','fov')]
                    if fh:fields=[t.flip(-1) for t in fields]
                    if fv:fields=[t.flip(-2) for t in fields]
                    samples.append(fields);pw.writerow([epoch,idx,n,meta['centre_y'],meta['centre_x'],meta['top'],meta['left'],meta['used_vessel_centre'],k,fh,fv])
                optimizer.zero_grad(set_to_none=True)
                # Two microbatches retain effective batch16 and the O3 crop stream.
                for micro in range(0,len(samples),8):
                    batch=samples[micro:micro+8]
                    c,rgb,g,fov=[torch.stack([s[j] for s in batch]).to(ref.device) for j in range(4)]
                    time=torch.rand(len(c),device=ref.device,generator=trng)
                    time=torch.where(torch.rand(len(c),device=ref.device,generator=trng)<.25,torch.zeros_like(time),time)
                    loss,parts=objective(model,c,rgb,g,fov,time,cfg['fm_weight'])
                    assert bool(torch.isfinite(loss));(loss*len(batch)/len(samples)).backward()
                    aggregate+=np.array([loss.item(),*(parts[k].item() for k in ('fm','bce','dice','cldice','gate','preservation'))])*len(batch);seen+=len(batch)
                torch.nn.utils.clip_grad_norm_(model.field.velocity_net.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(model.field.error_net.parameters(),1.)
                optimizer.step();updates+=1
            scheduler.step();assert seen==64 and updates==4
            tw.writerow([epoch,*(aggregate/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:
                status('pilot_validation' if pilot else 'final_validation',dataset=ref.dataset,seed=seed,epoch=epoch,epochs=epochs,config=cfg['name'])
                grid=validation()
                if grid[1.]>raw['validation_dice']+1e-12:
                    raw=dict(epoch=epoch,beta=1.,validation_dice=grid[1.]);torch.save(dict(state=model.field.state_dict(),selection=raw,config=cfg),raw_path)
                beta=max(BETAS,key=lambda b:grid[b])
                if grid[beta]>best['validation_dice']+1e-12:
                    best=dict(epoch=epoch,beta=beta,validation_dice=grid[beta]);torch.save(dict(state=model.field.state_dict(),selection=best,config=cfg),selected_path)
                for beta in BETAS:vw.writerow([epoch,beta,grid[beta]])
                vf.flush();print(ref.dataset,'PILOT' if pilot else 'FINAL',seed,cfg['name'],epoch,'best',best,'raw',raw,flush=True)
    decision=dict(dataset=ref.dataset,seed=seed,anchor_seed=anchor_seed,pilot=pilot,config=cfg,epochs=epochs,selected=best,raw=raw,anchor_validation_dice=initial[0.],
                  selected_path=str(selected_path),raw_path=str(raw_path),anchor_path=str(anchor_path(ref.dataset,anchor_seed)),anchor_sha256=sha256_file(anchor_path(ref.dataset,anchor_seed)),selected_sha256=sha256_file(selected_path),raw_sha256=sha256_file(raw_path),locked_at=stamp(),
                  trainable_parameters=sum(p.numel() for p in model.field.parameters()),frozen_anchor_parameters=sum(p.numel() for p in model.anchor.parameters()),steps=2,tile=256,stride=128,lr=1e-4)
    write_json(out/f'selection_locked_{tag}.json',decision)
    del model,optimizer,scheduler,pools;gc.collect();torch.cuda.empty_cache();return decision

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

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    if not args.execute:print('Inert. --execute authorizes 8validation pilots +10finalfits, CHASE/DRIVE only.');return
    out=(args.output or ROOT/'outputs'/f'native_patch_flow_v2_{stamp()}').resolve();out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,datasets=list(DS),seeds=list(SEEDS),new_pilot_fits=8,new_final_fits=10,completed_pilots=0,completed_final_fits=0,HRF_started=False)
    def status(stage,**extra):state.update(stage=stage,updated_at=stamp(),**extra);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_native_patch_flow_v2_work.json',state)
    status('protecting_previous_results')
    protected=json.loads((OLD/f'protected_before_including_loaded_sources_{OT}.json').read_text())
    assert all(sha256_file(p)==h for p,h in protected.items())
    for path in OLD.rglob('*'):
        if path.is_file():protected[str(path)]=sha256_file(path)
    for path in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(path)]=sha256_file(path)
    for d in DS:
        for p in (O3/f'{d}_{AT}').rglob('*'):
            if p.is_file():protected[str(p)]=sha256_file(p)
    write_json(out/f'protected_before_{tag}.json',protected)
    for path in (ROOT/'flow_matching_refiner').glob('*.py'):
        dest=out/'source'/f'{path.stem}_{tag}.py';dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest)
    plan=dict(**state,configs=CONFIGS,pilot_seed=91011,pilot_epochs=40,final_epochs=120,
       method='Frozen matched-seed O3 anchor +newzero-initialized bounded flow; error-selective clipped target velocity, balancedFM, fullsegmentation losses on actual2step rollout and correct-pixel preservation1.',
       validation='4config pilots per dataset. Global config=max mean validation-selected Dice gain over anchor across BOTHdatasets; tie rawvalidation gain, then config order. All8pilots precede HP lock and finaltraining.',
       final_selection='Primary: highest validation meanDice over beta0,.25,.5,1 at epochs10..120 plus exactO3identity at epoch0; strict improvement/earliest ties. Rawlearnedbeta1 is separately selected and always reported. Identity choices explicitly counted.',
       inference='FrozenO3 onecall +2boundedEuler fieldcalls/tile; positiveHann separateanchor/flowresiduals; original512residualprojection, GT/FOV-free inference. Adds a frozen network; not a parameter/compute-matched pureFMclaim.',
       optimization='AdamW1e-4 wd1e-4 cosine withoutwarmup, effectivebatch16/micro8 with gradientaccumulation, fournative256patches/trainimage/epoch; same crop/rot90/flipstreamasO3 forfinalseeds; FP32.',
       statistical_scope='Primary selected pipeline vs Baseline,O3,V1raw x2datasets x3metrics(Dice/clDice/thinrecall)=18testHolmfamily. t/W andseed/cohort separate. Rawnewfield reporteddescriptively; n5means/sampleSD, seedt95CI, CHASE4subject/DRIVE20image sensitivity. Unadjustedcohortbootstrap256CHASE/50000DRIVE seed20261117.',
       locked_before_test='All10finalcheckpoint/beta selections globally locked before newrefiner test outputs; baselinereplay byinputloader only. Previously examined cohorts: exploratory, no fresh heldouttest.',
       OOF='Reuse retained seed42OOFbaseline trainingmaps; fixed fullbaseline val/test; no baseline,OOF orO3refit. Frozen O3 was trained on thoseOOFmaps, but is not separately cross-fitted for the newstage.',
       no_HRF=True,no_old_result_or_manuscript_edit=True)
    write_json(out/f'plan_{tag}.json',plan)
    try:
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
        write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),training='FP32',TF32=False))
        refs={};pilots=[]
        for d in DS:
            status('loading_audited_OOF_native',dataset=d);ref,audit=load_native_dataset(d,out/f'{d}_{tag}/input_audit','cuda');refs[d]=ref
            for p,h in audit['source_sha256_before'].items():
                if p in protected:assert protected[p]==h
                protected[p]=h
            write_json(out/f'{d}_{tag}/input_metadata_{tag}.json',dict(dataset=d,split=ref.manifest,audit=audit['output_audit'],OOF_verified=True))
            diagnose(ref,out/f'{d}_{tag}',tag,status)
            for cfg in CONFIGS:
                decision=train(ref,out/f'{d}_{tag}/pilots/{cfg["name"]}',tag,91011,42,cfg,40,status,True);pilots.append(decision);status('pilot_completed',completed_pilots=len(pilots))
        ranking=[]
        for cfg in CONFIGS:
            choices=[p for p in pilots if p['config']==cfg];assert len(choices)==2
            ranking.append(dict(config=cfg,mean_selected_validation_gain=sum(p['selected']['validation_dice']-p['anchor_validation_dice'] for p in choices)/2,mean_raw_validation_gain=sum(p['raw']['validation_dice']-p['anchor_validation_dice'] for p in choices)/2))
        chosen=max(ranking,key=lambda r:(r['mean_selected_validation_gain'],r['mean_raw_validation_gain']))['config']
        hp=out/f'hyperparameters_locked_before_final_training_{tag}.json';write_json(hp,dict(locked_at=stamp(),selected=chosen,ranking=ranking,pilots=pilots))
        decisions=[]
        for d in DS:
            for seed in SEEDS:
                decisions.append(train(refs[d],out/f'{d}_{tag}/final/seed_{seed}',tag,seed,seed,chosen,120,status));status('final_fit_completed',completed_final_fits=len(decisions))
        lock=out/f'all10_final_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=decisions,hyperparameter_lock_sha256=sha256_file(hp),plan_sha256=sha256_file(out/f'plan_{tag}.json')));lh=sha256_file(lock)
        entries=[]
        for d in DS:
            for decision in decisions:
                if decision['dataset']==d:entries.extend(evaluate(refs[d],decision,out,tag,status))
        assert len(entries)==140 and sha256_file(lock)==lh
        write_json(out/f'prediction_manifest_{tag}.json',dict(records=entries,global_lock_path=str(lock),global_lock_sha256=lh))
        status('checking_previous_sources')
        after={p:sha256_file(p) for p in protected};assert after==protected
        write_json(out/f'protected_before_including_loaded_sources_{tag}.json',protected);write_json(out/f'protected_after_{tag}.json',after)
        write_json(out/f'completed_training_inference_{tag}.json',dict(completed_at=stamp(),pilot_fits=8,final_fits=10,final_epochs=120,prediction_files=140,identity_selections=sum(d['selected']['beta']==0 for d in decisions),all_choices_validation_only=True,config=chosen,old_files_unchanged=True,HRF_started=False,lock_sha256=lh))
        status('training_inference_complete');print('V2_COMPLETE',str(out),flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(traceback=traceback.format_exc(),failed_at=stamp()));status('failed');raise

if __name__=='__main__':main()
