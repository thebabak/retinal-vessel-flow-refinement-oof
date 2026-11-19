"""Ten new native-patch FM fits on CHASE/DRIVE only; preserve previous work."""
from __future__ import annotations
import argparse,csv,gc,json,random,shutil,traceback
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json

DATASETS=('CHASE','DRIVE')
SEEDS=(42,123,2026,3000,3001)
TILE=256;STRIDE=128;BATCH=16;PATCHES=4;EPOCHS=120;STEPS=8
ALPHAS=(0.,.25,.5,.75,1.)
ARM='native_patch_flow'
NORMALIZED=ROOT/'outputs/private_test_comparisons_2026-11-17_00-38-33/normalized_saved_comparison_manifest_2026-11-17_00-39-15.json'

def score(ref,prob,name):
    d=ref.data[name]
    return ref.score(prob,d['g'])['dice'] if ref.dataset=='CHASE' else ref.score(prob,d['g'],d['fov']>.5)['dice']

def protected_files():
    paths=set((ROOT/'flow_matching_refiner').glob('*.py'))|set(ROOT.glob('*.tex'))
    for source in (NORMALIZED,ROOT/'tmp/current_CHASE_DRIVE_prediction_statistics.json'):
        paths.add(source)
    normalized=json.loads(NORMALIZED.read_text())
    for entry in normalized['inputs']:paths.add(Path(entry['path']))
    # Preserve ALL earlier comparison prediction arrays, not just favorable arms.
    for category in ('retained','controls','segflow'):
        for entry in normalized[category]:paths.add(Path(entry['path']))
    previous=json.loads((ROOT/'tmp/current_CHASE_DRIVE_prediction_statistics.json').read_text())
    paths.update(p for p in Path(previous['output']).iterdir() if p.is_file())
    return {str(p):sha256_file(p) for p in sorted(paths)}

def train_one(ref,out,tag,seed,pools,status):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow import flow_loss,predict_flow_tiled
    from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,sample_training_patch,project_residual_512,apply_residual
    dest=out/ARM/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ErrorSelectiveModel(False).to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    rng=random.Random(seed+141001)  # Same native patch/augmentation stream as O3.
    time_rng=torch.Generator(device=ref.device).manual_seed(seed+20261117)
    base_val=float(np.mean([score(ref,ref.data[n]['coarse'],n) for n in ref.manifest['validation']]))
    raw=dict(epoch=None,validation_dice=-1.)
    blend=dict(epoch=0,alpha=0.,validation_dice=base_val)
    blendpath=dest/f'selected_blend_model_{tag}.pt'
    torch.save(dict(state=model.state_dict(),selection=blend),blendpath)
    validation_meta=[]
    def validation(epoch):
        values={a:[] for a in ALPHAS}
        for name in ref.manifest['validation']:
            d=ref.data[name]
            _,delta,meta=predict_flow_tiled(model,d['n_coarse'],d['n_rgb'],ref.device,
                     tile=TILE,stride=STRIDE,batch=BATCH,steps=STEPS,return_residual=True)
            probability=apply_residual(d['coarse'],project_residual_512(delta,ref.dataset))
            for a in ALPHAS:
                p=d['coarse'] if a==0 else (1-a)*d['coarse']+a*probability
                values[a].append(score(ref,p,name))
            validation_meta.append(dict(epoch=epoch,image=name,**meta))
        return {a:float(np.mean(v)) for a,v in values.items()}
    with (dest/f'training_{tag}.csv').open('x',newline='',encoding='utf-8') as tf, \
         (dest/f'validation_{tag}.csv').open('x',newline='',encoding='utf-8') as vf, \
         (dest/f'patch_draws_{tag}.csv').open('x',newline='',encoding='utf-8') as pf:
        tw,vw,pw=csv.writer(tf),csv.writer(vf),csv.writer(pf)
        tw.writerow(['epoch','loss','fm','bce','dice_loss','soft_cldice','gate','lr','patches','updates'])
        vw.writerow(['epoch','alpha','validation_dice']);vw.writerow([0,0,base_val])
        pw.writerow(['epoch','patch_index','image','centre_y','centre_x','top','left','used_vessel_centre','rotation90','flip_horizontal','flip_vertical'])
        for epoch in range(1,EPOCHS+1):
            status('training',dataset=ref.dataset,seed=seed,epoch=epoch,completed_fits=status.state.get('completed_fits',0))
            model.train();names=ref.manifest['train']*PATCHES;rng.shuffle(names)
            aggregate=np.zeros(6);seen=0;updates=0;lr=opt.param_groups[0]['lr']
            for start in range(0,len(names),BATCH):
                samples=[]
                for index,name in enumerate(names[start:start+BATCH],start=start):
                    d=ref.data[name]
                    c,meta=sample_training_patch(d['n_coarse'],d['n_rgb'],d['n_g'],d['n_fov'],
                               tile=TILE,rng=rng,vessel_probability=.5,center_pools=pools[name])
                    k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5
                    fields=[torch.rot90(c[key],k,(-2,-1)) for key in ('coarse','rgb','gt','fov')]
                    if fh:fields=[v.flip(-1) for v in fields]
                    if fv:fields=[v.flip(-2) for v in fields]
                    samples.append(fields)
                    pw.writerow([epoch,index,name,meta['centre_y'],meta['centre_x'],meta['top'],meta['left'],meta['used_vessel_centre'],k,fh,fv])
                coarse,rgb,gt,fov=[torch.stack([s[j] for s in samples]).to(ref.device).clone() for j in range(4)]
                time=torch.rand(len(coarse),device=ref.device,generator=time_rng)
                time=torch.where(torch.rand(len(coarse),device=ref.device,generator=time_rng)<.25,torch.zeros_like(time),time)
                loss,parts=flow_loss(model,coarse,rgb,gt,fov,time)
                if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite FM objective')
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(model.error_net.parameters(),1.)
                opt.step();seen+=len(coarse);updates+=1
                aggregate+=np.array([loss.item(),*(parts[k].item() for k in ('fm','bce','dice','cldice','gate'))])*len(coarse)
            schedule.step();assert seen==len(ref.manifest['train'])*PATCHES
            assert updates==(seen+BATCH-1)//BATCH
            tw.writerow([epoch,*(aggregate/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:
                status('validation_native_flow',dataset=ref.dataset,seed=seed,epoch=epoch)
                values=validation(epoch)
                if values[1.]>raw['validation_dice']+1e-12:
                    raw=dict(epoch=epoch,validation_dice=values[1.])
                    torch.save(dict(state=model.state_dict(),selection=raw),dest/f'best_raw_model_{tag}.pt')
                a=max(ALPHAS,key=lambda a:values[a])
                if values[a]>blend['validation_dice']+1e-12:
                    blend=dict(epoch=epoch,alpha=a,validation_dice=values[a])
                    torch.save(dict(state=model.state_dict(),selection=blend),blendpath)
                for a in ALPHAS:vw.writerow([epoch,a,values[a]])
                vf.flush()
                print(ref.dataset,seed,epoch,'loss',float(aggregate[0]/seen),'raw_validation',values[1.],'best',raw,'blend',blend,flush=True)
    final=dest/f'final_model_{tag}.pt';torch.save(dict(state=model.state_dict(),epoch=EPOCHS),final)
    decision=dict(dataset=ref.dataset,arm=ARM,refiner_seed=seed,baseline_seed=42,raw=raw,blend=blend,locked_at=stamp(),
        raw_path=str(dest/f'best_raw_model_{tag}.pt'),blend_path=str(blendpath),
        raw_sha256=sha256_file(dest/f'best_raw_model_{tag}.pt'),blend_sha256=sha256_file(blendpath),final_sha256=sha256_file(final),
        training_parameters=sum(p.numel() for p in model.parameters()),inference_velocity_parameters=sum(p.numel() for p in model.velocity_net.parameters()),inference_gate_parameters=0,
        tile=TILE,stride=STRIDE,steps=STEPS,train_batch=BATCH,infer_batch=BATCH,epochs=EPOCHS,training_patches_per_image_per_epoch=PATCHES)
    write_json(dest/f'validation_tiling_{tag}.json',validation_meta)
    write_json(dest/f'selection_locked_{tag}.json',decision)
    del model,opt,schedule;gc.collect();torch.cuda.empty_cache()
    return decision

def evaluate_one(ref,out,tag,decision,cutoff,status):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_flow import predict_flow_tiled
    from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,project_residual_512,apply_residual
    from flow_matching_refiner.structure_metrics_all import metrics
    seed=decision['refiner_seed'];dest=out/ARM/f'seed_{seed}'
    model=ErrorSelectiveModel(False).to(ref.device);model.eval().requires_grad_(False)
    for kind in ('raw','blend'):assert sha256_file(decision[f'{kind}_path'])==decision[f'{kind}_sha256']
    rawstate=torch.load(decision['raw_path'],map_location=ref.device,weights_only=True)['state']
    alpha=decision['blend']['alpha']
    blendstate=torch.load(decision['blend_path'],map_location=ref.device,weights_only=True)['state'] if alpha else None
    rows=[];entries=[]
    for name in ref.manifest['test']:
        status('locked_test_inference',dataset=ref.dataset,seed=seed,image=name)
        d=ref.data[name];model.load_state_dict(rawstate)
        p_native,delta,meta=predict_flow_tiled(model,d['n_coarse'],d['n_rgb'],ref.device,tile=TILE,stride=STRIDE,batch=BATCH,steps=STEPS,return_residual=True)
        raw=apply_residual(d['coarse'],project_residual_512(delta,ref.dataset))
        native={'raw_probability_native':p_native.numpy(),'raw_residual_native':delta.numpy()}
        blendmeta=None
        if alpha:
            if decision['blend']['epoch']==decision['raw']['epoch']:bd=delta;blendmeta=dict(meta,reused_raw=True)
            else:
                model.load_state_dict(blendstate)
                _,bd,blendmeta=predict_flow_tiled(model,d['n_coarse'],d['n_rgb'],ref.device,tile=TILE,stride=STRIDE,batch=BATCH,steps=STEPS,return_residual=True)
            candidate=apply_residual(d['coarse'],project_residual_512(bd,ref.dataset))
            blend=(1-alpha)*d['coarse']+alpha*candidate
            native['blend_candidate_residual_native']=bd.numpy()
        else:blend=d['coarse'].clone()
        nativepath=dest/f'{name}_native_predictions_{tag}.npz';np.savez_compressed(nativepath,**native)
        predpath=dest/f'{name}_predictions_{tag}.npz'
        np.savez_compressed(predpath,baseline=d['coarse'].numpy(),raw=raw.numpy(),blend=blend.numpy(),ground_truth=d['g'].numpy(),fov=d['fov'].numpy().astype(bool))
        with np.load(predpath,allow_pickle=False) as z:
            assert all(z[k].dtype==np.float32 and z[k].shape==(1,512,512) and np.isfinite(z[k]).all() for k in ('baseline','raw','blend','ground_truth'))
            assert np.array_equal(z['raw'],raw.numpy()) and np.array_equal(z['blend'],blend.numpy())
            for kind,calls in (('raw',meta),('blend',blendmeta)):
                m=metrics(z[kind][0],z['ground_truth'][0],z['baseline'][0],cutoff,z['fov'][0])
                rows.append(dict(dataset=ref.dataset,method='NativePatchFM' if kind=='raw' else 'NativePatchFM_validation_blend',seed=seed,image=name,output=kind,selected_epoch=decision[kind]['epoch'],alpha=1. if kind=='raw' else alpha,velocity_calls=0 if calls is None else calls['velocity_calls'],**m))
        entries.append(dict(dataset=ref.dataset,seed=seed,image=name,path=str(predpath),sha256=sha256_file(predpath),native_path=str(nativepath),native_sha256=sha256_file(nativepath),raw_checkpoint_sha256=decision['raw_sha256'],blend_checkpoint_sha256=decision['blend_sha256'],raw=meta,blend=blendmeta))
    with (dest/f'test_metrics_{tag}.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(dest/f'inference_{tag}.json',entries)
    write_json(dest/f'completed_{tag}.json',dict(completed_at=stamp(),rows=len(rows),test_images=len(ref.manifest['test']),FP32_roundtrip_verified=True,primary_raw=True,secondary_blend=True))
    del model,rawstate,blendstate;gc.collect();torch.cuda.empty_cache()
    return rows,entries

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if not args.execute:print('No training/inference without --execute. Datasets fixed to CHASE and DRIVE.');return
    out=(args.output or ROOT/'outputs'/f'native_patch_flow_study_{stamp()}').resolve();out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,stage='preparing',datasets=list(DATASETS),seeds=list(SEEDS),new_fits=10,completed_fits=0,epochs=EPOCHS,HRF_started=False)
    def status(stage,**extra):
        state.update(stage=stage,updated_at=stamp(),**extra)
        write_json(out/f'controller_status_{tag}.json',state)
        write_json(ROOT/'tmp/current_native_patch_flow_work.json',state)
    status.state=state;status('hashing_protected_inputs')
    old=protected_files();write_json(out/f'protected_before_{tag}.json',old)
    sources={}
    for path in (ROOT/'flow_matching_refiner').glob('*.py'):
        dest=out/'source'/f'{path.stem}_{tag}.py';dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest);sources[path.name]=sha256_file(path)
    plan=dict(**state,source_sha256=sources,baseline_seed=42,new_baseline_fits=0,new_OOF_fits=0,
       reuse='Exact retained seed42 OOF training maps; held-out validation/test maps from fixed full baseline; native RGB/train/validation GT; no HRF loading',
       architecture='Same ungated ErrorSelectiveModel(False) as option3; auxiliary error network trained but never gates inference',
       flow='Coarse logits -> full GT +/-6 logit straight path; t uniform[0,1] with 25% atom at0; normalized velocity MSE',
       loss='FM +0.5 BCE+0.5 Dice+0.1 softclDice+0.1 balanced auxiliary gate BCE, valid FOV only; no new thin/anchor terms',
       inference='Eight explicit Euler steps per native tile, deterministic coarse-logit start; no target/GT/FOV argument',
       native_tiles=dict(tile=TILE,stride=STRIDE,batch=BATCH,patches_per_training_image_per_epoch=PATCHES,vessel_center_probability=.5),
       matched_training='Same120epochs, AdamW3e-4 weight_decay1e-4 cosine/no warmup; batch16; same native patch+rot90/flip RNG seed+141001 as O3; independent time RNG seed+20261117',
       selection='Raw checkpoint highest validation image-macro Dice among epochs10..120, earliest ties. Blend alpha0,.25,.5,.75,1 separate secondary checkpoint. All10 selections globally locked before ANY new refiner test inference/metric. Input loader replays frozen baseline predictions for source verification only.',
       projection='Positive-Hann integrated logit residual native fusion, no halo; project residual only with Pillow float nearest to512; reapply to original512coarse logits, exact zero residual identity',
       primary_evaluation='Original512 GT/FOV threshold>=0.5; original training-derived thin cutoff; raw model primary; blended results explicitly secondary',
       inference_statistics='Five training-seed means and sampleSD; raw NativePatchFM-baseline and raw NativePatchFM-O3 for Dice/clDice/thin recall on bothdatasets (12test family); paired seed t/W and separate4subject/20image sensitivities; unadjusted t95CI; old test cohort previously inspected',
       limitations=['Exploratory new method on previously examined fixed cohorts','Single frozen baseline/OOFset, not five full pipelines','Native coarse maps are interpolated512 probabilities, not native baseline inference','Retained local DRIVE test-label provenance not independently verified','FM versus Direct adds flow loss/random times/eight-step integration; no isolated single-factor mechanism claim'],
       old_results_and_manuscript_unchanged=True)
    write_json(out/f'study_plan_{tag}.json',plan)
    try:
        import numpy as np
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        from flow_matching_refiner.native_patch_model import training_center_pools
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available(),'CUDA is required for the authorized training run'
        write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),deterministic_cudnn=True,TF32=False))
        refs={};decisions=[];cutoffs={}
        # No new test prediction is requested until every dataset/seed is locked.
        for dataset in DATASETS:
            job=out/f'{dataset}_{tag}';job.mkdir()
            status('loading_audited_native_OOF',dataset=dataset)
            ref,audit=load_native_dataset(dataset,job/'input_audit','cuda');refs[dataset]=ref
            assert ref.dataset in DATASETS and all(ref.data[n]['n_g'] is not None for n in ref.manifest['train']+ref.manifest['validation'])
            for p,h in audit['source_sha256_before'].items():
                if p in old:assert old[p]==h
                old[p]=h
            thinfile=ROOT/f'outputs/reviewer_completion_2026-11-16_19-37-22/controlled_ablations_2026-11-16_19-41-32/{dataset}_2026-11-16_19-41-32/thin_definition_2026-11-16_19-41-32.json'
            thin=json.loads(thinfile.read_text());assert thin['cutoff']==1. and thin['training_image_ids']==ref.manifest['train']
            cutoffs[dataset]=thin['cutoff'];old[str(thinfile)]=sha256_file(thinfile)
            write_json(job/f'input_metadata_{tag}.json',dict(dataset=dataset,split=ref.manifest,thin_cutoff=thin,source_audit=audit['output_audit'],source_audit_sha256=sha256_file(audit['output_audit'])))
            pools={n:training_center_pools(ref.data[n]['n_g'],ref.data[n]['n_fov']) for n in ref.manifest['train']}
            for seed in SEEDS:
                decisions.append(train_one(ref,job,tag,seed,pools,status));status('fit_completed',completed_fits=len(decisions),dataset=dataset,seed=seed)
            del pools;gc.collect();torch.cuda.empty_cache()
        assert len(decisions)==10 and {(d['dataset'],d['refiner_seed']) for d in decisions}=={(d,s) for d in DATASETS for s in SEEDS}
        lock=out/f'all10_choices_locked_before_any_test_{tag}.json'
        write_json(lock,dict(locked_at=stamp(),decisions=decisions,study_plan_sha256=sha256_file(out/f'study_plan_{tag}.json')))
        lockhash=sha256_file(lock);status('all10_choices_locked',lock_sha256=lockhash)
        entries=[];rows=[]
        for dataset in DATASETS:
            ref=refs[dataset];job=out/f'{dataset}_{tag}'
            for decision in decisions:
                if decision['dataset']==dataset:
                    rr,ee=evaluate_one(ref,job,tag,decision,cutoffs[dataset],status);rows.extend(rr);entries.extend(ee)
        assert len(entries)==140 and len(rows)==280 and sha256_file(lock)==lockhash
        with (out/f'all_new_test_metrics_{tag}.csv').open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        write_json(out/f'prediction_manifest_{tag}.json',dict(datasets=list(DATASETS),seeds=list(SEEDS),records=entries,global_lock_path=str(lock),global_lock_sha256=lockhash))
        status('checking_protected_sources',completed_fits=10)
        after={p:sha256_file(p) for p in old};assert after==old,'Old source/result changed'
        write_json(out/f'protected_before_including_loaded_sources_{tag}.json',old)
        write_json(out/f'protected_after_{tag}.json',after)
        write_json(out/f'completed_training_and_inference_{tag}.json',dict(completed_at=stamp(),new_fits=10,optimizer_updates=4800,test_prediction_files=140,raw_and_secondary_blend_records=280,all_FP32_roundtrips_verified=True,all10_choices_locked_before_any_test=True,lock_path=str(lock),lock_sha256=lockhash,datasets=list(DATASETS),seeds=list(SEEDS),HRF_started=False,previous_files_unchanged=True,protected_files=len(old)))
        status('training_and_inference_complete',completed_fits=10);print('NATIVE_FLOW_COMPLETE',str(out),flush=True)
    except BaseException:
        write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
