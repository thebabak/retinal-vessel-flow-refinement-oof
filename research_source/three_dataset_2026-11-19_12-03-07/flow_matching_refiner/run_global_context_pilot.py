"""Whole-image pretrained context: paired pilot, conditional five-seed CHASE study.

Run only with --execute. All fitting choices use validation, never test.
"""
from __future__ import annotations
import argparse,csv,gc,json,random,shutil,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.geometry_flow_data import load_cases
from flow_matching_refiner.global_context_data import build_context,load_test_cases,rgb_sha,WEIGHTS
from flow_matching_refiner.global_context_flow import GlobalContextCorrection,sample_context,predict_delta,self_test
from flow_matching_refiner.native_rollout_pilot import objective,eligible
from flow_matching_refiner.run_native_rollout_pilot import prepare_training,draw,score,rank,savecsv,PARTS,state_sha
from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual

SEEDS=(42,123,2026,3000,3001);KEYS=('dice','cldice','thin_skeleton_recall','precision');BETAS=(.125,.25,.5,1.);STEPS=(2,4,8);EPOCHS=80
CONFIG=dict(name='whole_image_resnet18_preservation1',preservation_weight=1.)

def train(ref,cases,context,seed,mode,out,tag,status,pilot):
    out.mkdir(parents=True,exist_ok=False)
    assert all(rgb_sha(c['rgb'])==context[n]['rgb_sha256'] for n,c in cases.items())
    torch.manual_seed(seed+88177);torch.cuda.manual_seed_all(seed+88177);model=GlobalContextCorrection(mode).to(ref.device);initial=state_sha(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS);rng=random.Random(seed+44111);trng=torch.Generator(device=ref.device).manual_seed(seed+88178)
    cache,pools=prepare_training(ref,cases);validation=[cases[n] for n in ref.manifest['validation']];base={k:float(np.mean([score(ref,c,c['source512'])[k] for c in validation])) for k in KEYS};best=None;raw=None;vr=[]
    write_json(out/f'input_lineage_{tag}.json',dict(training=[cases[n]['record'] for n in ref.manifest['train']],validation=[cases[n]['record'] for n in ref.manifest['validation']],context=[dict(image=n,path=v['path'],sha256=v['sha256'],RGB_array_sha256=v['rgb_sha256']) for n,v in context.items()],pool_counts={n:{k:len(v) for k,v in pp.items()} for n,pp in pools.items()},baseline_validation=base,GT_train_only=True))
    with (out/f'training_{tag}.csv').open('x',newline='') as tf,(out/f'patch_draws_{tag}.csv').open('x',newline='') as pf:
        tw=csv.writer(tf);pw=csv.writer(pf);tw.writerow(['epoch','loss',*PARTS,'lr','patches','optimizer_updates']);pw.writerow(['epoch','patch_index','image','top','left','centre_y','centre_x','requested_group','fallback_random','rotation90','flip_horizontal','flip_vertical','source_array_sha256','context_sha256','context_augmentation_index'])
        for epoch in range(1,EPOCHS+1):
            status('pilot_training' if pilot else 'five_seed_training',mode=mode,seed=seed,epoch=epoch,epochs=EPOCHS);model.train();names=ref.manifest['train']*4;rng.shuffle(names);total=np.zeros(1+len(PARTS));seen=0;updates=0;lr=optimizer.param_groups[0]['lr']
            for start in range(0,len(names),8):
                samples=[];align=[];contexts=[]
                for j,n in enumerate(names[start:start+8],start=start):
                    arrays,al,meta=draw(cases[n],cache[n],pools[n],rng);samples.append(arrays);align.append(al);aug=(meta['rotation90'],meta['flip_horizontal'],meta['flip_vertical']);ci=context[n]['augs'].index(aug);contexts.append(sample_context(context[n]['spatial'][ci],context[n]['global_vector'][ci],cases[n]['source'].shape[-2:],meta['top'],meta['left'],*aug));pw.writerow([epoch,j,n,*(meta[k] for k in ('top','left','centre_y','centre_x','requested_group','fallback_random','rotation90','flip_horizontal','flip_vertical')),cases[n]['source_array_sha256'],context[n]['sha256'],ci])
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0,len(samples),4):
                    batch=samples[micro:micro+4];tensors=[torch.stack([s[k] for s in batch]).to(ref.device) for k in range(6)];cc=contexts[micro:micro+4];model.bind(torch.stack([v[0] for v in cc]).to(ref.device),torch.stack([v[1] for v in cc]).to(ref.device));time=torch.rand(len(batch),device=ref.device,generator=trng)
                    value,parts=objective(model,*tensors,align[micro:micro+4],time,CONFIG);assert torch.isfinite(value);(value*len(batch)/len(samples)).backward();model.unbind();total+=np.asarray([value.item(),*(parts[k].item() for k in PARTS)])*len(batch);seen+=len(batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
            schedule.step();assert seen==64 and updates==8;tw.writerow([epoch,*(total/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:print('CONTEXT CHASE', 'PILOT' if pilot else 'FINAL',mode,seed,'epoch',epoch,'loss',float(total[0]/seen),flush=True)
            if epoch%20:continue
            cp=out/f'field_epoch{epoch}_{tag}.pt';torch.save(dict(state=model.state_dict(),mode=mode,seed=seed,config=CONFIG,epoch=epoch),cp);ch=sha256_file(cp)
            for steps in (STEPS if mode=='flow' else (1,)):
                values={b:[] for b in BETAS}
                for c in validation:
                    status('validation_only_selection',mode=mode,seed=seed,epoch=epoch,steps=steps,image=c['name']);ct=context[c['name']];delta,_=predict_delta(model,c['source'],c['rgb'],ct['spatial'][0],ct['global_vector'][0],ref.device,steps=steps);d512=project_residual_512(delta,'CHASE')
                    for beta in BETAS:
                        p=apply_residual(c['source512'],beta*d512);m=score(ref,c,p);mm={k:m[k] for k in KEYS};values[beta].append(mm);vr.append(dict(epoch=epoch,steps=steps,beta=beta,image=c['name'],changed_pixels=int(((p>=.5)!=(c['source512']>=.5)).sum()),**mm))
                for beta,vs in values.items():
                    mm={k:float(np.mean([v[k] for v in vs])) for k in KEYS};q=dict(epoch=epoch,steps=steps,beta=beta,metrics=mm,structural_gain=rank(mm,base),joint_gate=eligible(mm,base),checkpoint=str(cp),checkpoint_sha256=ch);rr=mm['dice']+.1*mm['cldice']+.02*mm['thin_skeleton_recall']
                    if raw is None or rr>raw['rank']+1e-12:raw=dict(**q,rank=rr)
                    if q['joint_gate'] and (best is None or q['structural_gain']>best['structural_gain']+1e-12):best=q
                print('CONTEXT VALIDATION',mode,seed,epoch,'steps',steps,'qualified',best,'best_raw',raw,flush=True)
            savecsv(out/f'validation_epoch{epoch}_{tag}.csv',[r for r in vr if r['epoch']==epoch])
    assert raw is not None;savecsv(out/f'validation_all_{tag}.csv',vr);selection=best if best else dict(epoch=0,steps=4 if mode=='flow' else 1,beta=0.,metrics=base,structural_gain=0.,joint_gate=False,checkpoint=raw['checkpoint'],checkpoint_sha256=raw['checkpoint_sha256'])
    decision=dict(mode=mode,seed=seed,pilot=pilot,config=CONFIG,epochs=EPOCHS,initial_state_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),baseline_validation=base,selection=selection,best_raw=raw,pilot_gate_passed=best is not None,locked_at=stamp(),selection_uses_test=False)
    write_json(out/f'selection_locked_{tag}.json',decision);del model,optimizer,schedule,cache,pools;gc.collect();torch.cuda.empty_cache();return decision

def paired_train(ref,cases,ctx,seed,out,tag,status,pilot):
    pair=[train(ref,cases,ctx,seed,mode,out/mode/f'seed_{seed}',tag,status,pilot) for mode in ('flow','direct')]
    assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256'] and pair[0]['parameter_count']==pair[1]['parameter_count']
    assert sha256_file(out/'flow'/f'seed_{seed}'/f'patch_draws_{tag}.csv')==sha256_file(out/'direct'/f'seed_{seed}'/f'patch_draws_{tag}.csv');return pair

def run_test(systems,testcases,context,out,tag,status,lock):
    records=[]
    for d in systems:
        model=GlobalContextCorrection(d['mode']).cuda();q=d['selection'];assert sha256_file(q['checkpoint'])==q['checkpoint_sha256'];model.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=True)['state']);model.eval();dest=out/'test'/d['mode']/f'seed_{d["seed"]}';dest.mkdir(parents=True,exist_ok=False)
        for n,c in testcases[d['seed']].items():
            status('locked_test_inference',mode=d['mode'],seed=d['seed'],image=n);ct=context[n];assert rgb_sha(c['rgb'])==ct['rgb_sha256']
            if q['beta']==0:delta=torch.zeros_like(c['source']);meta=dict(identity=True,GT_free=True)
            else:delta,meta=predict_delta(model,c['source'],c['rgb'],ct['spatial'][0],ct['global_vector'][0],'cuda',steps=q['steps'])
            p=apply_residual(c['source512'],q['beta']*project_residual_512(delta,'CHASE'));fov=torch.ones_like(c['g512'],dtype=torch.bool);ref=SimpleNamespace(data={n:dict(g=c['g512'],fov=fov)});path=dest/f'{n}_predictions_{tag}.npz';np.savez_compressed(path,O3=c['source512'].numpy(),refined=p.numpy(),coarse=c['coarse512'].numpy(),delta_native=delta.numpy(),ground_truth=c['g512'].numpy(),fov=fov.numpy())
            records.append(dict(mode=d['mode'],seed=d['seed'],image=n,path=str(path),sha256=sha256_file(path),source=c['record'],selection=q,inference=meta,metrics=score(ref,c,p),O3_metrics=score(ref,c,c['source512'])))
        del model;gc.collect();torch.cuda.empty_cache()
    assert len(records)==80;write_json(out/f'prediction_manifest_{tag}.json',dict(records=records,global_lock=str(lock),global_lock_sha256=sha256_file(lock)));return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if not args.execute:print('Inert without --execute. CHASE only.');return
    out=args.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and not out.exists();out.mkdir()
    state=dict(output=str(out),tag=tag,dataset='CHASE',HRF_started=False,DRIVE_started=False)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_global_context_pilot.json',state)
    try:
        self_test();torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();ref,cases,protected=load_cases(42)
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
        protected[str(WEIGHTS)]=sha256_file(WEIGHTS)
        for name in ('template261006.tex','last.tex','tmp/current_native_rollout_delivery.json','tmp/current_geometry_flow_delivery.json'):
            p=ROOT/name
            if p.is_file():protected[str(p)]=sha256_file(p)
        write_json(out/f'protected_before_{tag}.json',protected);snap=out/'source';snap.mkdir()
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):shutil.copy2(p,snap/f'{p.stem}_{tag}.py')
        plan=dict(**state,locked_at=stamp(),pilot_seed=42,epochs=EPOCHS,config=CONFIG,validation_epochs=[20,40,60,80],validation_steps_flow=STEPS,validation_betas=BETAS,seeds=SEEDS,gate='Mean validation Dice>=O3-1e-12; clDice and thin recall>O3+1e-12; precision secondary only. Choose maximal mean structural gain, ties keep earlier. Identity fallback. Pilot Flow must pass before ten NEW matched final fits; direct pilot is reported regardless.',pretrained='Frozen torchvision ResNet18 ImageNet1K V1. Whole RGB224 resize, no center crop, ImageNet normalization. Same augmented full-image layer3 spatial and layer4 pooled512 context for Flow/direct. Train-only patch source chain is held-subject OOF. No retinal training for encoder.',training='80epochs x64 native256patches, effectivebatch8/micro4, AdamW3e-4 wd1e-4 cosine. Fixed prior rollout endpoint/CFM/imitation/preservation1 objective; direct same endpoint loss without CFM/imitation. Identical crop schedule, initialization and trainable parameter count.',selection='Four validation images/two subjects only. Flow2/4/8steps; direct1. Both evaluate4betas,4epochs. All ten choices locked before any new test source/context reading or test inference.',comparison='Frozen ImageNet encoder overhead shared; parameter match does not imply equal compute or model selection budgets. Five refiner seeds share frozen upstream fitting. Previously reused validation/test cohorts; exploratory development, not independent confirmation.',evaluation='512comparison grid; threshold>=.5; whole CHASE image; GT skeleton EDT radius<=1 for thin recall. Native radius2 is auxiliary training only.',previous_results_unchanged=True)
        write_json(out/f'plan_{tag}.json',plan);write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False));status('building_train_validation_context');context,cm=build_context(cases,out/'RGB_context_train_validation',tag,status)
        pilot=paired_train(ref,cases,context,42,out/'pilot',tag,status,True);gate=out/f'pilot_gate_locked_{tag}.json';passed=pilot[0]['pilot_gate_passed'];write_json(gate,dict(locked_at=stamp(),passed=passed,decisions=pilot,no_test_scores_used=True));gh=sha256_file(gate);print('CONTEXT PILOT GATE',passed,flush=True);systems=[];records=[]
        if passed:
            for seed in SEEDS:
                if seed!=42:ref,cases,pp=load_cases(seed);protected.update(pp)
                systems.extend(paired_train(ref,cases,context,seed,out/'final',tag,status,False));del cases;gc.collect();torch.cuda.empty_cache()
            lock=out/f'all10_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=systems,pilot_gate_sha256=gh));testcases={s:load_test_cases(s) for s in SEEDS};tc,tm=build_context(testcases[42],out/'RGB_context_test',tag,status,False)
            for cs in testcases.values():
                for c in cs.values():protected[c['record']['path']]=c['record']['sha256'];pr=c['record']['provenance'];protected[pr['O3_checkpoint']]=pr['O3_sha256']
            records=run_test(systems,testcases,tc,out,tag,status,lock)
        assert sha256_file(gate)==gh;status('verifying_previous_files');assert all(sha256_file(p)==h for p,h in protected.items());write_json(out/f'protected_before_including_sources_{tag}.json',protected)
        result=dict(completed_at=stamp(),dataset='CHASE',pilot_fits=2,pilot_gate_passed=passed,final_fits=len(systems),prediction_files=len(records),previous_files_unchanged=True,HRF_started=False,DRIVE_started=False,gate=str(gate),gate_sha256=gh);write_json(out/f'completed_experiment_{tag}.json',result);status('complete',**result);print('CONTEXT EXPERIMENT COMPLETE',result,flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
