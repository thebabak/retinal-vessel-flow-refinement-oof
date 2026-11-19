"""CHASE-only native Flow pilot; strict validation gate precedes any five-seed run."""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,random,shutil,traceback
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.native_rollout_pilot import CorrectionField,objective,predict_delta,eligible,CAP
from flow_matching_refiner.native_rollout_sources import build_cases
from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
from flow_matching_refiner.topology_oof_flow import prepare_supervision,patch_supervision
from flow_matching_refiner.structure_metrics_all import metrics

SEEDS=(42,123,2026,3000,3001);KEYS=('dice','cldice','thin_skeleton_recall');BETAS=(.125,.25,.5,1.);STEPS=(2,4,8);EPOCHS=80
CONFIGS=[dict(name='preservation1',preservation_weight=1.),dict(name='preservation4',preservation_weight=4.)]
PARTS=('fm','onpolicy','bce','eval_bce','dice','eval_dice','native_cldice','native_thin','native_skeleton','cldice','thin','skeleton','preservation','dice_regret','eval_dice_regret','structure')
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def savecsv(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def score(ref,c,p):
    d=ref.data[c['name']];return metrics(p.numpy()[0],d['g'].numpy()[0],c['coarse512'].numpy()[0],1.,d['fov'].numpy()[0])
def rank(mm,base):return .5*(mm['cldice']-base['cldice']+mm['thin_skeleton_recall']-base['thin_skeleton_recall'])
def prepare_training(ref,cases):
    cache=prepare_supervision(ref);pools={}
    for name in ref.manifest['train']:
        c=cases[name];p=c['source'].numpy()[0]>=.5;g=c['g'].numpy()[0]>=.5;valid=c['fov'].numpy()[0]>.5;sk=skeletonize(g&valid);radius=ndi.distance_transform_edt(np.pad(g&valid,1))[1:-1,1:-1];thin=sk&(radius<=2.)
        c['sk']=torch.from_numpy(sk.copy())[None].float();c['thin']=torch.from_numpy(thin.copy())[None].float();c['source_array_sha256']=arrsha(c['source'].numpy())
        masks=dict(missed_thin=thin&~p,near_false_positive=p&~g&ndi.binary_dilation(g,iterations=6),correct_thin=thin&p,random=valid)
        pools[name]={k:np.flatnonzero(v) for k,v in masks.items()};assert len(pools[name]['random'])>0
    return cache,pools

def draw(c,cache,pools,rng):
    u=rng.random();group='missed_thin' if u<.4 else 'near_false_positive' if u<.65 else 'correct_thin' if u<.85 else 'random';indices=pools[group];fallback=not len(indices)
    if fallback:indices=pools['random']
    h,w=c['g'].shape[-2:];cy,cx=divmod(int(indices[rng.randrange(len(indices))]),w);top=min(h-256,max(0,cy-128));left=min(w-256,max(0,cx-128));meta=dict(top=top,left=left)
    alignment,nsk=patch_supervision(cache,meta);yi=torch.searchsorted(torch.from_numpy(cache['ys']),alignment['iy']+top);xi=torch.searchsorted(torch.from_numpy(cache['xs']),alignment['ix']+left);alignment['source']=c['source512'][:,yi[:,None],xi[None,:]].contiguous();assert torch.equal(alignment['gt'],c['g'][:,top+alignment['iy'][:,None],left+alignment['ix'][None,:]])
    k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5;arrays=[]
    for key in ('source','rgb','g','fov','sk','thin'):
        a=torch.rot90(c[key][:,top:top+256,left:left+256],k,(-2,-1))
        if fh:a=a.flip(-1)
        if fv:a=a.flip(-2)
        arrays.append(a.contiguous())
    alignment.update(rotation90=k,flip_horizontal=fh,flip_vertical=fv)
    return arrays,alignment,dict(top=top,left=left,centre_y=cy,centre_x=cx,requested_group=group,fallback_random=fallback,rotation90=k,flip_horizontal=fh,flip_vertical=fv)

def train(ref,cases,seed,mode,cfg,out,tag,status,pilot):
    out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed+88177);torch.cuda.manual_seed_all(seed+88177);model=CorrectionField(mode).to(ref.device);initial_hash=state_sha(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,EPOCHS);rng=random.Random(seed+44111);trng=torch.Generator(device=ref.device).manual_seed(seed+88178)
    cache,pools=prepare_training(ref,cases);validation=[cases[n] for n in ref.manifest['validation']];base={k:float(np.mean([score(ref,c,c['source512'])[k] for c in validation])) for k in KEYS};best=None;raw=None;validation_rows=[]
    old_lock=ROOT/'outputs/local_flow_acceptor_CHASE_v2_2026-11-17_22-19-21/all10_system_choices_locked_before_test_2026-11-17_22-19-21.json';old_base=next(d['O3_validation'] for d in json.loads(old_lock.read_text())['decisions'] if d['mode']=='flow' and d['seed']==seed);assert base==old_base,'Full Option3 validation source changed'
    write_json(out/f'input_lineage_{tag}.json',dict(training=[cases[n]['record'] for n in ref.manifest['train']],validation=[cases[n]['record'] for n in ref.manifest['validation']],pool_counts={n:{k:len(v) for k,v in pp.items()} for n,pp in pools.items()},baseline_validation=base,training_labels_only_for_crops=True))
    with (out/f'training_{tag}.csv').open('x',newline='') as tf,(out/f'patch_draws_{tag}.csv').open('x',newline='') as pf:
        tw=csv.writer(tf);pw=csv.writer(pf);tw.writerow(['epoch','loss',*PARTS,'lr','patches','optimizer_updates']);pw.writerow(['epoch','patch_index','image','top','left','centre_y','centre_x','requested_group','fallback_random','rotation90','flip_horizontal','flip_vertical','source_array_sha256'])
        for epoch in range(1,EPOCHS+1):
            status('pilot_training' if pilot else 'five_seed_training',mode=mode,seed=seed,config=cfg['name'],epoch=epoch,epochs=EPOCHS);model.train();names=ref.manifest['train']*4;rng.shuffle(names);total=np.zeros(1+len(PARTS));seen=0;updates=0;lr=optimizer.param_groups[0]['lr']
            for start in range(0,len(names),8):
                samples=[];align=[]
                for j,n in enumerate(names[start:start+8],start=start):
                    arrays,al,meta=draw(cases[n],cache[n],pools[n],rng);samples.append(arrays);align.append(al);pw.writerow([epoch,j,n,*(meta[k] for k in ('top','left','centre_y','centre_x','requested_group','fallback_random','rotation90','flip_horizontal','flip_vertical')),cases[n]['source_array_sha256']])
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0,len(samples),4):
                    batch=samples[micro:micro+4];tensors=[torch.stack([s[k] for s in batch]).to(ref.device) for k in range(6)];time=torch.rand(len(batch),device=ref.device,generator=trng)
                    value,parts=objective(model,*tensors,align[micro:micro+4],time,cfg);assert torch.isfinite(value);(value*len(batch)/len(samples)).backward();total+=np.asarray([value.item(),*(parts[k].item() for k in PARTS)])*len(batch);seen+=len(batch)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
            schedule.step();assert seen==64 and updates==8;tw.writerow([epoch,*(total/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:print('CHASE', 'PILOT' if pilot else 'FINAL',mode,seed,cfg['name'],'epoch',epoch,'loss',float(total[0]/seen),flush=True)
            if epoch%20!=0:continue
            cp=out/f'field_epoch{epoch}_{tag}.pt';torch.save(dict(state=model.state_dict(),mode=mode,seed=seed,config=cfg,epoch=epoch),cp);cpsha=sha256_file(cp)
            for steps in (STEPS if mode=='flow' else (1,)):
                values={b:[] for b in BETAS}
                for c in validation:
                    status('validation_only_selection',mode=mode,seed=seed,config=cfg['name'],epoch=epoch,steps=steps,image=c['name']);delta,_=predict_delta(model,c['source'],c['rgb'],ref.device,steps=steps);d512=project_residual_512(delta,'CHASE')
                    for beta in BETAS:
                        p=apply_residual(c['source512'],beta*d512);m=score(ref,c,p);mm={k:m[k] for k in KEYS};values[beta].append(mm);validation_rows.append(dict(epoch=epoch,steps=steps,beta=beta,image=c['name'],changed_pixels=int(((p>=.5)!=(c['source512']>=.5)).sum()),**mm))
                for beta,vs in values.items():
                    mm={k:float(np.mean([v[k] for v in vs])) for k in KEYS};q=dict(epoch=epoch,steps=steps,beta=beta,metrics=mm,structural_gain=rank(mm,base),joint_gate=eligible(mm,base),checkpoint=str(cp),checkpoint_sha256=cpsha);rawrank=mm['dice']+.1*mm['cldice']+.02*mm['thin_skeleton_recall']
                    if raw is None or rawrank>raw['rank']+1e-12:raw=dict(**q,rank=rawrank)
                    if q['joint_gate'] and (best is None or q['structural_gain']>best['structural_gain']+1e-12):best=q
                print('CHASE VALIDATION',mode,seed,cfg['name'],epoch,'steps',steps,'qualified',best,'best_raw',raw,flush=True)
            savecsv(out/f'validation_epoch{epoch}_{tag}.csv',[r for r in validation_rows if r['epoch']==epoch])
    assert raw is not None;savecsv(out/f'validation_all_{tag}.csv',validation_rows);selection=best if best else dict(epoch=0,steps=4 if mode=='flow' else 1,beta=0.,metrics=base,structural_gain=0.,joint_gate=False,checkpoint=raw['checkpoint'],checkpoint_sha256=raw['checkpoint_sha256'])
    decision=dict(mode=mode,seed=seed,pilot=pilot,config=cfg,epochs=EPOCHS,initial_state_sha256=initial_hash,parameter_count=sum(p.numel() for p in model.parameters()),baseline_validation=base,selection=selection,best_raw=raw,pilot_gate_passed=best is not None,locked_at=stamp(),input_lineage=str(out/f'input_lineage_{tag}.json'),selection_uses_test=False)
    write_json(out/f'selection_locked_{tag}.json',decision);del model,optimizer,schedule,cache,pools;gc.collect();torch.cuda.empty_cache();return decision

def run_test(ref,systems,source_cases,out,tag,status,lock):
    records=[]
    for d in systems:
        model=CorrectionField(d['mode']).to(ref.device);q=d['selection'];assert sha256_file(q['checkpoint'])==q['checkpoint_sha256'];model.load_state_dict(torch.load(q['checkpoint'],map_location=ref.device,weights_only=True)['state']);model.eval();dest=out/'test'/d['mode']/f'seed_{d["seed"]}';dest.mkdir(parents=True,exist_ok=False)
        for n in ref.manifest['test']:
            status('locked_test_inference',mode=d['mode'],seed=d['seed'],image=n);c=source_cases[d['seed']][n]
            if q['beta']==0:delta=torch.zeros_like(c['source']);meta=dict(identity=True,GT_free=True)
            else:delta,meta=predict_delta(model,c['source'],c['rgb'],ref.device,steps=q['steps'])
            p=apply_residual(c['source512'],q['beta']*project_residual_512(delta,'CHASE'));data=ref.data[n];path=dest/f'{n}_predictions_{tag}.npz';np.savez_compressed(path,O3=c['source512'].numpy(),refined=p.numpy(),coarse=c['coarse512'].numpy(),delta_native=delta.numpy(),ground_truth=data['g'].numpy(),fov=data['fov'].numpy())
            records.append(dict(mode=d['mode'],seed=d['seed'],image=n,path=str(path),sha256=sha256_file(path),source=c['record'],selection=q,inference=meta,metrics=score(ref,c,p),O3_metrics=score(ref,c,c['source512'])))
        del model;gc.collect();torch.cuda.empty_cache()
    write_json(out/f'prediction_manifest_{tag}.json',dict(records=records,global_lock=str(lock),global_lock_sha256=sha256_file(lock)));assert len(records)==80;return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if not args.execute:print('Inert; --execute required. CHASE only.');return
    out=args.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and not (out/f'plan_{tag}.json').exists();out.mkdir(exist_ok=True)
    state=dict(output=str(out),tag=tag,dataset='CHASE',HRF_started=False,DRIVE_started=False)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_native_rollout_pilot.json',state)
    prior=ROOT/'outputs/local_flow_acceptor_CHASE_v2_2026-11-17_22-19-21';protected=json.loads((prior/'protected_before_including_loaded_sources_2026-11-17_22-19-21.json').read_text())
    for p in prior.rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    assert all(sha256_file(p)==h for p,h in protected.items());write_json(out/f'protected_before_{tag}.json',protected)
    source=out/'source';source.mkdir()
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):shutil.copy2(p,source/f'{p.stem}_{tag}.py')
    plan=dict(**state,configs=CONFIGS,pilot_seed=42,epochs=EPOCHS,steps=STEPS,betas=BETAS,seeds=SEEDS,strict_gate='mean validation Dice>=O3 (tolerance1e-12), clDice>O3+1e-12 and thin recall>O3+1e-12. Both pilots completed; select maximal mean structural gain. No qualified candidate => stop, no final fitting or new test inference.',OOF='Native O3 maps regenerated through held-subject coarse baseline, fold O3 fit and nested training dependencies; old source cache not used.',training='Train-only missed-thin/near-FP/correct-thin/random crop probabilities.4/.25/.2/.15; 64native256patches/epoch; effectivebatch8, microbatch4; AdamW3e-4 wd1e-4 cosine. Flow4Euler actual-rollout endpoint training, CFM plus detached mid-rollout corrective imitation. Direct same architecture/data/init/loss except no FM/imitation.',comparison='If pilot passes: ten NEW80epoch fits, five matched seeds each Flow/direct, chosen pilot preservation weight fixed. Each uses own-seed frozen fold/fullO3. All ten choices lock before test. Parameter counts matched; inference calls differ.',evaluation='Fixed512primary grid, threshold>=.5, whole-image CHASE, thin GT skeleton radius1; native radius2 is an auxiliary training support only.',limitations='Repeatedly used four validation images and previously examined four-subject test cohort; this is exploratory model development, not independent confirmation. Shared fold baseline models; five conditional refiner seeds.',old_results_unchanged=True)
    write_json(out/f'plan_{tag}.json',plan)
    try:
        from flow_matching_refiner.native_patch_data import load_native_dataset
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
        write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False));status('loading_CHASE_only');ref,audit=load_native_dataset('CHASE',out/'input_audit','cuda');protected.update(audit['source_sha256_before'])
        cases,recs,pp=build_cases(ref,42,out/'sources/seed_42',tag,status);protected.update(pp);pilot=[]
        for cfg in CONFIGS:pilot.append(train(ref,cases,42,'flow',cfg,out/'pilot'/cfg['name'],tag,status,True))
        qualified=[d for d in pilot if d['pilot_gate_passed']];chosen=max(qualified,key=lambda d:d['selection']['structural_gain']) if qualified else None;gate=out/f'pilot_gate_locked_{tag}.json';write_json(gate,dict(locked_at=stamp(),passed=bool(chosen),selected=chosen,decisions=pilot,no_test_scores_used=True));gh=sha256_file(gate);status('pilot_gate_passed' if chosen else 'pilot_gate_failed',gate_passed=bool(chosen));print('PILOT GATE',bool(chosen),chosen,flush=True)
        systems=[];test_records=[]
        if chosen:
            for seed in SEEDS:
                if seed!=42:cases,recs,pp=build_cases(ref,seed,out/f'sources/seed_{seed}',tag,status);protected.update(pp)
                pair=[]
                for mode in ('flow','direct'):pair.append(train(ref,cases,seed,mode,chosen['config'],out/'final'/mode/f'seed_{seed}',tag,status,False));systems.append(pair[-1])
                assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256'] and pair[0]['parameter_count']==pair[1]['parameter_count']
                assert sha256_file(out/'final/flow'/f'seed_{seed}'/f'patch_draws_{tag}.csv')==sha256_file(out/'final/direct'/f'seed_{seed}'/f'patch_draws_{tag}.csv')
                del cases;gc.collect();torch.cuda.empty_cache()
            lock=out/f'all10_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=systems,pilot_gate_sha256=gh));testcases={}
            for seed in SEEDS:testcases[seed],rr,pp=build_cases(ref,seed,out/f'test_sources/seed_{seed}',tag,status,parts=('test',),test_unlocked=True);protected.update(pp)
            test_records=run_test(ref,systems,testcases,out,tag,status,lock)
        assert sha256_file(gate)==gh;status('verifying_previous_files');assert all(sha256_file(p)==h for p,h in protected.items());write_json(out/f'protected_before_including_sources_{tag}.json',protected)
        result=dict(completed_at=stamp(),dataset='CHASE',pilot_fits=len(pilot),pilot_gate_passed=bool(chosen),final_fits=len(systems),prediction_files=len(test_records),previous_files_unchanged=True,HRF_started=False,DRIVE_started=False,gate=str(gate),gate_sha256=gh)
        write_json(out/f'completed_experiment_{tag}.json',result);status('complete',**result);print('NATIVE FLOW PILOT COMPLETE',result,flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
