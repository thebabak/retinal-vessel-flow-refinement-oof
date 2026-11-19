"""One diagnostic-selected CHASE pilot; validation only, no new test inference."""
import sys,json,csv,gc,random,traceback,shutil
from pathlib import Path
import numpy as np
import torch
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.geometry_flow_data import load_cases
from flow_matching_refiner.geometry_guided_flow import GeometryFlow,training_geometry,geometry_crop,objective,eligible
from flow_matching_refiner.run_native_rollout_pilot import prepare_training,draw,score,rank,savecsv,PARTS as OLD_PARTS
from flow_matching_refiner.native_rollout_pilot import predict_delta
from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
TAG='2026-11-18_01-11-21';P=ROOT/'outputs'/f'geometry_flow_CHASE_pilot_{TAG}';KEYS=('dice','precision','cldice','thin_skeleton_recall');PARTS=(*OLD_PARTS,'base_loss','guide_bce','guide_direction','guide_radius','boundary_FP');BETAS=(.125,.25,.5,1.);STEPS=(2,4,8)
def run():
    assert '--execute' in sys.argv and P.is_dir() and not (P/f'plan_{TAG}.json').exists();diagnosis=json.loads((P/'diagnosis'/f'diagnosis_{TAG}.json').read_text());assert diagnosis['boundary_adjacent_fraction']>.5
    state=dict(output=str(P),tag=TAG,dataset='CHASE',seed=42,HRF_started=False,DRIVE_started=False,new_test_inference=False)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(P/f'status_{TAG}.json',state);write_json(ROOT/'tmp/current_geometry_flow.json',state)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();ref,cases,protected=load_cases()
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    for name in ('template261006.tex','last.tex','tmp/current_native_rollout_delivery.json'):
        path=ROOT/name
        if path.is_file():protected[str(path)]=sha256_file(path)
    write_json(P/f'protected_before_{TAG}.json',protected);snap=P/'source';snap.mkdir()
    for name in ('geometry_guided_flow.py','geometry_flow_data.py','run_geometry_guided_pilot.py'):
        p=ROOT/'flow_matching_refiner'/name;shutil.copy2(p,snap/f'{p.stem}_{TAG}.py')
    plan=dict(**state,locked_at=stamp(),pilot_only=True,epochs=80,config=dict(preservation_weight=1.,guide_BCE=.25,direction=.05,radius=.1,boundary_FP=.5),validation_steps=STEPS,validation_betas=BETAS,selection='Mean validation Dice and precision >=O3 (1e-12 tolerance), clDice and thin recall strictly higher. Max mean structural gain; exact O3 fallback. Epochs20,40,60,80. No hyperparameter retuning after the pilot.',method='Learned guide from RGB+OOF-O3 predicts axial direction, normalized log radius and vessel support. It conditions the velocity field; positive velocity is gated by .25+.75*predicted vessel support, negative corrections remain available. GT directions/radii/support train on16training images only.',target_geometry='Full-image GT structure tensor sigma1/3, direction supervision on GT skeleton coherence>.4; radius log1p(min(EDT,12))/log13. Axial direction vectors transform with rotations/reflections.',training='AdamW3e-4 wd1e-4 cosine;80epochs,64native256patches/epoch, effectivebatch8/micro4. Same error-aware crop recipe as preceding pilot. Extra guide losses and native/512 GT-background boundary penalty.',scope='One single-seed pilot after diagnosis, not a five-seed confirmation. No new test evaluation, no DRIVE/HRF fitting, no paper edits.',limitations='Repeatedly reused4validation images/2subjects. Geometry/capacity/auxiliary changes are combined; no isolated causal or Flow-specific advantage can be inferred. A matched GeometryFlow direct architecture exists but is not fitted in this pilot.')
    write_json(P/f'plan_{TAG}.json',plan);write_json(P/f'environment_{TAG}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True))
    torch.manual_seed(42+88177);torch.cuda.manual_seed_all(42+88177);model=GeometryFlow().cuda();direct=GeometryFlow('direct');count=sum(p.numel() for p in model.parameters());assert count==sum(p.numel() for p in direct.parameters());del direct
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80);rng=random.Random(42+44111);trng=torch.Generator(device='cuda').manual_seed(42+88178);cache,pools=prepare_training(ref,cases);targets={n:training_geometry(cases[n]['g'].numpy()[0]>.5) for n in ref.manifest['train']};validation=[cases[n] for n in ref.manifest['validation']];base={k:float(np.mean([score(ref,c,c['source512'])[k] for c in validation])) for k in KEYS};best=None;raw=None;vr=[];dest=P/'pilot';dest.mkdir()
    write_json(dest/f'input_lineage_{TAG}.json',dict(training=[cases[n]['record'] for n in ref.manifest['train']],validation=[cases[n]['record'] for n in ref.manifest['validation']],GT_geometry_targets_train_only=True,pool_counts={n:{k:len(v) for k,v in pp.items()} for n,pp in pools.items()}))
    with (dest/f'training_{TAG}.csv').open('x',newline='') as tf,(dest/f'patch_draws_{TAG}.csv').open('x',newline='') as pf:
        tw=csv.writer(tf);pw=csv.writer(pf);tw.writerow(['epoch','loss',*PARTS,'lr','patches','optimizer_updates']);pw.writerow(['epoch','patch_index','image','top','left','centre_y','centre_x','requested_group','fallback_random','rotation90','flip_horizontal','flip_vertical','source_array_sha256'])
        for epoch in range(1,81):
            status('geometry_pilot_training',epoch=epoch,epochs=80);model.train();names=ref.manifest['train']*4;rng.shuffle(names);sums=np.zeros(1+len(PARTS));seen=0;updates=0;lr=optimizer.param_groups[0]['lr']
            for start in range(0,64,8):
                samples=[];align=[];geo=[]
                for j,n in enumerate(names[start:start+8],start=start):
                    arrays,al,meta=draw(cases[n],cache[n],pools[n],rng);samples.append(arrays);align.append(al);geo.append(geometry_crop(targets[n],meta));pw.writerow([epoch,j,n,*(meta[k] for k in ('top','left','centre_y','centre_x','requested_group','fallback_random','rotation90','flip_horizontal','flip_vertical')),cases[n]['source_array_sha256']])
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0,8,4):
                    batch=samples[micro:micro+4];tensors=[torch.stack([s[k] for s in batch]).cuda() for k in range(6)];time=torch.rand(4,device='cuda',generator=trng);value,parts=objective(model,*tensors,align[micro:micro+4],time,dict(preservation_weight=1.),torch.stack(geo[micro:micro+4]).cuda());assert torch.isfinite(value);(value*.5).backward();sums+=np.asarray([value.item(),*(parts[k].item() for k in PARTS)])*4;seen+=4
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
            scheduler.step();tw.writerow([epoch,*(sums/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush();assert seen==64 and updates==8
            if epoch%10==0:print('GEOMETRY FLOW CHASE seed42',epoch,'loss',float(sums[0]/seen),flush=True)
            if epoch%20:continue
            cp=dest/f'field_epoch{epoch}_{TAG}.pt';torch.save(dict(state=model.state_dict(),epoch=epoch,parameter_count=count),cp);ch=sha256_file(cp)
            for steps in STEPS:
                values={b:[] for b in BETAS}
                for c in validation:
                    status('validation_only_selection',epoch=epoch,steps=steps,image=c['name']);delta,_=predict_delta(model,c['source'],c['rgb'],ref.device,steps=steps);d512=project_residual_512(delta,'CHASE')
                    for beta in BETAS:
                        p=apply_residual(c['source512'],beta*d512);m=score(ref,c,p);mm={k:m[k] for k in KEYS};values[beta].append(mm);vr.append(dict(epoch=epoch,steps=steps,beta=beta,image=c['name'],**mm))
                for beta,vs in values.items():
                    mm={k:float(np.mean([v[k] for v in vs])) for k in KEYS};q=dict(epoch=epoch,steps=steps,beta=beta,metrics=mm,joint_gate=eligible(mm,base),structural_gain=rank(mm,base),checkpoint=str(cp),checkpoint_sha256=ch);rankraw=mm['dice']+.1*mm['cldice']+.02*mm['thin_skeleton_recall']
                    if raw is None or rankraw>raw['rank']+1e-12:raw=dict(**q,rank=rankraw)
                    if q['joint_gate'] and (best is None or q['structural_gain']>best['structural_gain']+1e-12):best=q
                print('VALIDATION',epoch,'steps',steps,'qualified',best,'raw',raw,flush=True)
            savecsv(dest/f'validation_epoch{epoch}_{TAG}.csv',[r for r in vr if r['epoch']==epoch])
    savecsv(dest/f'validation_all_{TAG}.csv',vr);selected=best if best else dict(epoch=0,steps=4,beta=0.,metrics=base,joint_gate=False,structural_gain=0.,checkpoint=raw['checkpoint'],checkpoint_sha256=raw['checkpoint_sha256']);decision=dict(locked_at=stamp(),pilot_seed=42,epochs=80,parameter_count=count,baseline_validation=base,selection=selected,best_raw=raw,pilot_gate_passed=best is not None,validation_only=True,new_test_inference=False,HRF_started=False,DRIVE_started=False)
    write_json(P/f'selection_locked_{TAG}.json',decision);assert all(sha256_file(p)==h for p,h in protected.items());write_json(P/f'completed_experiment_{TAG}.json',dict(completed_at=stamp(),pilot_fits=1,previous_inputs_and_paper_unchanged=True,pilot_gate_passed=best is not None,HRF_started=False,DRIVE_started=False,new_test_inference=False));status('complete',pilot_gate_passed=best is not None);print('GEOMETRY PILOT COMPLETE',decision,flush=True)
if __name__=='__main__':
    try:run()
    except BaseException:write_json(P/f'failed_{TAG}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));raise
