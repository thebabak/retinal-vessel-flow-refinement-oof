"""Same context recipe; fold-size budget and precision selection generalized."""
import csv,gc,random
import numpy as np
import torch
from flow_matching_refiner.run_global_context_pilot import KEYS,BETAS,STEPS,EPOCHS,CONFIG
from flow_matching_refiner.run_native_rollout_pilot import prepare_training,draw,score,rank,savecsv,PARTS,state_sha
from flow_matching_refiner.global_context_data import rgb_sha
from flow_matching_refiner.global_context_flow import GlobalContextCorrection,sample_context,predict_delta
from flow_matching_refiner.native_rollout_pilot import objective
from flow_matching_refiner.precision_guard_selection import eligible
from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
from flow_matching_refiner.data import sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json

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
            schedule.step();assert seen==len(names) and updates==(len(names)+7)//8;tw.writerow([epoch,*(total/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
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

