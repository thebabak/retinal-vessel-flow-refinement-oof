"""One geometry pilot per missing dataset, matching CHASE's actual pilot scope."""
from __future__ import annotations
import numpy as np
import torch
from scipy import ndimage as ndi
from .common import ROOT,write_json,load,sha256_file,savecsv
from .adapters import module,native_ref,cleanup
from .sources import oof_cases
from .corrections import rollout_api

def diagnosis(ctx,ref,cases):
    """Replay this dataset's preceding locked model on validation only."""
    from flow_matching_refiner.native_rollout_pilot import CorrectionField,predict_delta
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
    from flow_matching_refiner.structure_metrics_all import metrics
    parent=ctx.root/'rollout'/ctx.dataset/'final/flow/seed_42'
    decision=load(parent/f'selection_locked_{ctx.tag}.json');q=decision['selection']
    assert sha256_file(q['checkpoint'])==q['checkpoint_sha256'];ctx.protect(q['checkpoint'])
    model=CorrectionField('flow').to(ref.device)
    model.load_state_dict(torch.load(q['checkpoint'],map_location=ref.device,weights_only=True)['state']);model.eval()
    dest=ctx.out/'diagnosis';dest.mkdir(exist_ok=True);rows=[];parts=[];records=[]
    for n in ref.manifest['validation']:
        ctx.status('validation_geometry_diagnosis',image=n)
        c=cases[n];assert c['record']['split']=='validation'
        if q['beta']==0:delta=torch.zeros_like(c['source'])
        else:delta,_=predict_delta(model,c['source'],c['rgb'],ref.device,steps=q['steps'])
        p=apply_residual(c['source512'],q['beta']*project_residual_512(delta,ref.dataset))
        v=ref.data[n]['fov'].numpy()[0]>.5;g=(ref.data[n]['g'].numpy()[0]>.5)&v
        o=(c['source512'].numpy()[0]>=.5)&v;f=(p.numpy()[0]>=.5)&v
        add=f&~o;remove=o&~f;fp=add&~g;tp=add&g
        near=fp&(ndi.distance_transform_edt(~o)<=1.5);labels,ncomp=ndi.label(add,np.ones((3,3)));classes=np.zeros_like(labels)
        for j in range(1,ncomp+1):
            ys,xs=np.where(labels==j);y0,y1=max(0,ys.min()-8),min(g.shape[0],ys.max()+9);x0,x1=max(0,xs.min()-8),min(g.shape[1],xs.max()+9)
            piece=labels[y0:y1,x0:x1]==j;old=o[y0:y1,x0:x1];contacts,_=ndi.label(old,np.ones((3,3)));touch=np.unique(contacts[ndi.binary_dilation(piece,np.ones((3,3)))&old]);touch=touch[touch>0]
            kind='disconnected_addition' if len(touch)==0 else 'multi_contact_connection' if len(touch)>1 else 'attached_extension'
            classes[labels==j]={'disconnected_addition':1,'multi_contact_connection':2,'attached_extension':3}[kind]
            parts.append(dict(image=n,component=j,added_pixels=len(ys),added_TP=int((tp&(labels==j)).sum()),added_FP=int((fp&(labels==j)).sum()),local_O3_contacts=len(touch),heuristic_type=kind))
        rows.append(dict(image=n,added_TP=int(tp.sum()),added_FP=int(fp.sum()),removed_TP=int((remove&g).sum()),removed_FP=int((remove&~g&v).sum()),boundary_adjacent_added_FP=int(near.sum()),beyond_boundary_added_FP=int((fp&~near).sum()),disconnected_added_FP=int((fp&(classes==1)).sum()),multi_contact_added_FP=int((fp&(classes==2)).sum()),attached_added_FP=int((fp&(classes==3)).sum())))
        path=dest/f'{n}_validation_replay_{ctx.tag}.npz';np.savez_compressed(path,O3=c['source512'].numpy(),Flow=p.numpy(),GT=g,fov=v,delta_native=delta.numpy())
        records.append(dict(image=n,path=str(path),sha256=sha256_file(path),Flow_metrics=metrics(p.numpy()[0],g,c['coarse512'].numpy()[0],1.,v),O3_metrics=metrics(c['source512'].numpy()[0],g,c['coarse512'].numpy()[0],1.,v),checkpoint=q))
    savecsv(dest/f'per_image_errors_{ctx.tag}.csv',rows)
    if parts:savecsv(dest/f'added_component_types_{ctx.tag}.csv',parts)
    total={k:sum(r[k] for r in rows) for k in rows[0] if k!='image'}
    result=dict(dataset=ref.dataset,records=records,total=total,boundary_adjacent_fraction=total['boundary_adjacent_added_FP']/max(1,total['added_FP']),source_seed=42,validation_images=ref.manifest['validation'],test_used=False,FOV_restricted=True,interpretation='Boundary adjacency and component contacts are descriptive, not anatomical proof. The missing-dataset pilot is explicitly requested regardless of whether the original CHASE >0.5 diagnostic heuristic holds.')
    write_json(dest/f'diagnosis_{ctx.tag}.json',result);del model;cleanup();return result

def geometry(ctx):
    ref=native_ref(ctx);fr,cases,_=oof_cases(ctx,ref,42);nr=rollout_api(ctx)
    diagnosis(ctx,ref,cases)
    changes=[
        ("assert diagnosis['boundary_adjacent_fraction']>.5", "assert diagnosis['dataset']==DATASET and diagnosis['test_used'] is False"),
        ("dataset='CHASE'",'dataset=DATASET'),
        ("write_json(ROOT/'tmp/current_geometry_flow.json',state)","ctx_status(stage,**kw)"),
        ('for start in range(0,64,8):','for start in range(0,len(names),8):'),
        ('for micro in range(0,8,4):','for micro in range(0,len(samples),4):'),
        ("time=torch.rand(4,device='cuda',generator=trng)","time=torch.rand(len(batch),device='cuda',generator=trng)"),
        ('(value*.5).backward()','(value*len(batch)/len(samples)).backward()'),
        ('*4;seen+=4','*len(batch);seen+=len(batch)'),
        ('seen==64 and updates==8',"seen==len(ref.manifest['train'])*4 and updates==(seen+7)//8"),
        ("project_residual_512(delta,'CHASE')","project_residual_512(delta,ref.dataset)"),
        ('HRF_started=False',f'HRF_started={ctx.dataset=="HRF"}'),
        ('DRIVE_started=False',f'DRIVE_started={ctx.dataset=="DRIVE"}'),
        ('No new test evaluation, no DRIVE/HRF fitting, no paper edits.','No new test evaluation. Explicit missing-dataset pilot extension; no five-seed geometry confirmation.'),
        ('64native256patches/epoch','4native256patches/image/epoch'),
        ('train on16training images only.',f'train on the {len(ref.manifest["train"])} fitting images only.'),
        ('Repeatedly reused4validation images/2subjects.',f'Reused {len(ref.manifest["validation"])} validation images; patient linkage is not assumed.'),
        ("cases[n]['g'].numpy()[0]>.5", "(cases[n]['g'].numpy()[0]>.5)&(cases[n]['fov'].numpy()[0]>.5)"),
        ('GEOMETRY FLOW CHASE seed42',f'GEOMETRY FLOW {ctx.dataset} seed42')]
    r=module(ctx,'run_geometry_guided_pilot',changes);r.P=ctx.out;r.TAG=ctx.tag
    r.DATASET=ctx.dataset;r.ctx_status=ctx.status
    r.load_cases=lambda:(fr,cases,{})
    r.prepare_training=nr.prepare_training;r.draw=nr.draw
    r.run()
    return ctx.done(dict(pilot_fits=1,epochs=80,validation_only=True,new_test_inference=False))
