"""Fresh subject-disjoint CHASE baseline and native Option3 source chain.

Retained model weights are never used. Validation selects checkpoints; outer
test subjects are excluded from every training and validation dependency.
"""
from __future__ import annotations
import csv,gc,itertools,json,math,random
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from flow_matching_refiner.data import ROOT,load_reference,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,predict_tiled,project_residual_512,apply_residual,training_center_pools
from flow_matching_refiner.run_native_patch import train_one

def subject(n):return int(n.split('_')[1][:2])
def groups(names,k,seed):
    ss=sorted({subject(n) for n in names});random.Random(seed).shuffle(ss);return [[n for n in sorted(names) if subject(n) in ss[j::k]] for j in range(k)]
def partitions(names):
    outer=groups(names,3,20261118);result=[]
    for j,test in enumerate(outer):
        remaining=sorted({subject(n) for n in names if n not in test});random.Random(20261118+101*(j+1)).shuffle(remaining);vs=set(remaining[:2]);validation=[n for n in sorted(names) if subject(n) in vs];train=[n for n in sorted(names) if n not in test+validation];inner=groups(train,4,20261218+j);result.append(dict(fold=j+1,train=train,validation=validation,test=test,OOF_groups=inner))
        assert all(not {subject(n) for n in result[-1][a]} & {subject(n) for n in result[-1][b]} for a,b in itertools.combinations(('train','validation','test'),2))
    assert sorted(n for d in result for n in d['test'])==sorted(names);return result
def lineage_guard(test,val,fit,held,dependencies):
    assert not set(test)&(set(fit)|set(val)|set(held)) and not set(val)&(set(fit)|set(held)) and not set(fit)&set(held)
    assert set(dependencies)==set(fit)
    for n,producer_fit in dependencies.items():assert n not in producer_fit and set(producer_fit)<=set(fit) and not set(producer_fit)&(set(test)|set(held)|set(val))

def load_raw(out):
    ref=load_reference('CHASE',42,out/'raw_loader','cuda');protected={str(ref.protocol_path):sha256_file(ref.protocol_path)};data={}
    for n,d in ref.data.items():
        rp=ROOT/'data'/f'{n}.jpg';gp=ROOT/'data'/f'{n}_1stHO.png';protected[str(rp)]=sha256_file(rp);protected[str(gp)]=sha256_file(gp)
        with Image.open(rp) as im:rgb=torch.from_numpy((np.asarray(im.convert('RGB'))/255.).astype(np.float32)).permute(2,0,1).contiguous()
        with Image.open(gp) as im:g=torch.from_numpy((np.asarray(im.convert('L'))>127).copy())[None].float();gg=torch.from_numpy((np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>127).copy())[None].float()
        assert g.shape==(1,960,999) and torch.equal(gg,d['g']);data[n]=dict(d,n_rgb=rgb,n_g=g,n_fov=torch.ones_like(g,dtype=torch.bool),fov=torch.ones_like(gg,dtype=torch.bool),coarse=torch.zeros_like(gg));data[n]['n_coarse']=torch.zeros_like(g)
    assert len(data)==28 and len({subject(n) for n in data})==14;return replace(ref,data=data),protected

def score_base(ref,p,n):return ref.score(p,ref.data[n]['g'])['dice']
def fit_baseline(ref,fit,val,test,excluded,out,tag,status):
    from torchvision.transforms.functional import rotate
    from torchvision.transforms import InterpolationMode
    assert fit and not set(fit)&(set(val)|set(test)|set(excluded)) and not set(val)&(set(test)|set(excluded));out.mkdir(parents=True,exist_ok=False)
    protocol=dict(fit_images=fit,validation_images=val,outer_test_excluded=test,held_images=excluded,seed=42,max_epochs=1000,patience=20,learning_rate=.001,weight_decay=.001,loss='Global soft Dice on whole512grid; unchanged baseline recipe',new_random_initialization=True)
    write_json(out/f'protocol_{tag}.json',protocol);torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=random.Random(42);model=ref.LNet_RA(in_channels=1).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);best=-1.;best_epoch=None;stale=0;log=[];cp=out/f'baseline_{tag}.pt'
    for epoch in range(1,1001):
        status('fresh_nested_baseline_training',model=str(out.name),epoch=epoch,fit_images=len(fit));model.train();names=fit.copy();rng.shuffle(names);losses=[]
        for start in range(0,len(names),4):
            samples=[]
            for n in names[start:start+4]:
                x=ref.data[n]['x'].clone();g=ref.data[n]['g'].clone();angle=rng.choice([0,15,30,45,90,100,120]);x=rotate(x,angle,interpolation=InterpolationMode.BILINEAR);g=rotate(g,angle,interpolation=InterpolationMode.NEAREST)
                for dim in (-1,-2):
                    if rng.random()<.5:x=x.flip(dim);g=g.flip(dim)
                samples.append((x,g))
            x=torch.stack([a[0] for a in samples]).cuda();g=torch.stack([a[1] for a in samples]).cuda();p=model(x);loss=1-(2*(p*g).sum()+1e-6)/(p.sum()+g.sum()+1e-6);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.item())
        model.eval()
        with torch.inference_mode():pred=model(torch.stack([ref.data[n]['x'] for n in val]).cuda()).cpu();v=float(np.mean([score_base(ref,pred[j],n) for j,n in enumerate(val)]))
        log.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_dice=v,optimizer_updates=math.ceil(len(fit)/4)))
        if v>best:best=v;best_epoch=epoch;stale=0;torch.save(dict(state=model.state_dict(),protocol=protocol,epoch=epoch,validation_dice=v),cp)
        else:stale+=1
        if epoch%20==0:print('SUBJECT CV BASELINE',out.name,epoch,'best',best,'stale',stale,flush=True)
        if stale>=20:break
    with (out/f'training_{tag}.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
    result=dict(**protocol,checkpoint=str(cp),checkpoint_sha256=sha256_file(cp),selected_epoch=best_epoch,validation_dice=best,epochs_run=len(log),locked_at=stamp());write_json(out/f'selection_locked_{tag}.json',result);del model,opt;gc.collect();torch.cuda.empty_cache();return result

@torch.inference_mode()
def baseline_predict(ref,decision,names):
    assert not set(names)&set(decision['fit_images']);assert sha256_file(decision['checkpoint'])==decision['checkpoint_sha256'];m=ref.LNet_RA(in_channels=1).cuda();m.load_state_dict(torch.load(decision['checkpoint'],map_location='cuda',weights_only=True)['state']);m.eval();maps={n:m(ref.data[n]['x'][None].cuda()).cpu()[0].contiguous() for n in names};del m;gc.collect();torch.cuda.empty_cache();return maps

def make_ref(ref,fit,val,maps):
    data={}
    for n in fit+val:
        c=maps[n].clone().float();d=dict(ref.data[n],coarse=c);d['n_coarse']=F.interpolate(c[None],size=d['n_rgb'].shape[-2:],mode='bilinear',align_corners=False)[0].contiguous();data[n]=d
    return replace(ref,data=data,manifest=dict(train=fit.copy(),validation=val.copy(),test=[]),namespace=dict(ref.namespace,data=data))

def fit_o3(ref,fit,val,test,held,maps,dependencies,baseline,out,tag,status):
    lineage_guard(test,val,fit,held,dependencies);out.mkdir(parents=True,exist_ok=False);fr=make_ref(ref,fit,val,maps);protocol=dict(fit_images=fit,validation_images=val,outer_test_excluded=test,OOF_held_images=held,coarse_training_fit_dependencies=dependencies,baseline_checkpoint=baseline['checkpoint'],baseline_sha256=baseline['checkpoint_sha256'],baseline_fit_images=baseline['fit_images'],seed=42,new_training=True)
    write_json(out/f'lineage_{tag}.json',protocol);pools={n:training_center_pools(fr.data[n]['n_g'],fr.data[n]['n_fov']) for n in fit};d=train_one(fr,out,tag,42,pools,status);cp=out/'native_patch_direct/seed_42'/f'best_raw_model_{tag}.pt';result=dict(**d,checkpoint=str(cp),checkpoint_sha256=d['raw_sha256'],protocol=protocol);write_json(out/f'completed_O3_{tag}.json',result);return result

@torch.inference_mode()
def option3_predict(ref,names,coarse,decision,out,tag,part):
    assert not set(names)&set(decision['protocol']['fit_images']);assert sha256_file(decision['checkpoint'])==decision['checkpoint_sha256'];model=ErrorSelectiveModel(False).cuda();model.load_state_dict(torch.load(decision['checkpoint'],map_location='cuda',weights_only=True)['state']);model.eval();out.mkdir(parents=True,exist_ok=True);cases={}
    for n in names:
        d=ref.data[n];c=coarse[n];nc=F.interpolate(c[None],size=d['n_rgb'].shape[-2:],mode='bilinear',align_corners=False)[0].contiguous();_,a,_=predict_tiled(model,nc,d['n_rgb'],'cuda',batch=8,return_residual=True);pn=apply_residual(nc,a);p512=apply_residual(c,project_residual_512(a,'CHASE'));path=out/f'{n}_source_{tag}.npz';np.savez_compressed(path,source_native=pn.numpy(),source512=p512.numpy(),coarse512=c.numpy(),anchor_residual_native=a.numpy());record=dict(image=n,split=part,path=str(path),sha256=sha256_file(path),provenance=dict(whole_chain_OOF=part=='train',fit_images=decision['protocol']['fit_images'],validation_images=decision['protocol']['validation_images'],outer_test_excluded=decision['protocol']['outer_test_excluded'],O3_checkpoint=decision['checkpoint'],O3_sha256=decision['checkpoint_sha256'],baseline_checkpoint=decision['protocol']['baseline_checkpoint'],baseline_sha256=decision['protocol']['baseline_sha256'],baseline_fit_images=decision['protocol']['baseline_fit_images'],coarse_training_fit_dependencies=decision['protocol']['coarse_training_fit_dependencies']));cases[n]=dict(name=n,source=pn,source512=p512,coarse512=c,rgb=d['n_rgb'],g=d['n_g'],fov=d['n_fov'].float(),record=record)
    del model;gc.collect();torch.cuda.empty_cache();return cases

def build_fold(ref,split,out,tag,status):
    out.mkdir(parents=True,exist_ok=False);a=split['train'];v=split['validation'];t=split['test'];folds=split['OOF_groups'];base={};maps={}
    # Full and four single-exclusion baselines, plus six pair-exclusion fits.
    for excluded_groups in [()]+[(j,) for j in range(4)]+list(itertools.combinations(range(4),2)):
        held=[n for j in excluded_groups for n in folds[j]];fit=[n for n in a if n not in held];label='full' if not excluded_groups else 'exclude_'+'_'.join(str(j+1) for j in excluded_groups);d=fit_baseline(ref,fit,v,t,held,out/'baselines'/label,tag,status);base[excluded_groups]=d;maps[excluded_groups]=baseline_predict(ref,d,held+v)
    cases={};o3={};fullmaps={};fulldeps={}
    for j,held in enumerate(folds):
        for n in held:fullmaps[n]=maps[(j,)][n];fulldeps[n]=base[(j,)]['fit_images']
    for n in v:fullmaps[n]=maps[()][n]
    o3[()]=fit_o3(ref,a,v,t,[],fullmaps,fulldeps,base[()],out/'Option3/full',tag,status)
    for j,held in enumerate(folds):
        fit=[n for n in a if n not in held];mp={};deps={}
        for k,other in enumerate(folds):
            if k==j:continue
            pair=tuple(sorted((j,k)))
            for n in other:mp[n]=maps[pair][n];deps[n]=base[pair]['fit_images']
        for n in v:mp[n]=maps[(j,)][n]
        o3[(j,)]=fit_o3(ref,fit,v,t,held,mp,deps,base[(j,)],out/f'Option3/OOF_group_{j+1}',tag,status);cases.update(option3_predict(ref,held,maps[(j,)],o3[(j,)],out/'source_maps',tag,'train'))
    cases.update(option3_predict(ref,v,maps[()],o3[()],out/'source_maps',tag,'validation'));assert set(cases)==set(a+v)
    frdata={n:dict(ref.data[n],coarse=cases[n]['coarse512']) for n in cases};fr=replace(ref,data=frdata,manifest=dict(train=a.copy(),validation=v.copy(),test=[]));inventory=dict(split=split,baselines=[dict(excluded_groups=list(k),**d) for k,d in base.items()],Option3=[dict(excluded_groups=list(k),**d) for k,d in o3.items()],source_records=[c['record'] for c in cases.values()],outer_full_baseline=base[()],outer_full_O3=o3[()]);write_json(out/f'upstream_inventory_{tag}.json',inventory);return fr,cases,inventory

def self_test():
    names=[f'Image_{i:02d}{eye}' for i in range(1,15) for eye in ('L','R')];pp=partitions(names);assert [len(p['test']) for p in pp]==[10,10,8];assert [len(p['train']) for p in pp]==[14,14,16]
    lineage_guard(['t'],['v'],['a','b','c'],['h'],{'a':['b','c'],'b':['a','c'],'c':['a','b']})
    for bad in ({'a':['t'],'b':['a'],'c':['b']},{'a':['h'],'b':['a'],'c':['b']},{'a':['a'],'b':['a'],'c':['b']}):
        try:lineage_guard(['t'],['v'],['a','b','c'],['h'],bad)
        except AssertionError:pass
        else:raise AssertionError('Leaky source dependency accepted')
    print('Subject partitions and direct/indirect leakage guards passed.',flush=True)
if __name__=='__main__':self_test()
