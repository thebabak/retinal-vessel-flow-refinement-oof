"""CHASE-only third-level baseline and pair-excluded O3 sources for an OOF critic.
Old weights/data remain read-only. Validation and all indirect producer fits exclude each held subject.
"""
from __future__ import annotations
import csv,gc,itertools,json,random
from dataclasses import replace
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
CROSS=ROOT/'outputs/strict_o3_crossfit_2026-11-17_03-16-57';CT='2026-11-17_03-16-57'
OOF=ROOT/'revise/oof_comparison_2026-10-09_23-30-40'

def train_triple_baseline(ref,fit,excluded,out,tag,status):
    import numpy as np
    import torch
    from torchvision.transforms.functional import rotate
    from torchvision.transforms import InterpolationMode
    out.mkdir(parents=True,exist_ok=False)
    assert len(fit)==4 and not set(fit)&set(excluded) and not set(excluded)&set(ref.manifest['validation'])
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=random.Random(42)
    model=ref.LNet_RA(in_channels=1).to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=-1.;best_soft=-1.;stale=0;best_epoch=None;checkpoint=out/f'baseline_{tag}.pt';log=[]
    protocol=dict(dataset=ref.dataset,fit_images=fit,excluded_images=excluded,validation_images=ref.manifest['validation'],test_excluded=ref.manifest['test'],seed=42,max_epochs=1000,patience=20,loss='1-global softDice entire512grid, matching retained baseline recipe',learning_rate=.001,weight_decay=.001,min_optimizer_updates=200,checkpoint_tie_break='validation softDice')
    write_json(out/f'protocol_{tag}.json',protocol)
    for epoch in range(1,1001):
        status('nested_baseline_training',dataset=ref.dataset,epoch=epoch,excluded_folds=out.name)
        model.train();names=fit.copy();rng.shuffle(names);losses=[]
        for start in range(0,len(names),4):
            samples=[]
            for name in names[start:start+4]:
                d=ref.data[name];x=d['x'].clone();g=d['g'].clone();angle=rng.choice([0,15,30,45,90,100,120])
                x=rotate(x,angle,interpolation=InterpolationMode.BILINEAR);g=rotate(g,angle,interpolation=InterpolationMode.NEAREST)
                for dim in (-1,-2):
                    if rng.random()<.5:x=x.flip(dim);g=g.flip(dim)
                samples.append((x,g))
            x=torch.stack([v[0] for v in samples]).to(ref.device);g=torch.stack([v[1] for v in samples]).to(ref.device)
            p=model(x);loss=1-(2*(p*g).sum()+1e-6)/(p.sum()+g.sum()+1e-6)
            assert bool(torch.isfinite(loss));opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.item()))
        model.eval();vs=[];soft_scores=[]
        with torch.no_grad():
            names=ref.manifest['validation'];x=torch.stack([ref.data[n]['x'] for n in names]).to(ref.device);p=model(x).cpu()
            for j,n in enumerate(names):
                d=ref.data[n];score=ref.score(p[j],d['g']) if ref.dataset=='CHASE' else ref.score(p[j],d['g'],d['fov']>.5)
                vs.append(score['dice']);soft_scores.append(float((2*(p[j]*d['g']).sum()+1e-6)/(p[j].sum()+d['g'].sum()+1e-6)))
        val=float(np.mean(vs));log.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_dice=val,validation_softdice=float(np.mean(soft_scores)),optimizer_updates=1))
        soft=float(np.mean(soft_scores))
        if val>best or (val==best and soft>best_soft):
            best=val;best_soft=soft;best_epoch=epoch;stale=0;torch.save(dict(state=model.state_dict(),epoch=epoch,validation_dice=val,protocol=protocol),checkpoint)
        else:stale+=1
        if epoch%10==0:print(ref.dataset,'INNER',out.name,epoch,'val',val,'best',best,'stale',stale,flush=True)
        if stale>=20 and epoch>=200:break
    with (out/f'training_{tag}.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
    model.load_state_dict(torch.load(checkpoint,map_location=ref.device,weights_only=True)['state']);model.eval().requires_grad_(False)
    maps={}
    with torch.inference_mode():
        for n in excluded:
            a=model(ref.data[n]['x'][None].to(ref.device)).cpu()[0].numpy().copy()
            assert a.dtype==np.float32 and a.shape==(1,512,512) and np.isfinite(a).all();maps[n]=a
    archive=out/f'excluded_fold_coarse_maps_{tag}.npz';np.savez_compressed(archive,**maps)
    with np.load(archive,allow_pickle=False) as z:assert set(z.files)==set(excluded) and all(np.array_equal(z[n],maps[n]) for n in excluded)
    decision=dict(**protocol,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),selected_epoch=best_epoch,selected_validation_dice=best,selected_validation_softdice=best_soft,epochs_run=len(log),maps_path=str(archive),maps_sha256=sha256_file(archive),locked_before_prediction=True)
    write_json(out/f'completed_{tag}.json',decision)
    del model,opt;gc.collect();torch.cuda.empty_cache();return decision,maps

def baseline_predictions(ref,path,names):
    import torch
    cp=torch.load(path,map_location=ref.device,weights_only=False)
    model=ref.LNet_RA(in_channels=1).to(ref.device);model.load_state_dict(cp['state']);model.eval().requires_grad_(False)
    with torch.inference_mode():maps={n:model(ref.data[n]['x'][None].to(ref.device)).cpu()[0] for n in names}
    del model;gc.collect();torch.cuda.empty_cache();return maps

def with_coarse(ref,fit,held,maps):
    import torch
    from torch.nn import functional as F
    names=fit+ref.manifest['validation']+held;data={}
    for n in names:
        d=dict(ref.data[n]);d['coarse']=maps[n].clone() if isinstance(maps[n],torch.Tensor) else torch.from_numpy(maps[n].copy())
        d['n_coarse']=F.interpolate(d['coarse'][None],size=d['n_rgb'].shape[-2:],mode='bilinear',align_corners=False)[0].contiguous();data[n]=d
    return replace(ref,data=data,manifest=dict(train=fit,validation=ref.manifest['validation'].copy(),test=held))

def predict_anchor(ref,model,d):
    from flow_matching_refiner.native_patch_model import predict_tiled,project_residual_512,apply_residual
    _,delta,meta=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device,return_residual=True)
    return apply_residual(d['coarse'],project_residual_512(delta,'CHASE')),meta

def build_sources(ref,out,tag,status):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,training_center_pools
    from flow_matching_refiner.run_native_patch import train_one
    assert ref.dataset=='CHASE'
    old=json.loads((CROSS/f'all40_O3_choices_locked_before_OOF_{CT}.json').read_text())
    folds=json.loads((OOF/'CHASE/seed_42/oof_protocol_2026-10-09_23-30-40.json').read_text())['folds']
    assert len(folds)==4 and all(len(f)==4 for f in folds)
    assert all(not {n[:-1] for n in folds[i]}&{n[:-1] for n in folds[j]} for i,j in itertools.combinations(range(4),2))
    root=out/'nested_sources';root.mkdir(exist_ok=False);triples={};triple_maps={}
    pairs={tuple(v-1 for v in d['excluded_folds']):d for d in old['inner_baselines'] if d['dataset']=='CHASE'}
    assert len(pairs)==6
    for excluded in itertools.combinations(range(4),3):
        held=[n for j in excluded for n in folds[j]];fit=[n for n in ref.manifest['train'] if n not in held]
        d,m=train_triple_baseline(ref,fit,held,root/f'triple_baseline_{"_".join(str(j+1) for j in excluded)}',tag,status)
        triples[excluded]=d;triple_maps[excluded]=m;status('triple_baseline_complete',completed_triple_baselines=len(triples))
    pairrefs={};decisions=[]
    for pair in itertools.combinations(range(4),2):
        status('pair_O3_preparation',excluded_pair=[j+1 for j in pair]);pd=pairs[pair]
        assert sha256_file(pd['checkpoint'])==pd['checkpoint_sha256'] and sha256_file(pd['maps_path'])==pd['maps_sha256']
        held=[n for j in pair for n in folds[j]];fit=[n for n in ref.manifest['train'] if n not in held]
        with np.load(pd['maps_path'],allow_pickle=False) as z:maps={n:z[n].copy() for n in held}
        maps.update(baseline_predictions(ref,pd['checkpoint'],ref.manifest['validation']));lineage=[]
        for n in fit:
            j=next(i for i,f in enumerate(folds) if n in f);triple=tuple(sorted((*pair,j)));d=triples[triple]
            assert not (set(held)|{n})&set(d['fit_images']);maps[n]=triple_maps[triple][n]
            lineage.append(dict(image=n,producer_checkpoint=d['checkpoint'],producer_sha256=d['checkpoint_sha256'],producer_fit_images=d['fit_images'],excluded_folds=[v+1 for v in triple]))
        fr=with_coarse(ref,fit,held,maps);job=root/f'pair_O3_{pair[0]+1}_{pair[1]+1}'
        job.mkdir(exist_ok=False)
        protocol=dict(excluded_pair=[j+1 for j in pair],fit_images=fit,held_images=held,validation_images=ref.manifest['validation'],test_excluded=ref.manifest['test'],training_input_lineage=lineage,validation_and_held_baseline=pd,strict_indirect_exclusions=True)
        write_json(job/f'protocol_{tag}.json',protocol)
        pools={n:training_center_pools(fr.data[n]['n_g'],fr.data[n]['n_fov']) for n in fit}
        d=train_one(fr,job,tag,42,pools,status);path=job/'native_patch_direct/seed_42'/f'best_raw_model_{tag}.pt'
        decisions.append(dict(pair=[j+1 for j in pair],protocol=protocol,checkpoint=str(path),checkpoint_sha256=sha256_file(path),selection=d));pairrefs[pair]=fr
        status('pair_O3_complete',completed_pair_O3_fits=len(decisions));del pools;gc.collect();torch.cuda.empty_cache()
    lock=root/f'all6_pair_O3_locked_before_maps_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=decisions,triple_baselines=list(triples.values())))
    records=[]
    def save(role,key,n,p,coarse,source,protocol):
        dest=root/'cache'/role/key;dest.mkdir(parents=True,exist_ok=True);path=dest/f'{n}_anchor_{tag}.npz'
        np.savez_compressed(path,O3=p.numpy(),coarse=coarse.numpy());records.append(dict(role=role,key=key,image=n,path=str(path),sha256=sha256_file(path),source_checkpoint=source['checkpoint'],source_sha256=source['checkpoint_sha256'],protocol=protocol))
    for dec in decisions:
        pair=tuple(j-1 for j in dec['pair']);fr=pairrefs[pair];net=ErrorSelectiveModel(False).to(ref.device);net.load_state_dict(torch.load(dec['checkpoint'],map_location=ref.device,weights_only=True)['state']);net.eval().requires_grad_(False)
        for n in fr.manifest['test']:
            status('pair_O3_cache',pair=dec['pair'],image=n);p,meta=predict_anchor(fr,net,fr.data[n]);save('pair',f'{pair[0]+1}_{pair[1]+1}',n,p,fr.data[n]['coarse'],dec,dec['protocol'])
        del net;gc.collect();torch.cuda.empty_cache()
    for k in range(4):
        dec=next(d for d in old['decisions'] if d['dataset']=='CHASE' and d['outer_fold']==k+1 and d['refiner_seed']==42)
        fp=json.loads(Path(dec['fold_protocol']).read_text());bpath=fp['outer_baseline_checkpoint'];assert sha256_file(bpath)==fp['outer_baseline_sha256']
        maps=baseline_predictions(ref,bpath,ref.manifest['validation']);maps.update({n:ref.data[n]['coarse'] for n in folds[k]})
        fr=with_coarse(ref,[],folds[k],maps);net=ErrorSelectiveModel(False).to(ref.device);net.load_state_dict(torch.load(dec['raw_path'],map_location=ref.device,weights_only=True)['state']);net.eval().requires_grad_(False)
        source=dict(checkpoint=dec['raw_path'],checkpoint_sha256=dec['raw_sha256'])
        for n in ref.manifest['validation']+folds[k]:
            p,meta=predict_anchor(fr,net,fr.data[n]);save('outer',str(k+1),n,p,fr.data[n]['coarse'],source,fp)
        del net;gc.collect();torch.cuda.empty_cache()
    assert len(records)==80
    result=dict(completed_at=stamp(),folds=folds,records=records,new_triple_baselines=4,new_pair_O3_fits=6,global_lock=str(lock),global_lock_sha256=sha256_file(lock),old_crossfit=str(CROSS),HRF_started=False,DRIVE_started=False)
    write_json(root/f'completed_nested_sources_{tag}.json',result);return result
