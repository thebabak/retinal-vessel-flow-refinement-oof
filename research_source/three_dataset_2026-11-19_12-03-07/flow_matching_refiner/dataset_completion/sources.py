"""Four-way nested source chains with direct and indirect label exclusions."""
from __future__ import annotations
import itertools,random,math,copy
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,SEEDS,load,write_json,sha256_file,csvrows
from .adapters import module,native_ref,with_coarse,cleanup
AT='2026-11-16_16-53-08'
O3=ROOT/f'outputs/native_patch_study_{AT}'

def grouped(names,k,seed,dataset):
    if dataset=='HRF':
        groups=[[] for _ in range(k)]
        for j,kind in enumerate(('dr','g','h')):
            a=sorted(n for n in names if n.endswith('_'+kind));random.Random(seed+j).shuffle(a)
            for z,n in enumerate(a):groups[z%k].append(n)
        return [sorted(a) for a in groups]
    a=sorted(names);random.Random(seed).shuffle(a);return [sorted(a[j::k]) for j in range(k)]

def source_api(ctx):
    r=module(ctx,'subject_cv_sources',[
        ("def score_base(ref,p,n):return ref.score(p,ref.data[n]['g'])['dice']", "def score_base(ref,p,n):return ref.score(p,ref.data[n]['g'],ref.data[n]['fov']>.5)['dice']"),
        ("project_residual_512(a,'CHASE')","project_residual_512(a,ref.dataset)"),
        ('if stale>=20:break','if stale>=20 and epoch*math.ceil(len(fit)/4)>=MIN_UPDATES:break')])
    r.MIN_UPDATES=0
    return r

def cached_baseline(ctx,r,ref,fit,val,test,held,out,min_updates=0):
    p=out/f'selection_locked_{ctx.tag}.json'
    if p.exists():
        d=load(p);assert d['fit_images']==fit and d['validation_images']==val and d['outer_test_excluded']==test and d['held_images']==held
        assert sha256_file(d['checkpoint'])==d['checkpoint_sha256'];return d
    r.MIN_UPDATES=min_updates
    if min_updates:
        # Preserve the special CHASE critic-source guard: 200 optimizer
        # updates and a validation-softDice tie break, not 200 HRF epochs.
        tr=module(ctx,'local_acceptor_sources_v2',[
            ('assert len(fit)==4 and','assert len(fit)>0 and'),
            ('optimizer_updates=1','optimizer_updates=(len(fit)+3)//4'),
            ('if stale>=20 and epoch>=200:break','if stale>=20 and epoch*((len(fit)+3)//4)>=200:break')])
        d,_=tr.train_triple_baseline(ref,fit,held,out,ctx.tag,ctx.status)
        d=dict(d,outer_test_excluded=test,held_images=held)
        write_json(p,d);return d
    return r.fit_baseline(ref,fit,val,test,held,out,ctx.tag,ctx.status)

def fit_o3(ctx,r,ref,fit,val,test,held,maps,deps,baseline,out,seed):
    from flow_matching_refiner.run_native_patch import train_one
    from flow_matching_refiner.native_patch_model import training_center_pools
    done=out/f'completed_O3_{ctx.tag}.json'
    if done.exists():
        d=load(done);assert sha256_file(d['checkpoint'])==d['checkpoint_sha256'];return d
    r.lineage_guard(test,val,fit,held,deps)
    out.mkdir(parents=True,exist_ok=True);fr=r.make_ref(ref,fit,val,maps)
    protocol=dict(fit_images=fit,validation_images=val,outer_test_excluded=test,OOF_held_images=held,coarse_training_fit_dependencies=deps,baseline_checkpoint=baseline['checkpoint'],baseline_sha256=baseline['checkpoint_sha256'],baseline_fit_images=baseline['fit_images'],seed=seed,new_training=True)
    write_json(out/f'lineage_{ctx.tag}.json',protocol)
    inner=out/'native_patch_direct'/f'seed_{seed}';selection=inner/f'selection_locked_{ctx.tag}.json'
    if selection.exists():d=load(selection)
    else:
        assert not inner.exists(),f'Incomplete O3 fit preserved: {inner}'
        pools={n:training_center_pools(fr.data[n]['n_g'],fr.data[n]['n_fov']) for n in fit}
        d=train_one(fr,out,ctx.tag,seed,pools,ctx.status)
    cp=inner/f'best_raw_model_{ctx.tag}.pt';assert sha256_file(cp)==d['raw_sha256']
    assert len(csvrows(inner/f'training_{ctx.tag}.csv'))==120
    result=dict(**d,checkpoint=str(cp),checkpoint_sha256=d['raw_sha256'],protocol=protocol)
    write_json(done,result);return result

def source_cases(ctx,r,ref,names,maps,decision,out,part):
    # Per-image atomic cache permits inference recovery without any refitting.
    cases={}
    for n in names:
        rec=out/f'{n}_record_{ctx.tag}.json'
        if rec.exists():
            rr=load(rec);assert sha256_file(rr['path'])==rr['sha256']
            with np.load(rr['path'],allow_pickle=False) as z:pn=torch.from_numpy(z['source_native'].copy());p=torch.from_numpy(z['source512'].copy());c=torch.from_numpy(z['coarse512'].copy())
            d=ref.data[n];cases[n]=dict(name=n,source=pn,source512=p,coarse512=c,rgb=d['n_rgb'],g=d['n_g'],fov=d['n_fov'].float(),record=rr)
        else:
            one=r.option3_predict(ref,[n],maps,decision,out,ctx.tag,part);write_json(rec,one[n]['record']);cases.update(one)
    return cases

def strict_sources(ctx):
    ref=native_ref(ctx);r=source_api(ctx);a=ref.manifest['train'];v=ref.manifest['validation'];t=ref.manifest['test'];folds=grouped(a,4,20261218,ref.dataset)
    write_json(ctx.out/f'four_fold_plan_{ctx.tag}.json',dict(folds=folds,train=a,validation=v,test=t,original_full_O3_models_retained=True,HRF_original_coarse_folds=3 if ref.dataset=='HRF' else 4,new_nested_groups=4,selection='external validation only; no held-fold scoring'))
    bases={};maps={}
    for size in (1,2,3):
        for keys in itertools.combinations(range(4),size):
            held=[n for j in keys for n in folds[j]];fit=[n for n in a if n not in held];label='exclude_'+'_'.join(str(j+1) for j in keys)
            b=cached_baseline(ctx,r,ref,fit,v,t,held,ctx.out/'baselines'/label,200 if size==3 else 0);bases[keys]=b;maps[keys]=r.baseline_predict(ref,b,held+v)
    entries=[];outer_cases={};pair_cases={}
    for size in (1,2):
        for keys in itertools.combinations(range(4),size):
            held=[n for j in keys for n in folds[j]];fit=[n for n in a if n not in held];mp={};deps={}
            for j,group in enumerate(folds):
                if j in keys:continue
                inner=tuple(sorted((*keys,j)))
                for n in group:mp[n]=maps[inner][n];deps[n]=bases[inner]['fit_images']
            for n in v:mp[n]=maps[keys][n]
            for seed in (SEEDS if size==1 else (42,)):
                out=ctx.out/'Option3'/('exclude_'+'_'.join(str(j+1) for j in keys))/f'seed_{seed}'
                decision=fit_o3(ctx,r,ref,fit,v,t,held,mp,deps,bases[keys],out,seed)
                cases=source_cases(ctx,r,ref,held,maps[keys],decision,out/'sources','train')
                cases.update(source_cases(ctx,r,ref,v,maps[keys],decision,out/'sources','validation'))
                if size==1:
                    outer_cases[keys[0],seed]=[c['record'] for c in cases.values()]
                    for n in held:
                        rec=cases[n]['record'];entry=dict(dataset=ref.dataset,seed=seed,image=n,outer_fold=keys[0]+1,held_images=held,fit_images=fit,validation_images=v,path=rec['path'],sha256=rec['sha256'],anchor_model_path=decision['checkpoint'],anchor_model_sha256=decision['checkpoint_sha256'],fold_protocol=str(out/f'lineage_{ctx.tag}.json'))
                        entries.append(entry)
                else:pair_cases[keys]=[c['record'] for c in cases.values()]
                del cases;cleanup()
    write_json(ctx.out/f'OOF_routes_{ctx.tag}.json',dict(records=entries,folds=folds,baseline_decisions=[dict(excluded_groups=list(k),**b) for k,b in bases.items()],outer_case_records=[dict(group=j,seed=s,records=cc) for (j,s),cc in outer_cases.items()],pair_case_records=[dict(groups=list(k),records=cc) for k,cc in pair_cases.items()]))
    return ctx.done(dict(nested_baseline_fits=14,outer_O3_fits=20,pair_O3_fits=6,OOF_image_seed_records=len(entries),strict_indirect_exclusion_verified=True))

def read_strict(ctx,ref):
    out=ctx.root/'strict_sources'/ctx.dataset;done=load(out/f'completed_{ctx.tag}.json');assert done['strict_indirect_exclusion_verified']
    m=load(out/f'OOF_routes_{ctx.tag}.json');routes={};mp={}
    for e in m['records']:
        assert sha256_file(e['path'])==e['sha256'] and sha256_file(e['anchor_model_path'])==e['anchor_model_sha256']
        assert not set(e['held_images'])&set(e['fit_images']+e['validation_images'])
        lineage=load(e['fold_protocol'])
        for n,dep in lineage['coarse_training_fit_dependencies'].items():assert n not in dep and not set(e['held_images'])&set(dep)
        routes[ctx.dataset,e['seed'],e['image']]=e
        if e['seed']==42:
            with np.load(e['path'],allow_pickle=False) as z:mp[e['image']]=torch.from_numpy(z['coarse512'].copy())
    assert set(mp)==set(ref.manifest['train'])
    return with_coarse(ref,mp),routes,m

def full_cases(ctx,ref,seed,parts=('validation','test')):
    r=source_api(ctx);cp=O3/f'{ctx.dataset}_{AT}/native_patch_direct/seed_{seed}/best_raw_model_{AT}.pt';ctx.protect(cp)
    # Match the retained full-anchor inference batch, including FP32 ordering.
    original_predict=r.predict_tiled
    def predict_full(*args,**kwargs):
        kwargs['batch']=16
        return original_predict(*args,**kwargs)
    r.predict_tiled=predict_full
    p=load(cp.parent/f'selection_locked_{AT}.json');baseline=ref.baseline_checkpoint;ctx.protect(baseline)
    decision=dict(checkpoint=str(cp),checkpoint_sha256=sha256_file(cp),protocol=dict(fit_images=ref.manifest['train'],validation_images=ref.manifest['validation'],outer_test_excluded=ref.manifest['test'],baseline_checkpoint=str(baseline),baseline_sha256=sha256_file(baseline),baseline_fit_images=ref.manifest['train'],coarse_training_fit_dependencies={}))
    cases={}
    for part in parts:
        names=ref.manifest[part];mp={n:ref.data[n]['coarse'] for n in names}
        cases.update(source_cases(ctx,r,ref,names,mp,decision,ctx.out/'full_sources'/f'seed_{seed}'/part,part))
    return cases

def load_case_records(ref,records):
    cases={}
    for rr in records:
        assert sha256_file(rr['path'])==rr['sha256']
        with np.load(rr['path'],allow_pickle=False) as z:pn=torch.from_numpy(z['source_native'].copy());p=torch.from_numpy(z['source512'].copy());c=torch.from_numpy(z['coarse512'].copy())
        n=rr['image'];d=ref.data[n];cases[n]=dict(name=n,source=pn,source512=p,coarse512=c,rgb=d['n_rgb'],g=d['n_g'],fov=d['n_fov'].float(),record=rr)
    return cases

def oof_cases(ctx,ref,seed):
    fr,routes,m=read_strict(ctx,ref);rr=[e for e in m['records'] if e['seed']==seed];records=[load_case_record(e) for e in rr]
    cases=load_case_records(ref,records);cases.update(full_cases(ctx,ref,seed,('validation',)))
    return fr,cases,m

def load_case_record(e):
    # The cached original record includes complete producer dependencies.
    p=Path(e['path']).with_name(Path(e['path']).name.replace('_source_','_record_')).with_suffix('.json')
    return load(p)
