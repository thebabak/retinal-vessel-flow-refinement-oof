"""Native rollout/global-context extensions and fresh, nested outer CV."""
from __future__ import annotations
import math,gc,random,itertools
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,SEEDS,load,write_json,sha256_file,savecsv
from .adapters import module,native_ref,completed_fit,lock,supervision,cleanup
from .sources import grouped,source_api,cached_baseline,fit_o3,source_cases,full_cases,oof_cases,load_case_records

def rollout_api(ctx):
    old="    old_lock=ROOT/'outputs/local_flow_acceptor_CHASE_v2_2026-11-17_22-19-21/all10_system_choices_locked_before_test_2026-11-17_22-19-21.json';old_base=next(d['O3_validation'] for d in json.loads(old_lock.read_text())['decisions'] if d['mode']=='flow' and d['seed']==seed);assert base==old_base,'Full Option3 validation source changed'"
    r=module(ctx,'run_native_rollout_pilot',[(old,"    assert all(c['record']['split']=='validation' for c in validation)"),("project_residual_512(delta,'CHASE')","project_residual_512(delta,ref.dataset)"),('seen==64 and updates==8',"seen==len(ref.manifest['train'])*4 and updates==(seen+7)//8"),("print('CHASE',",'print(ref.dataset,'),("print('CHASE VALIDATION',","print(ref.dataset+' VALIDATION',")])
    r.prepare_supervision=supervision
    return r

def context_api(ctx,precision=False):
    r=module(ctx,'subject_cv_context_training' if precision else 'run_global_context_pilot',[("project_residual_512(delta,'CHASE')","project_residual_512(delta,ref.dataset)"),("print('CONTEXT CHASE',","print('CONTEXT '+ref.dataset,")]+([] if precision else [('seen==64 and updates==8',"seen==len(ref.manifest['train'])*4 and updates==(seen+7)//8")]))
    native=rollout_api(ctx);r.prepare_training=native.prepare_training;r.draw=native.draw
    return r

def contexts(ctx,cases,dest):
    from flow_matching_refiner.global_context_data import build_context,AUGS,rgb_sha,WEIGHTS
    ctx.protect(WEIGHTS);p=dest/f'context_manifest_{ctx.tag}.json'
    if p.exists():
        m=load(p);cache={}
        for rr in m['records']:
            assert sha256_file(rr['path'])==rr['sha256']
            with np.load(rr['path'],allow_pickle=False) as z:sp=z['spatial'].copy();v=z['global_vector'].copy()
            n=rr['image'];assert rgb_sha(cases[n]['rgb'])==rr['RGB_array_sha256']
            cache[n]=dict(spatial=torch.from_numpy(sp),global_vector=torch.from_numpy(v),augs=AUGS,rgb_sha256=rr['RGB_array_sha256'],path=rr['path'],sha256=rr['sha256'])
        return cache
    cache,_=build_context(cases,dest,ctx.tag,ctx.status);return cache

def evaluate(ctx,ref,decisions,cases_by_seed,context=None,outer_fold=None):
    from flow_matching_refiner.native_rollout_pilot import CorrectionField,predict_delta as native_predict
    from flow_matching_refiner.global_context_flow import GlobalContextCorrection,predict_delta as context_predict
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
    from flow_matching_refiner.structure_metrics_all import metrics
    entries=[]
    for d in decisions:
        dest=ctx.out/(f'outer_fold_{outer_fold}' if outer_fold else 'fixed')/'test'/d['mode']/f'seed_{d["seed"]}';dest.mkdir(parents=True,exist_ok=True)
        model=(GlobalContextCorrection(d['mode']) if context is not None else CorrectionField(d['mode'])).to(ref.device)
        for n in ref.manifest['test']:
            rp=dest/f'{n}_record_{ctx.tag}.json'
            if rp.exists():
                rr=load(rp);assert sha256_file(rr['path'])==rr['sha256'];entries.append(rr);continue
            c=cases_by_seed[d['seed']][n];values={};metas={};native={}
            for kind,q in (('selected',d['selection']),('raw',d['best_raw'])):
                ctx.status('locked_test_inference',outer_fold=outer_fold,mode=d['mode'],seed=d['seed'],image=n,output=kind)
                assert sha256_file(q['checkpoint'])==q['checkpoint_sha256']
                model.load_state_dict(torch.load(q['checkpoint'],map_location=ref.device,weights_only=True)['state']);model.eval()
                if q['beta']==0:delta=torch.zeros_like(c['source']);meta=dict(identity=True,GT_free=True)
                elif context is None:delta,meta=native_predict(model,c['source'],c['rgb'],ref.device,q['steps'])
                else:
                    ct=context[n];delta,meta=context_predict(model,c['source'],c['rgb'],ct['spatial'][0],ct['global_vector'][0],ref.device,q['steps'])
                values[kind]=apply_residual(c['source512'],q['beta']*project_residual_512(delta,ctx.dataset)).numpy();metas[kind]=meta;native[kind+'_delta_native']=delta.numpy()
            dd=ref.data[n];path=dest/f'{n}_predictions_{ctx.tag}.npz';np.savez_compressed(path,O3=c['source512'].numpy(),coarse=c['coarse512'].numpy(),ground_truth=dd['g'].numpy(),fov=dd['fov'].numpy(),**values)
            npth=dest/f'{n}_native_{ctx.tag}.npz';np.savez_compressed(npth,**native)
            with np.load(path,allow_pickle=False) as z:
                assert all(np.array_equal(z[k],values[k]) for k in values)
                mm={k:metrics(z[k][0],z['ground_truth'][0],z['coarse'][0],1.,z['fov'][0]) for k in ('coarse','O3','selected','raw')}
            rr=dict(dataset=ctx.dataset,outer_fold=outer_fold,seed=d['seed'],mode=d['mode'],image=n,path=str(path),sha256=sha256_file(path),native_path=str(npth),native_sha256=sha256_file(npth),metrics=mm,selection=d['selection'],raw_selection=d['best_raw'],source=c['record'],inference=metas)
            write_json(rp,rr);entries.append(rr)
        del model;cleanup()
    return entries

def paired(ctx,r,ref,cases,ct,seed,out,pilot,cfg=None):
    ds=[]
    for mode in ('flow','direct'):
        p=out/mode/f'seed_{seed}'
        def train():
            if ct is None:return r.train(ref,cases,seed,mode,cfg,p,ctx.tag,ctx.status,pilot)
            return r.train(ref,cases,{n:ct[n] for n in cases},seed,mode,p,ctx.tag,ctx.status,pilot)
        ds.append(completed_fit(p,ctx.tag,train,80))
    assert ds[0]['initial_state_sha256']==ds[1]['initial_state_sha256'] and ds[0]['parameter_count']==ds[1]['parameter_count']
    assert sha256_file(out/'flow'/f'seed_{seed}'/f'patch_draws_{ctx.tag}.csv')==sha256_file(out/'direct'/f'seed_{seed}'/f'patch_draws_{ctx.tag}.csv')
    return ds

def fixed_correction(ctx):
    ref=native_ref(ctx);fr,cases,_=oof_cases(ctx,ref,42);global_context=ctx.phase=='context'
    r=context_api(ctx) if global_context else rollout_api(ctx);ct=contexts(ctx,{n:dict(rgb=d['n_rgb']) for n,d in ref.data.items()},ctx.out/'RGB_context') if global_context else None
    pilots=[]
    if global_context:pilots=paired(ctx,r,fr,cases,ct,42,ctx.out/'pilots',True)
    else:
        for cfg in r.CONFIGS:
            p=ctx.out/'pilots'/cfg['name'];pilots.append(completed_fit(p,ctx.tag,lambda:r.train(fr,cases,42,'flow',cfg,p,ctx.tag,ctx.status,True),80))
    cfg=None if global_context else max(pilots,key=lambda d:(d['pilot_gate_passed'],d['selection']['structural_gain'],d['best_raw']['rank']))['config']
    lock(ctx,'pilot_configuration_before_final',pilots,chosen_configuration=cfg,complete_all_five_seeds_even_if_pilot_gate_fails=True,reason='User explicitly requested completion of every CHASE comparison; failure remains reported and identity fallback unchanged.')
    ds=[]
    for seed in SEEDS:
        fr,cases,_=oof_cases(ctx,ref,seed);ds.extend(paired(ctx,r,fr,cases,ct,seed,ctx.out/'final',False,cfg))
        del fr,cases;cleanup()
    lp=lock(ctx,'all10_choices_before_test',ds);testcases={s:full_cases(ctx,ref,s,('test',)) for s in SEEDS}
    entries=evaluate(ctx,ref,ds,testcases,ct);write_json(ctx.out/f'prediction_manifest_{ctx.tag}.json',dict(records=entries,selection_lock=str(lp),selection_lock_sha256=sha256_file(lp)))
    return ctx.done(dict(pilot_fits=2,final_fits=10,test_prediction_files=len(entries),full_five_seed_comparison=True))

def cv_partitions(ref):
    # Use the recorded native annotation dimensions; never infer availability
    # from whether an image belonged to the original fixed test partition.
    names=sorted(n for n,d in ref.data.items() if d['n_g'] is not None)
    if ref.dataset=='DRIVE':
        assert len(names)==40
    else:assert len(names)==45
    outer=grouped(names,3,20261118,ref.dataset);splits=[]
    for j,test in enumerate(outer):
        left=[n for n in names if n not in test];validation=[]
        if ref.dataset=='HRF':
            for k,kind in enumerate(('dr','g','h')):
                a=sorted(n for n in left if n.endswith('_'+kind));random.Random(20261219+101*j+k).shuffle(a);validation+=a[:2]
        else:
            random.Random(20261219+101*j).shuffle(left);validation=left[:4]
        validation=sorted(validation);train=sorted(n for n in left if n not in validation);folds=grouped(train,4,20261218+j,ref.dataset)
        assert all(not set(a)&set(b) for a,b in itertools.combinations((train,validation,test),2));assert min(map(len,folds))>0
        splits.append(dict(fold=j+1,train=train,validation=validation,test=test,OOF_groups=folds))
    assert sorted(n for sp in splits for n in sp['test'])==names;return splits

def fresh_sources(ctx):
    ref=native_ref(ctx);r=source_api(ctx);splits=cv_partitions(ref)
    write_json(ctx.out/f'partitions_{ctx.tag}.json',dict(dataset=ctx.dataset,folds=splits,DRIVE_pool='All40 retained images and native-size labels; separate from the original official fixed split' if ctx.dataset=='DRIVE' else None,unit='pathology-stratified image' if ctx.dataset=='HRF' else 'image; patient linkage unavailable',retinal_weights_reused=False,correction_seeds=list(SEEDS),upstream_seed=42,not_five_whole_pipelines=True))
    for sp in splits:
        out=ctx.out/f'outer_fold_{sp["fold"]}';out.mkdir(exist_ok=True);a=sp['train'];v=sp['validation'];t=sp['test'];folds=sp['OOF_groups'];bases={};maps={}
        for keys in [()]+[(j,) for j in range(4)]+list(itertools.combinations(range(4),2)):
            held=[n for j in keys for n in folds[j]];fit=[n for n in a if n not in held];label='full' if not keys else 'exclude_'+'_'.join(str(j+1) for j in keys)
            b=cached_baseline(ctx,r,ref,fit,v,t,held,out/'baselines'/label);bases[keys]=b;maps[keys]=r.baseline_predict(ref,b,held+v)
        mp={};deps={};o3={};cases={}
        for j,held in enumerate(folds):
            for n in held:mp[n]=maps[j,][n];deps[n]=bases[j,]['fit_images']
        mp.update({n:maps[()][n] for n in v})
        o3[()]=fit_o3(ctx,r,ref,a,v,t,[],mp,deps,bases[()],out/'Option3/full',42)
        for j,held in enumerate(folds):
            fit=[n for n in a if n not in held];mp={};deps={}
            for k,group in enumerate(folds):
                if k==j:continue
                pair=tuple(sorted((j,k)))
                for n in group:mp[n]=maps[pair][n];deps[n]=bases[pair]['fit_images']
            mp.update({n:maps[j,][n] for n in v});o3[j,]=fit_o3(ctx,r,ref,fit,v,t,held,mp,deps,bases[j,],out/f'Option3/OOF_group_{j+1}',42)
            cases.update(source_cases(ctx,r,ref,held,maps[j,],o3[j,],out/'source_maps','train'))
        cases.update(source_cases(ctx,r,ref,v,maps[()],o3[()],out/'source_maps','validation'))
        write_json(out/f'upstream_inventory_{ctx.tag}.json',dict(split=sp,baselines=[dict(excluded_groups=list(k),**b) for k,b in bases.items()],Option3=[dict(excluded_groups=list(k),**b) for k,b in o3.items()],source_records=[c['record'] for c in cases.values()],outer_full_baseline=bases[()],outer_full_O3=o3[()]))
        del cases;cleanup()
    return ctx.done(dict(outer_folds=3,fresh_baseline_fits=33,fresh_O3_fits=15,retinal_learned_sources_reused=False))

def fresh_cv(ctx):
    ref=native_ref(ctx);r=context_api(ctx,True);src=ctx.root/'fresh_sources'/ctx.dataset;assert (src/f'completed_{ctx.tag}.json').exists();splits=load(src/f'partitions_{ctx.tag}.json')['folds']
    pool=sorted({n for sp in splits for n in sp['train']+sp['validation']+sp['test']});ct=contexts(ctx,{n:dict(rgb=ref.data[n]['n_rgb']) for n in pool},ctx.out/'RGB_context');ds=[];inventories={}
    for sp in splits:
        u=load(src/f'outer_fold_{sp["fold"]}'/f'upstream_inventory_{ctx.tag}.json');inventories[sp['fold']]=u;cases=load_case_records(ref,u['source_records']);data={n:dict(ref.data[n],coarse=c['coarse512']) for n,c in cases.items()};fr=replace(ref,data=data,manifest=dict(train=sp['train'],validation=sp['validation'],test=[]))
        for seed in SEEDS:
            pair=paired(ctx,r,fr,cases,ct,seed,ctx.out/f'outer_fold_{sp["fold"]}'/'context_refiners',False)
            ds.extend(dict(outer_fold=sp['fold'],**d) for d in pair)
        del fr,cases,data;cleanup()
    lp=lock(ctx,'all30_choices_before_outer_test',ds);entries=[];api=source_api(ctx)
    for sp in splits:
        u=inventories[sp['fold']];coarse=api.baseline_predict(ref,u['outer_full_baseline'],sp['test']);cases=source_cases(ctx,api,ref,sp['test'],coarse,u['outer_full_O3'],ctx.out/f'outer_fold_{sp["fold"]}'/'test_sources','outer_test')
        fr=replace(ref,manifest=dict(train=sp['train'],validation=sp['validation'],test=sp['test']));dd=[d for d in ds if d['outer_fold']==sp['fold']]
        entries.extend(evaluate(ctx,fr,dd,{s:cases for s in SEEDS},ct,sp['fold']))
    expected=sum(len(sp['test']) for sp in splits)*10;assert len(entries)==expected
    write_json(ctx.out/f'outer_prediction_manifest_{ctx.tag}.json',dict(records=entries,selection_lock=str(lp),selection_lock_sha256=sha256_file(lp)))
    return ctx.done(dict(final_fits=30,outer_folds=3,test_prediction_files=len(entries),five_correction_seeds=True))
