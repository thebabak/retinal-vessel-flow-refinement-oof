"""FOV-aware threshold/AP and connection diagnostics of fresh validation folds."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from .common import ROOT,load,write_json,savecsv,sha256_file,csvrows,SEEDS
from .adapters import module,native_ref,cleanup
from .sources import load_case_records

def precision_reselection(ctx):
    from flow_matching_refiner.precision_guard_selection import eligible,gain
    src=ctx.root/'context'/ctx.dataset;records=[];decisions=[]
    for seed in SEEDS:
        for mode in ('flow','direct'):
            folder=src/'final'/mode/f'seed_{seed}';p=folder/f'selection_locked_{ctx.tag}.json';v=folder/f'validation_all_{ctx.tag}.csv';ctx.protect(p);ctx.protect(v);d=load(p);base=d['baseline_validation'];vv=csvrows(v);best=None;local=[]
            images={r['image'] for r in vv}
            for epoch in (20,40,60,80):
                for steps in ((2,4,8) if mode=='flow' else (1,)):
                    cp=folder/f'field_epoch{epoch}_{ctx.tag}.pt';digest=ctx.protect(cp)
                    for beta in (.125,.25,.5,1.):
                        rr=[r for r in vv if int(r['epoch'])==epoch and int(r['steps'])==steps and float(r['beta'])==beta];assert {r['image'] for r in rr}==images and len(rr)==len(images)
                        mm={k:float(np.mean([float(r[k]) for r in rr])) for k in ('dice','precision','cldice','thin_skeleton_recall')};q=dict(mode=mode,seed=seed,epoch=epoch,steps=steps,beta=beta,**mm,qualifies=eligible(mm,base),structural_gain=gain(mm,base));local.append(q)
                        if q['qualifies'] and (best is None or q['structural_gain']>best['structural_gain']+1e-12):best=dict(**q,checkpoint=str(cp),checkpoint_sha256=digest)
            chosen=best or dict(mode=mode,seed=seed,epoch=0,steps=0,beta=0.,**base,qualifies=False,structural_gain=0.,checkpoint=None,checkpoint_sha256=None)
            decisions.append(dict(mode=mode,seed=seed,selection=chosen,baseline_validation=base,qualifying_candidates=sum(r['qualifies'] for r in local)));records.extend(local)
    assert len(records)==320 and len(decisions)==10;savecsv(ctx.out/f'precision_guard_candidates_{ctx.tag}.csv',records);write_json(ctx.out/f'precision_guard_choices_{ctx.tag}.json',dict(decisions=decisions,training_performed=False,test_used=False,scope='Four-metric retrospective validation re-selection of the fixed-cohort context checkpoints. Separate from the fresh outer CV.'))
    return ctx.done(dict(training_fits=0,systems=10,validation_candidates=320))

def encoded_diagnosis(ctx):
    import sys
    changes=[('load_reference("CHASE", 42','load_reference(DATASET, 42'),('dataset="CHASE"','dataset=DATASET'),('f"CHASE:{name}:{seed}"','f"{DATASET}:{name}:{seed}"'),('reference.score(probability, gt)','reference.score(probability, gt,reference.data[name]["fov"])')]
    r=module(ctx,'diagnose_v1',changes);r.DATASET=ctx.dataset;r.OLD_TAG=ctx.tag;r.OLD_JOB=ctx.root/'encoded'/ctx.dataset/ctx.dataset/'training_seed_42'
    previous=sys.argv;sys.argv=['diagnose_v1','--execute','--output',str(ctx.out)]
    try:r.main()
    finally:sys.argv=previous
    return ctx.done(dict(training_fits=0,validation_solver_diagnostics=True,representation_diagnostics=True))

def bias_probe(ctx):
    from flow_matching_refiner.run_dice_guarded_oof_flow import model_for
    from flow_matching_refiner.dice_guarded_oof_flow import feasible,retained,selection_score
    from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
    from flow_matching_refiner.structure_metrics_all import metrics
    ref=native_ref(ctx)
    if ctx.dataset=='HRF':
        prior=next(d for d in load(ctx.root/'topology/HRF'/f'all5_choices_before_test_{ctx.tag}.json')['decisions'] if d['seed']==42)
        p=ctx.root/'retained/HRF/pilots/regret2_structure1'/f'selection_locked_{ctx.tag}.json'
    else:
        tag='2026-11-17_18-49-15';prior=next(d for d in load(ROOT/f'outputs/topology_oof_flow_{tag}/all10_choices_locked_before_test_{tag}.json')['decisions'] if d['dataset']=='DRIVE' and d['seed']==42)
        tag='2026-11-17_19-36-12';p=ROOT/f'outputs/dice_guarded_oof_flow_{tag}/DRIVE_{tag}/pilots/regret2_structure1/selection_locked_{tag}.json'
    dec=load(p);ctx.protect(p);assert ctx.protect(dec['raw_path'])==dec['raw_sha256'];base=prior['anchor_validation_metrics'];target=prior['selected']['validation_metrics'];model=model_for(ctx.dataset,42,91011,2.,'cuda');model.field.load_state_dict(torch.load(dec['raw_path'],map_location='cuda',weights_only=True)['state']);cache=[]
    for n in ref.manifest['validation']:
        d=ref.data[n];a,f,_=predict_tiled(model,d['n_coarse'],d['n_rgb'],'cuda');cache.append((d,project_residual_512(a,ctx.dataset),project_residual_512(f,ctx.dataset)))
    rr=[]
    for beta in (0.,.0625,.125,.25,.5,1.):
        for bias in (0.,-.01,-.025,-.05):
            values=[]
            for d,a,f in cache:
                p=apply_residual(d['coarse'],a+beta*f+bias);values.append(metrics(p.numpy()[0],d['g'].numpy()[0],d['coarse'].numpy()[0],1.,d['fov'].numpy()[0]))
            avg={k:float(np.mean([m[k] for m in values])) for k in ('dice','precision','cldice','thin_skeleton_recall')};rr.append(dict(beta=beta,bias=bias,**avg,score=selection_score(avg,base),pareto=feasible(avg,base),full_retention=retained(avg,base,target)))
    savecsv(ctx.out/f'validation_calibration_candidates_{ctx.tag}.csv',rr);write_json(ctx.out/f'probe_result_{ctx.tag}.json',dict(candidates=rr,base=base,target=target,source=dec,validation_only=True,no_weight_update=True,not_a_test_efficacy_result=True))
    return ctx.done(dict(training_fits=0,validation_operating_points=24))

def validation_cases(ctx,ref,fold,part='validation'):
    src=ctx.root/'fresh_sources'/ctx.dataset/f'outer_fold_{fold}';inv=load(src/f'upstream_inventory_{ctx.tag}.json')
    if part=='validation':records=[r for r in inv['source_records'] if r['split']=='validation']
    else:
        assert part=='test'
        folder=ctx.root/'fresh_cv'/ctx.dataset/f'outer_fold_{fold}'/'test_sources';records=[load(p) for p in folder.glob(f'*_record_{ctx.tag}.json')];assert {r['image'] for r in records}==set(inv['split']['test'])
    cases=load_case_records(ref,records)
    cm=load(ctx.root/'fresh_cv'/ctx.dataset/'RGB_context'/f'context_manifest_{ctx.tag}.json')
    from flow_matching_refiner.calibration_metrics import truth_cache
    values={}
    for n,c in cases.items():
        r=next(q for q in cm['records'] if q['image']==n);assert sha256_file(r['path'])==r['sha256']
        with np.load(r['path'],allow_pickle=False) as z:sp=torch.from_numpy(z['spatial'][0].copy());v=torch.from_numpy(z['global_vector'][0].copy())
        g=ref.data[n]['g'].numpy()[0]>=.5;f=ref.data[n]['fov'].numpy()[0]>.5;truth=truth_cache(g&f);truth['fov']=f
        values[n]=dict(name=n,rgb=c['rgb'],source=c['source'],p=c['source512'].numpy()[0],coarse=c['coarse512'].numpy()[0],g=g,truth=truth,context=sp,vector=v,record=c['record'],context_record=r)
    return values

def calibration(ctx):
    from flow_matching_refiner import calibration_metrics as cm
    ref=native_ref(ctx);r=module(ctx,'run_validation_calibration',[("project_residual_512(delta,'CHASE')","project_residual_512(delta,DATASET)")]);r.OLD=ctx.root/'fresh_cv'/ctx.dataset;r.OT=ctx.tag;r.DATASET=ctx.dataset
    def ap(p,c):return float(average_precision_score(c['g'][c['fov']],np.asarray(p,dtype=np.float32)[c['fov']]))
    def cheap(p,c):return cm.cheap_curve(np.where(c['fov'],p,0).astype(np.float32),c)
    def score(p,c,t):return cm.score_mask((np.asarray(p,dtype=np.float32)>=np.float32(t))&c['fov'],c)
    r.ap=ap;r.cheap_curve=cheap;r.score_prob=score
    choices=[];bases=[]
    for fold in (1,2,3):
        cases=validation_cases(ctx,ref,fold);basepath=ctx.out/f'fold{fold}_Option3_calibration_locked_{ctx.tag}.json'
        base=load(basepath) if basepath.exists() else r.calibrate_base(fold,cases,ctx.out,ctx.tag);bases.append(base)
        for mode in ('flow','direct'):
            p=ctx.out/f'fold_{fold}/{mode}/seed_42'/f'selection_locked_{ctx.tag}.json'
            d=load(p) if p.exists() else r.infer_and_select(fold,mode,42,cases,base,ctx.out,ctx.tag,ctx.status,ctx.protected);choices.append(d)
    flow=[d for d in choices if d['mode']=='flow'];mm=cm.mean_scores([d['selection']['metrics'] for d in flow]);bb=cm.mean_scores([bases[d['outer_fold']-1]['calibrated_selection']['metrics'] for d in flow]);passed=cm.qualifies(mm,bb) and any(d['selection']['beta']>0 for d in flow)
    write_json(ctx.out/f'pilot_gate_before_additional_seeds_{ctx.tag}.json',dict(passed=passed,Flow_validation=mm,O3_calibrated_validation=bb,pilot_decisions=choices,rule='Same aggregate four-metric rule as CHASE; only a nonzero Flow correction can trigger the remaining stored seeds.',test_used_for_selection=False))
    entries=[]
    if passed:
        from flow_matching_refiner.global_context_flow import GlobalContextCorrection,predict_delta
        from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
        from flow_matching_refiner.structure_metrics_all import metrics
        for seed in SEEDS[1:]:
            for fold in (1,2,3):
                cases=validation_cases(ctx,ref,fold)
                for mode in ('flow','direct'):
                    p=ctx.out/f'fold_{fold}/{mode}/seed_{seed}'/f'selection_locked_{ctx.tag}.json'
                    choices.append(load(p) if p.exists() else r.infer_and_select(fold,mode,seed,cases,bases[fold-1],ctx.out,ctx.tag,ctx.status,ctx.protected))
        assert len(choices)==30
        lock=ctx.out/f'all30_calibrations_before_new_outer_test_{ctx.tag}.json';write_json(lock,dict(decisions=choices,baseline_calibrations=bases,test_used_for_selection=False))
        for fold in (1,2,3):
            cases=validation_cases(ctx,ref,fold,'test');base=bases[fold-1]['calibrated_selection']
            for dec in [d for d in choices if d['outer_fold']==fold]:
                q=dec['selection'];model=None
                if q['beta']:
                    assert ctx.protect(q['checkpoint'])==q['checkpoint_sha256'];model=GlobalContextCorrection(dec['mode']).cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=True)['state'])
                dest=ctx.out/f'outer_fold_{fold}'/'test'/dec['mode']/f'seed_{dec["seed"]}';dest.mkdir(parents=True,exist_ok=True)
                for n,c in cases.items():
                    ctx.status('locked_calibrated_outer_inference',outer_fold=fold,mode=dec['mode'],seed=dec['seed'],image=n)
                    if model is None:p=c['p'].copy()
                    else:
                        delta,_=predict_delta(model,c['source'],c['rgb'],c['context'],c['vector'],'cuda',q['solver_steps']);p=apply_residual(torch.from_numpy(c['p']),q['beta']*project_residual_512(delta,ctx.dataset)[0]).numpy()
                    f=c['truth']['fov'];mask=(p>=np.float32(q['threshold']))&f;om=(c['p']>=np.float32(base['threshold']))&f;path=dest/f'{n}_predictions_{ctx.tag}.npz';np.savez_compressed(path,selected=mask[None].astype(np.float32),probability=p[None],O3=om[None].astype(np.float32),O3_probability=c['p'][None],coarse=c['coarse'][None],ground_truth=c['g'][None].astype(np.float32),fov=f[None]);value=score(p,c['truth'],q['threshold']);assert value=={k:metrics(mask,c['g'],c['coarse'],1.,f)[k] for k in value}
                    entries.append(dict(outer_fold=fold,mode=dec['mode'],seed=dec['seed'],image=n,path=str(path),sha256=sha256_file(path),selection=q,O3_calibration=base,average_precision=ap(p,c['truth']),metrics=value))
                if model is not None:del model;cleanup()
        write_json(ctx.out/f'calibrated_outer_prediction_manifest_{ctx.tag}.json',dict(records=entries,selection_lock=str(lock),selection_lock_sha256=sha256_file(lock)))
    write_json(ctx.out/f'all_validation_calibration_choices_{ctx.tag}.json',dict(decisions=choices,baseline_calibrations=bases,training_performed=False,test_used_for_selection=False,pilot_gate_passed=passed,scope='Same conditional calibration protocol as CHASE; AP-max candidates are optimistic validation diagnostics.'))
    return ctx.done(dict(training_fits=0,validation_systems=len(choices),threshold_candidates=1296*len(choices),FOV_correct_AP=True,pilot_gate_passed=passed,test_prediction_files=len(entries),qualifying_folds={m:sum(d['mode']==m and d['seed']==42 and d['selection']['joint_gate'] for d in choices) for m in ('flow','direct')}))

def gap(ctx):
    from flow_matching_refiner.run_gap_external_validation import diagnose_fov
    from flow_matching_refiner.calibration_metrics import KEYS,qualifies
    ref=native_ref(ctx);individual=[];folds=[]
    for fold in (1,2,3):
        cases=validation_cases(ctx,ref,fold)
        for n,c in cases.items():
            ctx.status('fresh_fold_gap_diagnosis',outer_fold=fold,image=n)
            rgb=ref.data[n]['rgb'].numpy().transpose(1,2,0);result,arrays=diagnose_fov(c['p'],rgb,c['g'],c['truth']['fov']);p=ctx.out/f'fold{fold}_{n}_gap_arrays_{ctx.tag}.npz';np.savez_compressed(p,**arrays)
            individual.append(dict(outer_fold=fold,image=n,result=result,path=str(p),sha256=sha256_file(p),source=c['record']))
        rr=[r for r in individual if r['outer_fold']==fold];s={k:sum(r['result']['stats'][k] for r in rr) for k in rr[0]['result']['stats']};missing=s['missing_thin_points'];gf=s['bounded_gap_thin_points']/missing if missing else 0.;cf=s['proposal_covered_missing_thin_points']/missing if missing else 0.
        folds.append(dict(outer_fold=fold,images=list(cases),counts=s,bounded_gap_fraction=gf,candidate_coverage=cf,geometric_gate=gf>=.2 and cf>=.1))
    mean={kind:{k:float(np.mean([r['result']['metrics'][kind][k] for r in individual])) for k in KEYS} for kind in ('O3','oracle_GT_filtered_proposals','unfiltered_proposal_union')};oracle=qualifies(mean['oracle_GT_filtered_proposals'],mean['O3']);passed=sum(f['geometric_gate'] for f in folds)>=2 and oracle
    write_json(ctx.out/f'diagnosis_results_{ctx.tag}.json',dict(folds=folds,individual=individual,validation_macro_metrics=mean,feasibility_gate=passed,gate_rule='At least2/3folds with gapshare>=.20 and GT-free coverage>=.10, plus four-metric oracle gain',oracle_uses_GT=True,unfiltered_not_a_learned_model=True,pilot_training_started=False,scope='Mirror CHASE diagnosis. No connection-Flow model existed on CHASE; diagnostic gains do not establish learned efficacy.'))
    return ctx.done(dict(training_fits=0,validation_images=len(individual),feasibility_gate=passed,new_connection_Flow_trained=False))
