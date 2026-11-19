"""Inference-only validation calibration pilot, with a conditional five-seed arm.

No optimizer, backward pass or training call is used. All data/checkpoints are
from the retained subject-disjoint CHASE source chain.
"""
from __future__ import annotations
import argparse,csv,gc,json,shutil,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.global_context_flow import GlobalContextCorrection,predict_delta
from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
from flow_matching_refiner.run_native_rollout_pilot import state_sha
from flow_matching_refiner.calibration_metrics import KEYS,THRESHOLDS,truth_cache,cheap_curve,score_prob,mean_scores,ap,qualifies,gain,selection_key,self_test

OLD=ROOT/'outputs/precision_guarded_subject_CV_CHASE_2026-11-18_02-14-28';OT=OLD.name[-19:];SEEDS=(42,123,2026,3000,3001);EPOCHS=(20,40,60,80);BETAS=(.125,.25,.5,1.)
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def savecsv(p,rows):
    with Path(p).open('x',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def protect(path,protected):p=Path(path);h=sha256_file(p);protected[str(p)]=h;return h

def read_cases(fold,part,protected,test_unlocked=False):
    assert part=='validation' or (part=='test' and test_unlocked);sp=load(OLD/f'plan_{OT}.json')['folds'][fold-1]
    if part=='validation':rs=[r for r in load(OLD/f'outer_fold_{fold}'/f'upstream_inventory_{OT}.json')['source_records'] if r['split']=='validation']
    else:rs=load(OLD/f'outer_fold_{fold}'/f'outer_test_source_manifest_{OT}.json')
    assert {r['image'] for r in rs}==set(sp[part]);cm=load(OLD/'frozen_RGB_context'/f'context_manifest_{OT}.json');cases={}
    for r in rs:
        assert protect(r['path'],protected)==r['sha256'];n=r['image'];pr=r['provenance'];assert not set(sp['test'])&set(pr['fit_images']+pr['validation_images']);assert protect(pr['O3_checkpoint'],protected)==pr['O3_sha256'];assert protect(pr['baseline_checkpoint'],protected)==pr['baseline_sha256']
        rp=ROOT/'data'/f'{n}.jpg';gp=ROOT/'data'/f'{n}_1stHO.png';protect(rp,protected);protect(gp,protected)
        with Image.open(rp) as im:rgb=torch.from_numpy((np.asarray(im.convert('RGB'))/255.).astype(np.float32)).permute(2,0,1).contiguous()
        with Image.open(gp) as im:g=(np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>127).copy()
        with np.load(r['path'],allow_pickle=False) as z:source=torch.from_numpy(z['source_native'].copy());p=z['source512'][0].copy();coarse=z['coarse512'][0].copy()
        cr=next(v for v in cm['records'] if v['image']==n);assert protect(cr['path'],protected)==cr['sha256']
        with np.load(cr['path'],allow_pickle=False) as z:context=torch.from_numpy(z['spatial'][0].copy());vector=torch.from_numpy(z['global_vector'][0].copy())
        cases[n]=dict(name=n,rgb=rgb,source=source,p=p,coarse=coarse,g=g,truth=truth_cache(g),context=context,vector=vector,record=r,context_record=cr)
    return cases

def curve_search(probabilities,cases,reference,meta):
    names=list(cases);curves={n:cheap_curve(probabilities[n],cases[n]['truth']) for n in names};meanAP=float(np.mean([ap(probabilities[n],cases[n]['truth']) for n in names]));rows=[];best=None
    for i,t in enumerate(THRESHOLDS):
        cheap={k:float(np.mean([curves[n][k][i] for n in names])) for k in ('dice','precision','thin_skeleton_recall')};passed=cheap['dice']>=reference['dice']-1e-12 and cheap['precision']>=reference['precision']-1e-12 and cheap['thin_skeleton_recall']>reference['thin_skeleton_recall']+1e-12;mm=dict(cheap,cldice=None)
        if passed:
            mm=mean_scores([score_prob(probabilities[n],cases[n]['truth'],t) for n in names]);assert all(abs(mm[k]-cheap[k])<1e-12 for k in cheap)
        qualified=passed and qualifies(mm,reference);row=dict(**meta,threshold=float(t),macro_average_precision=meanAP,**{k:mm[k] for k in KEYS},cheap_constraints_passed=passed,hard_cldice_computed=passed,qualifies=bool(qualified));rows.append(row)
        if qualified:
            q=dict(**meta,threshold=float(t),macro_average_precision=meanAP,metrics=mm,structural_gain=gain(mm,reference),joint_gate=True)
            if best is None or selection_key(q,reference)>selection_key(best,reference):best=q
    return best,rows,meanAP

def calibrate_base(fold,cases,out,tag):
    p={n:c['p'] for n,c in cases.items()};fixed=mean_scores([score_prob(c['p'],c['truth'],.5) for c in cases.values()]);best,rows,meanAP=curve_search(p,cases,fixed,dict(outer_fold=fold,mode='O3',seed=42,epoch=0,beta=0.,solver_steps=0));chosen=best if best else dict(outer_fold=fold,mode='O3',seed=42,epoch=0,beta=0.,solver_steps=0,threshold=.5,macro_average_precision=meanAP,metrics=fixed,structural_gain=0.,joint_gate=False);result=dict(outer_fold=fold,fixed_threshold_metrics=fixed,calibrated_selection=chosen,calibration_changed=chosen['threshold']!=.5,locked_at=stamp());savecsv(out/f'fold{fold}_Option3_thresholds_{tag}.csv',rows);write_json(out/f'fold{fold}_Option3_calibration_locked_{tag}.json',result);return result

def infer_and_select(fold,mode,seed,cases,base,out,tag,status,protected):
    dest=out/f'fold_{fold}/{mode}/seed_{seed}';dest.mkdir(parents=True,exist_ok=False);reference=base['calibrated_selection']['metrics'];best=None;bestAP=None;rows=[];aps=[];cache_records=[];steps=4 if mode=='flow' else 1
    for epoch in EPOCHS:
        cp=OLD/f'outer_fold_{fold}/context_refiners'/mode/f'seed_{seed}'/f'field_epoch{epoch}_{OT}.pt';cp_sha=protect(cp,protected);obj=torch.load(cp,map_location='cuda',weights_only=True);assert obj['epoch']==epoch and obj['mode']==mode and obj['seed']==seed;model=GlobalContextCorrection(mode).cuda().eval().requires_grad_(False);model.load_state_dict(obj['state']);initial=state_sha(model);deltas={}
        for n,c in cases.items():
            status('validation_raw_inference',outer_fold=fold,mode=mode,seed=seed,epoch=epoch,image=n);delta,_=predict_delta(model,c['source'],c['rgb'],c['context'],c['vector'],'cuda',steps=steps);deltas[n]=project_residual_512(delta,'CHASE').numpy()[0].copy()
        assert state_sha(model)==initial;path=dest/f'epoch{epoch}_validation_delta512_{tag}.npz';np.savez_compressed(path,**deltas);cache=dict(outer_fold=fold,mode=mode,seed=seed,epoch=epoch,checkpoint=str(cp),checkpoint_sha256=cp_sha,state_unchanged_during_inference=True,path=str(path),sha256=sha256_file(path));cache_records.append(cache);del model,obj;gc.collect();torch.cuda.empty_cache()
        for beta in BETAS:
            p={n:apply_residual(torch.from_numpy(c['p']),beta*torch.from_numpy(deltas[n])).numpy() for n,c in cases.items()};meta=dict(outer_fold=fold,mode=mode,seed=seed,epoch=epoch,beta=beta,solver_steps=steps);q,rr,meanAP=curve_search(p,cases,reference,meta);rows.extend(rr);a=dict(**meta,macro_average_precision=meanAP,delta_cache=str(path),delta_cache_sha256=cache['sha256'],checkpoint=str(cp),checkpoint_sha256=cp_sha);aps.append(a)
            if bestAP is None or meanAP>bestAP['macro_average_precision']+1e-12:bestAP=a
            if q is not None and (best is None or selection_key(q,reference)>selection_key(best,reference)):best=dict(**q,checkpoint=str(cp),checkpoint_sha256=cp_sha,delta_cache=str(path),delta_cache_sha256=cache['sha256'])
        print('CALIBRATION VALIDATION',fold,mode,seed,epoch,'qualified',best is not None,'best AP',bestAP['macro_average_precision'],flush=True)
    assert len(rows)==1296 and len(aps)==16;selected=best if best else dict(outer_fold=fold,mode=mode,seed=seed,epoch=0,beta=0.,solver_steps=0,threshold=base['calibrated_selection']['threshold'],macro_average_precision=base['calibrated_selection']['macro_average_precision'],metrics=reference,structural_gain=0.,joint_gate=False,checkpoint=None,checkpoint_sha256=None,delta_cache=None,delta_cache_sha256=None);decision=dict(locked_at=stamp(),outer_fold=fold,mode=mode,seed=seed,selection=selected,baseline_calibration=base,qualifying_candidates=sum(r['qualifies'] for r in rows),best_validation_AP_diagnostic=bestAP,test_used_for_selection=False,training_performed=False,weights_updated=False);savecsv(dest/f'validation_threshold_candidates_{tag}.csv',rows);savecsv(dest/f'validation_AP_candidates_{tag}.csv',aps);write_json(dest/f'validation_delta_cache_manifest_{tag}.json',cache_records);write_json(dest/f'selection_locked_{tag}.json',decision);return decision

def validation_PR_figures(fold,cases,decisions,out,tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    choices={d['mode']:d['best_validation_AP_diagnostic'] for d in decisions};probs={'O3':{n:c['p'] for n,c in cases.items()}}
    for mode,q in choices.items():
        with np.load(q['delta_cache'],allow_pickle=False) as z:probs[mode]={n:apply_residual(torch.from_numpy(c['p']),q['beta']*torch.from_numpy(z[n])).numpy() for n,c in cases.items()}
    fig,axes=plt.subplots(1,4,figsize=(14,3.4));curve_records=[]
    for ax,(n,c) in zip(axes,cases.items()):
        for mode,color in [('O3','#6b7280'),('flow','#166b9c'),('direct','#bf7132')]:
            pp=probs[mode][n];precision,recall,t=precision_recall_curve(c['g'].reshape(-1),pp.reshape(-1),drop_intermediate=True);value=ap(pp,c['truth']);ax.plot(recall,precision,color=color,lw=1.3,label=mode+f' AP={value:.5f}');curve_records.append(dict(outer_fold=fold,image=n,mode=mode,average_precision=value));path=out/'PR_curves'/f'fold{fold}_{n}_{mode}_PR_{tag}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,precision=precision,recall=recall,threshold=t)
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_title(n);ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower left')
    fig.suptitle(f'Fold{fold} validation only | seed42 | neural curves use AP-max diagnostic configurations');fig.tight_layout();fig.savefig(out/f'fold{fold}_validation_PR_{tag}.png',dpi=180);plt.close(fig);savecsv(out/f'fold{fold}_diagnostic_AP_per_image_{tag}.csv',curve_records)

def run_outer(systems,bases,out,tag,status,protected,lock):
    records=[];manifest=[]
    for fold in (1,2,3):
        cases=read_cases(fold,'test',protected,True);base=bases[fold]['calibrated_selection'];base_metrics={}
        for n,c in cases.items():base_metrics[n]=dict(fixed=score_prob(c['p'],c['truth'],.5),calibrated=score_prob(c['p'],c['truth'],base['threshold']),AP=ap(c['p'],c['truth']))
        for d in [v for v in systems if v['outer_fold']==fold]:
            q=d['selection'];model=None
            if q['beta']>0:
                assert sha256_file(q['checkpoint'])==q['checkpoint_sha256'];model=GlobalContextCorrection(d['mode']).cuda().eval().requires_grad_(False);model.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=True)['state'])
            dest=out/f'outer_test/fold_{fold}'/d['mode']/f'seed_{d["seed"]}';dest.mkdir(parents=True,exist_ok=False)
            for n,c in cases.items():
                status('locked_outer_inference',outer_fold=fold,mode=d['mode'],seed=d['seed'],image=n)
                if model is None:delta=np.zeros_like(c['p']);p=c['p'].copy()
                else:dd,_=predict_delta(model,c['source'],c['rgb'],c['context'],c['vector'],'cuda',q['solver_steps']);delta=project_residual_512(dd,'CHASE').numpy()[0].copy();p=apply_residual(torch.from_numpy(c['p']),q['beta']*torch.from_numpy(delta)).numpy()
                mm=score_prob(p,c['truth'],q['threshold']);value=ap(p,c['truth']);path=dest/f'{n}_probabilities_{tag}.npz';np.savez_compressed(path,probability=p,O3_probability=c['p'],delta512=delta,ground_truth=c['g'],selected_mask=p>=np.float32(q['threshold']),O3_calibrated_mask=c['p']>=np.float32(base['threshold']));manifest.append(dict(outer_fold=fold,mode=d['mode'],seed=d['seed'],image=n,path=str(path),sha256=sha256_file(path),selection=q,metrics=mm,average_precision=value,source_record=c['record']));records.append(dict(outer_fold=fold,seed=d['seed'],method='Flow' if d['mode']=='flow' else 'Direct',image=n,subject=int(n.split('_')[1][:2]),threshold=q['threshold'],**mm,average_precision=value))
                if d['mode']=='flow':
                    for method,key,t in [('O3_fixed','fixed',.5),('O3_calibrated','calibrated',base['threshold'])]:records.append(dict(outer_fold=fold,seed=d['seed'],method=method,image=n,subject=int(n.split('_')[1][:2]),threshold=t,**base_metrics[n][key],average_precision=base_metrics[n]['AP']))
            del model;gc.collect();torch.cuda.empty_cache()
        del cases;gc.collect()
    assert len(records)==560 and len(manifest)==280;savecsv(out/f'outer_per_image_metrics_{tag}.csv',records);write_json(out/f'outer_prediction_manifest_{tag}.json',dict(global_lock=str(lock),global_lock_sha256=sha256_file(lock),records=manifest));return records

def main():
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    if not args.execute:print('Inert without --execute. Inference only.');return
    out=args.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and out.is_dir() and not (out/f'plan_{tag}.json').exists();state=dict(output=str(out),tag=tag,training_performed=False,HRF_started=False,DRIVE_started=False)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_validation_calibration.json',state)
    try:
        self_test();torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();protected={};oldplan=OLD/f'plan_{OT}.json';protect(oldplan,protected);protect(OLD/'frozen_RGB_context'/f'context_manifest_{OT}.json',protected)
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):protect(p,protected)
        for name in ('template261006.tex','last.tex','tmp/current_precision_subject_cv_delivery.json'):
            p=ROOT/name
            if p.is_file():protect(p,protected)
        snap=out/'source';snap.mkdir()
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):shutil.copy2(p,snap/f'{p.stem}_{tag}.py')
        plan=dict(locked_at=stamp(),old_study=str(OLD),pilot_seed=42,pilot_folds=[1,2,3],seeds=SEEDS,checkpoint_epochs=EPOCHS,threshold_grid=[float(t) for t in THRESHOLDS],beta_grid=BETAS,flow_Euler_steps=4,direct_calls=1,neural_configurations_per_arm_fold_seed=1296,raw_ranking_metric='Macro individual-image sklearn average_precision_score, non-interpolated AP. Not trapezoidal PR area. Independent of mask threshold.',baseline='Calibrate Option3 first using the same four-metric rule against its fixed0.5output. If no qualifying threshold, retain0.5. This baseline is then the reference for BOTH neural arms. Calibration-only baseline gains are never credited to Flow.',gate='Mean validation Dice and precision>=calibratedOption3 tolerance1e-12, clDice and thin recall strictly higher. Select max structural gain, then Dice/precision, threshold closest0.5, weaker beta, earlier checkpoint. No candidate => exact calibratedOption3 fallback.',pilot_trigger='Aggregate chosen Flow versus calibratedOption3 over the12inner validation images from3folds. If all4goals met with a nonzero Flow correction, process the4additional stored seeds. Otherwise stop after6pilot searches; no outer-test access.',test='Lock all30choices before loading any outer-test arrays/labels. Compare fixedOption3, calibratedOption3, calibratedFlow and calibrateddirect;280retained probability files,560rows. Conditional5seed comparisons; not5new full-pipeline trainings.',fairness='Flow/direct use same4checkpoint x4beta x81threshold grid and identical selection predicate. Flow4Euler is fixed at training rollout, not searched; direct is one call.',AP_diagnostic='AP-max neural configuration chosen using validation only for descriptive PR plots; it is not used to select classifier output or activate test evaluation.',limitations='These subjects and test cohorts were used in prior development; calibration is exploratory re-evaluation. Reusing trained checkpoints changes no weights, but new inference/metric calculations are performed. No independent external confirmation.',training_performed=False,previous_papers_and_results_unchanged=True)
        write_json(out/f'plan_{tag}.json',plan);write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False));bases={};systems=[]
        for fold in (1,2,3):
            cases=read_cases(fold,'validation',protected);bases[fold]=calibrate_base(fold,cases,out,tag);pair=[infer_and_select(fold,mode,42,cases,bases[fold],out,tag,status,protected) for mode in ('flow','direct')];systems.extend(pair);validation_PR_figures(fold,cases,pair,out,tag);del cases;gc.collect()
        f=[d for d in systems if d['mode']=='flow'];mm=mean_scores([d['selection']['metrics'] for d in f]);bb=mean_scores([bases[d['outer_fold']]['calibrated_selection']['metrics'] for d in f]);passed=qualifies(mm,bb) and any(d['selection']['beta']>0 for d in f);gate=out/f'pilot_gate_locked_{tag}.json';write_json(gate,dict(locked_at=stamp(),passed=passed,Flow_validation=mm,calibrated_O3_validation=bb,decisions=systems,test_accessed=False,training_performed=False));gh=sha256_file(gate);print('CALIBRATION PILOT GATE',passed,flush=True);outer=[]
        if passed:
            for seed in SEEDS[1:]:
                for fold in (1,2,3):
                    cases=read_cases(fold,'validation',protected);systems.extend(infer_and_select(fold,mode,seed,cases,bases[fold],out,tag,status,protected) for mode in ('flow','direct'));del cases;gc.collect()
            assert len(systems)==30;lock=out/f'all30_choices_locked_before_outer_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=systems,baseline_calibrations=bases,pilot_gate_sha256=gh,plan_sha256=sha256_file(out/f'plan_{tag}.json')));outer=run_outer(systems,bases,out,tag,status,protected,lock)
        assert sha256_file(gate)==gh;assert all(sha256_file(p)==h for p,h in protected.items());write_json(out/f'protected_inputs_unchanged_{tag}.json',protected);result=dict(completed_at=stamp(),pilot_passed=passed,searched_arm_fold_seeds=len(systems),validation_configurations=len(systems)*1296,outer_metric_rows=len(outer),training_performed=False,old_files_unchanged=True,HRF_started=False,DRIVE_started=False,external_run_condition_pending=bool(outer));write_json(out/f'completed_experiment_{tag}.json',result);status('complete',**result);print('CALIBRATION STUDY COMPLETE',result,flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise
if __name__=='__main__':main()
