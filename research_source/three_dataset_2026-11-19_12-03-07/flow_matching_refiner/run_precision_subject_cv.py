"""New CHASE subject-grouped nested source chain and five matched refiner seeds."""
import argparse,csv,gc,json,shutil,traceback
from pathlib import Path
import numpy as np
import torch
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.subject_cv_sources import load_raw,partitions,build_fold,baseline_predict,option3_predict,self_test
from flow_matching_refiner.subject_cv_context_training import train
from flow_matching_refiner.global_context_data import build_context,WEIGHTS
from flow_matching_refiner.global_context_flow import GlobalContextCorrection,predict_delta
from flow_matching_refiner.precision_guard_selection import run_existing,save_csv
from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual
from flow_matching_refiner.structure_metrics_all import metrics
SEEDS=(42,123,2026,3000,3001)

def paired(fr,cases,context,fold,seed,out,tag,status):
    decisions=[]
    for mode in ('flow','direct'):
        status('outer_fold_refiner_training',outer_fold=fold,mode=mode,seed=seed);d=train(fr,cases,{n:context[n] for n in cases},seed,mode,out/mode/f'seed_{seed}',tag,status,False);decisions.append(dict(outer_fold=fold,**d))
    assert decisions[0]['initial_state_sha256']==decisions[1]['initial_state_sha256'] and decisions[0]['parameter_count']==decisions[1]['parameter_count'];assert sha256_file(out/'flow'/f'seed_{seed}'/f'patch_draws_{tag}.csv')==sha256_file(out/'direct'/f'seed_{seed}'/f'patch_draws_{tag}.csv');return decisions

def evaluate(ref,folds,upstream,systems,context,out,tag,status,lock):
    records=[];pm=[]
    for sp in folds:
        j=sp['fold'];job=out/f'outer_fold_{j}';u=upstream[j];status('locked_outer_inference',outer_fold=j);coarse=baseline_predict(ref,u['outer_full_baseline'],sp['test']);cases=option3_predict(ref,sp['test'],coarse,u['outer_full_O3'],job/'outer_test_sources',tag,'outer_test');write_json(job/f'outer_test_source_manifest_{tag}.json',[c['record'] for c in cases.values()])
        for seed in SEEDS:
            for mode in ('flow','direct'):
                d=next(s for s in systems if s['outer_fold']==j and s['seed']==seed and s['mode']==mode);q=d['selection'];model=GlobalContextCorrection(mode).cuda();assert sha256_file(q['checkpoint'])==q['checkpoint_sha256'];model.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=True)['state']);dest=job/'outer_test_predictions'/mode/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=False)
                for n,c in cases.items():
                    status('locked_outer_inference',outer_fold=j,mode=mode,seed=seed,image=n)
                    if q['beta']==0:delta=torch.zeros_like(c['source']);imeta=dict(identity=True,GT_free=True)
                    else:delta,imeta=predict_delta(model,c['source'],c['rgb'],context[n]['spatial'][0],context[n]['global_vector'][0],'cuda',q['steps'])
                    p=apply_residual(c['source512'],q['beta']*project_residual_512(delta,'CHASE'));gt=ref.data[n]['g'].numpy();fov=ref.data[n]['fov'].numpy();path=dest/f'{n}_prediction_{tag}.npz';np.savez_compressed(path,refined=p.numpy(),O3=c['source512'].numpy(),coarse=c['coarse512'].numpy(),delta_native=delta.numpy(),ground_truth=gt,fov=fov);mm=metrics(p.numpy()[0],gt[0],c['coarse512'].numpy()[0],1.,fov[0]);om=metrics(c['source512'].numpy()[0],gt[0],c['coarse512'].numpy()[0],1.,fov[0]);pm.append(dict(outer_fold=j,seed=seed,mode=mode,image=n,path=str(path),sha256=sha256_file(path),metrics=mm,O3_metrics=om,selection=q,source=c['record'],inference=imeta))
                    records.append(dict(outer_fold=j,seed=seed,method='Flow' if mode=='flow' else 'Direct',image=n,subject=int(n.split('_')[1][:2]),**mm))
                    if mode=='flow':records.append(dict(outer_fold=j,seed=seed,method='O3',image=n,subject=int(n.split('_')[1][:2]),**om))
                del model;gc.collect();torch.cuda.empty_cache()
        del cases;gc.collect()
    assert len(pm)==280 and len(records)==420;write_json(out/f'outer_prediction_manifest_{tag}.json',dict(global_lock=str(lock),global_lock_sha256=sha256_file(lock),records=pm));save_csv(out/f'outer_per_image_metrics_{tag}.csv',records);return pm

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not a.execute:print('Inert without --execute. CHASE only.');return
    out=a.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and out.is_dir() and not (out/f'plan_{tag}.json').exists();state=dict(output=str(out),tag=tag,HRF_started=False,DRIVE_started=False,dataset='CHASE',completed_outer_folds=0)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_precision_subject_cv.json',state)
    try:
        self_test();torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();status('reading_raw_CHASE');ref,protected=load_raw(out);folds=partitions(list(ref.data));rp=out/f'existing_validation_precision_guard_selections_{tag}.json';assert rp.is_file()
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
        protected[str(WEIGHTS)]=sha256_file(WEIGHTS)
        for name in ('template261006.tex','last.tex','tmp/current_global_context_delivery.json'):
            p=ROOT/name
            if p.is_file():protected[str(p)]=sha256_file(p)
        snap=out/'source';snap.mkdir()
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):shutil.copy2(p,snap/f'{p.stem}_{tag}.py')
        write_json(out/f'protected_before_{tag}.json',protected);plan=dict(locked_at=stamp(),folds=folds,seeds=SEEDS,outer_folds=3,subjects=14,images=28,planned_fresh_baseline_fits=33,planned_fresh_Option3_fits=15,planned_context_refiner_fits=30,source='No prior learned retinal checkpoints or predictions are reused; fresh random initialization. ImageNet context alone is reused frozen.',OOF='Within each outer training set:4subject groups;4single-exclusion and6pair-exclusion baselines plus1full baseline. Each held-group O3 uses coarse inputs excluding held and own-input groups. Outer test subjects excluded from every dependency, including all checkpoint selections.',selection='All model checkpoints selected only on4inner validation images/2subjects. Context candidate gate:Dice and precision>=O3 tolerance1e-12, clDice and thin recall strictly higher. Max mean structural gain; exact identity fallback. All30refiner decisions locked before any outer-test predictions or scores.',training='Baseline seed42 max1000epochs patience20, unchanged AdamW1e-3/wd1e-3 softDice/rotation/flip recipe. Option3 seed42 native120epoch exact prior recipe. Flow/direct80epochs native256patches4/image, effectivebatch8/micro4, unchanged context-objective recipe. Same per-image training budget, initialization/crop streams matched within each refiner pair.',five_seed_scope='Five independently initialized correction-stage seeds share each outer-fold upstream models. They are not five independently trained whole pipelines.',evaluation='Primary original512comparison grid, whole CHASE image, threshold>=.5, thin GT skeleton EDT radius<=1 pixel fixed from preceding method specification.',inference='GT-free baseline/RGB context/Option3/Flow/direct inference. Frozen context may be cached for every RGB independently before fitting; no label or fitted normalization is used.',limitations='All14subjects were encountered in prior method development. Newly excluded folds prevent fitting/selection leakage in this run but do not create a new external cohort. Exploratory grouped nested-source cross-validation.',previous_paper_and_results_unchanged=True,HRF_started=False,DRIVE_started=False);write_json(out/f'plan_{tag}.json',plan);write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False))
        ctx,cm=build_context({n:dict(rgb=d['n_rgb']) for n,d in ref.data.items()},out/'frozen_RGB_context',tag,status);upstream={};systems=[]
        for sp in folds:
            j=sp['fold'];status('building_fresh_outer_source_chain',outer_fold=j);fr,cases,inv=build_fold(ref,sp,out/f'outer_fold_{j}',tag,status);upstream[j]=inv
            for seed in SEEDS:systems.extend(paired(fr,cases,ctx,j,seed,out/f'outer_fold_{j}/context_refiners',tag,status))
            write_json(out/f'outer_fold_{j}'/f'correction_choices_locked_{tag}.json',dict(locked_at=stamp(),decisions=[d for d in systems if d['outer_fold']==j]));status('outer_fold_fitting_complete',completed_outer_folds=j);del cases,fr;gc.collect();torch.cuda.empty_cache()
        assert len(systems)==30;lock=out/f'all30_choices_locked_before_outer_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=systems,plan_sha256=sha256_file(out/f'plan_{tag}.json')));pm=evaluate(ref,folds,upstream,systems,ctx,out,tag,status,lock);status('verifying_old_files');assert all(sha256_file(p)==h for p,h in protected.items());result=dict(completed_at=stamp(),fresh_baseline_fits=33,fresh_Option3_fits=15,fresh_context_refiner_fits=30,outer_prediction_files=len(pm),image_seed_method_rows=420,outer_test_subjects=14,previous_files_unchanged=True,HRF_started=False,DRIVE_started=False,global_lock_sha256=sha256_file(lock));write_json(out/f'completed_experiment_{tag}.json',result);status('complete',**result);print('SUBJECT CV COMPLETE',result,flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise
if __name__=='__main__':main()
