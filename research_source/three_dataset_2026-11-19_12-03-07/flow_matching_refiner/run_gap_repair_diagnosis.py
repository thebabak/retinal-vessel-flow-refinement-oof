"""Validation-only, preregistered feasibility gate for a connection-flow pilot."""
from __future__ import annotations
import argparse,csv,gc,json,shutil,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.run_validation_calibration import OLD,OT,read_cases
from flow_matching_refiner.gap_repair_diagnosis import CONFIG,diagnose,self_test
from flow_matching_refiner.calibration_metrics import KEYS,qualifies

def csvwrite(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def figure(path,rgb,arrays,result,name,fold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13.5,3.7))
    axes[0].imshow(rgb);axes[0].set_title('RGB / Option-3 gaps')
    overlay=np.zeros((*arrays['gt'].shape,4),np.float32);overlay[arrays['missing_thin']]=[1,.1,.1,.9]
    overlay[arrays['bounded_gaps']]=[0,1,1,1];axes[0].imshow(overlay)
    axes[1].imshow(arrays['binary'],cmap='gray',vmin=0,vmax=1);axes[1].set_title('Option 3 at threshold 0.5')
    axes[2].imshow(rgb);overlay=np.zeros_like(overlay);overlay[arrays['proposal_tube']]=[1,.8,0,.7];axes[2].imshow(overlay);axes[2].set_title('GT-free candidate tubes')
    axes[3].imshow(arrays['oracle_mask'],cmap='gray',vmin=0,vmax=1);axes[3].set_title('GT-filtered oracle (not a model)')
    for ax in axes:ax.axis('off')
    fig.suptitle(f'Fold {fold} validation | {name} | red: missed thin points; cyan: short GT-bounded gaps',fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=155);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not a.execute:print('Inert without --execute. No training or test access.');return
    out=a.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and out.is_dir() and not (out/f'plan_{tag}.json').exists()
    state=dict(output=str(out),tag=tag,training_performed=False,HRF_started=False,DRIVE_started=False)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_gap_repair_diagnosis.json',state)
    try:
        self_test();protected={}
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
        for n in ('template261006.tex','last.tex','tmp/current_validation_calibration_delivery.json'):
            p=ROOT/n
            if p.is_file():protected[str(p)]=sha256_file(p)
        plan=dict(locked_at=stamp(),stage='feasibility diagnosis BEFORE any new model training',source_study=str(OLD),folds=[1,2,3],source_seed=42,validation_images_per_fold=4,proposal_config=CONFIG,
          inference_proposals='Probability >=0.5 skeleton endpoints; distinct 8-connected prediction components; distance2..32; outward tangent cosine>=0; min component8pixels. GT-free minimum-cost route uses probabilities and RGB green dark contrast, bbox margin6, <=64path pixels; tube radius2.',
          diagnostic_gap_definition='GT-skeleton missed 8-connected components of <=32points with >=2disconnected covered-GT-skeleton anchor groups in one-pixel neighbourhood. Strict subset connects distinct prediction components. This is a proxy, not clinical annotation.',
          feasibility_gate=dict(bounded_gap_share_of_missed_thin_at_least=.20,GT_free_candidate_coverage_of_missed_thin_at_least=.10,minimum_qualifying_validation_folds=2,pooled_GT_filtered_oracle_four_metric_gain_required=True),
          feasibility_scope='Exploratory engineering thresholds, fixed before reading diagnostic results; not clinical or statistical cutoffs. Require at least20percent of missed thin points to be short anchored gaps and10percent to lie in GT-free candidate tubes in at least2folds. Aggregate oracle must preserve Dice/precision and improve clDice/thin. Gate only authorizes a pilot, never an efficacy claim.',
          oracle='Add only GT-positive pixels within the GT-free proposal tubes. Explicitly label-unavailable at inference. Unfiltered union also scored to expose false-bridge cost. Neither is Flow or a learned refiner.',
          pilot_if_gate_passes='New connection/path-conditioned Flow and matched direct control, seed42, retained nested-OOF training sources; settings selected only with same inner validation; preserve Dice/precision and improve clDice/thin versus Option3. No old paper/results edits. No DRIVE/HRF automatically from diagnostic oracle.',
          stops_if_gate_fails=True,outer_test_access_allowed=False,previous_subject_reuse='These validation subjects were used in earlier model development; exploratory diagnosis only.',training_performed=False)
        write_json(out/f'plan_{tag}.json',plan);snap=out/'source';snap.mkdir()
        for n in ('gap_repair_diagnosis','run_gap_repair_diagnosis','calibration_metrics','run_validation_calibration'):
            shutil.copy2(ROOT/'flow_matching_refiner'/f'{n}.py',snap/f'{n}_{tag}.py')
        rows=[];individual=[];folds=[];(out/'figures').mkdir();(out/'diagnostic_arrays').mkdir()
        for fold in (1,2,3):
            status('validation_diagnosis',outer_fold=fold);cases=read_cases(fold,'validation',protected);fr=[]
            for n,c in cases.items():
                status('validation_diagnosis',outer_fold=fold,image=n)
                rgb=np.asarray(Image.open(ROOT/'data'/f'{n}.jpg').convert('RGB').resize((512,512),Image.Resampling.BILINEAR),dtype=np.float32)/255.
                result,arrays=diagnose(c['p'],rgb,c['g']);path=out/'diagnostic_arrays'/f'fold{fold}_{n}_gap_diagnostic_{tag}.npz';np.savez_compressed(path,**arrays)
                record=dict(outer_fold=fold,image=n,source_record=c['record'],result=result,arrays=str(path),arrays_sha256=sha256_file(path));individual.append(record);fr.append(record)
                row=dict(outer_fold=fold,image=n,**result['stats'])
                for method,m in result['metrics'].items():row.update({method+'_'+k:v for k,v in m.items()})
                rows.append(row);fp=out/'figures'/f'fold{fold}_{n}_gap_diagnostic_{tag}.png';figure(fp,rgb,arrays,result,n,fold)
                write_json(out/f'fold{fold}_{n}_gap_records_{tag}.json',record)
            sums={k:sum(r['result']['stats'][k] for r in fr) for k in fr[0]['result']['stats']}
            missing=sums['missing_thin_points'];gap_fraction=sums['bounded_gap_thin_points']/missing if missing else 0.;coverage=sums['proposal_covered_missing_thin_points']/missing if missing else 0.
            means={method:{k:float(np.mean([r['result']['metrics'][method][k] for r in fr])) for k in KEYS} for method in fr[0]['result']['metrics']}
            passed=gap_fraction>=.20 and coverage>=.10
            folds.append(dict(outer_fold=fold,images=list(cases),counts=sums,bounded_gap_fraction_of_missed_thin=gap_fraction,candidate_fraction_of_missed_thin=coverage,geometric_feasibility_passed=passed,mean_metrics=means))
            print('GAP DIAGNOSIS FOLD',fold,'bounded share',gap_fraction,'candidate coverage',coverage,'feasible',passed,flush=True);del cases;gc.collect()
        macro={method:{k:float(np.mean([r['result']['metrics'][method][k] for r in individual])) for k in KEYS} for method in individual[0]['result']['metrics']}
        oracle_ok=qualifies(macro['oracle_GT_filtered_proposals'],macro['O3']);passing=sum(f['geometric_feasibility_passed'] for f in folds);passed=passing>=2 and oracle_ok
        aggregate_counts={k:sum(r['result']['stats'][k] for r in individual) for k in individual[0]['result']['stats']}
        assert all(sha256_file(p)==h for p,h in protected.items());write_json(out/f'protected_inputs_unchanged_{tag}.json',protected);csvwrite(out/f'validation_gap_per_image_{tag}.csv',rows)
        foldrows=[]
        for f in folds:
            rr=dict(outer_fold=f['outer_fold'],bounded_gap_fraction_of_missed_thin=f['bounded_gap_fraction_of_missed_thin'],candidate_fraction_of_missed_thin=f['candidate_fraction_of_missed_thin'],geometric_feasibility_passed=f['geometric_feasibility_passed'],**f['counts'])
            for method,m in f['mean_metrics'].items():rr.update({method+'_'+k:v for k,v in m.items()})
            foldrows.append(rr)
        csvwrite(out/f'validation_gap_by_fold_{tag}.csv',foldrows)
        result=dict(completed_at=stamp(),plan=plan,folds=folds,aggregate_counts=aggregate_counts,validation_macro_metrics=macro,oracle_four_metric_goal=oracle_ok,qualifying_folds=passing,connection_flow_pilot_authorized_by_feasibility=passed,pilot_training_started=False,training_performed=False,outer_test_accessed=False,HRF_started=False,DRIVE_started=False,previous_files_unchanged=True,protected_file_count=len(protected),images_analyzed=len(individual))
        write_json(out/f'diagnosis_results_{tag}.json',result);write_json(out/f'completed_diagnosis_{tag}.json',dict(completed_at=stamp(),feasibility_gate_passed=passed,training_performed=False,images=12,outer_test_accessed=False));status('diagnosis_complete',feasibility_gate_passed=passed);print('GAP DIAGNOSIS COMPLETE',passed,flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
