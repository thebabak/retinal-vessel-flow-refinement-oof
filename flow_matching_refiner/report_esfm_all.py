"""FOV-aware verification/reporting for completed DRIVE and HRF ESFM extensions.

Only completed new experiments may be reported. No training, model inference,
selection, or changes to earlier CHASE/article/result files occur here.
Bootstrap clusters are IMAGES, because independent patient IDs are unverified.
Five refiner fits share one frozen baseline; seeds are not bootstrap units.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import assert_output_path

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np

from flow_matching_refiner.data import sha256_file
from flow_matching_refiner.structure_metrics_all import DEFINITION,dependency_versions,metrics

SEEDS=[42,123,2026,3000,3001]
ARMS=["direct","endpoint_iterative","full_fm","selective_no_gate","esfm"]
OUTPUTS=["raw","blend"]
COMPARATORS=["direct","baseline","endpoint_iterative","full_fm","selective_no_gate"]
SUMMARY_METRICS=["dice","iou","cldice","thin_skeleton_recall","precision","recall",
                 "new_error_count","new_error_rate","fixed_error_count","fixed_error_rate",
                 "betti0_absolute_error","fp","fn","thin_skeleton_missed"]
INTERVAL_METRICS=["dice","cldice","thin_skeleton_recall"]
BOOTSTRAP_REPETITIONS=50000
BOOTSTRAP_SEED=20261115
INTERPRETATION=(
    "Exploratory unadjusted image-cluster percentile intervals on previously examined cohorts. "
    "One resized image is the resampling unit; patient-level independence is not established "
    "for DRIVE/HRF by the retained records. All five fixed fitted-refiner outputs for an image "
    "remain together. No pixels or refiner seeds are independently resampled. These intervals "
    "condition on the fitted models, common split and shared baseline; they do not include "
    "full-pipeline or between-training uncertainty. Seed sample SD is descriptive and separate. "
    "There are no p-values, multiplicity adjustments, equivalence claims or formal significance claims."
)


def _json(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def _plane(array,name):
    value=np.asarray(array)
    if value.ndim==3 and value.shape[0]==1:value=value[0]
    if value.ndim!=2 or not np.isfinite(value).all():
        raise ValueError(f"Expected finite single-image array: {name}")
    return value


def _number(row,name):
    value=float(row[name])
    if not np.isfinite(value):raise ValueError(f"Nonfinite CSV metric {name}")
    return value


def _check_metrics(row,actual,context):
    for name,value in actual.items():
        if name not in row or _number(row,name)!=value:
            raise AssertionError(f"Metric mismatch {context}/{name}: CSV {row.get(name)}, recomputed {value}")


def _write_json(path,value):
    with path.open("x",encoding="utf-8") as handle:json.dump(value,handle,indent=2,allow_nan=False)


def _write_csv(path,rows):
    if not rows:raise ValueError("Empty report CSV")
    with path.open("x",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _dataset(plan):
    value=str(plan["dataset"]).upper().replace("-","").replace("_","")
    if value.startswith("DRIVE"):return "DRIVE"
    if value.startswith("HRF"):return "HRF"
    raise ValueError("This new extension reporter supports DRIVE/HRF; preserved CHASE uses its original report")


def image_interval(differences,draw_indices):
    """Image-bootstrap interval conditional on all fixed refiner predictions."""
    values=np.asarray(differences,dtype=np.float64)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError("Finite paired image differences are required")
    if draw_indices.shape!=(BOOTSTRAP_REPETITIONS,len(values)):
        raise ValueError("Bootstrap inventory must be 50000 draws of N image indices")
    if not np.issubdtype(draw_indices.dtype,np.integer) or draw_indices.min()<0 or draw_indices.max()>=len(values):
        raise ValueError("Invalid image-bootstrap index")
    resampled=values[draw_indices].mean(axis=1)
    lo,hi=np.quantile(resampled,[.025,.975],method="linear")
    return dict(
        mean_difference=float(values.mean()),ci95=[float(lo),float(hi)],
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_unit="image",bootstrap_cluster_count=int(len(values)),
        bootstrap_method="paired image means across five fixed refiner fits, resampled N times with replacement",
        percentile_method="numpy.quantile linear at 0.025 and 0.975",
        interval_adjustment="none; exploratory unadjusted percentile interval",
        resamples_training_seeds=False,patient_independence_verified=False,
        interval_inferential_role="exploratory; no formal significance claim",
    )


def report(experiment):
    out = assert_output_path(experiment, must_exist=True);tag=out.name[-19:]
    complete_path=out/f"completed_training_evaluation_{tag}.json"
    if not complete_path.is_file():raise RuntimeError("Completed training/evaluation marker is required")
    complete=_json(complete_path)
    plan_path=out/f"plan_{tag}.json";plan=_json(plan_path)
    dataset=_dataset(plan);n_images={"DRIVE":20,"HRF":12}[dataset]
    expected_records=n_images*(1+len(ARMS)*len(SEEDS)*len(OUTPUTS))
    assert complete.get("fits")==25 and complete.get("records")==expected_records
    for flag in ("choices_locked_before_test","previous_files_byte_identical","OOF_maps_audited"):
        if complete.get(flag) is not True:raise AssertionError(f"Completion audit flag not true: {flag}")
    assert plan["refiner_training_seeds"]==SEEDS and plan["arms"]==ARMS
    paths={name:out/f"{prefix}_{tag}.{extension}" for name,prefix,extension in (
        ("summary_json","summary_statistics","json"),("seed_means_csv","seed_means","csv"),
        ("seed_differences_csv","seed_differences","csv"),("image_differences_csv","image_differences","csv"),
        ("difference_arrays_json","difference_arrays","json"),("verification_json","verified_report","json"))}
    if any(path.exists() for path in paths.values()):raise FileExistsError("Existing timestamped reports must be preserved")
    thin_path=out/f"thin_definition_{tag}.json";thin=_json(thin_path)
    cutoff=float(thin["cutoff"])
    assert thin["quantile"]==1/3 and thin.get("fov_supplied") is True
    env_path=out/f"environment_{tag}.json";env=_json(env_path)
    if env["structural_libraries"]!=dependency_versions():raise RuntimeError("Structural-library version differs from executed experiment")
    locks_path=out/f"all_selections_before_test_{tag}.json";locks=_json(locks_path)
    assert sha256_file(locks_path)==complete["lock_sha256"]
    decisions={(d["arm"],int(d["refiner_seed"])):d for d in locks["decisions"]}
    assert len(decisions)==len(locks["decisions"])==25
    assert set(decisions)==set(itertools.product(ARMS,SEEDS))
    csv_path=out/f"all_test_records_{tag}.csv"
    with csv_path.open(newline="",encoding="utf-8") as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==expected_records
    baseline_rows=[r for r in rows if r["arm"]=="baseline"]
    names=sorted(r["image"] for r in baseline_rows)
    assert len(names)==len(set(names))==n_images
    image_subject={r["image"]:int(r["subject"]) for r in baseline_rows}
    index={}
    for row in rows:
        key=(row["arm"],int(row["refiner_seed"]),row["image"],row["output"])
        assert key not in index
        assert row["image"] in names and int(row["subject"])==image_subject[row["image"]]
        if "dataset" in row:assert _dataset({"dataset":row["dataset"]})==dataset
        index[key]=row
    expected={("baseline",-1,name,"raw") for name in names}
    expected.update(itertools.product(ARMS,SEEDS,names,OUTPUTS))
    assert set(index)==expected
    hashes={str(p.relative_to(out)):sha256_file(p) for p in
            (complete_path,plan_path,thin_path,env_path,locks_path,csv_path)}
    references={};verified_files=0;verified_rows=0
    for arm,seed in itertools.product(ARMS,SEEDS):
        dest=out/arm/f"seed_{seed}";decision=decisions[arm,seed]
        job_complete=_json(dest/f"completed_{tag}.json")
        assert job_complete["records"]==2*n_images and job_complete["test_images"]==n_images
        for filename,digest_key in ((f"best_raw_model_{tag}.pt","raw_sha256"),
                                    (f"selected_blend_model_{tag}.pt","blend_sha256"),
                                    (f"final_model_{tag}.pt","final_sha256")):
            path=dest/filename;digest=sha256_file(path)
            assert digest==decision[digest_key]
            hashes[str(path.relative_to(out))]=digest
        for name in names:
            path=dest/f"{name}_predictions_{tag}.npz";hashes[str(path.relative_to(out))]=sha256_file(path)
            with np.load(path,allow_pickle=False) as saved:
                baseline=_plane(saved["baseline"],"baseline")
                truth=_plane(saved["ground_truth"],"ground_truth")
                fov=_plane(saved["fov"],"fov")>=.5
                assert baseline.shape==truth.shape==fov.shape and np.isin(truth,[0,1]).all() and fov.any()
                assert (baseline>=0).all() and (baseline<=1).all()
                if name not in references:
                    references[name]=(baseline.copy(),truth.copy(),fov.copy())
                    _check_metrics(index["baseline",-1,name,"raw"],metrics(baseline,truth,baseline,cutoff,fov),f"baseline/{name}")
                    verified_rows+=1
                else:
                    assert all(np.array_equal(a,b) for a,b in zip((baseline,truth,fov),references[name]))
                for kind in OUTPUTS:
                    probability=_plane(saved[kind],kind)
                    assert probability.shape==truth.shape and (probability>=0).all() and (probability<=1).all()
                    row=index[arm,seed,name,kind]
                    _check_metrics(row,metrics(probability,truth,baseline,cutoff,fov),f"{arm}/{seed}/{name}/{kind}")
                    selected=decision[kind]
                    assert int(row["selected_epoch"])==int(selected["epoch"])
                    alpha=1. if kind=="raw" else float(selected["alpha"])
                    assert _number(row,"alpha")==alpha
                    if kind=="raw":assert int(selected["epoch"])>0
                    if kind=="blend" and alpha==0:
                        assert np.array_equal(probability,baseline)
                        assert int(row["velocity_calls"])==int(row["gate_calls"])==0
                    verified_rows+=1
            verified_files+=1
    assert verified_files==25*n_images and verified_rows==expected_records
    baseline_means={metric:statistics.mean(_number(row,metric) for row in baseline_rows) for metric in SUMMARY_METRICS}
    arms={};seed_rows=[dict(dataset=dataset,arm="baseline",output="raw",refiner_seed=-1,n_images=n_images,**baseline_means)]
    for arm,kind in itertools.product(ARMS,OUTPUTS):
        seed_means=[]
        for seed in SEEDS:
            values={metric:statistics.mean(_number(index[arm,seed,name,kind],metric) for name in names)
                    for metric in SUMMARY_METRICS}
            seed_means.append(dict(refiner_seed=seed,metrics=values))
            seed_rows.append(dict(dataset=dataset,arm=arm,output=kind,refiner_seed=seed,n_images=n_images,**values))
        arms.setdefault(arm,{})[kind]=dict(
            n_refiner_seeds=5,n_images=n_images,
            mean={metric:statistics.mean(s["metrics"][metric] for s in seed_means) for metric in SUMMARY_METRICS},
            sample_sd={metric:statistics.stdev(s["metrics"][metric] for s in seed_means) for metric in SUMMARY_METRICS},
            seed_means=seed_means,
            sd_definition="sample SD of five image-macro refiner-training means conditional on one shared frozen baseline",
        )
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    draws=rng.integers(0,n_images,size=(BOOTSTRAP_REPETITIONS,n_images),dtype=np.int64)
    draw_hash=hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    comparisons=[];image_rows=[];seed_difference_rows=[]
    for kind,comparator in itertools.product(OUTPUTS,COMPARATORS):
        comparison_id=f"{kind}_esfm_minus_{comparator}";metric_results={}
        for metric in INTERVAL_METRICS:
            per_image=[]
            for name in names:
                a=statistics.mean(_number(index["esfm",seed,name,kind],metric) for seed in SEEDS)
                b=_number(index["baseline",-1,name,"raw"],metric) if comparator=="baseline" else statistics.mean(
                    _number(index[comparator,seed,name,kind],metric) for seed in SEEDS)
                item=dict(image=name,subject_label=image_subject[name],esfm_mean=a,comparator_mean=b,difference=a-b)
                per_image.append(item)
                image_rows.append(dict(dataset=dataset,comparison=comparison_id,output=kind,metric=metric,
                                       lhs="esfm",rhs=comparator,image=name,subject_label=image_subject[name],
                                       fixed_refiner_seeds=5,esfm_mean=a,comparator_mean=b,difference=a-b))
            per_seed=[]
            for seed in SEEDS:
                a=arms["esfm"][kind]["seed_means"][SEEDS.index(seed)]["metrics"][metric]
                b=baseline_means[metric] if comparator=="baseline" else arms[comparator][kind]["seed_means"][SEEDS.index(seed)]["metrics"][metric]
                item=dict(refiner_seed=seed,esfm_mean=a,comparator_mean=b,difference=a-b)
                per_seed.append(item)
                seed_difference_rows.append(dict(dataset=dataset,comparison=comparison_id,output=kind,metric=metric,
                                                 lhs="esfm",rhs=comparator,refiner_seed=seed,n_images=n_images,
                                                 esfm_mean=a,comparator_mean=b,difference=a-b))
            interval=image_interval([item["difference"] for item in per_image],draws)
            macro_a=arms["esfm"][kind]["mean"][metric]
            macro_b=baseline_means[metric] if comparator=="baseline" else arms[comparator][kind]["mean"][metric]
            assert abs(interval["mean_difference"]-(macro_a-macro_b))<1e-12
            interval.update(
                per_image=per_image,per_seed=per_seed,
                paired_seed_difference_mean=statistics.mean(item["difference"] for item in per_seed),
                paired_seed_difference_sample_sd=statistics.stdev(item["difference"] for item in per_seed),
                analysis_role="primary" if kind=="raw" and comparator=="direct" and metric=="dice" else "secondary",
                positive_difference_means="higher ESFM score",bootstrap_index_sha256=draw_hash,
            )
            metric_results[metric]=interval
        comparisons.append(dict(id=comparison_id,output=kind,lhs="esfm",rhs=comparator,arm_a="esfm",arm_b=comparator,
                                primary=kind=="raw" and comparator=="direct",
                                mandatory_baseline_comparison=comparator=="baseline",
                                primary_metric="dice" if kind=="raw" and comparator=="direct" else None,
                                metrics=metric_results))
    summary=dict(
        version=tag,dataset=dataset,scope="new FOV-aware DRIVE/HRF OOF extension; earlier CHASE/paper/results preserved",
        n_test_images=n_images,bootstrap_unit="image",patient_independence_verified=False,
        refiner_training_seeds=SEEDS,shared_frozen_baseline_seed=42,n_shared_frozen_baseline_checkpoints=1,
        primary_result="raw validation-selected learned ESFM versus raw validation-selected learned direct refiner",
        mandatory_baseline_result="raw ESFM minus shared fixed baseline; secondary exploratory contrast",
        secondary_result="all other controls, structural contrasts and blend-selected policies; unadjusted intervals",
        metric_aggregation="equal-image means within each refiner seed, then mean/sample SD across five fixed-baseline refiner fits",
        count_metric_units="mean count per image inside FOV; no pooled micro Dice",
        structural_definition=DEFINITION,thin_definition=thin,
        baseline=dict(n_images=n_images,n_refiner_seeds=0,means=baseline_means,seed_sd=None,
                      explanation="one fixed baseline; no fabricated refiner-seed SD"),
        arms=arms,comparisons=comparisons,uncertainty_interpretation=INTERPRETATION,
        bootstrap=dict(repetitions=BOOTSTRAP_REPETITIONS,seed=BOOTSTRAP_SEED,image_order=names,
                       shared_index_sha256=draw_hash,resamples_training_seeds=False),
        files={name:str(path) for name,path in paths.items()},
    )
    verification=dict(
        version=tag,dataset=dataset,completed_training_evaluation_marker_verified=True,
        csv_rows=expected_records,verified_metric_rows=verified_rows,
        verified_prediction_archives=verified_files,unique_images=n_images,models=25,
        all_saved_metrics_exactly_reproduced=True,all_raw_models_learned_epoch_gt_zero=True,
        all_selection_lock_and_selected_model_hashes_verified=True,
        all_model_baseline_GT_FOV_arrays_match=True,cutoff_fixed_from_recorded_training_GT_FOV=True,
        structural_library_versions=dependency_versions(),verified_input_sha256=hashes,
        report_source_sha256=sha256_file(Path(__file__)),
        no_training_or_model_inference_performed_by_report=True,no_previous_file_modified=True,
        uncertainty_interpretation=INTERPRETATION,
    )
    _write_json(paths["summary_json"],summary)
    _write_csv(paths["seed_means_csv"],seed_rows)
    _write_csv(paths["seed_differences_csv"],seed_difference_rows)
    _write_csv(paths["image_differences_csv"],image_rows)
    _write_json(paths["difference_arrays_json"],dict(dataset=dataset,version=tag,comparisons=comparisons,
                                                   bootstrap=summary["bootstrap"],interpretation=INTERPRETATION))
    verification["report_output_sha256"]={name:sha256_file(path) for name,path in paths.items() if name!="verification_json"}
    _write_json(paths["verification_json"],verification)
    return summary


def _self_test():
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    indices=rng.integers(0,20,size=(BOOTSTRAP_REPETITIONS,20),dtype=np.int64)
    result=image_interval([.125]*20,indices)
    assert result["ci95"]==[.125,.125] and result["mean_difference"]==.125
    assert result["bootstrap_cluster_count"]==20 and result["resamples_training_seeds"] is False
    again=np.random.default_rng(BOOTSTRAP_SEED).integers(0,20,size=indices.shape,dtype=np.int64)
    assert np.array_equal(indices,again)
    assert _dataset({"dataset":"DRIVE"})=="DRIVE" and _dataset({"dataset":"HRF"})=="HRF"
    try:report(Path("_synthetic_nonexistent_experiment_2026-11-15_00-00-00"))
    except RuntimeError:pass
    else:raise AssertionError("Incomplete experiment was accepted")
    print("Synthetic image-bootstrap/guard checks passed; no experiment data read or written.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:_self_test()
    elif args.experiment is None:parser.error("--experiment is required")
    else:
        result=report(args.experiment)
        print(json.dumps(dict(dataset=result["dataset"],version=result["version"],files=result["files"]),indent=2))
