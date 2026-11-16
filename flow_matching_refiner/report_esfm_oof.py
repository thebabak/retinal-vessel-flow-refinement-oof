"""Verify and report the completed CHASE OOF ESFM experiment.

No training, inference, checkpoint selection, or test tuning occurs here.
Report creation requires the experiment's completed_training_evaluation marker.
Patient-bootstrap intervals condition on the five fitted refiners and shared
baseline. They neither resample training seeds nor claim pipeline independence.
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
from flow_matching_refiner.structure_metrics import dependency_versions, metrics


SEEDS = [42, 123, 2026, 3000, 3001]
ARMS = ["direct", "endpoint_iterative", "full_fm", "selective_no_gate", "esfm"]
OUTPUTS = ["raw", "blend"]
SUMMARY_METRICS = [
    "dice", "iou", "cldice", "thin_skeleton_recall", "precision", "recall",
    "new_error_count", "new_error_rate", "fixed_error_count", "fixed_error_rate",
    "betti0_absolute_error", "fp", "fn", "thin_skeleton_missed",
]
INTERVAL_METRICS = ["dice", "cldice", "thin_skeleton_recall"]
COMPARATORS = ["direct", "baseline", "endpoint_iterative", "full_fm", "selective_no_gate"]
INTERPRETATION = (
    "Exploratory descriptive analysis on an already inspected test cohort with only four patients. "
    "Intervals are unadjusted percentile intervals from exact enumeration of the ordinary "
    "four-patient cluster bootstrap. Both eyes and all five fixed fitted-refiner outputs stay "
    "together inside each patient cluster. No pixels, eyes, or training seeds are independently "
    "resampled. Refiner seed SD is reported separately; the five fits share one frozen baseline "
    "and common OOF inputs. No hypothesis-test p-values, equivalence claims, multiplicity-adjusted "
    "inference, or formal significance declarations are supplied."
)


def exact_patient_interval(differences):
    """Enumerate all 4**4 ordered cluster draws with equal bootstrap weights."""
    values = [float(x) for x in differences]
    if len(values) != 4 or not np.isfinite(values).all():
        raise ValueError("Exactly four finite paired patient differences are required")
    bootstrap = [statistics.mean(values[index] for index in draw)
                 for draw in itertools.product(range(4), repeat=4)]
    lo, hi = np.quantile(np.asarray(bootstrap), [.025, .975], method="linear")
    return {
        "mean_difference": statistics.mean(values),
        "ci95": [float(lo), float(hi)],
        "exact_bootstrap_ordered_resamples": 256,
        "bootstrap_cluster_count": 4,
        "bootstrap_method": "all ordered size-4 draws with replacement from four paired patients",
        "percentile_method": "numpy.quantile linear at 0.025 and 0.975",
        "interval_adjustment": "none; exploratory unadjusted percentile interval",
        "interval_inferential_role": "exploratory; no formal significance claim",
        "resamples_training_seeds": False,
    }


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _plane(array, name):
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"Invalid single-image array: {name}")
    return value


def _number(row, key):
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite CSV field {key}")
    return value


def _check_metrics(row, actual, context):
    for name, value in actual.items():
        if name not in row or _number(row, name) != value:
            raise AssertionError(f"Metric mismatch {context}: {name}, CSV={row.get(name)} new={value}")


def _write_json_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _write_csv_new(path, rows):
    if not rows:
        raise ValueError("Cannot emit an empty report table")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output):
    """Verify completed records and create four new timestamped report files.

    Returns the JSON-serializable summary written to summary_statistics_TAG.json.
    Output names retain the experiment tag. Existing reports are never replaced.
    """
    out = assert_output_path(output, must_exist=True)
    tag = out.name[-19:]
    completion_path = out/f"completed_training_evaluation_{tag}.json"
    if not completion_path.is_file():
        raise RuntimeError("Training/evaluation is incomplete: completion marker is required")
    completion = _json(completion_path)
    if completion.get("fits") != 25 or completion.get("records") != 408:
        raise AssertionError("Completion inventory differs from the planned 25 fits / 408 rows")
    for flag in ("choices_locked_before_test", "previous_files_byte_identical", "OOF_maps_audited"):
        if completion.get(flag) is not True:
            raise AssertionError(f"Experiment audit flag is not true: {flag}")
    paths = {
        "summary_json": out/f"summary_statistics_{tag}.json",
        "seed_means_csv": out/f"seed_means_{tag}.csv",
        "patient_differences_csv": out/f"patient_differences_{tag}.csv",
        "verification_json": out/f"verified_report_{tag}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("A timestamped report already exists; earlier reports must be preserved")
    plan_path = out/f"plan_{tag}.json"
    plan = _json(plan_path)
    assert plan["refiner_training_seeds"] == SEEDS and plan["arms"] == ARMS
    assert plan["primary_output"] == "raw validation-selected model alpha1"
    thin_path = out/f"thin_definition_{tag}.json"
    thin_info = _json(thin_path)
    cutoff = float(thin_info["cutoff"])
    assert thin_info["quantile"] == 1/3
    env_path = out/f"environment_{tag}.json"
    environment = _json(env_path)
    if environment["structural_libraries"] != dependency_versions():
        raise RuntimeError("Structural metric library versions differ from the executed experiment")
    locks_path = out/f"all_selections_before_test_{tag}.json"
    locks = _json(locks_path)
    assert sha256_file(locks_path) == completion["lock_sha256"]
    decisions = locks["decisions"]
    expected_jobs = set(itertools.product(ARMS, SEEDS))
    decision_map = {(d["arm"], int(d["refiner_seed"])):d for d in decisions}
    assert len(decisions) == len(decision_map) == 25
    assert set(decision_map) == expected_jobs
    csv_path = out/f"all_test_records_{tag}.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 408
    baseline_rows = [r for r in rows if r["arm"] == "baseline"]
    assert len(baseline_rows) == 8
    image_names = sorted(r["image"] for r in baseline_rows)
    assert len(set(image_names)) == 8
    subjects = sorted({int(r["subject"]) for r in baseline_rows})
    assert len(subjects) == 4
    subject_images = {subject:sorted(r["image"] for r in baseline_rows if int(r["subject"])==subject)
                      for subject in subjects}
    assert all(len(names)==2 for names in subject_images.values())
    image_subject = {r["image"]:int(r["subject"]) for r in baseline_rows}
    index = {}
    for row in rows:
        key = (row["arm"], int(row["refiner_seed"]), row["image"], row["output"])
        if key in index:
            raise AssertionError(f"Duplicate CSV record: {key}")
        index[key] = row
        assert row["image"] in image_names and int(row["subject"]) == image_subject[row["image"]]
    expected = {("baseline",-1,name,"raw") for name in image_names}
    expected.update((arm,seed,name,kind) for arm,seed,name,kind in
                    itertools.product(ARMS,SEEDS,image_names,OUTPUTS))
    assert set(index) == expected
    hashes = {str(p.relative_to(out)):sha256_file(p) for p in
              [completion_path, plan_path, thin_path, env_path, locks_path, csv_path]}
    reference_arrays = {}
    verified_files = 0
    verified_rows = 0
    for arm, seed in itertools.product(ARMS, SEEDS):
        dest = out/arm/f"seed_{seed}"
        decision = decision_map[arm, seed]
        job_completion = _json(dest/f"completed_{tag}.json")
        assert job_completion["records"] == 16 and job_completion["test_images"] == 8
        for kind, filename, digest_key in (
            ("raw", f"best_raw_model_{tag}.pt", "raw_sha256"),
            ("blend", f"selected_blend_model_{tag}.pt", "blend_sha256"),
            ("final", f"final_model_{tag}.pt", "final_sha256"),
        ):
            model_path = dest/filename
            digest = sha256_file(model_path)
            assert digest == decision[digest_key]
            hashes[str(model_path.relative_to(out))] = digest
        for name in image_names:
            prediction_path = dest/f"{name}_predictions_{tag}.npz"
            hashes[str(prediction_path.relative_to(out))] = sha256_file(prediction_path)
            with np.load(prediction_path, allow_pickle=False) as saved:
                base = _plane(saved["baseline"], "baseline")
                truth = _plane(saved["ground_truth"], "ground_truth")
                assert base.shape == truth.shape and np.isin(truth, [0,1]).all()
                assert bool(saved["fov"].all()), "CHASE whole-image FOV must be all true"
                if name not in reference_arrays:
                    reference_arrays[name] = (base.copy(), truth.copy())
                    baseline = index["baseline",-1,name,"raw"]
                    _check_metrics(baseline, metrics(base,truth,base,cutoff), f"baseline/{name}")
                    verified_rows += 1
                else:
                    ref_base, ref_truth = reference_arrays[name]
                    assert np.array_equal(base,ref_base) and np.array_equal(truth,ref_truth)
                for kind in OUTPUTS:
                    probability = _plane(saved[kind], kind)
                    assert probability.shape == truth.shape and (probability>=0).all() and (probability<=1).all()
                    row = index[arm,seed,name,kind]
                    _check_metrics(row, metrics(probability,truth,base,cutoff), f"{arm}/{seed}/{name}/{kind}")
                    expected_epoch = int(decision[kind]["epoch"])
                    expected_alpha = 1. if kind=="raw" else float(decision["blend"]["alpha"])
                    assert int(row["selected_epoch"]) == expected_epoch
                    assert _number(row,"alpha") == expected_alpha
                    if kind == "raw":
                        assert expected_epoch > 0
                    if kind == "blend" and expected_alpha == 0:
                        assert np.array_equal(probability, base), "Identity-selected blend must retain exact baseline"
                        assert int(row["velocity_calls"]) == int(row["gate_calls"]) == 0
                    verified_rows += 1
            verified_files += 1
    assert verified_files == 200 and verified_rows == 408
    baseline_means = {metric:statistics.mean(_number(r,metric) for r in baseline_rows)
                      for metric in SUMMARY_METRICS}
    arms = {}
    seed_csv = [dict(arm="baseline",output="raw",refiner_seed=-1,n_images=8,**baseline_means)]
    for arm in ARMS:
        arms[arm] = {}
        for kind in OUTPUTS:
            seed_means = []
            for seed in SEEDS:
                matches = [index[arm,seed,name,kind] for name in image_names]
                values = {metric:statistics.mean(_number(r,metric) for r in matches)
                          for metric in SUMMARY_METRICS}
                seed_means.append({"refiner_seed":seed,"metrics":values})
                seed_csv.append(dict(arm=arm,output=kind,refiner_seed=seed,n_images=8,**values))
            arms[arm][kind] = {
                "n_refiner_seeds":5, "n_images":8,
                "mean":{metric:statistics.mean(s["metrics"][metric] for s in seed_means)
                        for metric in SUMMARY_METRICS},
                "sample_sd":{metric:statistics.stdev(s["metrics"][metric] for s in seed_means)
                             for metric in SUMMARY_METRICS},
                "sd_definition":"sample SD of five image-macro mean scores from refiner fits sharing a frozen baseline",
                "seed_means":seed_means,
            }
    comparisons = []
    patient_csv = []
    for kind, comparator in itertools.product(OUTPUTS, COMPARATORS):
        comparison_id = f"{kind}_esfm_minus_{comparator}"
        metric_results = {}
        for metric in INTERVAL_METRICS:
            patient_results = []
            for subject in subjects:
                names = subject_images[subject]
                a = statistics.mean(_number(index["esfm",seed,name,kind],metric)
                                    for seed,name in itertools.product(SEEDS,names))
                if comparator == "baseline":
                    b = statistics.mean(_number(index["baseline",-1,name,"raw"],metric) for name in names)
                else:
                    b = statistics.mean(_number(index[comparator,seed,name,kind],metric)
                                        for seed,name in itertools.product(SEEDS,names))
                difference = a-b
                patient_results.append(dict(subject=subject, images=names, esfm_mean=a,
                                            comparator_mean=b, difference=difference))
                patient_csv.append(dict(comparison=comparison_id,output=kind,metric=metric,
                                        arm_a="esfm",arm_b=comparator,subject=subject,n_images=2,
                                        n_fixed_refiner_seeds=5,arm_a_mean=a,arm_b_mean=b,difference=difference))
            interval = exact_patient_interval([p["difference"] for p in patient_results])
            a_macro = arms["esfm"][kind]["mean"][metric]
            b_macro = baseline_means[metric] if comparator=="baseline" else arms[comparator][kind]["mean"][metric]
            assert abs(interval["mean_difference"]-(a_macro-b_macro)) < 1e-12
            interval.update(per_patient=patient_results,
                            analysis_role="primary" if kind=="raw" and comparator=="direct" and metric=="dice" else "secondary",
                            positive_difference_means="higher ESFM score")
            metric_results[metric] = interval
        comparisons.append(dict(
            id=comparison_id, output=kind, lhs="esfm", rhs=comparator,
            arm_a="esfm", arm_b=comparator,
            primary=kind=="raw" and comparator=="direct",
            primary_metric="dice" if kind=="raw" and comparator=="direct" else None,
            metrics=metric_results,
        ))
    summary = {
        "version":tag,
        "scope":"new CHASE-DB1 OOF experiment; old manuscript and results are unchanged",
        "dataset":"CHASE-DB1", "n_test_images":8, "n_test_patients":4,
        "refiner_training_seeds":SEEDS, "shared_frozen_baseline_seed":42,
        "n_shared_frozen_baseline_checkpoints":1,
        "primary_result":"raw learned validation-selected ESFM versus raw learned validation-selected direct refiner",
        "secondary_result":"blend-selected policies and all structural contrasts; no adjustment for multiplicity",
        "metric_aggregation":"equal image weighting inside each seed, then mean and sample SD over five refiner-seed means",
        "count_metric_units":"mean count per image; counts are not pooled into a new micro Dice score",
        "thin_definition":thin_info,
        "baseline":{"n_images":8,"n_refiner_seeds":0,"means":baseline_means,"seed_sd":None,
                    "explanation":"one fixed baseline; no fabricated refiner-seed variability"},
        "arms":arms,
        "comparisons":comparisons,
        "uncertainty_interpretation":INTERPRETATION,
        "files":{name:str(path) for name,path in paths.items()},
    }
    verification = {
        "version":tag,"completed_training_evaluation_marker_verified":True,
        "csv_rows":408,"verified_metric_rows":408,"verified_prediction_archives":200,
        "unique_images":8,"patients":subjects,"images_per_patient":subject_images,
        "models":25,"all_raw_models_learned_epoch_gt_zero":True,
        "all_selection_lock_and_selected_model_hashes_verified":True,
        "all_saved_metrics_exactly_reproduced":True,
        "all_model_baseline_and_GT_arrays_match":True,
        "cutoff_fixed_from_recorded_training_GT":True,
        "structural_library_versions":dependency_versions(),
        "verified_input_sha256":hashes,
        "report_source_sha256":sha256_file(Path(__file__)),
        "no_training_or_model_inference_performed_by_report":True,
        "no_previous_file_modified":True,
        "uncertainty_interpretation":INTERPRETATION,
    }
    _write_json_new(paths["summary_json"],summary)
    _write_csv_new(paths["seed_means_csv"],seed_csv)
    _write_csv_new(paths["patient_differences_csv"],patient_csv)
    verification["report_output_sha256"] = {name:sha256_file(path) for name,path in paths.items() if name!="verification_json"}
    _write_json_new(paths["verification_json"],verification)
    return summary


def _self_test():
    constant = exact_patient_interval([.1,.1,.1,.1])
    assert abs(constant["mean_difference"]-.1) < 1e-15
    assert all(abs(x-.1)<1e-15 for x in constant["ci95"])
    symmetric = exact_patient_interval([-3.,-1.,1.,3.])
    assert symmetric["mean_difference"] == 0.
    assert abs(sum(symmetric["ci95"])) < 1e-12
    assert symmetric["exact_bootstrap_ordered_resamples"] == 256
    try:
        exact_patient_interval([1.,2.,3.])
    except ValueError:
        pass
    else:
        raise AssertionError("Bootstrap did not reject the wrong cluster count")
    print("Synthetic exact-bootstrap checks passed; no experiment files opened or written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",nargs="?",type=Path)
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    elif args.output is None:
        parser.error("output directory required unless --self-test is used")
    else:
        result = report(args.output)
        print(json.dumps({"version":result["version"],"files":result["files"],"baseline":result["baseline"]},indent=2))
