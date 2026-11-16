"""Verify completed v3 refinement variants and compare with preserved ESFM runs.

This reporter never trains, runs model inference, chooses a checkpoint, or edits
an earlier experiment. It requires the complete final-training/test inventory,
the pilot-configuration lock, and the global selection lock made before any new
test scoring. Historical arms are contextual inputs, not newly fitted models.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import resolve_recorded_path, recorded_is_absolute, assert_output_path

import argparse
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics

import numpy as np

from flow_matching_refiner.data import sha256_file
from flow_matching_refiner.report_esfm_all import image_interval
from flow_matching_refiner.report_esfm_oof import exact_patient_interval
from flow_matching_refiner.structure_metrics import metrics as whole_image_metrics
from flow_matching_refiner.structure_metrics_all import dependency_versions, metrics as fov_metrics

SEEDS = [42, 123, 2026, 3000, 3001]
ARMS = ["direct_thin_anchor", "esfm_endpoint_anchor"]
EXISTING_ARMS = ["direct", "endpoint_iterative", "full_fm", "selective_no_gate", "esfm"]
OUTPUTS = ["raw", "blend"]
SUMMARY_METRICS = [
    "dice", "iou", "cldice", "thin_skeleton_recall", "precision", "recall",
    "new_error_count", "new_error_rate", "fixed_error_count", "fixed_error_rate",
    "betti0_absolute_error", "fp", "fn", "thin_skeleton_missed",
]
INTERVAL_METRICS = ["dice", "cldice", "thin_skeleton_recall"]
N_IMAGES = {"CHASE-DB1": 8, "DRIVE": 20, "HRF": 12}
BOOTSTRAP_REPETITIONS = 50000
BOOTSTRAP_SEED = 20261115
CONTRASTS = [
    ("direct_thin_anchor", "direct", "existing", True),
    ("esfm_endpoint_anchor", "esfm", "existing", True),
    ("direct_thin_anchor", "baseline", "baseline", False),
    ("esfm_endpoint_anchor", "baseline", "baseline", False),
    ("esfm_endpoint_anchor", "direct_thin_anchor", "new", False),
]
INTERPRETATION = (
    "Exploratory unadjusted paired percentile intervals on previously examined cohorts, "
    "conditional on five fixed fitted refiners in each compared arm, common input maps and "
    "one frozen baseline checkpoint per dataset. CHASE keeps both eyes and all fitted seed "
    "outputs within each of four patient clusters and enumerates all 256 ordered cluster "
    "resamples; exhaustive computation is not exact confidence coverage. DRIVE and HRF use "
    "50000 paired image-bootstrap draws with seed 20261115; independent patient IDs are "
    "unverified. No pixels or training seeds are independently resampled. Training-seed "
    "sample SD is separate. The two raw Dice changes relative to each variant's existing "
    "counterpart are co-primary descriptive contrasts. No pooling, p-values, multiplicity "
    "adjustment, equivalence conclusion, or formal significance claim is supplied."
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _json(path):
    return json.loads(resolve_recorded_path(path).read_text(encoding="utf-8-sig"))


def _dataset(value):
    key = str(value).upper().replace("-", "").replace("_", "")
    if key.startswith("CHASE"):
        return "CHASE-DB1"
    if key.startswith("DRIVE"):
        return "DRIVE"
    if key.startswith("HRF"):
        return "HRF"
    raise ValueError("Unsupported dataset: " + str(value))


def _tag(path):
    value = Path(path).name[-19:]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", value):
        raise ValueError("Expected timestamped experiment folder")
    return value


def _number(row, metric):
    value = float(row[metric])
    if not np.isfinite(value):
        raise ValueError("Nonfinite metric: " + metric)
    return value


def _plane(array, name):
    value = np.asarray(array)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Expected one finite image array: " + name)
    return value


def _same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def _check_metrics(row, actual, context):
    for metric, value in actual.items():
        _require(metric in row and _number(row, metric) == value,
                 f"Saved metric mismatch: {context}/{metric}")


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _write_csv(path, rows):
    _require(bool(rows), "Cannot write an empty report CSV")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rows(path, dataset, arms, n_images):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected_count = n_images * (1 + len(arms) * len(SEEDS) * len(OUTPUTS))
    _require(len(rows) == expected_count, "Unexpected per-image CSV inventory")
    baseline_rows = [row for row in rows if row["arm"] == "baseline"]
    names = sorted(row["image"] for row in baseline_rows)
    _require(len(names) == len(set(names)) == n_images, "Invalid baseline image inventory")
    subjects = {row["image"]: int(row["subject"]) for row in baseline_rows}
    index = {}
    for row in rows:
        key = (row["arm"], int(row["refiner_seed"]), row["image"], row["output"])
        _require(key not in index, "Duplicate per-image row")
        _require(row["image"] in names and int(row["subject"]) == subjects[row["image"]], "Image/subject mismatch")
        if "dataset" in row:
            _require(_dataset(row["dataset"]) == dataset, "CSV dataset mismatch")
        index[key] = row
    expected = {("baseline", -1, name, "raw") for name in names}
    expected.update(itertools.product(arms, SEEDS, names, OUTPUTS))
    _require(set(index) == expected, "Incomplete or unexpected arm/seed/image/output grid")
    return rows, baseline_rows, names, subjects, index


def _relative(root, name):
    # Relative verification keys stay within the earlier experiment; absolute
    # historical source keys resolve only through configured original bytes.
    if recorded_is_absolute(name):
        return resolve_recorded_path(name)
    path = (root / name).resolve()
    _require(path.is_relative_to(root), "Recorded source path escapes its experiment")
    return path


def _lock_reference(completion, plan, stem):
    aliases = [stem]
    if stem == "global_lock":
        aliases.append("global_selection_lock")
    for alias in aliases:
        path = completion.get(alias + "_path", plan.get(alias + "_path"))
        digest = completion.get(alias + "_sha256", plan.get(alias + "_sha256"))
        if path is not None and digest is not None:
            path = resolve_recorded_path(path)
            _require(path.is_file() and sha256_file(path) == digest, alias + " digest mismatch")
            return path, _json(path)
    raise AssertionError("Missing " + stem + " path/hash reference")


def _global_decisions(lock):
    if isinstance(lock.get("decisions"), list):
        return [dict(item, dataset=_dataset(item["dataset"])) for item in lock["decisions"]]
    mapping = lock.get("decisions_by_dataset", lock.get("datasets"))
    _require(isinstance(mapping, dict), "Global selection lock requires all dataset decisions")
    rows = []
    for dataset, value in mapping.items():
        if isinstance(value, dict):
            value = value["decisions"]
        _require(isinstance(value, list), "Invalid global decisions list")
        rows.extend(dict(item, dataset=_dataset(dataset)) for item in value)
    return rows


def _verify_selection_locks(out, tag, dataset, completion, plan, hashes):
    path = out / f"all_selections_before_test_{tag}.json"
    local = _json(path)
    _require(sha256_file(path) == completion["lock_sha256"], "Local selection lock changed")
    decisions = {(d["arm"], int(d["refiner_seed"])): d for d in local["decisions"]}
    _require(len(decisions) == len(local["decisions"]) == 10, "Expected ten final selections")
    _require(set(decisions) == set(itertools.product(ARMS, SEEDS)), "Final selection grid differs")
    global_path, global_lock = _lock_reference(completion, plan, "global_lock")
    config_path, config_lock = _lock_reference(completion, plan, "config_lock")
    global_rows = _global_decisions(global_lock)
    global_index = {(d["dataset"], d["arm"], int(d["refiner_seed"])): d for d in global_rows}
    expected = set(itertools.product(N_IMAGES, ARMS, SEEDS))
    _require(len(global_rows) == len(global_index) == 30 and set(global_index) == expected,
             "Global final selection lock must contain all thirty final fits")
    for key, decision in decisions.items():
        global_decision = dict(global_index[(dataset,) + key])
        global_decision.pop("dataset", None)
        local_decision = dict(decision)
        local_decision.pop("dataset", None)
        _require(global_decision == local_decision, "Global/local final selection differs")
    for source in (path, global_path, config_path):
        hashes[str(source)] = sha256_file(source)
    _require(bool(config_lock), "Empty pilot configuration lock")
    return decisions, dict(local=str(path), global_lock=str(global_path), config_lock=str(config_path),
                           global_final_fits=30, globally_locked_before_test_declared=True,
                           configuration_locked_before_final_training_declared=True)


def _score(dataset, probability, truth, baseline, cutoff, fov):
    if dataset == "CHASE-DB1":
        _require(fov.all(), "Preserved CHASE whole-image evaluation requires an all-valid FOV")
        return whole_image_metrics(probability, truth, baseline, cutoff)
    return fov_metrics(probability, truth, baseline, cutoff, fov)


def _read_existing(old, dataset, n_images, new_thin, hashes):
    tag = _tag(old)
    files = {key: old / f"{prefix}_{tag}.{ext}" for key, prefix, ext in (
        ("summary", "summary_statistics", "json"), ("verification", "verified_report", "json"),
        ("csv", "all_test_records", "csv"), ("completion", "completed_training_evaluation", "json"),
        ("thin", "thin_definition", "json"))}
    saved = {key: _json(path) for key, path in files.items() if key != "csv"}
    summary, verification = saved["summary"], saved["verification"]
    _require(_dataset(summary["dataset"]) == dataset, "Existing summary dataset mismatch")
    _require(verification["all_saved_metrics_exactly_reproduced"] is True, "Existing results lack metric verification")
    _require(saved["completion"]["fits"] == 25, "Expected preserved five-arm experiment")
    _require(float(new_thin["cutoff"]) == float(saved["thin"]["cutoff"]), "Thin cutoff differs from retained comparison")
    _require(new_thin["training_image_ids"] == saved["thin"]["training_image_ids"], "Thin training IDs differ")
    _require(set(summary["arms"]) == set(EXISTING_ARMS), "Missing contextual existing arm")
    _require(sha256_file(files["summary"]) == verification["report_output_sha256"]["summary_json"],
             "Existing summary differs from its verified digest")
    for relative, digest in verification["verified_input_sha256"].items():
        path = _relative(old, relative)
        _require(path.is_file() and sha256_file(path) == digest, "Existing verified source changed: " + str(path))
        hashes[str(path)] = digest
    for path in files.values():
        hashes[str(path)] = sha256_file(path)
    rows, baselines, names, subjects, index = _rows(files["csv"], dataset, EXISTING_ARMS, n_images)
    references = {}
    for arm, seed, name in itertools.product(EXISTING_ARMS, SEEDS, names):
        path = old / arm / f"seed_{seed}" / f"{name}_predictions_{tag}.npz"
        _require(str(path) in hashes, "Existing archive not bound to retained verification")
        with np.load(path, allow_pickle=False) as arrays:
            inputs = tuple(np.asarray(arrays[key]).copy() for key in ("baseline", "ground_truth", "fov"))
            if name not in references:
                references[name] = inputs
            else:
                _require(all(_same_bytes(a, b) for a, b in zip(inputs, references[name])),
                         "Existing arms do not share byte-identical baseline/GT/FOV")
    # Old means/SDs are reused as saved; only their row binding is checked.
    for arm, output, seed, metric in itertools.product(EXISTING_ARMS, OUTPUTS, SEEDS, SUMMARY_METRICS):
        mean = statistics.mean(_number(index[arm, seed, name, output], metric) for name in names)
        old_seed = next(item for item in summary["arms"][arm][output]["seed_means"] if int(item["refiner_seed"]) == seed)
        _require(mean == old_seed["metrics"][metric], "Existing CSV/summary seed binding changed")
    contextual = dict(old_experiment=str(old), version=tag, summary_sha256=hashes[str(files["summary"])],
                      per_image_csv_sha256=hashes[str(files["csv"])],
                      arms=copy.deepcopy(summary["arms"]), baseline=copy.deepcopy(summary["baseline"]),
                      scope="all five earlier arms reused exactly; no earlier model is retrained or replaced")
    return contextual, references, names, subjects, index


def report(experiment):
    out = assert_output_path(experiment, must_exist=True)
    tag = _tag(out)
    completion_path = out / f"completed_training_evaluation_{tag}.json"
    if not completion_path.is_file():
        raise RuntimeError("Completed final training/test evaluation marker is required")
    completion = _json(completion_path)
    plan_path = out / f"plan_{tag}.json"
    plan = _json(plan_path)
    dataset = _dataset(plan["dataset"])
    n_images = N_IMAGES[dataset]
    expected_records = 21 * n_images
    _require(completion.get("fits") == 10 and completion.get("records") == expected_records,
             "Expected ten final fits and 21N final records")
    for flag in ("configs_locked_before_final_training", "choices_locked_before_test", "previous_files_byte_identical", "OOF_maps_audited"):
        _require(completion.get(flag) is True, "Missing execution audit flag: " + flag)
    if "old_sources_byte_identical" in completion:
        _require(completion["old_sources_byte_identical"] is True, "Prior sources changed")
    _require(plan["arms"] == ARMS and plan["refiner_training_seeds"] == SEEDS, "Unexpected final arm/seed grid")
    old = resolve_recorded_path(plan["old_experiment"])
    _require(out != old and not out.is_relative_to(old), "New reports must not be written inside a preserved experiment")
    paths = {name: out / f"{prefix}_{tag}.{extension}" for name, prefix, extension in (
        ("summary_json", "summary_statistics", "json"), ("seed_means_csv", "seed_means", "csv"),
        ("seed_differences_csv", "seed_differences", "csv"), ("image_differences_csv", "image_differences", "csv"),
        ("patient_differences_csv", "patient_differences", "csv"),
        ("difference_arrays_json", "difference_arrays", "json"), ("verification_json", "verified_report", "json"))}
    _require(not any(path.exists() for path in paths.values()), "Existing timestamped reports must be preserved")
    thin_path = out / f"thin_definition_{tag}.json"
    thin = _json(thin_path)
    cutoff = float(thin["cutoff"])
    _require(np.isfinite(cutoff) and cutoff >= 0 and thin["quantile"] == 1/3, "Invalid frozen thin definition")
    environment_path = out / f"environment_{tag}.json"
    environment = _json(environment_path)
    _require(environment["structural_libraries"] == dependency_versions(), "Structural-library versions changed")
    hashes = {str(path): sha256_file(path) for path in (completion_path, plan_path, thin_path, environment_path)}
    decisions, selection_audit = _verify_selection_locks(out, tag, dataset, completion, plan, hashes)
    contextual, references, old_names, old_subjects, old_index = _read_existing(old, dataset, n_images, thin, hashes)
    csv_path = out / f"all_test_records_{tag}.csv"
    hashes[str(csv_path)] = sha256_file(csv_path)
    rows, baseline_rows, names, subjects, index = _rows(csv_path, dataset, ARMS, n_images)
    _require(names == old_names and subjects == old_subjects, "New/retained image identity differs")
    if dataset == "CHASE-DB1":
        patients = sorted(set(subjects.values()))
        _require(len(patients) == 4 and all(sum(v == p for v in subjects.values()) == 2 for p in patients),
                 "CHASE requires four two-eye patient clusters")
    else:
        patients = []
        _require(thin.get("fov_supplied") is True, "FOV-aware thin definition required")
    verified_rows = 0
    verified_archives = 0
    for arm, seed in itertools.product(ARMS, SEEDS):
        destination = out / arm / f"seed_{seed}"
        decision = decisions[arm, seed]
        job_complete_path = destination / f"completed_{tag}.json"
        job = _json(job_complete_path)
        _require(job["records"] == 2*n_images and job["test_images"] == n_images, "Incomplete final prediction job")
        hashes[str(job_complete_path)] = sha256_file(job_complete_path)
        for filename, key in ((f"best_raw_model_{tag}.pt", "raw_sha256"),
                              (f"selected_blend_model_{tag}.pt", "blend_sha256"),
                              (f"final_model_{tag}.pt", "final_sha256")):
            path = destination / filename
            digest = sha256_file(path)
            _require(digest == decision[key], "Locked selected/final model changed")
            hashes[str(path)] = digest
        for name in names:
            path = destination / f"{name}_predictions_{tag}.npz"
            hashes[str(path)] = sha256_file(path)
            with np.load(path, allow_pickle=False) as arrays:
                inputs = tuple(np.asarray(arrays[key]) for key in ("baseline", "ground_truth", "fov"))
                _require(all(_same_bytes(a, b) for a, b in zip(inputs, references[name])),
                         f"New/retained baseline, GT or FOV bytes differ: {arm}/{seed}/{name}")
                baseline, truth, fov = (_plane(array, label) for array, label in zip(inputs, ("baseline", "GT", "FOV")))
                fov = fov >= .5
                _require(baseline.shape == truth.shape == fov.shape and np.isin(truth, [0, 1]).all() and fov.any(), "Invalid input masks")
                _require(((baseline >= 0) & (baseline <= 1)).all(), "Invalid baseline probability")
                if arm == ARMS[0] and seed == SEEDS[0]:
                    _check_metrics(index["baseline", -1, name, "raw"], _score(dataset, baseline, truth, baseline, cutoff, fov), "baseline/" + name)
                    for metric in SUMMARY_METRICS:
                        _require(_number(index["baseline", -1, name, "raw"], metric) ==
                                 _number(old_index["baseline", -1, name, "raw"], metric), "Retained baseline metric changed")
                    verified_rows += 1
                for output in OUTPUTS:
                    probability = _plane(arrays[output], output)
                    _require(probability.shape == truth.shape and ((probability >= 0) & (probability <= 1)).all(), "Invalid prediction")
                    row = index[arm, seed, name, output]
                    _check_metrics(row, _score(dataset, probability, truth, baseline, cutoff, fov), f"{arm}/{seed}/{name}/{output}")
                    selected = decision[output]
                    _require(int(row["selected_epoch"]) == int(selected["epoch"]), "Selected epoch differs")
                    alpha = 1. if output == "raw" else float(selected["alpha"])
                    _require(_number(row, "alpha") == alpha, "Selected blend differs")
                    if output == "raw":
                        _require(int(selected["epoch"]) > 0, "Primary raw model cannot be epoch-zero identity")
                    elif alpha == 0:
                        _require(_same_bytes(np.asarray(arrays[output]), inputs[0]), "Identity blend differs from baseline")
                        _require(int(row["velocity_calls"]) == int(row["gate_calls"]) == 0, "Identity uses learned inference")
                    verified_rows += 1
            verified_archives += 1
    _require(verified_rows == expected_records and verified_archives == 10*n_images, "Final verification inventory mismatch")
    baseline_means = {metric: statistics.mean(_number(row, metric) for row in baseline_rows) for metric in SUMMARY_METRICS}
    _require(baseline_means == contextual["baseline"]["means"], "Summary baseline differs from retained result")
    arms = {}
    seed_rows = [dict(dataset=dataset, arm="baseline", source="baseline", output="raw", refiner_seed=-1, n_images=n_images, **baseline_means)]
    for arm, output in itertools.product(ARMS, OUTPUTS):
        seed_means = []
        for seed in SEEDS:
            values = {metric: statistics.mean(_number(index[arm, seed, name, output], metric) for name in names) for metric in SUMMARY_METRICS}
            seed_means.append(dict(refiner_seed=seed, metrics=values))
            seed_rows.append(dict(dataset=dataset, arm=arm, source="new", output=output, refiner_seed=seed, n_images=n_images, **values))
        arms.setdefault(arm, {})[output] = dict(n_refiner_seeds=5, n_images=n_images,
            mean={metric: statistics.mean(item["metrics"][metric] for item in seed_means) for metric in SUMMARY_METRICS},
            sample_sd={metric: statistics.stdev(item["metrics"][metric] for item in seed_means) for metric in SUMMARY_METRICS},
            seed_means=seed_means, sd_definition="sample SD of five final training means conditional on one frozen baseline")
    draws = None
    draw_hash = None
    if dataset != "CHASE-DB1":
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_images, size=(BOOTSTRAP_REPETITIONS, n_images), dtype=np.int64)
        draw_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    comparisons = []
    image_rows = []
    patient_rows = []
    seed_difference_rows = []
    def get_row(source, arm, seed, name, output):
        return (old_index if source == "existing" else index)[arm, seed, name, output]
    for output, (lhs, rhs, rhs_source, primary_candidate) in itertools.product(OUTPUTS, CONTRASTS):
        suffix = ("existing_" if rhs_source == "existing" else "") + rhs
        comparison_id = f"{output}_{lhs}_minus_{suffix}"
        metric_results = {}
        for metric in INTERVAL_METRICS:
            per_image = []
            for name in names:
                a = statistics.mean(_number(index[lhs, seed, name, output], metric) for seed in SEEDS)
                b = _number(index["baseline", -1, name, "raw"], metric) if rhs_source == "baseline" else statistics.mean(
                    _number(get_row(rhs_source, rhs, seed, name, output), metric) for seed in SEEDS)
                item = dict(image=name, subject_label=subjects[name], lhs_mean=a, rhs_mean=b, difference=a-b)
                per_image.append(item)
                image_rows.append(dict(dataset=dataset, comparison=comparison_id, output=output, metric=metric, lhs=lhs, rhs=rhs,
                    lhs_source="new", rhs_source=rhs_source, **item))
            per_seed = []
            for seed in SEEDS:
                a = arms[lhs][output]["seed_means"][SEEDS.index(seed)]["metrics"][metric]
                rhs_stats = contextual["arms"] if rhs_source == "existing" else arms
                b = baseline_means[metric] if rhs_source == "baseline" else rhs_stats[rhs][output]["seed_means"][SEEDS.index(seed)]["metrics"][metric]
                item = dict(refiner_seed=seed, lhs_mean=a, rhs_mean=b, difference=a-b)
                per_seed.append(item)
                seed_difference_rows.append(dict(dataset=dataset, comparison=comparison_id, output=output, metric=metric,
                    lhs=lhs, rhs=rhs, lhs_source="new", rhs_source=rhs_source, n_images=n_images, **item))
            if dataset == "CHASE-DB1":
                per_patient = []
                for subject in patients:
                    items = [item for item in per_image if item["subject_label"] == subject]
                    item = dict(subject=subject, n_images=2, lhs_mean=statistics.mean(q["lhs_mean"] for q in items),
                                rhs_mean=statistics.mean(q["rhs_mean"] for q in items),
                                difference=statistics.mean(q["difference"] for q in items))
                    per_patient.append(item)
                    patient_rows.append(dict(dataset=dataset, comparison=comparison_id, output=output, metric=metric,
                        lhs=lhs, rhs=rhs, lhs_source="new", rhs_source=rhs_source, **item))
                interval = exact_patient_interval([item["difference"] for item in per_patient])
                interval.update(bootstrap_unit="patient", per_patient=per_patient, patient_independence_verified=True)
            else:
                interval = image_interval([item["difference"] for item in per_image], draws)
                interval["bootstrap_index_sha256"] = draw_hash
            macro_a = arms[lhs][output]["mean"][metric]
            rhs_stats = contextual["arms"] if rhs_source == "existing" else arms
            macro_b = baseline_means[metric] if rhs_source == "baseline" else rhs_stats[rhs][output]["mean"][metric]
            _require(abs(interval["mean_difference"] - (macro_a-macro_b)) < 1e-12, "Paired/macro difference disagrees")
            interval.update(per_image=per_image, per_seed=per_seed,
                paired_seed_difference_mean=statistics.mean(q["difference"] for q in per_seed),
                paired_seed_difference_sample_sd=statistics.stdev(q["difference"] for q in per_seed),
                analysis_role="co-primary descriptive" if primary_candidate and output == "raw" and metric == "dice" else "secondary",
                positive_difference_means="higher new-variant score")
            metric_results[metric] = interval
        comparisons.append(dict(id=comparison_id, output=output, lhs=lhs, rhs=rhs, lhs_source="new", rhs_source=rhs_source,
            primary=primary_candidate and output == "raw", primary_metric="dice" if primary_candidate and output == "raw" else None,
            mandatory_baseline_comparison=rhs_source == "baseline", metrics=metric_results))
    summary = dict(version=tag, dataset=dataset, scope="new separately locked final v3 variants; prior outputs preserved",
        n_test_images=n_images, n_test_patients=4 if dataset == "CHASE-DB1" else None,
        bootstrap_unit="patient" if dataset == "CHASE-DB1" else "image", patient_independence_verified=dataset == "CHASE-DB1",
        refiner_training_seeds=SEEDS, shared_frozen_baseline_seed=42, n_shared_frozen_baseline_checkpoints=1,
        new_final_fits=10, contextual_existing_fits=25,
        primary_result="two raw Dice contrasts: each new objective variant minus its preserved own-method counterpart",
        secondary_result="new-minus-baseline mandatory contrasts, new-versus-new contrast, structural metrics and all blends; exploratory unadjusted",
        metric_aggregation="equal-image mean within each seed; mean and sample SD over five final refiner fits",
        thin_definition=thin, baseline=copy.deepcopy(contextual["baseline"]), arms=arms,
        contextual_reference=contextual, comparisons=comparisons, selection_audit=selection_audit,
        uncertainty_interpretation=INTERPRETATION,
        bootstrap=dict(unit="patient" if dataset == "CHASE-DB1" else "image", repetitions=256 if dataset == "CHASE-DB1" else BOOTSTRAP_REPETITIONS,
                       seed=None if dataset == "CHASE-DB1" else BOOTSTRAP_SEED, image_order=names, shared_index_sha256=draw_hash,
                       resamples_training_seeds=False), files={key: str(path) for key, path in paths.items()})
    verification = dict(version=tag, dataset=dataset, completed_training_evaluation_marker_verified=True,
        csv_rows=expected_records, verified_metric_rows=verified_rows, verified_prediction_archives=verified_archives,
        unique_images=n_images, models=10, contextual_existing_models=25,
        all_saved_metrics_exactly_reproduced=True, all_raw_models_learned_epoch_gt_zero=True,
        all_selection_lock_and_selected_model_hashes_verified=True,
        global_thirty_final_fit_selection_lock_verified=True, pilot_configuration_lock_digest_verified=True,
        all_new_and_existing_baseline_GT_FOV_arrays_byte_identical=True,
        all_five_existing_arm_summaries_copied_exactly=True,
        earlier_inputs_match_their_preserved_verification_hashes=True,
        structural_library_versions=dependency_versions(), verified_input_sha256=hashes,
        report_source_sha256=sha256_file(Path(__file__)),
        no_training_or_model_inference_performed_by_report=True, no_previous_file_modified=True,
        uncertainty_interpretation=INTERPRETATION)
    _write_json(paths["summary_json"], summary)
    _write_csv(paths["seed_means_csv"], seed_rows)
    _write_csv(paths["seed_differences_csv"], seed_difference_rows)
    _write_csv(paths["image_differences_csv"], image_rows)
    if dataset == "CHASE-DB1":
        _write_csv(paths["patient_differences_csv"], patient_rows)
    _write_json(paths["difference_arrays_json"], dict(dataset=dataset, version=tag, comparisons=comparisons,
                bootstrap=summary["bootstrap"], interpretation=INTERPRETATION))
    verification["report_output_sha256"] = {key: sha256_file(path) for key, path in paths.items()
                                           if key != "verification_json" and path.is_file()}
    _write_json(paths["verification_json"], verification)
    return summary


def _self_test():
    _require(_dataset("CHASE-DB1") == "CHASE-DB1" and _dataset("DRIVE") == "DRIVE", "Dataset aliases")
    a = np.array([1, 2], dtype=np.float32)
    _require(_same_bytes(a, a.copy()) and not _same_bytes(a, a.astype(np.float64)), "Array byte comparator")
    for value in ("bad", "non_timestamp"):
        try:
            _tag(Path(value))
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted invalid experiment tag")
    interval = exact_patient_interval([.125]*4)
    _require(interval["ci95"] == [.125, .125] and interval["exact_bootstrap_ordered_resamples"] == 256, "Patient interval")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 20, size=(BOOTSTRAP_REPETITIONS, 20), dtype=np.int64)
    _require(image_interval([.125]*20, draws)["ci95"] == [.125, .125], "Image interval")
    try:
        report(Path("_synthetic_incomplete_v3_2026-11-15_00-00-00"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete experiment accepted")
    print("Synthetic reporter guards passed; no experiment loaded or report generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    elif args.experiment is None:
        parser.error("--experiment is required")
    else:
        result = report(args.experiment)
        print(json.dumps(dict(dataset=result["dataset"], version=result["version"], files=result["files"]), indent=2))
