"""Verify a completed native-patch Direct study without model inference.

All comparative metrics use the preserved 512 GT/FOV grid. Native outputs are
retained and their residual/probability arithmetic is checked, but no native-GT
segmentation metric is computed. Existing original/v3 arms remain unchanged.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import resolve_recorded_path, recorded_is_absolute, assert_output_path

import argparse
from collections import Counter
import copy
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np

from flow_matching_refiner.data import sha256_file
from flow_matching_refiner.report_esfm_all import image_interval
from flow_matching_refiner.report_esfm_oof import exact_patient_interval
from flow_matching_refiner.report_refinement_v3 import (
    _check_metrics, _dataset, _global_decisions, _json, _lock_reference, _number,
    _plane, _read_existing, _require, _rows, _same_bytes, _tag, _write_csv, _write_json,
)
from flow_matching_refiner.structure_metrics_all import dependency_versions, metrics

ARM = "native_patch_direct"
ARMS = [ARM]
SEEDS = [42, 123, 2026, 3000, 3001]
V3_ARMS = ["direct_thin_anchor", "esfm_endpoint_anchor"]
OUTPUTS = ["raw", "blend"]
N_IMAGES = {"CHASE-DB1": 8, "DRIVE": 20, "HRF": 12}
N_TRAIN = {"CHASE-DB1": 16, "DRIVE": 16, "HRF": 27}
SUMMARY_METRICS = [
    "dice", "iou", "cldice", "thin_skeleton_recall", "precision", "recall",
    "new_error_count", "new_error_rate", "fixed_error_count", "fixed_error_rate",
    "betti0_absolute_error", "fp", "fn", "thin_skeleton_missed",
]
INTERVAL_METRICS = ["dice", "cldice", "thin_skeleton_recall"]
BOOTSTRAP_REPETITIONS = 50000
BOOTSTRAP_SEED = 20261115
ALPHAS = [0., .25, .5, .75, 1.]
CONTRASTS = [
    ("direct", "existing", True), ("baseline", "baseline", False),
    ("esfm", "existing", False),
    ("endpoint_iterative", "existing", False), ("full_fm", "existing", False),
    ("selective_no_gate", "existing", False),
    ("direct_thin_anchor", "v3", False), ("esfm_endpoint_anchor", "v3", False),
]
INTERPRETATION = (
    "Exploratory unadjusted paired percentile intervals on previously examined cohorts. "
    "The primary descriptive contrast is raw native-patch Direct minus preserved original "
    "Direct in Dice; baseline comparison is mandatory. All other original arms and both v3 "
    "objective variants are secondary, as are all blended and structural comparisons. Five fixed "
    "fitted-refiner outputs stay together within each resampling unit. CHASE enumerates all "
    "256 ordered resamples of four paired two-eye patient clusters; exhaustive computation "
    "does not imply exact confidence coverage. DRIVE and HRF use 50000 paired image draws "
    "with seed 20261115; independent patient IDs are unverified. No pixels or training seeds "
    "are independently resampled. Seed sample SD is separate. All comparisons condition on "
    "one frozen baseline per dataset, the retained OOF maps, trained refiners and fixed split. "
    "No pooling, p-values, multiplicity adjustment, equivalence or significance claim is supplied."
)


def _bind(path, digest, hashes):
    path = resolve_recorded_path(path)
    if str(path) in hashes:
        _require(hashes[str(path)] == digest, "Conflicting retained digest: " + str(path))
    else:
        _require(path.is_file() and sha256_file(path) == digest, "Retained input changed: " + str(path))
        hashes[str(path)] = digest


def _read_v3(old, dataset, names, subjects, original_refs, original_summary, thin, hashes):
    tag = _tag(old)
    summary_path = old / f"summary_statistics_{tag}.json"
    verification_path = old / f"verified_report_{tag}.json"
    csv_path = old / f"all_test_records_{tag}.csv"
    thin_path = old / f"thin_definition_{tag}.json"
    saved = _json(summary_path)
    verified = _json(verification_path)
    _require(_dataset(saved["dataset"]) == dataset and set(saved["arms"]) == set(V3_ARMS), "Unexpected v3 comparison arms")
    _require(verified["all_saved_metrics_exactly_reproduced"] is True and verified["models"] == 10,
             "Existing v3 experiment lacks complete metric verification")
    _require(_json(thin_path)["cutoff"] == thin["cutoff"], "Existing v3 cutoff differs")
    _require(saved["baseline"] == original_summary["baseline"], "Existing v3 baseline differs")
    _require(saved["contextual_reference"]["arms"] == original_summary["arms"], "Existing v3 original controls changed")
    _bind(summary_path, verified["report_output_sha256"]["summary_json"], hashes)
    for recorded, digest in verified["verified_input_sha256"].items():
        source = resolve_recorded_path(recorded) if recorded_is_absolute(recorded) else old / recorded
        _bind(source, digest, hashes)
    for path in (summary_path, verification_path, csv_path, thin_path):
        hashes[str(path)] = sha256_file(path)
    _, _, old_names, old_subjects, index = _rows(csv_path, dataset, V3_ARMS, N_IMAGES[dataset])
    _require(names == old_names and subjects == old_subjects, "Existing v3 case identities differ")
    for arm, seed, name in itertools.product(V3_ARMS, SEEDS, names):
        path = old / arm / f"seed_{seed}" / f"{name}_predictions_{tag}.npz"
        _require(str(path) in hashes, "V3 archive is not bound to its verification")
        with np.load(path, allow_pickle=False) as arrays:
            inputs = tuple(np.asarray(arrays[key]) for key in ("baseline", "ground_truth", "fov"))
            _require(all(_same_bytes(a, b) for a, b in zip(inputs, original_refs[name])),
                     "V3 and original baseline/GT/FOV bytes differ")
    for arm, output, seed, metric in itertools.product(V3_ARMS, OUTPUTS, SEEDS, SUMMARY_METRICS):
        value = statistics.mean(_number(index[arm, seed, name, output], metric) for name in names)
        row = next(item for item in saved["arms"][arm][output]["seed_means"] if int(item["refiner_seed"]) == seed)
        _require(value == row["metrics"][metric], "V3 CSV/summary binding differs")
    return dict(old_experiment=str(old), version=tag, summary_sha256=hashes[str(summary_path)],
                per_image_csv_sha256=hashes[str(csv_path)], arms=copy.deepcopy(saved["arms"]),
                baseline=copy.deepcopy(saved["baseline"]),
                scope="both v3 objective variants reused exactly; no refitting or source replacement"), index


def _selection_locks(out, tag, dataset, completion, plan, hashes):
    local_path = out / f"all_selections_before_test_{tag}.json"
    local = _json(local_path)
    _bind(local_path, completion["lock_sha256"], hashes)
    decisions = {(d["arm"], int(d["refiner_seed"])): d for d in local["decisions"]}
    _require(len(decisions) == len(local["decisions"]) == 5 and set(decisions) == set(itertools.product(ARMS, SEEDS)),
             "Expected five final native-patch selections")
    global_path, global_lock = _lock_reference(completion, plan, "global_lock")
    rows = _global_decisions(global_lock)
    global_index = {(d["dataset"], d["arm"], int(d["refiner_seed"])): d for d in rows}
    _require(len(rows) == len(global_index) == 15 and set(global_index) == set(itertools.product(N_IMAGES, ARMS, SEEDS)),
             "Global selection lock does not cover all fifteen final fits")
    for key, decision in decisions.items():
        recorded = dict(global_index[(dataset,) + key])
        recorded.pop("dataset", None)
        _require(recorded == decision, "Global/local native-patch selection differs")
        for field, value in (("tile", 256), ("stride", 128), ("train_batch", 16), ("infer_batch", 16),
                             ("training_patches_per_image_per_epoch", 4), ("training_parameters", 315974)):
            _require(decision[field] == value, "Locked native-patch setting differs: " + field)
    study_plan_path = resolve_recorded_path(plan["study_plan"])
    _bind(study_plan_path, global_lock["fixed_study_plan_sha256"], hashes)
    hashes[str(global_path)] = sha256_file(global_path)
    return decisions, dict(local=str(local_path), global_lock=str(global_path),
                           fixed_study_plan=str(study_plan_path), global_final_fits=15,
                           globally_locked_before_test_declared=True, no_pilot_or_hyperparameter_search=True)


def _native_audit(environment, plan, dataset, names, thin, hashes):
    path = resolve_recorded_path(environment["input_audit"])
    _require(path == resolve_recorded_path(plan["native_source_audit"]), "Native source audit references differ")
    audit = _json(path)
    for flag in ("original_512_fields_unchanged", "exact_512_RGB_GT_FOV_green_replay_verified", "historical_sources_unchanged"):
        _require(audit.get(flag) is True, "Native input audit did not pass: " + flag)
    _require(audit["status"] == "passed" and audit["native_patch_schema"]["version"] == 1,
             "Native source audit/schema failed")
    _require(audit["source_sha256_before"] == audit["source_sha256_after"], "Native raw input sources changed")
    _require(sorted(audit["split"]["test"]) == names and audit["split"]["train"] == thin["training_image_ids"],
             "Native audit partitions differ")
    _require(len(audit["split"]["train"]) == N_TRAIN[dataset], "Unexpected training inventory")
    for recorded, digest in audit["source_sha256_before"].items():
        _bind(recorded, digest, hashes)
    _bind(audit["reference_512_audit"], audit["reference_512_audit_sha256"], hashes)
    hashes[str(path)] = sha256_file(path)
    return audit


def _native_shape(audit, name):
    width, height = audit["native_inventory"][name]["native_rgb_width_height"]
    return int(height), int(width)


def _check_training_logs(destination, tag, dataset, decision, audit, hashes):
    files = {key: destination / f"{prefix}_{tag}.{extension}" for key, prefix, extension in (
        ("training", "training", "csv"), ("patches", "patch_draws", "csv"),
        ("validation", "validation", "csv"), ("tiling", "validation_tiling", "json"))}
    with files["training"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require([int(row["epoch"]) for row in rows] == list(range(1, 121)), "Incomplete training epochs")
    patches = N_TRAIN[dataset] * 4
    updates = (N_TRAIN[dataset] + 3) // 4
    for row in rows:
        _require(int(row["patches_seen"]) == patches and int(row["optimizer_updates"]) == updates,
                 "Training pixel/update budget differs")
        for field in ("loss", "bce", "dice", "soft_cldice", "gate", "lr"):
            _number(row, field)
    with files["patches"].open(newline="", encoding="utf-8") as stream:
        draws = list(csv.DictReader(stream))
    _require(len(draws) == 120 * patches, "Incomplete patch-draw inventory")
    counts = Counter()
    ordinal = set()
    for row in draws:
        epoch, index, name = int(row["epoch"]), int(row["patch_index"]), row["image"]
        _require(1 <= epoch <= 120 and 0 <= index < patches and (epoch, index) not in ordinal, "Invalid patch epoch/index")
        ordinal.add((epoch, index))
        _require(name in audit["split"]["train"], "Validation/test image used for training patch")
        height, width = _native_shape(audit, name)
        cy, cx, top, left = (int(row[key]) for key in ("centre_y", "centre_x", "top", "left"))
        _require(0 <= cy < height and 0 <= cx < width, "Patch centre outside native image")
        _require(top == min(max(cy-128, 0), max(height-256, 0)) and
                 left == min(max(cx-128, 0), max(width-256, 0)), "Patch crop alignment differs")
        _require(int(row["rotation90"]) in range(4) and row["flip_horizontal"] in ("True", "False") and
                 row["flip_vertical"] in ("True", "False"), "Invalid patch augmentation record")
        counts[epoch, name] += 1
    _require(set(counts) == set(itertools.product(range(1, 121), audit["split"]["train"])) and
             all(value == 4 for value in counts.values()), "Expected four draws per training image/epoch")
    with files["validation"].open(newline="", encoding="utf-8") as stream:
        validation = list(csv.DictReader(stream))
    validation_index = {(int(row["epoch"]), float(row["alpha"])): _number(row, "validation_dice") for row in validation}
    expected = {(0, 0.)} | set(itertools.product(range(10, 121, 10), ALPHAS))
    _require(len(validation) == len(validation_index) and set(validation_index) == expected, "Validation grid differs")
    for kind in OUTPUTS:
        selected = decision[kind]
        alpha = 1. if kind == "raw" else float(selected["alpha"])
        key = (int(selected["epoch"]), alpha)
        _require(key in validation_index and validation_index[key] == selected["validation_dice"], "Selection not recorded in validation grid")
        candidates = [value for (epoch, a), value in validation_index.items() if (kind == "blend" or (epoch > 0 and a == 1.))]
        _require(max(candidates) <= selected["validation_dice"] + 1e-12, "Saved selection is not validation-best")
    tiling = _json(files["tiling"])
    _require(len(tiling) == 12 * len(audit["split"]["validation"]), "Validation tiling inventory differs")
    _require({(int(row["epoch"]), row["image"]) for row in tiling} ==
             set(itertools.product(range(10, 121, 10), audit["split"]["validation"])),
             "Validation tiling image/epoch grid differs")
    for row in tiling:
        _require(row["image"] in audit["split"]["validation"] and int(row["epoch"]) in range(10, 121, 10),
                 "Tiling validation record uses an unexpected image/epoch")
    for path in files.values():
        hashes[str(path)] = sha256_file(path)


def _check_tiling(meta, height, width):
    from flow_matching_refiner.native_patch_model import tile_starts
    tiles = len(tile_starts(max(height, 256), 256, 128)) * len(tile_starts(max(width, 256), 256, 128))
    for field, value in (("native_height", height), ("native_width", width), ("tile", 256), ("stride", 128),
                         ("batch", 16), ("total_tiles", tiles), ("velocity_calls", tiles),
                         ("velocity_forward_batches", (tiles+15)//16), ("gate_calls", 0), ("halo", 0)):
        _require(meta[field] == value, "Tiling metadata mismatch: " + field)
    _require(_number(meta, "coverage_min") > 0 and meta["hann_1d_floor"] == 1e-3,
             "Tiled output lacks positive fixed-window coverage")
    return tiles, (tiles+15)//16


def report(experiment):
    out = assert_output_path(experiment, must_exist=True)
    tag = _tag(out)
    completion_path = out / f"completed_training_evaluation_{tag}.json"
    if not completion_path.is_file():
        raise RuntimeError("Completed native-patch training/test marker is required")
    complete = _json(completion_path)
    plan_path = out / f"plan_{tag}.json"
    plan = _json(plan_path)
    dataset = _dataset(plan["dataset"])
    n_images = N_IMAGES[dataset]
    _require(complete["fits"] == 5 and complete["records"] == 11*n_images and complete["native_prediction_archives"] == 5*n_images,
             "Expected five native-patch fits and 11N rows")
    for flag in ("choices_locked_before_test", "global_choices_locked_before_test", "fixed_settings_before_training",
                 "previous_files_byte_identical", "OOF_maps_audited", "native_source_audit_passed", "original512_fields_unchanged"):
        _require(complete.get(flag) is True, "Native-patch completion flag failed: " + flag)
    for field, value in (("arms", ARMS), ("refiner_training_seeds", SEEDS), ("epochs", 120), ("batch", 16),
                         ("tile", 256), ("stride", 128), ("infer_batch", 16), ("patches_per_image", 4),
                         ("comparison_grid", 512), ("threshold", .5), ("fixed_settings_before_training", True)):
        _require(plan[field] == value, "Native-patch plan setting differs: " + field)
    _require(complete["primary_evaluation_grid"] == 512, "Native and comparison metrics must remain separate")
    paths = {key: out / f"{prefix}_{tag}.{extension}" for key, prefix, extension in (
        ("summary_json", "summary_statistics", "json"), ("seed_means_csv", "seed_means", "csv"),
        ("seed_differences_csv", "seed_differences", "csv"), ("image_differences_csv", "image_differences", "csv"),
        ("difference_arrays_json", "difference_arrays", "json"), ("verification_json", "verified_report", "json"))}
    if dataset == "CHASE-DB1":
        paths["patient_differences_csv"] = out / f"patient_differences_{tag}.csv"
    _require(not any(path.exists() for path in paths.values()), "Existing timestamped reports must be preserved")
    old = resolve_recorded_path(plan["old_experiment"])
    v3_old = resolve_recorded_path(plan["v3_experiment"])
    _require(out != old and out != v3_old and not out.is_relative_to(old) and not out.is_relative_to(v3_old),
             "Reports must not be written inside earlier experiments")
    thin_path = out / f"thin_definition_{tag}.json"
    thin = _json(thin_path)
    _require(thin["quantile"] == 1/3 and thin["comparison_grid_only"] is True and thin["native_cutoff_derived"] is False,
             "Thin definition must remain the frozen 512-grid definition")
    cutoff = float(thin["cutoff"])
    environment_path = out / f"environment_{tag}.json"
    environment = _json(environment_path)
    _require(environment["structural_libraries"] == dependency_versions(), "Metric-library versions differ")
    hashes = {str(path): sha256_file(path) for path in (completion_path, plan_path, thin_path, environment_path)}
    decisions, selection_audit = _selection_locks(out, tag, dataset, complete, plan, hashes)
    original, references, names, subjects, old_index = _read_existing(old, dataset, n_images, thin, hashes)
    v3_reference, v3_index = _read_v3(v3_old, dataset, names, subjects, references, original, thin, hashes)
    audit = _native_audit(environment, plan, dataset, names, thin, hashes)
    csv_path = out / f"all_test_records_{tag}.csv"
    _, baseline_rows, new_names, new_subjects, index = _rows(csv_path, dataset, ARMS, n_images)
    _require(new_names == names and new_subjects == subjects, "Native comparison case IDs differ")
    hashes[str(csv_path)] = sha256_file(csv_path)
    patients = sorted(set(subjects.values())) if dataset == "CHASE-DB1" else []
    if patients:
        _require(len(patients) == 4 and all(sum(value == p for value in subjects.values()) == 2 for p in patients),
                 "CHASE requires four two-eye patient clusters")
    import torch
    from torch.nn import functional as F
    from flow_matching_refiner.native_patch_model import apply_residual, project_residual_512
    torch.set_num_threads(4)
    _require(environment["torch"] == torch.__version__, "Torch version differs from prediction arithmetic")
    native_reference_paths = {}
    for name in names:
        path = out / "native_references" / f"{name}_native_reference_{tag}.npz"
        native_reference_paths[name] = path
        hashes[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as arrays:
            coarse, truth, fov = (np.asarray(arrays[key]) for key in ("coarse_native", "ground_truth_native", "fov_native"))
            height, width = _native_shape(audit, name)
            _require(coarse.dtype == np.float32 and coarse.shape == truth.shape == fov.shape == (1, height, width), "Native reference shape/dtype differs")
            _require(truth.dtype == fov.dtype == np.bool_ and fov.any(), "Native reference annotations must be binary original-grid masks")
            expected_coarse = F.interpolate(torch.from_numpy(references[name][0])[None], size=(height, width), mode="bilinear", align_corners=False)[0].numpy()
            _require(_same_bytes(coarse, expected_coarse), "Native coarse is not the documented bilinear 512 projection")
            _require(audit["native_inventory"][name]["native_gt_available"] is True, "Native test GT unexpectedly unavailable")
            for native_mask, old_mask in ((truth, references[name][1]), (fov, references[name][2])):
                projected = project_residual_512(torch.from_numpy(native_mask.astype(np.float32)), dataset).numpy() >= .5
                _require(np.array_equal(projected, old_mask >= .5), "Native annotation projection differs from original 512 target/FOV")
    verified_rows = 0
    verified_archives = 0
    for seed in SEEDS:
        destination = out / ARM / f"seed_{seed}"
        decision = decisions[ARM, seed]
        complete_path = destination / f"completed_{tag}.json"
        job = _json(complete_path)
        _require(job["records"] == 2*n_images and job["test_images"] == n_images and job["native_outputs_stored"] is True,
                 "Incomplete native-patch prediction job")
        hashes[str(complete_path)] = sha256_file(complete_path)
        _check_training_logs(destination, tag, dataset, decision, audit, hashes)
        inference_path = destination / f"inference_{tag}.json"
        inference_rows = _json(inference_path)
        inference = {item["image"]: item for item in inference_rows}
        _require(len(inference_rows) == len(inference) == n_images and sorted(inference) == names, "Inference metadata grid differs")
        hashes[str(inference_path)] = sha256_file(inference_path)
        for filename, key in ((f"best_raw_model_{tag}.pt", "raw_sha256"), (f"selected_blend_model_{tag}.pt", "blend_sha256"),
                              (f"final_model_{tag}.pt", "final_sha256")):
            _bind(destination / filename, decision[key], hashes)
        for name in names:
            meta = inference[name]
            prediction_path = destination / f"{name}_predictions_{tag}.npz"
            native_path = destination / f"{name}_native_predictions_{tag}.npz"
            _require(resolve_recorded_path(meta["prediction_path"]) == prediction_path and resolve_recorded_path(meta["native_path"]) == native_path and
                     resolve_recorded_path(meta["native_reference"]) == native_reference_paths[name], "Inference archive references differ")
            _bind(prediction_path, meta["prediction_sha256"], hashes)
            _bind(native_path, meta["native_sha256"], hashes)
            height, width = _native_shape(audit, name)
            alpha = float(decision["blend"]["alpha"])
            _require(meta["alpha"] == alpha and alpha in ALPHAS, "Inference alpha differs from lock")
            raw_calls, raw_batches = _check_tiling(meta["raw"], height, width)
            blend_calls, blend_batches = _check_tiling(meta["blend"], height, width) if alpha else (0, 0)
            if not alpha:
                _require(meta["blend"] is None, "Identity blend must not run tiled inference")
            with np.load(prediction_path, allow_pickle=False) as arrays, np.load(native_path, allow_pickle=False) as native, \
                 np.load(native_reference_paths[name], allow_pickle=False) as reference:
                inputs = tuple(np.asarray(arrays[key]) for key in ("baseline", "ground_truth", "fov"))
                _require(all(_same_bytes(a, b) for a, b in zip(inputs, references[name])), "Original comparison baseline/GT/FOV bytes changed")
                baseline, truth, fov = (_plane(value, label) for value, label in zip(inputs, ("baseline", "GT", "FOV")))
                fov = fov >= .5
                raw_delta = np.asarray(native["raw_residual_native"])
                raw_native_probability = np.asarray(native["raw_probability_native"])
                _require(raw_delta.dtype == raw_native_probability.dtype == np.float32 and raw_delta.shape == raw_native_probability.shape == (1, height, width),
                         "Native residual/probability shape or dtype differs")
                _require(np.isfinite(raw_delta).all() and np.isfinite(raw_native_probability).all(), "Nonfinite native output")
                expected_native = apply_residual(torch.from_numpy(reference["coarse_native"]), torch.from_numpy(raw_delta)).numpy()
                _require(_same_bytes(raw_native_probability, expected_native), "Native probability differs from the fixed residual application rule")
                baseline_tensor = torch.from_numpy(inputs[0])
                expected_raw = apply_residual(baseline_tensor, project_residual_512(torch.from_numpy(raw_delta), dataset))
                _require(_same_bytes(np.asarray(arrays["raw"]), expected_raw.numpy()), "512 raw output does not reproduce projected native residual")
                if alpha:
                    candidate_delta = np.asarray(native["blend_candidate_residual_native"])
                    _require(candidate_delta.dtype == np.float32 and candidate_delta.shape == raw_delta.shape and np.isfinite(candidate_delta).all(), "Invalid native blend candidate")
                    candidate = apply_residual(baseline_tensor, project_residual_512(torch.from_numpy(candidate_delta), dataset))
                    expected_blend = ((1-alpha)*baseline_tensor + alpha*candidate).numpy()
                    if decision["raw"]["epoch"] == decision["blend"]["epoch"]:
                        _require(_same_bytes(candidate_delta, raw_delta), "Same-epoch blend candidate differs from raw residual")
                else:
                    _require("blend_candidate_residual_native" not in native.files, "Identity blend must not contain a learned native candidate")
                    expected_blend = inputs[0]
                _require(_same_bytes(np.asarray(arrays["blend"]), expected_blend), "512 blend does not reproduce locked projected-residual policy")
                if seed == SEEDS[0]:
                    _check_metrics(index["baseline", -1, name, "raw"], metrics(baseline, truth, baseline, cutoff, fov), "baseline/" + name)
                    for metric in SUMMARY_METRICS:
                        _require(_number(index["baseline", -1, name, "raw"], metric) == _number(old_index["baseline", -1, name, "raw"], metric),
                                 "Baseline per-image metric changed")
                    verified_rows += 1
                for output in OUTPUTS:
                    probability = _plane(arrays[output], output)
                    _require(probability.shape == (512, 512) and ((probability >= 0) & (probability <= 1)).all(), "Invalid 512 prediction probability")
                    row = index[ARM, seed, name, output]
                    _check_metrics(row, metrics(probability, truth, baseline, cutoff, fov), f"{seed}/{name}/{output}")
                    selected = decision[output]
                    _require(int(row["selected_epoch"]) == int(selected["epoch"]) and
                             _number(row, "alpha") == (1. if output == "raw" else alpha), "Recorded output selection differs")
                    calls, batches = (raw_calls, raw_batches) if output == "raw" else (blend_calls, blend_batches)
                    _require(int(row["velocity_calls"]) == calls and int(row["tile_forward_batches"]) == batches and int(row["gate_calls"]) == 0,
                             "Recorded tiled inference cost differs")
                    _require(int(row["native_height"]) == height and int(row["native_width"]) == width, "Recorded native dimensions differ")
                    if output == "raw":
                        _require(int(selected["epoch"]) > 0, "Primary raw checkpoint must be learned")
                    verified_rows += 1
            verified_archives += 1
    _require(verified_rows == 11*n_images and verified_archives == 5*n_images, "Verification inventory differs")
    baseline_means = {metric: statistics.mean(_number(row, metric) for row in baseline_rows) for metric in SUMMARY_METRICS}
    _require(baseline_means == original["baseline"]["means"], "Baseline summary differs")
    arms = {ARM: {}}
    seed_rows = [dict(dataset=dataset, arm="baseline", source="baseline", output="raw", refiner_seed=-1, n_images=n_images, **baseline_means)]
    for output in OUTPUTS:
        means = []
        for seed in SEEDS:
            values = {metric: statistics.mean(_number(index[ARM, seed, name, output], metric) for name in names) for metric in SUMMARY_METRICS}
            means.append(dict(refiner_seed=seed, metrics=values))
            seed_rows.append(dict(dataset=dataset, arm=ARM, source="new", output=output, refiner_seed=seed, n_images=n_images, **values))
        arms[ARM][output] = dict(n_refiner_seeds=5, n_images=n_images, seed_means=means,
            mean={metric: statistics.mean(item["metrics"][metric] for item in means) for metric in SUMMARY_METRICS},
            sample_sd={metric: statistics.stdev(item["metrics"][metric] for item in means) for metric in SUMMARY_METRICS},
            sd_definition="sample SD of five image-macro native-patch training means conditional on one frozen baseline")
    draws = None
    draw_hash = None
    if dataset != "CHASE-DB1":
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n_images, size=(BOOTSTRAP_REPETITIONS, n_images), dtype=np.int64)
        draw_hash = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
    comparisons, image_rows, patient_rows, seed_difference_rows = [], [], [], []
    for output, (rhs, source, primary_candidate) in itertools.product(OUTPUTS, CONTRASTS):
        comparison_id = f"{output}_{ARM}_minus_{source + '_' if source != 'baseline' else ''}{rhs}"
        metric_results = {}
        rhs_index = v3_index if source == "v3" else old_index
        rhs_arms = v3_reference["arms"] if source == "v3" else original["arms"]
        for metric in INTERVAL_METRICS:
            per_image = []
            for name in names:
                a = statistics.mean(_number(index[ARM, seed, name, output], metric) for seed in SEEDS)
                b = _number(index["baseline", -1, name, "raw"], metric) if source == "baseline" else statistics.mean(
                    _number(rhs_index[rhs, seed, name, output], metric) for seed in SEEDS)
                item = dict(image=name, subject_label=subjects[name], lhs_mean=a, rhs_mean=b, difference=a-b)
                per_image.append(item)
                image_rows.append(dict(dataset=dataset, comparison=comparison_id, output=output, metric=metric, lhs=ARM, rhs=rhs,
                                       lhs_source="new", rhs_source=source, **item))
            per_seed = []
            for seed in SEEDS:
                a = arms[ARM][output]["seed_means"][SEEDS.index(seed)]["metrics"][metric]
                b = baseline_means[metric] if source == "baseline" else rhs_arms[rhs][output]["seed_means"][SEEDS.index(seed)]["metrics"][metric]
                item = dict(refiner_seed=seed, lhs_mean=a, rhs_mean=b, difference=a-b)
                per_seed.append(item)
                seed_difference_rows.append(dict(dataset=dataset, comparison=comparison_id, output=output, metric=metric,
                    lhs=ARM, rhs=rhs, lhs_source="new", rhs_source=source, n_images=n_images, **item))
            if dataset == "CHASE-DB1":
                per_patient = []
                for patient in patients:
                    pair = [item for item in per_image if item["subject_label"] == patient]
                    item = dict(subject=patient, n_images=2, lhs_mean=statistics.mean(row["lhs_mean"] for row in pair),
                                rhs_mean=statistics.mean(row["rhs_mean"] for row in pair), difference=statistics.mean(row["difference"] for row in pair))
                    per_patient.append(item)
                    patient_rows.append(dict(dataset=dataset, comparison=comparison_id, output=output, metric=metric,
                        lhs=ARM, rhs=rhs, lhs_source="new", rhs_source=source, **item))
                interval = exact_patient_interval([item["difference"] for item in per_patient])
                interval.update(bootstrap_unit="patient", per_patient=per_patient, patient_independence_verified=True)
            else:
                interval = image_interval([item["difference"] for item in per_image], draws)
                interval["bootstrap_index_sha256"] = draw_hash
            macro_rhs = baseline_means[metric] if source == "baseline" else rhs_arms[rhs][output]["mean"][metric]
            _require(abs(interval["mean_difference"]-(arms[ARM][output]["mean"][metric]-macro_rhs)) < 1e-12, "Paired/macro difference disagrees")
            interval.update(per_image=per_image, per_seed=per_seed,
                paired_seed_difference_mean=statistics.mean(item["difference"] for item in per_seed),
                paired_seed_difference_sample_sd=statistics.stdev(item["difference"] for item in per_seed),
                analysis_role="primary descriptive" if primary_candidate and output == "raw" and metric == "dice" else "secondary",
                positive_difference_means="higher native-patch score")
            metric_results[metric] = interval
        comparisons.append(dict(id=comparison_id, output=output, lhs=ARM, rhs=rhs, lhs_source="new", rhs_source=source,
            primary=primary_candidate and output == "raw", primary_metric="dice" if primary_candidate and output == "raw" else None,
            mandatory_baseline_comparison=source == "baseline", metrics=metric_results))
    summary = dict(version=tag, dataset=dataset, scope="one new native-patch Direct recipe; seven earlier controls preserved",
        n_test_images=n_images, n_test_patients=4 if dataset == "CHASE-DB1" else None,
        refiner_training_seeds=SEEDS, shared_frozen_baseline_seed=42, n_shared_frozen_baseline_checkpoints=1,
        new_final_fits=5, contextual_existing_fits=35,
        primary_result="raw native-patch Direct minus preserved original Direct in 512-grid Dice",
        secondary_result="mandatory frozen baseline; all other original arms and both v3 objective variants; structural metrics and all blends; exploratory",
        metric_aggregation="equal-image mean within each seed; mean and sample SD over five native-patch fits",
        evaluation_grid=512, native_grid_metrics_computed=False,
        prediction_projection="native logit residual only; Pillowfloat nearest CHASE/DRIVE or OpenCV INTER_NEAREST HRF, applied to original512coarselogit",
        zero_residual_policy="exact original coarse probability wherever delta==0; otherwise sigmoid(clipped coarse logit + delta)",
        native_probability_projection_used_for_comparisons=False, thin_definition=thin,
        baseline=copy.deepcopy(original["baseline"]), arms=arms, contextual_reference=original, v3_reference=v3_reference,
        selection_audit=selection_audit, fixed_plan=copy.deepcopy(plan), comparisons=comparisons,
        bootstrap_unit="patient" if dataset == "CHASE-DB1" else "image", patient_independence_verified=dataset == "CHASE-DB1",
        uncertainty_interpretation=INTERPRETATION,
        bootstrap=dict(unit="patient" if dataset == "CHASE-DB1" else "image", repetitions=256 if dataset == "CHASE-DB1" else BOOTSTRAP_REPETITIONS,
            seed=None if dataset == "CHASE-DB1" else BOOTSTRAP_SEED, image_order=names, shared_index_sha256=draw_hash, resamples_training_seeds=False),
        files={key: str(path) for key, path in paths.items()})
    verification = dict(version=tag, dataset=dataset, completed_training_evaluation_marker_verified=True,
        csv_rows=11*n_images, verified_metric_rows=verified_rows, verified_prediction_archives=verified_archives,
        verified_native_prediction_archives=verified_archives, verified_native_reference_archives=n_images,
        unique_images=n_images, models=5, contextual_existing_models=35,
        all_saved_512_metrics_exactly_reproduced=True, all_saved_metrics_exactly_reproduced=True,
        projected_native_residual_reproduces_every_512_raw_and_blended_prediction=True,
        native_probability_logit_residual_arithmetic_verified=True, native_grid_metrics_computed=False,
        native_GT_and_FOV_project_to_preserved_512_annotations=True,
        all_new_and_existing_baseline_GT_FOV_arrays_byte_identical=True,
        all_seven_existing_arm_summaries_copied_exactly=True,
        all_selection_lock_and_selected_model_hashes_verified=True, global_fifteen_final_fit_selection_lock_verified=True,
        training_patch_ids_coordinates_and_update_pixel_budgets_verified=True,
        fixed_settings_before_training_declared=True, structural_library_versions=dependency_versions(),
        verified_input_sha256=hashes, report_source_sha256=sha256_file(Path(__file__)),
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
    verification["report_output_sha256"] = {key: sha256_file(path) for key, path in paths.items() if key != "verification_json"}
    _write_json(paths["verification_json"], verification)
    return summary


def _self_test():
    _require(_dataset("CHASE") == "CHASE-DB1", "Dataset aliases")
    _check_tiling(dict(native_height=584, native_width=565, tile=256, stride=128, batch=16, total_tiles=16,
        velocity_calls=16, velocity_forward_batches=1, gate_calls=0, halo=0, coverage_min=1e-6, hann_1d_floor=1e-3), 584, 565)
    _require(exact_patient_interval([.125]*4)["ci95"] == [.125, .125], "Patient bootstrap guard")
    try:
        report(Path("_synthetic_native_incomplete_2026-11-16_00-00-00"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete native study accepted")
    print("Synthetic native-patch reporter guards passed; no real data or report loaded.")


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
        saved = report(args.experiment)
        print(json.dumps(dict(dataset=saved["dataset"], version=saved["version"], files=saved["files"]), indent=2))
