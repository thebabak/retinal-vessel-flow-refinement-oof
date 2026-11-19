"""Audited CHASE cross-fitted coarse maps for the new ESFM study.

Importing this file does not import torch, load data, or execute an experiment.
``load_chase_oof`` explicitly replays existing baseline checkpoints only.  It
never trains a baseline or refiner and never computes segmentation metrics.
All historical inputs are read-only; the sole output is a new timestamped audit.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from flow_matching_refiner.data import ROOT, ReferenceData, load_reference, sha256_file


OOF_TAG = "2026-10-09_23-30-40"
OOF_ROOT = ROOT / "revise" / f"oof_comparison_{OOF_TAG}" / "CHASE" / "seed_42"


def stamp() -> str:
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _array_hash(array: Any) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _subjects(names: list[str], data: dict[str, Any]) -> set[int]:
    return {int(data[name]["subject"]) for name in names}


def _check_probability(array: Any, name: str) -> None:
    import numpy as np
    _require(array.dtype == np.dtype("float32"), f"Not FP32: {name}")
    _require(array.shape == (1, 512, 512), f"Unexpected probability shape: {name}")
    _require(bool(np.isfinite(array).all()), f"Non-finite probabilities: {name}")
    _require(bool(((array >= 0) & (array <= 1)).all()), f"Out-of-range probabilities: {name}")


def _write_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def load_chase_oof(
    workdir: str | Path, device: str = "cuda",
) -> tuple[ReferenceData, dict[str, Any]]:
    """Return reference data with only training coarse maps replaced by OOF maps.

    The held-out fold predictions are recomputed from four retained fold
    checkpoints and must match both the per-fold and merged archives bitwise.
    Validation/test coarse maps always come from the full-training seed-42
    baseline.  Checkpoint selection used the original validation patients;
    those patients are disjoint from every held-out training patient.

    Ground truth is accessed for source-identity checks, not model selection or
    metric evaluation.  The caller must still restrict fitting to ``train``.
    CPU execution is supported but exact archive replay may fail if its floating
    point implementation differs from the original CUDA execution.
    """
    import numpy as np
    import torch

    workdir = Path(workdir).resolve()
    for historical in (OOF_ROOT, ROOT / "paired_reproduction" / "all_datasets_five_full_runs_2026-10-08_15-38-24"):
        _require(not workdir.is_relative_to(historical.resolve()), "Audit workspace cannot be inside historical experiment inputs")
    workdir.mkdir(parents=True, exist_ok=True)
    tag = stamp()
    audit_path = workdir / f"CHASE_OOF_source_audit_{tag}.json"
    _require(not audit_path.exists(), f"Audit output already exists: {audit_path}")

    oof_protocol_path = OOF_ROOT / f"oof_protocol_{OOF_TAG}.json"
    merged_path = OOF_ROOT / f"oof_training_maps_{OOF_TAG}.npz"
    _require(oof_protocol_path.is_file() and merged_path.is_file(), "Required OOF protocol/maps are missing; no training fallback")
    oof_protocol = json.loads(oof_protocol_path.read_text(encoding="utf-8-sig"))
    _require(oof_protocol.get("dataset") == "CHASE" and oof_protocol.get("training_seed") == 42,
             "OOF protocol dataset/seed mismatch")
    folds = oof_protocol.get("folds")
    _require(isinstance(folds, list) and len(folds) == 4, "Expected four retained CHASE OOF folds")

    reference = load_reference("CHASE", 42, workdir / "loader_workspace", device)
    data, split = reference.data, reference.manifest
    _require(oof_protocol["split"] == split, "OOF and retained full-baseline splits differ")
    _require([len(split[k]) for k in ("train", "validation", "test")] == [16, 4, 8], "Unexpected CHASE split sizes")
    outer_subjects = {k: _subjects(split[k], data) for k in split}
    _require(not any(outer_subjects[a] & outer_subjects[b]
                     for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
             "Patient overlap across outer partitions")
    flattened = [name for fold in folds for name in fold]
    _require(len(flattened) == len(set(flattened)) == 16 and set(flattened) == set(split["train"]),
             "Held-out folds do not cover training exactly once")

    protected_paths = {
        oof_protocol_path, merged_path, reference.baseline_checkpoint,
        reference.autoencoder_checkpoint, reference.protocol_path,
    }
    source_paths = [
        Path(__file__), ROOT / "flow_matching_refiner" / "data.py",
        ROOT / "paired_reproduction" / "official_model.py",
        ROOT / "paired_reproduction" / "run.py",
        ROOT / "paired_reproduction" / "run_all_datasets_five_full_runs.py",
        ROOT / "paired_reproduction" / "run_oof_comparison.py",
    ]
    for source in source_paths:
        protected_paths.add(source)
    for names in split.values():
        for name in names:
            protected_paths.update((ROOT / "data" / f"{name}.jpg", ROOT / "data" / f"{name}_1stHO.png"))
    for index in range(1, 5):
        folder = OOF_ROOT / f"fold_{index}"
        protected_paths.update((folder / "baseline.pt", folder / f"fold_inventory_{OOF_TAG}.json",
                                folder / f"held_out_predictions_{OOF_TAG}.npz"))
    for name in split["test"]:
        protected_paths.add(reference.baseline_checkpoint.parent / f"{name}_reference.npz")
    for path in protected_paths:
        _require(path.is_file(), f"Required source artifact missing: {path}")
    before = {str(path): sha256_file(path) for path in sorted(protected_paths)}
    report: dict[str, Any] = dict(
        status="auditing", started_at=tag, dataset="CHASE", baseline_seed=42,
        purpose="New ESFM input provenance audit; retained historical OOF maps reused",
        split=split, subjects={k: sorted(v) for k, v in outer_subjects.items()},
        oof_protocol=str(oof_protocol_path), merged_maps=str(merged_path),
        output_audit=str(audit_path), source_sha256_before=before,
        fold_training_performed=False, refiner_training_performed=False,
        segmentation_metrics_computed=False, autoencoder_used=False,
        fitting_policy="Only manifest['train'] may enter new refiner fitting",
        checkpoint_metadata_note="Embedded fold protocol.split records the outer split; fit_images records actual fold fitting IDs",
        limitations=[
            "Retained fold checkpoint selection used the original validation patients",
            "Fold baselines fit fewer images than the full-training test-map baseline",
            "Outer cohorts have been examined in earlier studies; this is not a new external test",
            "Fit exclusion is supported by retained metadata/source; replay verifies map identity, not every historical optimizer access",
        ],
    )

    try:
        full = reference.LNet_RA(in_channels=1).to(reference.device)
        checkpoint = torch.load(reference.baseline_checkpoint, map_location=reference.device, weights_only=False)
        _require(checkpoint.get("protocol", {}).get("split") == split, "Full checkpoint embedded split differs")
        full.load_state_dict(checkpoint["state"])
        full.eval().requires_grad_(False)
        full_maps: dict[str, Any] = {}
        with torch.inference_mode():
            for name, sample in data.items():
                array = full(sample["x"][None].to(reference.device)).cpu()[0].numpy().copy()
                _check_probability(array, f"full baseline {name}")
                full_maps[name] = array
            for name in split["test"]:
                with np.load(reference.baseline_checkpoint.parent / f"{name}_reference.npz", allow_pickle=False) as retained:
                    _require(np.array_equal(retained["coarse"], full_maps[name]), f"Archived full-test coarse mismatch: {name}")
                    _require(np.array_equal(retained["ground_truth"], data[name]["g"].numpy()), f"Archived test GT mismatch: {name}")
                    _require(np.array_equal(retained["fov"], np.ones_like(data[name]["g"].numpy(), dtype=bool)),
                             f"Archived test FOV mismatch: {name}")
        del full, checkpoint
        if reference.device.type == "cuda":
            torch.cuda.empty_cache()
        for name in data:
            # torch.from_numpy outside inference mode supplies ordinary tensors
            # to the later training code rather than inference-only tensors.
            data[name]["coarse"] = torch.from_numpy(full_maps[name].copy())
        report["full_baseline_map_sha256"] = {name: _array_hash(array) for name, array in full_maps.items()}
        report["archived_test_coarse_gt_fov_exact"] = True

        with np.load(merged_path, allow_pickle=False) as merged:
            _require(set(merged.files) == set(split["train"]), "Merged OOF keys differ from training IDs")
            oof_maps = {name: merged[name].copy() for name in merged.files}
        for name, array in oof_maps.items():
            _check_probability(array, f"merged OOF {name}")

        fold_reports = []
        for index, held in enumerate(folds, start=1):
            folder = OOF_ROOT / f"fold_{index}"
            cp_path = folder / "baseline.pt"
            inventory_path = folder / f"fold_inventory_{OOF_TAG}.json"
            predictions_path = folder / f"held_out_predictions_{OOF_TAG}.npz"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
            fit = [name for name in split["train"] if name not in set(held)]
            _require(inventory["held_out"] == held and inventory["fit"] == fit,
                     f"Fold {index} inventory fit/held IDs differ")
            _require(inventory["validation"] == split["validation"] and inventory.get("test_used_for_selection") is False,
                     f"Fold {index} selection inventory mismatch")
            _require(sha256_file(cp_path) == inventory["checkpoint_sha256"], f"Fold {index} checkpoint hash differs from retained inventory")
            held_subjects, fit_subjects = _subjects(held, data), _subjects(fit, data)
            _require(len(held) == 4 and len(held_subjects) == 2 and len(fit) == 12,
                     f"Fold {index} unexpected fit/held sizes")
            _require(not held_subjects & (fit_subjects | outer_subjects["validation"] | outer_subjects["test"]),
                     f"Fold {index} held-out patient overlaps fitting/selection/test")
            saved = torch.load(cp_path, map_location=reference.device, weights_only=False)
            embedded = saved.get("protocol", {})
            for key, expected in (("dataset", "CHASE"), ("training_seed", 42), ("split", split),
                                  ("fold", index), ("fit_images", fit), ("held_out_images", held),
                                  ("validation_images", split["validation"])):
                _require(embedded.get(key) == expected, f"Fold {index} embedded checkpoint metadata mismatch: {key}")
            model = reference.LNet_RA(in_channels=1).to(reference.device)
            model.load_state_dict(saved["state"])
            model.eval().requires_grad_(False)
            per_image = []
            with np.load(predictions_path, allow_pickle=False) as held_archive:
                _require(set(held_archive.files) == set(held), f"Fold {index} held archive keys differ")
                with torch.inference_mode():
                    for name in held:
                        archived = held_archive[name]
                        _check_probability(archived, f"fold {index} archived {name}")
                        _require(np.array_equal(archived, oof_maps[name]), f"Fold/merged map disagreement: {name}")
                        replayed = model(data[name]["x"][None].to(reference.device)).cpu()[0].numpy()
                        _check_probability(replayed, f"fold {index} replay {name}")
                        _require(np.array_equal(replayed, archived), f"Fold checkpoint replay not bitwise exact: {name}")
                        per_image.append(dict(
                            image=name, subject=int(data[name]["subject"]),
                            held_out_prediction_exact=True, merged_prediction_exact=True,
                            dtype="float32", shape=list(archived.shape),
                            probability_bytes_sha256=_array_hash(archived),
                            checkpoint_sha256=before[str(cp_path)],
                        ))
            fold_reports.append(dict(
                fold=index, checkpoint=str(cp_path), inventory=str(inventory_path),
                held_predictions=str(predictions_path), fit_images=fit, held_out_images=held,
                validation_images=split["validation"], test_images_excluded_from_fitting_and_selection=split["test"],
                fit_subjects=sorted(fit_subjects), held_out_subjects=sorted(held_subjects),
                validation_subjects=sorted(outer_subjects["validation"]),
                selected_epoch=int(saved["epoch"]), images=per_image,
                embedded_fit_held_validation_metadata_verified=True,
            ))
            del model, saved
            if reference.device.type == "cuda":
                torch.cuda.empty_cache()

        for name in split["train"]:
            data[name]["coarse"] = torch.from_numpy(oof_maps[name].copy())
        for partition in ("validation", "test"):
            _require(all(np.array_equal(data[name]["coarse"].numpy(), full_maps[name]) for name in split[partition]),
                     f"{partition} coarse maps were changed by OOF substitution")
        after = {str(path): sha256_file(path) for path in sorted(protected_paths)}
        _require(before == after, "Historical source artifact changed during OOF audit")
        report.update(
            status="passed", completed_at=stamp(), folds=fold_reports,
            fold_checkpoint_replays=4, bitwise_verified_oof_images=16,
            training_maps_source="audited retained patient-grouped OOF maps",
            validation_test_maps_source="unchanged full-training baseline seed 42",
            only_training_coarse_replaced=True, source_sha256_after=after,
            historical_sources_unchanged=True,
            environment=dict(torch=torch.__version__, cuda=torch.version.cuda,
                             device=str(reference.device),
                             gpu=torch.cuda.get_device_name(reference.device) if reference.device.type == "cuda" else None),
        )
        _write_new(audit_path, report)
        metadata = dict(reference.source_metadata)
        metadata.update(
            baseline_inference_performed=True, oof_checkpoint_replay_performed=True,
            oof_audit_path=str(audit_path), oof_audit_sha256=sha256_file(audit_path),
            coarse_training_source="audited OOF seed42 maps; held-out patients excluded from fold fitting/selection",
            coarse_validation_test_source="full-training baseline seed42",
            autoencoder_inference_performed=False,
        )
        return replace(reference, source_metadata=metadata), report
    except BaseException as error:
        report.update(status="failed", failed_at=stamp(), error_type=type(error).__name__, error=str(error))
        if not audit_path.exists():
            _write_new(audit_path, report)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="Explicitly replay retained baseline checkpoints and audit OOF inputs")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.audit:
        print("No data loaded or inference performed. Add --audit to execute the source audit.")
    else:
        destination = args.workdir or ROOT / "outputs" / f"esfm_oof_data_audit_{stamp()}"
        _reference, _report = load_chase_oof(destination, args.device)
        print(json.dumps(dict(status=_report["status"], audit=_report["output_audit"],
                              folds=_report["fold_checkpoint_replays"],
                              verified_oof_images=_report["bitwise_verified_oof_images"]), indent=2))
