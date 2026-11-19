"""Audited retained OOF inputs for the DRIVE/HRF ESFM extension.

Imports are inert and use the standard library only.  The explicit loader
replays existing full/fold baseline checkpoints but never fits a model or
computes segmentation metrics.  Prediction arrays remain in memory; only a
new timestamped provenance JSON is written.  The CHASE implementation and its
completed experiment remain unchanged.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from flow_matching_refiner.data import ROOT, ReferenceData, load_reference, sha256_file
from flow_matching_refiner.oof_data import (
    _array_hash, _check_probability, _require, _write_new, load_chase_oof, stamp,
)


OOF_TAG = "2026-10-09_23-30-40"
OOF_BASE = ROOT / "revise" / f"oof_comparison_{OOF_TAG}"
FULL_BASE = ROOT / "paired_reproduction" / "all_datasets_five_full_runs_2026-10-08_15-38-24"
SPECS = {
    "DRIVE": dict(sizes=(16, 4, 20), folds=4, held_size=4, fit_size=12,
                  source="run_drive.py", fold_rule="four image folds"),
    "HRF": dict(sizes=(27, 6, 12), folds=3, held_size=9, fit_size=18,
                source="run_hrf_baseline.py", fold_rule="three pathology-stratified image folds"),
}


def validate_membership(dataset: str, split: dict[str, list[str]], folds: list[list[str]]) -> None:
    """Check exact image inventories without assuming unknown patient identities."""
    spec = SPECS[dataset]
    _require(set(split) == {"train", "validation", "test"}, "Unexpected outer partition keys")
    _require(tuple(len(split[k]) for k in ("train", "validation", "test")) == spec["sizes"],
             f"Unexpected {dataset} outer split sizes")
    for key, names in split.items():
        _require(len(names) == len(set(names)), f"Duplicate image IDs in {key}")
    groups = {key: set(names) for key, names in split.items()}
    _require(not any(groups[a] & groups[b] for a, b in
                     (("train", "validation"), ("train", "test"), ("validation", "test"))),
             "Outer image partitions overlap")
    _require(isinstance(folds, list) and len(folds) == spec["folds"],
             f"Unexpected {dataset} fold count")
    held_all = [name for held in folds for name in held]
    _require(len(held_all) == len(set(held_all)) == len(split["train"])
             and set(held_all) == groups["train"], "OOF folds must cover every training image exactly once")
    for index, held in enumerate(folds, start=1):
        _require(len(held) == spec["held_size"], f"Unexpected held-out size in fold {index}")
    if dataset == "HRF":
        for key, expected in (("train", 9), ("validation", 2), ("test", 4)):
            counts = Counter(name.rsplit("_", 1)[-1] for name in split[key])
            _require(counts == {"h": expected, "dr": expected, "g": expected},
                     f"HRF outer {key} pathology counts differ")
        for index, held in enumerate(folds, start=1):
            counts = Counter(name.rsplit("_", 1)[-1] for name in held)
            _require(counts == {"h": 3, "dr": 3, "g": 3}, f"HRF fold {index} is not pathology-stratified")


def load_dataset_oof(
    dataset: str, workdir: str | Path, device: str = "cuda",
) -> tuple[ReferenceData, dict[str, Any]]:
    """Return data with audited seed-42 OOF maps replacing training inputs only.

    DRIVE and HRF use image-level test units: their retained metadata does not
    verify patient independence.  Existing numeric DRIVE ``subject`` fields
    are image identifiers, not newly verified patient IDs.  Both datasets retain
    their actual FOV masks, including for archival identity comparisons.

    CHASE delegates to the existing unchanged patient-grouped loader.  No new
    baseline training, statistics, or annotation conversion is performed.
    """
    if dataset == "CHASE":
        return load_chase_oof(workdir, device)
    if dataset not in SPECS:
        raise ValueError("dataset must be CHASE, DRIVE or HRF")
    import numpy as np
    import torch

    spec = SPECS[dataset]
    oof_root = OOF_BASE / dataset / "seed_42"
    workdir = Path(workdir).resolve()
    for historical in (OOF_BASE, FULL_BASE):
        _require(not workdir.is_relative_to(historical.resolve()), "New audit must not be inside historical inputs")
    workdir.mkdir(parents=True, exist_ok=True)
    tag = stamp()
    audit_path = workdir / f"{dataset}_OOF_source_audit_{tag}.json"
    _require(not audit_path.exists(), f"Audit already exists: {audit_path}")
    protocol_path = oof_root / f"oof_protocol_{OOF_TAG}.json"
    merged_path = oof_root / f"oof_training_maps_{OOF_TAG}.npz"
    _require(protocol_path.is_file() and merged_path.is_file(), "Retained OOF artifacts missing; no retraining fallback")
    oof_protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    _require(oof_protocol.get("dataset") == dataset and oof_protocol.get("training_seed") == 42,
             "OOF dataset/seed mismatch")
    validate_membership(dataset, oof_protocol["split"], oof_protocol["folds"])
    reference = load_reference(dataset, 42, workdir / "loader_workspace", device)
    data, split = reference.data, reference.manifest
    _require(oof_protocol["split"] == split, "OOF split and full-baseline split differ")
    _require(set(data) == {name for names in split.values() for name in names}, "Loaded data inventory differs")
    folds = oof_protocol["folds"]
    for name, sample in data.items():
        _require("fov" in sample and sample["fov"].shape == sample["g"].shape == (1, 512, 512),
                 f"Missing/invalid FOV or GT shape: {name}")
        _require(sample["fov"].dtype == torch.bool and bool(sample["fov"].any()), f"Invalid FOV: {name}")
        _require(bool(((sample["g"] == 0) | (sample["g"] == 1)).all()), f"Nonbinary GT: {name}")

    protected = {protocol_path, merged_path, reference.baseline_checkpoint,
                 reference.autoencoder_checkpoint, reference.protocol_path}
    sources = [
        Path(__file__), ROOT / "flow_matching_refiner" / "data.py",
        ROOT / "flow_matching_refiner" / "oof_data.py",
        ROOT / "paired_reproduction" / "official_model.py",
        ROOT / "paired_reproduction" / spec["source"],
        ROOT / "paired_reproduction" / "run.py",
        ROOT / "paired_reproduction" / "run_all_datasets_five_full_runs.py",
        ROOT / "paired_reproduction" / "run_oof_comparison.py",
    ]
    protected.update(sources)
    file_inventory = {}
    for name, sample in data.items():
        file_inventory[name] = {key: str(sample[key]) for key in ("image_path", "gt_path", "fov_path")}
        protected.update(Path(value) for value in file_inventory[name].values())
    provenance_path = reference.baseline_checkpoint.parent / "file_provenance.json"
    if provenance_path.is_file():
        protected.add(provenance_path)
    for index in range(1, len(folds) + 1):
        folder = oof_root / f"fold_{index}"
        protected.update((folder / "baseline.pt", folder / f"fold_inventory_{OOF_TAG}.json",
                          folder / f"held_out_predictions_{OOF_TAG}.npz"))
    for name in split["test"]:
        protected.add(reference.baseline_checkpoint.parent / f"{name}_reference.npz")
    for path in protected:
        _require(path.is_file(), f"Required source missing: {path}")
    before = {str(path): sha256_file(path) for path in sorted(protected)}
    report: dict[str, Any] = dict(
        status="auditing", started_at=tag, dataset=dataset, baseline_seed=42,
        split=split, outer_partition_sizes={k: len(v) for k, v in split.items()},
        test_unit="image", patient_independence_verified=False, fold_rule=spec["fold_rule"],
        numeric_subject_field_role="historical image identifier only; not a verified patient ID",
        output_audit=str(audit_path), oof_protocol=str(protocol_path), merged_maps=str(merged_path),
        source_sha256_before=before, raw_source_paths=file_inventory,
        fold_training_performed=False, refiner_training_performed=False,
        segmentation_metrics_computed=False, autoencoder_used=False,
        checkpoint_metadata_note="Embedded protocol.split is the outer split; actual fold fit IDs are in fit_images",
        ground_truth_and_fov_source=reference.protocol.get("annotation_source"),
        annotation_provenance=reference.protocol.get("annotation_provenance"),
        region=reference.protocol.get("region"),
        limitations=[
            "Image-disjoint folds do not establish patient independence",
            "Retained fold checkpoints were selected using the outer validation images",
            "Fold baselines fit fewer images than the full baseline producing validation/test maps",
            "Earlier examined cohorts are reused; no new external cohort is introduced",
            "Upstream local annotation conversion/acquisition provenance remains as documented in the retained protocol",
            "Checkpoint replay establishes prediction identity, not every historical optimizer access",
        ],
    )
    try:
        full = reference.LNet_RA(in_channels=1).to(reference.device)
        checkpoint = torch.load(reference.baseline_checkpoint, map_location=reference.device, weights_only=False)
        _require(checkpoint.get("protocol", {}).get("split") == split, "Full checkpoint split differs")
        full.load_state_dict(checkpoint["state"])
        full.eval().requires_grad_(False)
        full_maps = {}
        with torch.inference_mode():
            for name, sample in data.items():
                array = full(sample["x"][None].to(reference.device)).cpu()[0].numpy().copy()
                _check_probability(array, f"full baseline {dataset}/{name}")
                full_maps[name] = array
            for name in split["test"]:
                with np.load(reference.baseline_checkpoint.parent / f"{name}_reference.npz", allow_pickle=False) as archive:
                    _require(np.array_equal(archive["coarse"], full_maps[name]), f"Archived test coarse mismatch: {name}")
                    _require(np.array_equal(archive["ground_truth"], data[name]["g"].numpy()), f"Archived test GT mismatch: {name}")
                    _require(np.array_equal(archive["fov"], data[name]["fov"].numpy()), f"Archived test FOV mismatch: {name}")
        del full, checkpoint
        if reference.device.type == "cuda":
            torch.cuda.empty_cache()
        for name in data:
            data[name]["coarse"] = torch.from_numpy(full_maps[name].copy())
        report["full_baseline_map_sha256"] = {name: _array_hash(array) for name, array in full_maps.items()}
        report["ground_truth_and_fov_array_sha256"] = {
            name: {"gt": _array_hash(sample["g"].numpy()), "fov": _array_hash(sample["fov"].numpy())}
            for name, sample in data.items()
        }
        report["archived_test_coarse_gt_fov_exact"] = True
        with np.load(merged_path, allow_pickle=False) as merged:
            _require(set(merged.files) == set(split["train"]), "Merged map keys differ from training IDs")
            oof_maps = {name: merged[name].copy() for name in merged.files}
        for name, array in oof_maps.items():
            _check_probability(array, f"merged OOF {dataset}/{name}")

        fold_reports = []
        for index, held in enumerate(folds, start=1):
            folder = oof_root / f"fold_{index}"
            cp_path = folder / "baseline.pt"
            inv_path = folder / f"fold_inventory_{OOF_TAG}.json"
            maps_path = folder / f"held_out_predictions_{OOF_TAG}.npz"
            inventory = json.loads(inv_path.read_text(encoding="utf-8-sig"))
            fit = [name for name in split["train"] if name not in set(held)]
            _require(len(fit) == spec["fit_size"], f"Fold {index} fit size differs")
            _require(inventory["held_out"] == held and inventory["fit"] == fit, f"Fold {index} inventory IDs differ")
            _require(inventory["validation"] == split["validation"] and inventory.get("test_used_for_selection") is False,
                     f"Fold {index} checkpoint selection inventory differs")
            _require(not set(held) & (set(fit) | set(split["validation"]) | set(split["test"])),
                     f"Fold {index} held-out images overlap fitting/selection/test")
            _require(not set(split["test"]) & (set(fit) | set(inventory["validation"])),
                     f"Fold {index} test images used for fitting/selection")
            _require(sha256_file(cp_path) == inventory["checkpoint_sha256"], f"Fold {index} checkpoint hash differs")
            checkpoint = torch.load(cp_path, map_location=reference.device, weights_only=False)
            embedded = checkpoint.get("protocol", {})
            for key, expected in (("dataset", dataset), ("training_seed", 42), ("split", split),
                                  ("fold", index), ("fit_images", fit), ("held_out_images", held),
                                  ("validation_images", split["validation"])):
                _require(embedded.get(key) == expected, f"Fold {index} embedded metadata differs: {key}")
            model = reference.LNet_RA(in_channels=1).to(reference.device)
            model.load_state_dict(checkpoint["state"])
            model.eval().requires_grad_(False)
            images = []
            with np.load(maps_path, allow_pickle=False) as archive:
                _require(set(archive.files) == set(held), f"Fold {index} prediction keys differ")
                with torch.inference_mode():
                    for name in held:
                        archived = archive[name]
                        _check_probability(archived, f"fold {index}/{name}")
                        _require(np.array_equal(archived, oof_maps[name]), f"Fold/merged map mismatch: {name}")
                        replay = model(data[name]["x"][None].to(reference.device)).cpu()[0].numpy()
                        _check_probability(replay, f"replayed fold {index}/{name}")
                        _require(np.array_equal(replay, archived), f"Fold replay is not exact FP32: {name}")
                        images.append(dict(image=name, held_out_prediction_exact=True, merged_prediction_exact=True,
                                           dtype="float32", shape=list(archived.shape),
                                           probability_bytes_sha256=_array_hash(archived)))
            fold_reports.append(dict(
                fold=index, checkpoint=str(cp_path), inventory=str(inv_path), held_predictions=str(maps_path),
                checkpoint_sha256=before[str(cp_path)], selected_epoch=int(checkpoint["epoch"]),
                fit_images=fit, held_out_images=held, validation_images=split["validation"],
                test_images_excluded_from_fitting_and_selection=split["test"], images=images,
                embedded_fit_held_validation_metadata_verified=True,
            ))
            del model, checkpoint
            if reference.device.type == "cuda":
                torch.cuda.empty_cache()

        for name in split["train"]:
            data[name]["coarse"] = torch.from_numpy(oof_maps[name].copy())
        for key in ("validation", "test"):
            _require(all(np.array_equal(data[name]["coarse"].numpy(), full_maps[name]) for name in split[key]),
                     f"OOF substitution changed {key} coarse maps")
        after = {str(path): sha256_file(path) for path in sorted(protected)}
        _require(before == after, "Protected source changed during audit")
        report.update(
            status="passed", completed_at=stamp(), folds=fold_reports,
            fold_checkpoint_replays=len(folds), bitwise_verified_oof_images=len(split["train"]),
            only_training_coarse_replaced=True, training_maps_source="audited retained image-disjoint OOF maps",
            validation_test_maps_source="unchanged full-training baseline seed42",
            source_sha256_after=after, historical_sources_unchanged=True,
            prediction_arrays_written=False,
            environment=dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(reference.device),
                             gpu=torch.cuda.get_device_name(reference.device) if reference.device.type == "cuda" else None),
        )
        _write_new(audit_path, report)
        metadata = dict(reference.source_metadata)
        metadata.update(
            baseline_inference_performed=True, oof_checkpoint_replay_performed=True,
            oof_audit_path=str(audit_path), oof_audit_sha256=sha256_file(audit_path),
            coarse_training_source=f"audited historical {dataset} OOF seed42 maps; image-disjoint fitting/selection",
            coarse_validation_test_source="full-training baseline seed42",
            autoencoder_inference_performed=False, patient_independence_verified=False,
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
    parser.add_argument("--dataset", choices=["CHASE", "DRIVE", "HRF"], required=True)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.audit:
        print("No data loading or inference. Add --audit to replay and verify retained OOF inputs.")
    else:
        out = args.workdir or ROOT / "outputs" / f"esfm_{args.dataset}_oof_input_audit_{stamp()}"
        _, audit = load_dataset_oof(args.dataset, out, args.device)
        print(json.dumps(dict(status=audit["status"], dataset=args.dataset, audit=audit["output_audit"],
                              verified_oof_images=audit["bitwise_verified_oof_images"]), indent=2))
