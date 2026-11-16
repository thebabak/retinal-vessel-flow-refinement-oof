"""Audited native RGB/annotation inputs with explicitly low-resolution coarse maps.

Imports use the standard library only. ``load_native_dataset`` explicitly calls
the unchanged audited OOF loader, which replays retained 512-grid baselines but
never trains. This module then reads the exact original annotation versions,
checks their 512 conversion against the reference, and adds CPU native fields.
It never promotes a resized 512 label to native ground truth, evaluates a metric,
fits a model, caches native arrays on disk, or modifies an earlier artifact.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import current, raw_root, resolve_recorded_path, assert_output_path

import base64
from dataclasses import replace
import io
import json
from pathlib import Path
from typing import Any
import zlib

from flow_matching_refiner.data import ROOT, REFERENCE_ROOT, ReferenceData, sha256_file
from flow_matching_refiner.oof_data import _array_hash, _require, _write_new, stamp
from flow_matching_refiner.oof_data_all import load_dataset_oof


DATASETS = ("CHASE", "DRIVE", "HRF")
FIELDS_512 = ("x", "rgb", "g", "fov", "coarse")
NATIVE_SCHEMA = dict(version=1, tensor_device="cpu", fields=dict(
    n_rgb="torch.float32 [3,H,W] in [0,1]", n_g="torch.float32 [1,H,W] binary; None only for unavailable native test GT",
    n_fov="torch.bool [1,H,W]", n_coarse="torch.float32 [1,H,W] bilinear projected low-resolution probabilities"))


def _source_inventory(dataset: str) -> tuple[dict, dict, set[Path]]:
    """Resolve only source versions already bound to the fixed seed-42 run."""
    folder = REFERENCE_ROOT / dataset / "train_seed_42"
    protocol_path = folder / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    names = [n for part in protocol["split"].values() for n in part]
    _require(len(names) == len(set(names)), "Repeated source image ID")
    protected = {protocol_path, Path(__file__).resolve()}
    if dataset == "CHASE":
        records = {name: dict(image_path=str(raw_root("CHASE") / f"{name}.jpg"),
                             gt_path=str(raw_root("CHASE") / f"{name}_1stHO.png"),
                             fov_path=None) for name in names}
    else:
        path = folder / "file_provenance.json"
        records = json.loads(path.read_text(encoding="utf-8-sig"))
        records = {name: {key: str(resolve_recorded_path(value)) if value is not None else None
                          for key, value in record.items()} for name, record in records.items()}
        protected.add(path)
        _require(set(records) == set(names), "Retained raw-source inventory differs")
    for record in records.values():
        protected.update(Path(record[key]).resolve() for key in ("image_path", "gt_path", "fov_path")
                         if record.get(key) is not None)
    for path in protected:
        _require(path.is_file(), f"Required retained raw source missing: {path}")
    return protocol, records, protected


def _assert_workspace(workdir: Path, records: dict) -> None:
    protected_roots = [current().artifact_root, *(current().raw(k) for k in ("CHASE", "DRIVE", "DRIVE_TEST_GT", "HRF")),
                       ROOT / "paired_reproduction", ROOT / "revise",
                       ROOT / "diffusion_refiner"]
    protected_roots.extend(Path(record[key]).resolve().parent for record in records.values()
                           for key in ("image_path", "gt_path", "fov_path") if record.get(key))
    _require(not any(workdir.is_relative_to(path.resolve()) for path in protected_roots),
             "Native audit output must not be inside a retained input/source folder")


def _bitmap_canvas(obj: dict, size: dict) -> tuple[Any, dict]:
    """Reconstruct the retained Supervisely indexed bitmap without resizing."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(zlib.decompress(base64.b64decode(obj["bitmap"]["data"])))) as image:
        patch = np.asarray(image) > 0
    _require(patch.ndim == 2, "Expected the retained indexed binary bitmap")
    width, height = int(size["width"]), int(size["height"])
    x, y = obj["bitmap"]["origin"]
    _require(isinstance(x, int) and isinstance(y, int) and width > 0 and height > 0,
             "Invalid bitmap canvas/origin")
    _require(0 <= x <= width - patch.shape[1] and 0 <= y <= height - patch.shape[0],
             "Bitmap patch falls outside its declared native canvas")
    canvas = np.zeros((height, width), dtype=np.uint8)
    canvas[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
    return canvas, dict(origin_xy=[x, y], patch_width_height=[patch.shape[1], patch.shape[0]],
                        canvas_width_height=[width, height], positive_patch_pixels=int(patch.sum()))


def _read_native_arrays(dataset: str, record: dict, split: str) -> tuple[dict, dict]:
    """Read raw pixels, retaining the original PIL/cv2 512 conversion ordering."""
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(record["image_path"]) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    height, width = rgb.shape[:2]
    info = dict(image_path=record["image_path"], gt_path=record["gt_path"],
                fov_path=record.get("fov_path"), split=split,
                native_rgb_width_height=[width, height], annotation_origin=None)
    if dataset == "HRF":
        doc = json.loads(Path(record["gt_path"]).read_text(encoding="utf-8-sig"))
        objects = {o["classTitle"]: o for o in doc["objects"]}
        _require("vessels" in objects and "field of view" in objects, "Required HRF annotation class missing")
        mask, mask_origin = _bitmap_canvas(objects["vessels"], doc["size"])
        fov, fov_origin = _bitmap_canvas(objects["field of view"], doc["size"])
        info["annotation_origin"] = dict(vessels=mask_origin, field_of_view=fov_origin,
                                         reconstruction="place indexed >0 bitmap at origin on full uint8 canvas")
        small_g = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
        small_fov = cv2.resize(fov, (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
        mask, fov = mask.astype(bool), fov.astype(bool)
    else:
        with Image.open(record["gt_path"]) as image:
            luminance = image.convert("L")
            mask = np.asarray(luminance) > 127
            small_g = np.asarray(luminance.resize((512, 512), Image.Resampling.NEAREST)) > 127
        if dataset == "CHASE":
            fov = np.ones((height, width), dtype=bool)
            small_fov = np.ones((512, 512), dtype=bool)
            info["annotation_origin"] = dict(observer="1stHO", threshold="PIL L >127",
                                              fov="whole native RGB canvas; protocol has no FOV mask")
        else:
            with Image.open(record["fov_path"]) as image:
                luminance = image.convert("L")
                fov = np.asarray(luminance) > 127
                small_fov = np.asarray(luminance.resize((512, 512), Image.Resampling.NEAREST)) > 127
            info["annotation_origin"] = dict(observer="retained manual1 GIF for train/validation; local manual1 PNG for test",
                                              threshold="PIL L >127; no new annotation conversion")
    native_gt_available = mask.shape == (height, width)
    _require(native_gt_available or split == "test", "Native training/validation GT is not RGB-aligned; do not upscale labels")
    _require(fov.shape == (height, width) and bool(fov.any()), "Native FOV is missing, empty or not RGB-aligned")
    info.update(native_gt_width_height=[mask.shape[1], mask.shape[0]],
                native_fov_width_height=[fov.shape[1], fov.shape[0]],
                native_gt_available=native_gt_available,
                native_gt_unavailable_reason=None if native_gt_available else "retained test label is not native RGB size; no upscaling permitted",
                native_valid_pixels=int(fov.sum()), source_gt_positive_pixels=int(mask.sum()),
                native_gt_positive_pixels=int(mask.sum()) if native_gt_available else None,
                native_gt_positive_inside_fov=int((mask & fov).sum()) if native_gt_available else None)
    return dict(rgb=rgb, g=mask if native_gt_available else None, fov=fov,
                small_g=small_g, small_fov=small_fov), info


def _verify_reference_512(arrays: dict, sample: dict, name: str) -> dict:
    """Replay the exact legacy conversions; do not evaluate segmentation."""
    import cv2
    import numpy as np
    import torch

    rgb = arrays["rgb"]
    color = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    green = cv2.createCLAHE(clipLimit=5., tileGridSize=(32, 32)).apply(rgb[:, :, 1])
    green = np.uint8(np.clip((green / 255.) ** (1 / 1.2) * 255, 0, 255))
    green = cv2.resize(green, (512, 512), interpolation=cv2.INTER_LINEAR)
    replay = dict(rgb=torch.tensor(color / 255., dtype=torch.float32).permute(2, 0, 1),
                  x=torch.tensor(green / 255., dtype=torch.float32)[None],
                  g=torch.tensor(arrays["small_g"], dtype=torch.float32)[None],
                  fov=torch.tensor(arrays["small_fov"], dtype=torch.bool)[None])
    for key, expected in replay.items():
        _require(torch.equal(expected, sample[key].cpu()), f"Native-source {key} does not replay retained 512 pixels: {name}")
    return {key: _array_hash(value.numpy()) for key, value in replay.items()}


def _hash_512(data: dict) -> dict:
    return {name: {key: _array_hash(sample[key].detach().cpu().numpy()) for key in FIELDS_512 if key in sample}
            for name, sample in data.items()}


def _native_tensors(arrays: dict, coarse: Any) -> dict:
    import numpy as np
    import torch
    from torch.nn import functional as F

    height, width = arrays["rgb"].shape[:2]
    _require(coarse.dtype == torch.float32 and tuple(coarse.shape) == (1, 512, 512), "Expected retained FP32 512 coarse map")
    projected = F.interpolate(coarse.detach().cpu()[None], size=(height, width),
                              mode="bilinear", align_corners=False)[0].contiguous()
    _require(bool(torch.isfinite(projected).all()) and bool(((projected >= 0) & (projected <= 1)).all()),
             "Invalid projected coarse probabilities")
    # Match the legacy RGB division/conversion order, while keeping CPU fields.
    rgb = torch.from_numpy((arrays["rgb"] / 255.).astype(np.float32)).permute(2, 0, 1).contiguous()
    return dict(n_rgb=rgb,
                n_g=None if arrays["g"] is None else torch.from_numpy(arrays["g"].astype(np.float32))[None],
                n_fov=torch.from_numpy(arrays["fov"].copy())[None], n_coarse=projected)


def load_native_dataset(dataset: str, workdir: str | Path, device: str = "cuda") -> tuple[ReferenceData, dict]:
    """Add native CPU inputs to the unchanged audited 512-grid OOF reference.

    Only ``manifest['train']`` may supervise fitting. Validation selects on the
    original 512 targets; test remains locked until all choices are fixed. The
    native coarse map is always a bilinear projection of the retained 512 map,
    never a newly fitted or native-resolution baseline prediction. Genuine
    native test GT may be None if its retained source grid differs from RGB.
    """
    if dataset not in DATASETS:
        raise ValueError("dataset must be CHASE, DRIVE or HRF")
    protocol, records, protected = _source_inventory(dataset)
    workdir = Path(workdir).resolve()
    assert_output_path(workdir)
    _assert_workspace(workdir, records)
    workdir.mkdir(parents=True, exist_ok=True)
    tag = stamp()
    path = workdir / f"{dataset}_native_source_audit_{tag}.json"
    _require(not path.exists(), f"Audit already exists: {path}")
    before = {str(p): sha256_file(p) for p in sorted(protected)}
    audit = dict(status="auditing", started_at=tag, dataset=dataset, baseline_seed=42,
                 output_audit=str(path), source_sha256_before=before,
                 native_coarse_source="bilinear projection of retained audited 512 OOF train/full-baseline validation+test maps",
                 coarse_projection=dict(mode="bilinear", align_corners=False, source_grid=[512, 512]),
                 native_tensor_device="cpu", native_array_caches_written=False,
                 training_performed=False, segmentation_metrics_computed=False,
                 native_baseline_inference_performed=False,
                 native_patch_schema=NATIVE_SCHEMA,
                 primary_evaluation="native stitched logit residual projected with GT-matched nearest (PIL float32 for CHASE/DRIVE; cv2 for HRF), added to original 512 coarse logits; original GT/FOV retained",
                 secondary_native_evaluation_performed=False,
                 fitting_policy="Only train native RGB/GT/FOV/OOF coarse patches may supervise fitting",
                 limitations=["Coarse maps remain low-resolution projections, not native baseline predictions",
                              "Raw label resolution does not establish upstream annotation acquisition provenance",
                              "Previously examined cohorts and historical OOF validation choices are retained"])
    try:
        reference, base_audit = load_dataset_oof(dataset, workdir / "reference_512", device)
        _require(protocol["split"] == reference.manifest and set(records) == set(reference.data), "Native and OOF split inventories differ")
        for source, digest in base_audit["source_sha256_before"].items():
            _require(source not in before or before[source] == digest, "Raw source changed during reference load")
            before[source] = digest
        # The unchanged CHASE loader has no FOV field: its protocol is whole
        # image. Add canonical all-valid FOV to new dictionaries only.
        original_512 = _hash_512(reference.data)
        canonical = {name: dict(sample) for name, sample in reference.data.items()}
        if dataset == "CHASE":
            import torch
            for sample in canonical.values():
                sample.setdefault("fov", torch.ones_like(sample["g"], dtype=torch.bool))
        reference = replace(reference, data=canonical)
        before_512 = _hash_512(reference.data)
        _require(all(before_512[n][key] == digest for n, fields in original_512.items() for key, digest in fields.items()),
                 "Canonical FOV addition changed an original 512 field")
        enriched, inventory = {}, {}
        for name, sample in reference.data.items():
            split = next(k for k, values in reference.manifest.items() if name in values)
            record = records[name]
            if dataset != "CHASE":
                _require(all(Path(sample[key]).resolve() == Path(record[key]).resolve()
                             for key in ("image_path", "gt_path", "fov_path")), "Native source path differs from loaded legacy source")
            arrays, info = _read_native_arrays(dataset, record, split)
            info["exact_512_replay_sha256"] = _verify_reference_512(arrays, sample, name)
            info["source_512_coarse_sha256"] = before_512[name]["coarse"]
            enriched[name] = dict(sample, **_native_tensors(arrays, sample["coarse"]),
                                  native_gt_available=info["native_gt_available"],
                                  native_shape=tuple(arrays["rgb"].shape[:2]))
            inventory[name] = info
            del arrays
        _require(_hash_512(reference.data) == before_512 and _hash_512(enriched) == before_512,
                 "Original 512 reference fields changed")
        after = {source: sha256_file(source) for source in before}
        _require(after == before, "A protected raw source/checkpoint changed during native loading")
        availability = {part: dict(images=len(names), available=sum(inventory[n]["native_gt_available"] for n in names),
                                   unavailable=[n for n in names if not inventory[n]["native_gt_available"]])
                        for part, names in reference.manifest.items()}
        audit.update(status="passed", completed_at=stamp(), source_sha256_after=after,
                     split=reference.manifest, native_inventory=inventory,
                     native_gt_availability=availability,
                     all_native_gt_available=all(info["native_gt_available"] for info in inventory.values()),
                     original_512_fields_sha256=before_512, original_512_fields_unchanged=True,
                     CHASE_512_fov_added_from_whole_image_protocol=dataset == "CHASE",
                     exact_512_RGB_GT_FOV_green_replay_verified=True, historical_sources_unchanged=True,
                     reference_512_audit=base_audit["output_audit"],
                     reference_512_audit_sha256=sha256_file(base_audit["output_audit"]),
                     retained_512_baseline_checkpoint_replay_performed=True,
                     native_shapes_height_width=sorted({tuple(info["native_rgb_width_height"][::-1]) for info in inventory.values()}))
        _write_new(path, audit)
        metadata = dict(reference.source_metadata, native_source_audit=str(path),
                        native_source_audit_sha256=sha256_file(path), native_fields_device="cpu",
                        native_patch_schema_version=NATIVE_SCHEMA["version"],
                        native_coarse_is_projected_512=True,
                        all_native_gt_available=audit["all_native_gt_available"])
        return replace(reference, data=enriched, namespace=dict(reference.namespace, data=enriched, native_patch_schema=NATIVE_SCHEMA), source_metadata=metadata), audit
    except BaseException as error:
        audit.update(status="failed", failed_at=stamp(), error_type=type(error).__name__, error=str(error))
        if not path.exists():
            _write_new(path, audit)
        raise
