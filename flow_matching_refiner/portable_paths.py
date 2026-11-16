"""Standard-library path isolation for the public reproduction copy.

No configuration file, dataset, checkpoint or tensor is read at import time.
Historical digest checks resolve through explicit aliases to original artifact
bytes. They never substitute the adapted source tree for an archived source.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
import hashlib
import json
import os
import re
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
FULL_TAG = "2026-10-08_15-38-24"
OOF_TAG = "2026-10-09_23-30-40"
V3_TAG = "2026-11-15_23-28-26"
CONTROL_TAGS = {"CHASE": "2026-11-15_01-59-17", "DRIVE": "2026-11-15_02-35-09", "HRF": "2026-11-15_02-35-09"}
RAW_NAMES = ("CHASE", "DRIVE", "DRIVE_TEST_GT", "HRF")


def _normal(value: Any) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def recorded_is_absolute(value: Any) -> bool:
    return str(value).startswith("/") or Path(str(value)).is_absolute() or PureWindowsPath(str(value)).is_absolute()


def _under(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


@dataclass(frozen=True)
class PortablePaths:
    artifact_root: Path = CODE_ROOT / "artifacts"
    output_root: Path = CODE_ROOT / "runs"
    raw_roots: dict[str, Path] = field(default_factory=dict)
    controls: dict[str, Path] = field(default_factory=dict)
    v3_study: Path | None = None
    aliases: tuple[tuple[str, Path], ...] = ()

    def raw(self, dataset: str) -> Path:
        if dataset not in RAW_NAMES:
            raise ValueError("Unknown raw-data root: " + dataset)
        return self.raw_roots.get(dataset, self.artifact_root / "raw" / dataset)

    def control(self, dataset: str) -> Path:
        if dataset in self.controls:
            return self.controls[dataset]
        if dataset == "CHASE":
            return self.artifact_root / "outputs" / f"esfm_oof_chase_{CONTROL_TAGS[dataset]}"
        return self.artifact_root / "outputs" / f"esfm_all_datasets_{CONTROL_TAGS[dataset]}" / f"{dataset}_{CONTROL_TAGS[dataset]}"

    @property
    def reference_root(self) -> Path:
        return self.artifact_root / "paired_reproduction" / f"all_datasets_five_full_runs_{FULL_TAG}"

    @property
    def oof_root(self) -> Path:
        return self.artifact_root / "revise" / f"oof_comparison_{OOF_TAG}"

    @property
    def v3_root(self) -> Path:
        return self.v3_study or self.artifact_root / "outputs" / f"refinement_v3_{V3_TAG}"

    def validate(self) -> None:
        if CODE_ROOT.resolve() == self.artifact_root.resolve():
            raise ValueError("artifact_root and CODE_ROOT must be distinct")
        inputs = [self.artifact_root] + [self.raw(k) for k in RAW_NAMES] + list(self.controls.values())
        if self.v3_study:
            inputs.append(self.v3_study)
        if any(_under(self.output_root.resolve(), p.resolve()) for p in inputs):
            raise ValueError("output_root must be outside every retained artifact/raw input root")
        for prefix, target in self.aliases:
            if not prefix or not recorded_is_absolute(prefix):
                raise ValueError("Each recorded alias prefix must be an absolute historical path")
            staged_inside_code = self.artifact_root.resolve().is_relative_to(CODE_ROOT.resolve()) and _under(target.resolve(), self.artifact_root.resolve())
            if _under(target.resolve(), CODE_ROOT.resolve()) and not staged_inside_code:
                raise ValueError("Historical aliases may not target adapted source bytes")
            if not any(_under(target.resolve(), p.resolve()) for p in inputs):
                raise ValueError("Historical aliases must target original artifact/raw bytes, not adapted code")


_CONFIG = PortablePaths()


def current() -> PortablePaths:
    return _CONFIG


def raw_root(dataset: str) -> Path:
    return current().raw(dataset)


def output_root() -> Path:
    return current().output_root


def artifact_root() -> Path:
    return current().artifact_root


def _path(value: str | Path, base: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def configure(config_file: str | Path | None = None) -> PortablePaths:
    """Configure BEFORE importing runners/loaders; existing modules are refused."""
    global _CONFIG
    import sys
    watched = {"flow_matching_refiner.data", "flow_matching_refiner.oof_data",
               "flow_matching_refiner.oof_data_all", "flow_matching_refiner.native_patch_data",
               "flow_matching_refiner.run_refinement_v3", "flow_matching_refiner.run_native_patch"}
    if watched & set(sys.modules):
        raise RuntimeError("Configure paths before importing scientific loaders/runners; start a fresh process")
    document: dict[str, Any] = {}
    base = CODE_ROOT
    filename = config_file or os.environ.get("RETINAL_PATHS_CONFIG")
    if filename:
        path = Path(filename).expanduser().resolve()
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        base = path.parent
    art = _path(os.environ.get("RETINAL_ARTIFACT_ROOT", document.get("artifact_root", CODE_ROOT / "artifacts")), base)
    out = _path(os.environ.get("RETINAL_OUTPUT_ROOT", document.get("output_root", CODE_ROOT / "runs")), base)
    raw = {k: _path(os.environ.get("RETINAL_RAW_" + k, document.get("raw_roots", {}).get(k, art / "raw" / k)), art) for k in RAW_NAMES}
    controls = {k: _path(v, art) for k, v in document.get("controls", {}).items()}
    if not set(controls) <= {"CHASE", "DRIVE", "HRF"}:
        raise ValueError("controls keys must be CHASE, DRIVE or HRF")
    v3 = document.get("v3_study")
    aliases = document.get("recorded_path_aliases", {})
    if isinstance(aliases, list):
        aliases = {a["recorded_prefix"]: a["target"] for a in aliases}
    normalized_prefixes = [_normal(k).casefold() for k in aliases]
    if len(normalized_prefixes) != len(set(normalized_prefixes)):
        raise ValueError("Recorded alias prefixes must be unique after slash/case normalization")
    ordered = tuple(sorted(((_normal(k), _path(v, art)) for k, v in aliases.items()), key=lambda x: len(x[0]), reverse=True))
    value = PortablePaths(art, out, raw, controls, _path(v3, art) if v3 else None, ordered)
    value.validate()
    _CONFIG = value
    return value


def resolve_recorded_path(value: str | Path, *, base: Path | None = None, must_exist: bool = True) -> Path:
    """Resolve longest-prefix aliases without altering a historical JSON file."""
    cfg = current()
    recorded = _normal(value)
    path = None
    alias_target = None
    for prefix, target in cfg.aliases:
        if recorded.casefold() == prefix.casefold() or recorded.casefold().startswith(prefix.casefold() + "/"):
            suffix = recorded[len(prefix):].lstrip("/")
            path = target / suffix
            alias_target = target.resolve()
            break
    if path is None:
        if recorded_is_absolute(value):
            candidate = Path(str(value))
            if not candidate.is_absolute():
                raise FileNotFoundError("No configured alias resolves this historical platform path: " + str(value))
            path = candidate
        else:
            relative_root = (base or cfg.artifact_root).resolve()
            path = relative_root / str(value)
            if not _under(path.resolve(), relative_root):
                raise ValueError("Relative retained path escapes its base")
    path = path.resolve()
    if alias_target is not None and not _under(path, alias_target):
        raise ValueError("Recorded alias suffix or symlink escapes its original target")
    allowed = [CODE_ROOT, cfg.artifact_root, cfg.output_root] + [cfg.raw(k) for k in RAW_NAMES] + list(cfg.controls.values())
    if cfg.v3_study:
        allowed.append(cfg.v3_study)
    if not any(_under(path, p.resolve()) for p in allowed):
        raise ValueError("Unmapped historical input lies outside configured roots: " + str(value))
    if must_exist and not path.exists():
        raise FileNotFoundError("Required retained artifact is missing: " + str(path))
    return path


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assert_output_path(path: str | Path, *, must_exist: bool = False) -> Path:
    """Reject every generated file tree outside the dedicated output root."""
    candidate = Path(path).resolve()
    if must_exist and not candidate.is_dir():
        raise FileNotFoundError("Completed output experiment is missing: " + str(candidate))
    if not candidate.is_relative_to(current().output_root.resolve()):
        raise ValueError("Generated experiments/reports must be inside configured output_root")
    cfg = current()
    retained = [cfg.artifact_root, cfg.v3_root] + [cfg.raw(k) for k in RAW_NAMES] + [cfg.control(k) for k in ("CHASE", "DRIVE", "HRF")]
    if any(_under(candidate, p.resolve()) for p in retained):
        raise ValueError("Generated output may not be inside a retained input archive")
    current().validate()
    return candidate


def protected_inputs_and_code() -> dict[str, str]:
    """Read-only snapshot of configured input trees and the public source copy."""
    cfg = current()
    cfg.validate()
    roots = [cfg.artifact_root] + [cfg.raw(k) for k in RAW_NAMES] + list(cfg.controls.values())
    if cfg.v3_study:
        roots.append(cfg.v3_study)
    paths = set()
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError("Required configured input directory missing: " + str(root))
        exclude_adapted_subtree = _under(CODE_ROOT.resolve(), root.resolve())
        paths.update(p for p in root.rglob("*") if p.is_file() and
                     not (exclude_adapted_subtree and _under(p.resolve(), CODE_ROOT.resolve())))
    for name in ("flow_matching_refiner", "paired_reproduction", "diffusion_refiner"):
        paths.update((CODE_ROOT / name).glob("*.py"))
    return {str(p): digest(p) for p in sorted(paths)}


def _required_inputs() -> tuple[list[Path], list[Path]]:
    cfg = current();required=[];archives=[]
    for dataset in ("CHASE", "DRIVE", "HRF"):
        folder = cfg.reference_root / dataset / "train_seed_42"
        required += [folder / n for n in ("baseline.pt", "autoencoder.pt", "protocol.json")]
        if dataset != "CHASE":
            required.append(folder / "file_provenance.json")
        oof = cfg.oof_root / dataset / "seed_42"
        required += [oof / f"oof_protocol_{OOF_TAG}.json", oof / f"oof_training_maps_{OOF_TAG}.npz"]
        for fold in range(1, (3 if dataset == "HRF" else 4) + 1):
            f = oof / f"fold_{fold}"
            required += [f / "baseline.pt", f / f"fold_inventory_{OOF_TAG}.json", f / f"held_out_predictions_{OOF_TAG}.npz"]
        archives += [cfg.control(dataset), cfg.v3_root / f"{dataset}_{cfg.v3_root.name[-19:]}"]
    for folder in archives:
        tag = folder.name[-19:]
        required += [folder / f"{stem}_{tag}.{extension}" for stem, extension in (
            ("summary_statistics", "json"), ("verified_report", "json"), ("all_test_records", "csv"),
            ("completed_training_evaluation", "json"), ("thin_definition", "json"),
            ("plan", "json"), ("all_selections_before_test", "json"))]
    return required, archives


def preflight() -> dict[str, Any]:
    """Filesystem/JSON/SHA checks only: no third-party import or model/data evaluation."""
    cfg = current();cfg.validate()
    required, archives = _required_inputs()
    missing = [p for p in required if not p.is_file()]
    missing += [cfg.raw(k) for k in RAW_NAMES if not cfg.raw(k).is_dir()]
    if missing:
        raise FileNotFoundError("Complete retained archives/raw roots are required; missing " + str(len(missing)) + " inputs. First: " + str(missing[0]))
    bound: dict[Path, str] = {}
    def verify(path: Path, expected: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Malformed retained SHA256")
        if path in bound and bound[path] != expected:
            raise ValueError("Conflicting retained digests: " + str(path))
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Retained bytes do not match their original SHA256: " + str(path))
        bound[path] = expected
    documents = set()
    for folder in archives:
        documents.update(folder.rglob("*.json"))
    for path in sorted(documents):
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(obj, dict):
            continue
        for key in ("verified_input_sha256", "source_sha256_before", "source_sha256_after"):
            for recorded, expected in obj.get(key, {}).items():
                resolved = resolve_recorded_path(recorded, base=path.parent)
                verify(resolved, expected)
        if "report_output_sha256" in obj:
            value = obj["report_output_sha256"].get("summary_json")
            if value:
                tag = path.parent.name[-19:]
                verify(path.parent / f"summary_statistics_{tag}.json", value)
    # Baseline source/split checks need only JSON, not torch.load or image decoding.
    for dataset in ("CHASE", "DRIVE", "HRF"):
        full = json.loads((cfg.reference_root / dataset / "train_seed_42/protocol.json").read_text(encoding="utf-8-sig"))
        oof = json.loads((cfg.oof_root / dataset / "seed_42" / f"oof_protocol_{OOF_TAG}.json").read_text(encoding="utf-8-sig"))
        if full["split"] != oof["split"]:
            raise ValueError("Retained full/OOF split mismatch: " + dataset)
        for name in full["split"]["test"]:
            reference = cfg.reference_root / dataset / "train_seed_42" / f"{name}_reference.npz"
            if not reference.is_file():
                raise FileNotFoundError("Missing retained reference prediction: " + str(reference))
    return dict(status="passed", required_files=len(required), historical_digest_bindings_verified=len(bound),
                experiment_evaluation_performed=False, training_performed=False, data_or_tensor_loaded=False,
                scientific_settings="120epochs;seeds42,123,2026,3000,3001;V3separate30lock/native15lock unchanged")
