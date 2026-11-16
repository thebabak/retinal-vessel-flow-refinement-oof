"""Import-safe access to the retained reference experiment's exact data protocol.

Importing this module uses only the standard library.  Real image loading and
tensor creation happen only in :func:`load_reference`; that function does not
run a baseline, load model weights, fit a model, or evaluate predictions.

The legacy experiment drivers do not expose independent loader functions.
Their existing ``compose`` helper returns the source prefix ending immediately
before baseline construction.  We parse that prefix, redirect its output path,
remove the known metadata writes, and override only the execution device.  This
preserves the original preprocessing, image names, FOV handling and fixed split.
Structural assertions deliberately fail if the legacy source contract changes.
"""

from __future__ import annotations
from flow_matching_refiner.portable_paths import current, resolve_recorded_path, assert_output_path

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DRIVERS = ROOT / "paired_reproduction"
REFERENCE_ROOT = current().reference_root
DATASETS = ("CHASE", "DRIVE", "HRF")


def sha256_file(path: str | Path) -> str:
    """Hash artifact bytes without deserialising checkpoints or predictions."""
    digest = hashlib.sha256()
    with resolve_recorded_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceData:
    """Loaded data plus the unchanged reference split and checkpoint locations."""

    dataset: str
    baseline_seed: int
    data: dict[str, dict[str, Any]]
    manifest: dict[str, list[str]]
    device: Any
    namespace: dict[str, Any]
    baseline_checkpoint: Path
    autoencoder_checkpoint: Path
    protocol_path: Path
    protocol: dict[str, Any]
    source_metadata: dict[str, Any]

    @property
    def score(self) -> Any:
        return self.namespace["score"]

    @property
    def dice(self) -> Any:
        return self.namespace["dice"]

    @property
    def LNet_RA(self) -> Any:
        return self.namespace["LNet_RA"]

    @property
    def MaskAutoencoder(self) -> Any:
        return self.namespace["MaskAutoencoder"]

    @property
    def ImageEncoder(self) -> Any:
        return self.namespace["ImageEncoder"]

    @property
    def DiffusionUNet(self) -> Any:
        return self.namespace["DiffusionUNet"]


def _is_output_metadata_write(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    method = node.value.func
    if not isinstance(method, ast.Attribute) or method.attr != "write_text":
        return False
    destination = method.value
    return (
        isinstance(destination, ast.BinOp)
        and isinstance(destination.op, ast.Div)
        and isinstance(destination.left, ast.Name)
        and destination.left.id == "OUT"
        and isinstance(destination.right, ast.Constant)
        and destination.right.value in {"protocol.json", "file_provenance.json"}
    )


def _safe_loader_tree(prefix: str, dataset: str, source: Path, device: str) -> ast.Module:
    """Apply a narrow AST transformation to the known preprocessing-only prefix."""
    tree = ast.parse(prefix, filename=str(source))
    writes = [node for node in tree.body if _is_output_metadata_write(node)]
    expected = 1 if dataset == "CHASE" else 2
    if len(writes) != expected:
        raise RuntimeError(
            f"Legacy {dataset} loader changed: expected {expected} metadata writes, "
            f"found {len(writes)}. Review the adapter before executing."
        )
    tree.body = [node for node in tree.body if not _is_output_metadata_write(node)]
    changed_device = 0
    changed_output = 0
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "DEVICE":
            node.value = ast.Call(
                func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()),
                                   attr="device", ctx=ast.Load()),
                args=[ast.Constant(value=device)], keywords=[],
            )
            changed_device += 1
        elif target.id == "OUT":
            if not isinstance(node.value, ast.Name) or node.value.id != "experiment_output":
                raise RuntimeError("compose did not redirect the legacy output directory")
            changed_output += 1
    if (changed_device, changed_output) != (1, 1):
        raise RuntimeError("Legacy prefix no longer has the expected DEVICE/OUT assignments")

    # Remaining file mutations would violate the preprocessing-only contract.
    forbidden_calls = {
        "write_text", "write_bytes", "save", "savez", "savez_compressed",
        "savetxt", "dump", "dumps_to_file", "unlink", "replace", "rename",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden_calls:
                raise RuntimeError(
                    f"Unexpected potentially mutating call in legacy loader: {node.func.attr}"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "open":
                mode = node.args[1] if len(node.args) > 1 else None
                mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), mode)
                if mode is not None and (
                    not isinstance(mode, ast.Constant)
                    or not isinstance(mode.value, str)
                    or any(flag in mode.value for flag in "wax+")
                ):
                    raise RuntimeError("Unexpected writable open call in legacy loader")
    return ast.fix_missing_locations(tree)


def _compose_prefix(dataset: str, baseline_seed: int, workdir: Path) -> tuple[Path, str]:
    """Import only the standard-library composition helper under a private name."""
    source = DRIVERS / "run_all_datasets_five_full_runs.py"
    spec = importlib.util.spec_from_file_location("_flow_reference_compose", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference composition helper: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader_source, prefix, _unused_training = module.compose(dataset, baseline_seed, workdir)
    return Path(loader_source), prefix


def load_reference(
    dataset: str,
    baseline_seed: int,
    workdir: str | Path,
    device: str = "cuda",
) -> ReferenceData:
    """Load exact reference data; never retrain or substitute missing artifacts.

    ``data`` contains the original named partitions, including test ground truth
    for a later explicit evaluation.  Callers must select ``manifest['train']``
    exclusively for fitting and keep test observations out of model selection.
    The retained checkpoint protocol, not inferred filename ordering, must match
    the constructed split exactly.  No baseline inference occurs here.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {DATASETS}")
    if isinstance(baseline_seed, bool) or not isinstance(baseline_seed, int):
        raise TypeError("baseline_seed must be an integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty torch device string")

    reference = REFERENCE_ROOT / dataset / f"train_seed_{baseline_seed}"
    baseline_checkpoint = reference / "baseline.pt"
    autoencoder_checkpoint = reference / "autoencoder.pt"
    protocol_path = reference / "protocol.json"
    for required in (baseline_checkpoint, autoencoder_checkpoint, protocol_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required retained artifact is missing: {required}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    workdir = Path(workdir).resolve()
    assert_output_path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    source, prefix = _compose_prefix(dataset, baseline_seed, workdir)
    tree = _safe_loader_tree(prefix, dataset, source, device)
    namespace: dict[str, Any] = {
        "__file__": str(source),
        "__name__": "_flow_reference_loader",
        "experiment_output": workdir,
    }
    old_sys_path = list(sys.path)
    try:
        # The original prefix imports official_model before inserting ROOT.
        sys.path.insert(0, str(DRIVERS))
        exec(compile(tree, str(source), "exec"), namespace)
    finally:
        sys.path[:] = old_sys_path

    data = namespace["data"]
    manifest = namespace["manifest"]
    if protocol.get("split") != manifest:
        raise RuntimeError("Loaded image split does not match retained baseline protocol")
    partitions = [set(manifest[key]) for key in ("train", "validation", "test")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Reference partitions overlap")
    if set(data) != set().union(*partitions):
        raise RuntimeError("Reference data and split inventory do not match")

    return ReferenceData(
        dataset=dataset,
        baseline_seed=baseline_seed,
        data=data,
        manifest=manifest,
        device=namespace["DEVICE"],
        namespace=namespace,
        baseline_checkpoint=baseline_checkpoint,
        autoencoder_checkpoint=autoencoder_checkpoint,
        protocol_path=protocol_path,
        protocol=protocol,
        source_metadata={
            "loader_source": str(source),
            "loader_source_sha256": sha256_file(source),
            "compose_helper": str(DRIVERS / "run_all_datasets_five_full_runs.py"),
            "compose_helper_sha256": sha256_file(DRIVERS / "run_all_datasets_five_full_runs.py"),
            "baseline_sha256": sha256_file(baseline_checkpoint),
            "autoencoder_sha256": sha256_file(autoencoder_checkpoint),
            "protocol_sha256": sha256_file(protocol_path),
            "loader_modifications": [
                "Redirect OUT to the new run workspace",
                "Remove original protocol and file-provenance writes",
                "Use the explicitly requested device",
            ],
            "baseline_inference_performed": False,
            "training_performed": False,
        },
    )
