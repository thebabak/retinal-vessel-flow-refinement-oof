"""Canonical guarded entry point. Default operation is stdlib-only preflight."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

from flow_matching_refiner.portable_paths import configure, current, preflight, assert_output_path


@contextmanager
def _arguments(values):
    previous = sys.argv[:]
    try:
        sys.argv = ["retinal-reproduction"] + values
        yield
    finally:
        sys.argv = previous


def _output(path, phase):
    cfg = current()
    out = Path(path).expanduser().resolve() if path else cfg.output_root / (phase + "_" + datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S"))
    if not out.is_relative_to(cfg.output_root.resolve()):
        raise ValueError("Every generated experiment must be inside configured output_root")
    assert_output_path(out)
    if out.exists():
        raise FileExistsError("Use a new output directory; earlier results are immutable")
    if not re.fullmatch(r".*\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", out.name):
        raise ValueError("Output directory must end in a local YYYY-MM-DD_HH-mm-ss timestamp")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Ignored local JSON; see configs/paths.example.json")
    parser.add_argument("--phase", choices=["objectives", "native", "controls", "report"], required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly permit fitting or report evaluation after preflight")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", choices=["ALL", "CHASE", "DRIVE", "HRF"], default="ALL")
    parser.add_argument("--experiment", type=Path, help="For report: completed NEW experiment inside output_root")
    args = parser.parse_args(argv)
    configure(args.config)
    checked = preflight()
    print(json.dumps(dict(**checked, phase=args.phase, execute_requested=args.execute), indent=2), flush=True)
    if not args.execute:
        return checked
    if args.phase in ("objectives", "native"):
        if args.dataset != "ALL":
            raise ValueError("Preserved phase-wide selection requires ALL three datasets")
        out = _output(args.output, args.phase)
        module = importlib.import_module("flow_matching_refiner.run_refinement_v3" if args.phase == "objectives" else "flow_matching_refiner.run_native_patch")
        with _arguments(["--execute", "--output", str(out)]):
            return module.main()
    if args.phase == "controls":
        out = _output(args.output, "controls")
        datasets = [args.dataset] if args.dataset != "ALL" else ["CHASE", "DRIVE", "HRF"]
        for dataset in datasets:
            module = importlib.import_module("flow_matching_refiner.run_esfm_oof" if dataset == "CHASE" else "flow_matching_refiner.run_esfm_all")
            destination = out / (dataset + "_" + out.name[-19:])
            values = ["--execute", "--output", str(destination), "--epochs", "120", "--steps", "8"]
            if dataset != "CHASE":
                values += ["--dataset", dataset]
            with _arguments(values):
                module.main()
        return
    if args.experiment is None:
        raise ValueError("--phase report --execute requires --experiment")
    experiment = args.experiment.expanduser().resolve()
    if not experiment.is_relative_to(current().output_root.resolve()):
        raise ValueError("Reports may write only to a NEW-run experiment in output_root; retained input archives are read-only")
    assert_output_path(experiment, must_exist=True)
    tag = experiment.name[-19:]
    plan = json.loads((experiment / f"plan_{tag}.json").read_text(encoding="utf-8-sig"))
    arms = set(plan["arms"])
    if arms == {"direct_thin_anchor", "esfm_endpoint_anchor"}:
        name = "report_refinement_v3"
    elif arms == {"native_patch_direct"}:
        name = "report_native_patch"
    elif arms == {"direct", "endpoint_iterative", "full_fm", "selective_no_gate", "esfm"}:
        name = "report_esfm_oof" if plan["dataset"] in ("CHASE", "CHASE-DB1") else "report_esfm_all"
    else:
        raise ValueError("Unsupported report arm inventory")
    return importlib.import_module("flow_matching_refiner." + name).report(experiment)


if __name__ == "__main__":
    main()
