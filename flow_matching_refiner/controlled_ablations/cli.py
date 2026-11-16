"""Guarded portable entry point for the registered 16-arm ablation recipe.

Default operation describes the fixed plan without data access. --register
creates a fresh timestamped run registration only. --preflight checks retained
input bytes. Training or test evaluation requires the separate --execute flag.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from flow_matching_refiner.portable_paths import configure, current, assert_output_path, preflight
from .specs import plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp():
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S")


def register(output):
    """Register new local source paths/digests; never inspect scientific inputs."""
    output=assert_output_path(output)
    if not re.fullmatch(r".+\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}",output.name):
        raise ValueError("Output name must end in YYYY-MM-DD_HH-mm-ss")
    if output.exists():
        raise FileExistsError("Registration requires a new directory; previous runs are immutable")
    output.mkdir(parents=True,exist_ok=False)
    tag=output.name[-19:]
    record=dict(registered_at=timestamp(),**plan())
    design=output/f"prespecified_design_{tag}.json"
    design.write_text(json.dumps(record,indent=2),encoding="utf-8")
    (output/"active_design.json").write_text(json.dumps(dict(file=design.name,sha256=sha(design)),indent=2),encoding="utf-8")
    sources={str(path.resolve()):sha(path) for path in Path(__file__).resolve().parent.glob("*.py")}
    (output/f"source_lock_{tag}.json").write_text(json.dumps(sources,indent=2),encoding="utf-8")
    (output/f"public_registration_{tag}.json").write_text(json.dumps(dict(
        registered_at=timestamp(),protocol="Portable reproduction of the fixed 16-arm recipe",
        scientific_inputs_loaded=False,training_performed=False,evaluation_performed=False,
        config_artifact_root=str(current().artifact_root),config_output_root=str(current().output_root)),indent=2),encoding="utf-8")
    return output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,help="Local path configuration, matching configs/paths.example.json")
    parser.add_argument("--output",type=Path,help="Fresh or previously registered run inside configured output_root")
    parser.add_argument("--register",action="store_true",help="Create new run plan and source locks; does not train")
    parser.add_argument("--preflight",action="store_true",help="Run strict original-byte artifact checks without model execution")
    parser.add_argument("--execute",action="store_true",help="Explicitly permit registered fitting/evaluation after strict preflight")
    parser.add_argument("--phase",choices=["train","evaluate","all"],default="all")
    parser.add_argument("--max-fits",type=int)
    parser.add_argument("--stop-after-dataset",choices=["CHASE","DRIVE","HRF"])
    parser.add_argument("--cuda-device",type=int,default=0)
    args=parser.parse_args(argv)
    configure(args.config)
    if not (args.register or args.preflight or args.execute):
        result=dict(status="description_only",scientific_data_loaded=False,
            training_performed=False,evaluation_performed=False,plan=plan())
        print(json.dumps(result,indent=2));return result
    out=None
    if args.register:
        out=args.output or current().output_root/f"controlled_ablations_{timestamp()}"
        out=register(out)
        print(json.dumps(dict(status="registered_without_execution",output=str(out)),indent=2))
    checked=None
    if args.preflight or args.execute:
        checked=preflight()
        print(json.dumps(checked,indent=2),flush=True)
    if not args.execute:
        return dict(output=str(out) if out else None,preflight=checked,training_performed=False,evaluation_performed=False)
    if out is None:
        if args.output is None:raise ValueError("--execute needs --output for a registered run or --register")
        out=assert_output_path(args.output,must_exist=True)
    # Configure paths and check original bytes before importing scientific loaders.
    from .runner import main as run
    values=["--execute","--output",str(out),"--phase",args.phase,"--cuda-device",str(args.cuda_device)]
    if args.max_fits is not None:values += ["--max-fits",str(args.max_fits)]
    if args.stop_after_dataset:values += ["--stop-after-dataset",args.stop_after_dataset]
    return run(values)


if __name__=="__main__":
    main()
