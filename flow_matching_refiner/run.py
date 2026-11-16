"""Explicitly launched, isolated flow-matching training and evaluation.

Example: python -m flow_matching_refiner.run --execute --datasets CHASE
There is no training, inference, or data loading at import time.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2026, 3000, 3001]


def timestamp():
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if any(x < 1 for x in args.steps) or len(set(args.steps)) != len(args.steps):
        raise ValueError("steps must be distinct positive integers")
    for name in ("training_seeds", "sampling_seeds", "datasets"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            raise ValueError(f"Duplicate {name}")
    if not set(args.training_seeds).issubset(SEEDS):
        raise ValueError("Each training seed needs its matched retained baseline/AE")
    for name in ("sigma", "dice_weight", "weight_decay", "lr"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            raise ValueError(f"Invalid {name}")


def run_job(args, output, tag, dataset, seed):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from flow_matching_refiner.data import load_reference, sha256_file
    from flow_matching_refiner.model import (
        ConditionalVelocity, integrate, linear_path, masked_dice_loss,
    )

    job = output / dataset / f"training_seed_{seed}"
    job.mkdir(parents=True, exist_ok=False)
    status_path = job / f"status_{tag}.json"
    def status(stage, **extra):
        write_json(status_path, dict(stage=stage, dataset=dataset, training_seed=seed,
                                     updated_at=timestamp(), **extra))
    status("loading_reference")
    reference = load_reference(dataset, seed, job / "loader_workspace", args.device)
    device, data, split = reference.device, reference.data, reference.manifest
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed + 61001)
    random.seed(seed + 61002)
    np.random.seed(seed + 61003)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 61001)

    baseline = reference.LNet_RA(in_channels=1).to(device)
    # These locally generated checkpoints include a historical NumPy scalar.
    saved = torch.load(reference.baseline_checkpoint, map_location=device, weights_only=False)
    if saved.get("protocol", {}).get("split") != split:
        raise RuntimeError("Checkpoint's embedded split differs from reference protocol")
    baseline.load_state_dict(saved["state"])
    baseline.eval().requires_grad_(False)
    ae = reference.MaskAutoencoder().to(device)
    ae.load_state_dict(torch.load(reference.autoencoder_checkpoint, map_location=device, weights_only=True))
    ae.eval().requires_grad_(False)
    status("preparing_frozen_inputs")
    with torch.inference_mode():
        for name, sample in data.items():
            sample["coarse"] = baseline(sample["x"][None].to(device)).cpu()[0]
            sample["zc"] = ae.encode(sample["coarse"][None].to(device)).cpu()[0]
            # Only training targets are encoded for optimization.
            if name in split["train"]:
                sample["ztarget"] = ae.encode(sample["g"][None].to(device)).cpu()[0]
        # Require exact identity with archived test inputs, not a guessed table link.
        for name in split["test"]:
            sample = data[name]
            retained_path = reference.baseline_checkpoint.parent / f"{name}_reference.npz"
            with np.load(retained_path, allow_pickle=False) as retained:
                fov = sample.get("fov", torch.ones_like(sample["g"], dtype=torch.bool))
                for key, array in (("coarse", sample["coarse"]),
                                   ("ground_truth", sample["g"]), ("fov", fov)):
                    if not np.array_equal(retained[key], array.numpy()):
                        raise RuntimeError(f"Retained {key} mismatch: {name}")
    del baseline, saved
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Model initialization is independent of loader/baseline RNG consumption.
    torch.manual_seed(seed + 62001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 62001)
    model = ConditionalVelocity().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    order_rng = random.Random(seed + 62002)
    train_rng = torch.Generator(device=device).manual_seed(seed + 62003)
    protocol = dict(
        experiment="new conditional coarse-to-GT encoded-grid flow matching",
        dataset=dataset, training_seed=seed, split=split,
        baseline_autoencoder="frozen matched retained checkpoints; no retraining",
        reference_sources=reference.source_metadata,
        test_input_identity="coarse, GT and FOV exactly match archived reference arrays",
        source="x0 = E(coarse) + sigma * Gaussian noise", sigma=args.sigma,
        target="x1 = E(ground_truth), training partition only",
        path="xt = (1-t)*x0 + t*x1; t uniform in [0,1]",
        velocity_target="x1 - x0",
        condition="fixed spatial coarse encoding concatenated with xt; RGB global FiLM and continuous time",
        objective="velocity MSE + dice_weight * FOV-masked decoded endpoint Dice loss",
        endpoint_estimate="xt + (1-t)*v_theta(xt,t,coarse,RGB)",
        dice_weight=args.dice_weight, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay, gradient_clip=1.,
        schedule="cosine annealing; no warm-up", precision="FP32", augmentation=False,
        checkpoint_selection="fixed final epoch; no validation or test-based tuning",
        solver=args.solver, ode_interval=[0., 1.], steps=args.steps,
        nfe_per_step=1 if args.solver == "euler" else 2,
        sampling_seeds=args.sampling_seeds, cfg=False, threshold=.5,
        region=reference.protocol.get("region"),
        training_rng=dict(initialization=seed+62001, order=seed+62002, noise_and_time=seed+62003),
        environment=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                         device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=="cuda" else None),
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        limitations=[
            "Pilot on previously examined fixed cohorts; not an untouched confirmatory test",
            "Coarse training inputs are in-sample; no new OOF baseline fits",
            "Frozen AE reconstruction can limit achievable segmentation quality",
            "Spatial coarse conditioning and GT target differ from historical global_frozen diffusion",
            "This is not a single-factor flow-versus-diffusion ablation",
            "Noise sampling seeds are not independent model trainings",
            "No significance tests, confidence intervals, or latency benchmark are run",
        ],
        manuscript_unchanged=True, historical_results_unchanged=True,
        sources={str(p.relative_to(ROOT)):sha256_file(p) for p in
                 [ROOT/"flow_matching_refiner"/n for n in ("model.py","run.py","data.py")]},
    )
    write_json(job / f"protocol_{tag}.json", protocol)
    log_path = job / f"training_log_{tag}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "velocity_mse", "endpoint_dice_loss", "learning_rate"])
        for epoch in range(args.epochs):
            status("training", epoch=epoch+1, total_epochs=args.epochs)
            model.train()
            names = split["train"].copy()
            order_rng.shuffle(names)
            totals = np.zeros(3)
            observed = 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for start in range(0, len(names), args.batch_size):
                selected = [data[n] for n in names[start:start+args.batch_size]]
                def batch(key):
                    # clone removes inference-tensor status before autograd use.
                    return torch.stack([s[key] for s in selected]).to(device).clone()
                zc, target, rgb, truth = (batch(k) for k in ("zc", "ztarget", "rgb", "g"))
                valid = torch.stack([s.get("fov", torch.ones_like(s["g"])) for s in selected]).to(device)
                noise = torch.randn(zc.shape, generator=train_rng, device=device)
                source = zc + args.sigma * noise
                t = torch.rand(len(selected), generator=train_rng, device=device)
                state, velocity = linear_path(source, target, t)
                prediction = model(state, t, zc, model.image_condition(rgb))
                mse = F.mse_loss(prediction, velocity)
                endpoint = state + (1-t[:,None,None,None]) * prediction
                dl = masked_dice_loss(ae.decode(endpoint), truth, valid)
                loss = mse + args.dice_weight * dl
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                totals += np.asarray([loss.item(), mse.item(), dl.item()]) * len(selected)
                observed += len(selected)
            scheduler.step()
            means = totals / observed
            writer.writerow([epoch+1, *means.tolist(), learning_rate])
            handle.flush()
            if epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == args.epochs:
                print(f"{dataset} seed={seed} epoch={epoch+1}/{args.epochs} loss={means[0]:.6f}", flush=True)
    model_path = job / f"flow_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(), protocol=protocol, epoch=args.epochs), model_path)
    model.eval().requires_grad_(False)
    status("evaluating")
    records = []
    def metrics(probabilities, sample):
        if dataset == "CHASE":
            return reference.score(probabilities, sample["g"])
        return reference.score(probabilities, sample["g"], sample["fov"])
    with torch.inference_mode():
        for partition in ("validation", "test"):
            dest = job / partition
            dest.mkdir()
            for name in split[partition]:
                sample = data[name]
                zc = sample["zc"][None].to(device)
                condition = model.image_condition(sample["rgb"][None].to(device))
                coarse_metrics = metrics(sample["coarse"], sample)
                reconstruction = ae.decode(zc).cpu()[0]
                reconstruction_metrics = metrics(reconstruction, sample)
                predictions = []
                for steps in args.steps:
                    sampled = []
                    for sampling_seed in args.sampling_seeds:
                        # Stable distinct per-image streams, independent of Python's hash salt.
                        key = f"{dataset}:{name}:{sampling_seed}".encode()
                        noise_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63-1)
                        generator = torch.Generator(device=device).manual_seed(noise_seed)
                        noise = torch.randn(zc.shape, generator=generator, device=device)
                        endpoint, nfe = integrate(model, zc+args.sigma*noise, zc, condition, steps, args.solver)
                        probability = ae.decode(endpoint).cpu()[0]
                        if not bool(torch.isfinite(probability).all()):
                            raise RuntimeError("Non-finite prediction")
                        sampled.append(probability.numpy())
                        mm = metrics(probability, sample)
                        records.append(dict(
                            dataset=dataset, partition=partition, image=name,
                            subject=sample.get("subject", ""), training_seed=seed,
                            sampling_seed=sampling_seed, steps=steps, nfe=nfe,
                            **{"baseline_"+k:v for k,v in coarse_metrics.items()},
                            **{"ae_reconstruction_"+k:v for k,v in reconstruction_metrics.items()},
                            **{"flow_"+k:v for k,v in mm.items()},
                            delta_dice=mm["dice"]-coarse_metrics["dice"],
                        ))
                    predictions.append(np.stack(sampled))
                np.savez_compressed(
                    dest / f"{name}_predictions_{tag}.npz", probabilities=np.stack(predictions),
                    steps=np.asarray(args.steps), sampling_seeds=np.asarray(args.sampling_seeds),
                    baseline=sample["coarse"].numpy(), ground_truth=sample["g"].numpy(),
                    fov=sample.get("fov", torch.ones_like(sample["g"], dtype=torch.bool)).numpy(),
                    ae_reconstruction=reconstruction.numpy(),
                )
    with (job / f"per_image_{tag}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    # Verify persisted predictions and metrics before calling a job complete.
    for partition in ("validation", "test"):
        for name in split[partition]:
            with np.load(job/partition/f"{name}_predictions_{tag}.npz", allow_pickle=False) as saved:
                for i, steps in enumerate(args.steps):
                    for j, sampling_seed in enumerate(args.sampling_seeds):
                        row = next(r for r in records if r["partition"]==partition and r["image"]==name
                                   and r["steps"]==steps and r["sampling_seed"]==sampling_seed)
                        mm = metrics(torch.from_numpy(saved["probabilities"][i,j]), data[name])
                        if any(mm[k] != row["flow_"+k] for k in mm):
                            raise RuntimeError("Saved prediction metrics failed exact verification")
    expected = (len(split["validation"])+len(split["test"])) * len(args.steps)*len(args.sampling_seeds)
    if len(records) != expected:
        raise RuntimeError("Incomplete evaluation inventory")
    summaries=[]
    for partition in ("validation", "test"):
        for steps in args.steps:
            rows = [r for r in records if r["partition"]==partition and r["steps"]==steps]
            summaries.append(dict(
                dataset=dataset, partition=partition, training_seed=seed, steps=steps,
                nfe=steps*(2 if args.solver=="midpoint" else 1), images=len(split[partition]),
                sampling_seeds=len(args.sampling_seeds),
                baseline_dice=float(np.mean([r["baseline_dice"] for r in rows])),
                ae_reconstruction_dice=float(np.mean([r["ae_reconstruction_dice"] for r in rows])),
                flow_dice=float(np.mean([r["flow_dice"] for r in rows])),
                mean_delta_dice=float(np.mean([r["delta_dice"] for r in rows])),
            ))
    write_json(job/f"summary_{tag}.json", dict(
        results=summaries, independent_training_runs=1,
        uncertainty="No confidence interval or training-seed SD from a single training run",
        predictions_and_metrics_verified=True,
    ))
    if (sha256_file(reference.baseline_checkpoint) != protocol["reference_sources"]["baseline_sha256"]
        or sha256_file(reference.autoencoder_checkpoint) != protocol["reference_sources"]["autoencoder_sha256"]):
        raise RuntimeError("Retained source checkpoint changed during the experiment")
    status("complete", records=len(records))
    write_json(job/f"completed_{tag}.json", dict(
        completed_at=timestamp(), records=len(records), expected_records=expected,
        checkpoint_sha256=sha256_file(model_path), test_input_identity_verified=True,
        persisted_predictions_verified=True,
        retained_baseline_sha256_after=sha256_file(reference.baseline_checkpoint),
        retained_autoencoder_sha256_after=sha256_file(reference.autoencoder_checkpoint),
    ))
    print(json.dumps(summaries, indent=2), flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicitly train and evaluate")
    parser.add_argument("--datasets", nargs="+", choices=["CHASE","DRIVE","HRF"], default=["CHASE"])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", nargs="+", type=int, default=[20])
    parser.add_argument("--solver", choices=["euler","midpoint"], default="euler")
    parser.add_argument("--sigma", type=float, default=.1)
    parser.add_argument("--dice-weight", type=float, default=.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=.001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_args(args)
    if not args.execute:
        print("No experiment run. Add --execute to explicitly train and evaluate.")
        return
    tag = timestamp()
    output = (args.output or ROOT/"outputs"/f"flow_matching_{tag}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_dest = output / "source"
    source_dest.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, source_dest / f"{source.stem}_{tag}.py")
    write_json(output/f"plan_{tag}.json", dict(
        created_at=tag, arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        jobs=len(args.datasets)*len(args.training_seeds),
        manuscript_unchanged=True, github_not_modified=True,
    ))
    try:
        all_results=[]
        for dataset in args.datasets:
            for seed in args.training_seeds:
                all_results.extend(run_job(args, output, tag, dataset, seed))
        write_json(output/f"all_results_{tag}.json", all_results)
        write_json(output/f"completed_{tag}.json", dict(
            completed_at=timestamp(), jobs=len(args.datasets)*len(args.training_seeds),
            status="all requested jobs and persisted predictions verified",
        ))
    except BaseException:
        write_json(output/f"failed_{tag}.json", dict(failed_at=timestamp(), traceback=traceback.format_exc()))
        raise
    print(f"OUTPUT_DIRECTORY={output}", flush=True)


if __name__ == "__main__":
    main()
