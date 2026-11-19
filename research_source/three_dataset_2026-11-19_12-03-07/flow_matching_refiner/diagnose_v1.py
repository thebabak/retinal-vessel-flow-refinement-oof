"""Authorized validation-only diagnostic for the unchanged first FM pilot.

No training, test-prediction reevaluation, or old-file mutation is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OLD_TAG = "2026-11-14_23-46-03"
OLD_JOB = ROOT/"outputs"/f"flow_matching_{OLD_TAG}"/"CHASE"/"training_seed_42"


def retrospective_csv_diagnosis(rows):
    result = {}
    for partition in ("validation", "test"):
        selected = [r for r in rows if r["partition"] == partition]
        image_rows = []
        for name in sorted({r["image"] for r in selected}):
            matches = [r for r in selected if r["image"] == name]
            values = [float(r["flow_dice"]) for r in matches]
            baseline = float(matches[0]["baseline_dice"])
            ae = float(matches[0]["ae_reconstruction_dice"])
            flow = statistics.mean(values)
            image_rows.append(dict(
                image=name, baseline_dice=baseline, ae_reconstruction_dice=ae,
                flow_dice=flow, ae_minus_baseline=ae-baseline,
                flow_minus_ae=flow-ae, flow_minus_baseline=flow-baseline,
                noise_seed_sample_sd=statistics.stdev(values),
                noise_seed_range=max(values)-min(values),
                baseline_fp=float(matches[0]["baseline_fp"]),
                baseline_fn=float(matches[0]["baseline_fn"]),
                flow_fp=statistics.mean(float(r["flow_fp"]) for r in matches),
                flow_fn=statistics.mean(float(r["flow_fn"]) for r in matches),
            ))
        result[partition] = dict(
            source="retrospective analysis of existing v1 CSV only; no new prediction evaluation",
            per_image=image_rows,
            mean_ae_minus_baseline=statistics.mean(r["ae_minus_baseline"] for r in image_rows),
            mean_flow_minus_ae=statistics.mean(r["flow_minus_ae"] for r in image_rows),
            mean_flow_minus_baseline=statistics.mean(r["flow_minus_baseline"] for r in image_rows),
            mean_within_image_noise_seed_sd=statistics.mean(r["noise_seed_sample_sd"] for r in image_rows),
            max_noise_seed_range=max(r["noise_seed_range"] for r in image_rows),
            all_image_mean_flow_below_baseline=all(r["flow_minus_baseline"]<0 for r in image_rows),
            all_image_ae_below_baseline=all(r["ae_minus_baseline"]<0 for r in image_rows),
            mean_baseline_fp=statistics.mean(r["baseline_fp"] for r in image_rows),
            mean_baseline_fn=statistics.mean(r["baseline_fn"] for r in image_rows),
            mean_flow_fp=statistics.mean(r["flow_fp"] for r in image_rows),
            mean_flow_fn=statistics.mean(r["flow_fn"] for r in image_rows),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("No calculations performed. Add --execute for this authorized diagnostic.")
        return
    import numpy as np
    import torch
    from flow_matching_refiner.data import load_reference, sha256_file
    from flow_matching_refiner.model import ConditionalVelocity, integrate

    tag = datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S")
    dest = args.output.resolve()/f"v1_validation_diagnostic_{tag}"
    dest.mkdir(parents=True, exist_ok=False)
    old_csv = OLD_JOB/f"per_image_{OLD_TAG}.csv"
    old_checkpoint = OLD_JOB/f"flow_model_{OLD_TAG}.pt"
    with old_csv.open(newline="", encoding="utf-8") as handle:
        old_rows = list(csv.DictReader(handle))
    reference = load_reference("CHASE", 42, dest/"loader_workspace", "cuda")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = reference.device
    model = ConditionalVelocity().to(device)
    # v1 stored torch.__version__ as TorchVersion in its own protocol metadata.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(old_checkpoint, map_location=device, weights_only=True)
    protocol = checkpoint["protocol"]
    assert protocol["split"] == reference.manifest
    assert protocol["sigma"] == .1
    model.load_state_dict(checkpoint["state"])
    model.eval().requires_grad_(False)
    ae = reference.MaskAutoencoder().to(device)
    ae.load_state_dict(torch.load(reference.autoencoder_checkpoint, map_location=device, weights_only=True))
    ae.eval().requires_grad_(False)
    baseline = reference.LNet_RA(in_channels=1).to(device)
    baseline.load_state_dict(torch.load(reference.baseline_checkpoint, map_location=device, weights_only=False)["state"])
    baseline.eval().requires_grad_(False)
    source_paths = [Path(__file__), ROOT/"flow_matching_refiner/model.py", ROOT/"flow_matching_refiner/data.py",
                    ROOT/"diffusion_refiner/models.py", ROOT/"diffusion_refiner/utils.py",
                    ROOT/"paired_reproduction/official_model.py", old_csv, old_checkpoint,
                    reference.baseline_checkpoint, reference.autoencoder_checkpoint, reference.protocol_path]
    source_paths.extend(OLD_JOB/"validation"/f"{name}_predictions_{OLD_TAG}.npz" for name in reference.manifest["validation"])
    before = {str(p):sha256_file(p) for p in source_paths}
    rows = []
    exact20 = []
    predictions = {}
    def record(name, mode, probability, gt, sampling_seed="", steps=0):
        mm = reference.score(probability, gt)
        rows.append(dict(dataset="CHASE", partition="validation", image=name, mode=mode,
                         training_seed=42, sampling_seed=sampling_seed, steps=steps,
                         nfe=steps, **mm))
        return mm
    with torch.inference_mode():
        for name in reference.manifest["validation"]:
            sample = reference.data[name]
            coarse = baseline(sample["x"][None].to(device))
            zc = ae.encode(coarse)
            condition = model.image_condition(sample["rgb"][None].to(device))
            gt = sample["g"]
            old_path = OLD_JOB/"validation"/f"{name}_predictions_{OLD_TAG}.npz"
            with np.load(old_path, allow_pickle=False) as old:
                assert np.array_equal(coarse.cpu()[0].numpy(), old["baseline"])
                assert np.array_equal(gt.numpy(), old["ground_truth"])
                assert old["steps"].tolist() == [20]
                assert old["sampling_seeds"].tolist() == protocol["sampling_seeds"]
                record(name, "baseline", coarse.cpu()[0], gt)
                reconstruction = ae.decode(zc).cpu()[0]
                record(name, "ae_coarse_reconstruction", reconstruction, gt)
                gt_reconstruction = ae.decode(ae.encode(gt[None].to(device))).cpu()[0]
                record(name, "ae_gt_reconstruction_diagnostic_not_upper_bound", gt_reconstruction, gt)
                euler100 = []
                for index, seed in enumerate(protocol["sampling_seeds"]):
                    key = f"CHASE:{name}:{seed}".encode()
                    noise_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63-1)
                    generator = torch.Generator(device=device).manual_seed(noise_seed)
                    noise = torch.randn(zc.shape, generator=generator, device=device)
                    source = zc + protocol["sigma"] * noise
                    repeated, _ = integrate(model, source, zc, condition, 20, "euler")
                    repeated_probability = ae.decode(repeated).cpu()[0]
                    equal = np.array_equal(repeated_probability.numpy(), old["probabilities"][0,index])
                    if not equal:
                        raise RuntimeError(f"Original Euler20 probability mismatch: {name} seed {seed}")
                    repeated_metrics = record(name, "euler20_reproduced_exactly", repeated_probability, gt, seed, 20)
                    old_row = next(r for r in old_rows if r["partition"]=="validation" and r["image"]==name and int(r["sampling_seed"])==seed)
                    assert repeated_metrics["dice"] == float(old_row["flow_dice"])
                    exact20.append(dict(image=name, sampling_seed=seed, probability_array_exact_match=True,
                                        dice_exact_match=True, old_prediction_file_sha256=before[str(old_path)]))
                    endpoint, _ = integrate(model, source, zc, condition, 100, "euler")
                    probability = ae.decode(endpoint).cpu()[0]
                    record(name, "euler100_same_noise", probability, gt, seed, 100)
                    euler100.append(probability.numpy())
                no_noise, _ = integrate(model, zc, zc, condition, 20, "euler")
                no_noise_probability = ae.decode(no_noise).cpu()[0]
                record(name, "euler20_no_noise", no_noise_probability, gt, "none_sigma0", 20)
                output_path = dest/f"{name}_diagnostic_probabilities_{tag}.npz"
                np.savez_compressed(output_path, euler100=np.stack(euler100),
                                    no_noise_euler20=no_noise_probability.numpy(),
                                    ae_gt_reconstruction=gt_reconstruction.numpy(),
                                    ground_truth=gt.numpy(), baseline=coarse.cpu()[0].numpy(),
                                    sampling_seeds=np.asarray(protocol["sampling_seeds"]))
                predictions[name] = dict(path=str(output_path), sha256=sha256_file(output_path))
            print(f"DIAGNOSED {name}", flush=True)
    after = {str(p):sha256_file(p) for p in source_paths}
    assert before == after, "An original source changed during diagnostic"
    csv_path = dest/f"per_image_validation_diagnostics_{tag}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    means = {mode:statistics.mean(r["dice"] for r in rows if r["mode"]==mode) for mode in sorted({r["mode"] for r in rows})}
    euler20 = means["euler20_reproduced_exactly"]
    result = dict(
        created_at=tag, scope="Validation-only new inference diagnostics on unchanged v1 checkpoint",
        new_training=False, new_test_prediction_evaluation=False,
        validation_images=reference.manifest["validation"], sampling_seeds=protocol["sampling_seeds"],
        source_hashes_before=before, source_hashes_after=after, original_files_unchanged=True,
        validation_mean_dice=means,
        validation_euler100_minus_euler20=means["euler100_same_noise"]-euler20,
        validation_no_noise20_minus_euler20=means["euler20_no_noise"]-euler20,
        exact_original_euler20_checks=exact20,
        retrospective_original_csv_diagnosis=retrospective_csv_diagnosis(old_rows),
        prediction_files=predictions, rows=len(rows), csv=str(csv_path),
        interpretation=[
            "AE(coarse) round-trip loss occurs before learned flow integration",
            "AE(GT) Dice is a representation diagnostic and is not a strict achievable-performance upper bound",
            "Euler100 and sigma0 are diagnostics of the old model, not validation-tuned replacements",
            "Existing test CSV analysis is retrospective; this script performs no fresh test inference",
            "No confidence intervals, hypothesis tests, or additional trainings were performed",
        ],
    )
    json_path = dest/f"diagnosis_{tag}.json"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(output=str(dest), report=str(json_path), validation_mean_dice=means,
                         euler100_minus_euler20=result["validation_euler100_minus_euler20"],
                         no_noise20_minus_euler20=result["validation_no_noise20_minus_euler20"], rows=len(rows)), indent=2), flush=True)


if __name__ == "__main__":
    main()
