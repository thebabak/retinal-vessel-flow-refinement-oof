"""New AE-free flow experiment, with all validation choices locked before test.

No import-time data access or execution. Prior models and result files are read
only. Five refiner trainings share one frozen baseline, not five full pipelines.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import sys
import traceback
from zoneinfo import ZoneInfo

from flow_matching_refiner.data import ROOT, load_reference, sha256_file
from flow_matching_refiner.run import write_json

SEEDS = [42, 123, 2026, 3000, 3001]
ALPHAS = [0., .25, .5, .75, 1.]


def stamp():
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S")


def protect_inventory():
    files = set((ROOT/"outputs/flow_matching_2026-11-14_23-46-03").rglob("*"))
    files.update((ROOT/"outputs/flow_matching_delivery_2026-11-14_23-47-58").rglob("*"))
    files.update(ROOT.glob("*.tex"))
    for name in ("model.py", "run.py", "data.py", "__init__.py", "README.md", "test_model.py"):
        files.add(ROOT/"flow_matching_refiner"/name)
    for path in (ROOT/"tmp/mdpi_proof_revision_state.json",):
        if path.is_file():
            files.add(path)
    state_path = ROOT/"tmp/mdpi_proof_revision_state.json"
    if state_path.exists():
        def find_paths(value):
            if isinstance(value, dict):
                for x in value.values():
                    find_paths(x)
            elif isinstance(value, list):
                for x in value:
                    find_paths(x)
            elif isinstance(value, str) and len(value) < 500:
                candidate = Path(value)
                if candidate.is_absolute() and candidate.is_file():
                    files.add(candidate)
        find_paths(json.loads(state_path.read_text(encoding="utf-8-sig")))
    return {str(p):sha256_file(p) for p in sorted(files) if p.is_file()}


def prepare(output, tag):
    import numpy as np
    import torch
    reference = load_reference("CHASE", 42, output/"loader_workspace", "cuda")
    baseline = reference.LNet_RA(in_channels=1).to(reference.device)
    saved = torch.load(reference.baseline_checkpoint, map_location=reference.device, weights_only=False)
    assert saved["protocol"]["split"] == reference.manifest
    baseline.load_state_dict(saved["state"])
    baseline.eval().requires_grad_(False)
    with torch.inference_mode():
        for sample in reference.data.values():
            sample["coarse"] = baseline(sample["x"][None].to(reference.device)).cpu()[0].clone()
        for name in reference.manifest["test"]:
            sample = reference.data[name]
            with np.load(reference.baseline_checkpoint.parent/f"{name}_reference.npz", allow_pickle=False) as saved:
                assert np.array_equal(saved["coarse"], sample["coarse"].numpy()), name
                assert np.array_equal(saved["ground_truth"], sample["g"].numpy()), name
                assert np.array_equal(saved["fov"], np.ones_like(sample["g"].numpy(), dtype=bool)), name
    del baseline, saved
    torch.cuda.empty_cache()
    return reference


def train_one(reference, output, tag, seed, epochs, steps, eval_every):
    import numpy as np
    import torch
    from flow_matching_refiner.model import integrate, linear_path
    from flow_matching_refiner.pixel_model import (
        PixelVelocity, blend_probabilities, objective, probabilities, source_state,
    )

    device, data, split = reference.device, reference.data, reference.manifest
    job = output/f"refiner_seed_{seed}"
    job.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = PixelVelocity().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    order_rng = random.Random(seed+73001)
    sample_rng = torch.Generator(device=device).manual_seed(seed+73002)

    def validation():
        model.eval()
        values = {alpha:[] for alpha in ALPHAS}
        with torch.inference_mode():
            for start in range(0, len(split["validation"]), 2):
                names = split["validation"][start:start+2]
                coarse = torch.stack([data[n]["coarse"] for n in names]).to(device)
                rgb = torch.stack([data[n]["rgb"] for n in names]).to(device)
                endpoint, _ = integrate(model, source_state(coarse), coarse, rgb, steps)
                refined = probabilities(endpoint)
                for alpha in ALPHAS:
                    p = blend_probabilities(coarse, refined, alpha).cpu()
                    for index, name in enumerate(names):
                        values[alpha].append(reference.score(p[index], data[name]["g"])["dice"])
        return {alpha:float(np.mean(scores)) for alpha,scores in values.items()}

    baseline_val = float(np.mean([reference.score(data[n]["coarse"], data[n]["g"])["dice"]
                                 for n in split["validation"]]))
    best = dict(epoch=0, alpha=0., validation_dice=baseline_val)
    selected_path = job/f"selected_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(), selection=best.copy()), selected_path)
    log_path = job/f"training_log_{tag}.csv"
    val_path = job/f"validation_grid_{tag}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as log, val_path.open("w", newline="", encoding="utf-8") as val_log:
        writer, val_writer = csv.writer(log), csv.writer(val_log)
        writer.writerow(["epoch","loss","velocity_mse","bce","dice_loss","lr"])
        val_writer.writerow(["epoch","alpha","validation_dice","selected_epoch","selected_alpha"])
        val_writer.writerow([0,0.,baseline_val,0,0.])
        for epoch in range(1,epochs+1):
            write_json(output/f"status_{tag}.json", dict(
                stage="training", refiner_seed=seed, baseline_seed=42,
                epoch=epoch, max_epochs=epochs, updated_at=stamp()))
            model.train()
            names = split["train"].copy()
            order_rng.shuffle(names)
            sums = np.zeros(4)
            count = 0
            lr = optimizer.param_groups[0]["lr"]
            for start in range(0, len(names), 4):
                transformed = []
                for name in names[start:start+4]:
                    sample = data[name]
                    fields = [sample["coarse"], sample["rgb"], sample["g"]]
                    rotation = order_rng.randrange(4)
                    fields = [torch.rot90(x, rotation, (-2,-1)) for x in fields]
                    for dimension in (-1,-2):
                        if order_rng.random() < .5:
                            fields = [x.flip(dimension) for x in fields]
                    transformed.append(fields)
                coarse, rgb, truth = [torch.stack([x[i] for x in transformed]).to(device).clone()
                                      for i in range(3)]
                source, target = source_state(coarse), 2*truth-1
                t = torch.rand(len(coarse), device=device, generator=sample_rng)
                anchor = torch.rand(len(coarse), device=device, generator=sample_rng) < .25
                t = torch.where(anchor, torch.zeros_like(t), t)
                state, target_velocity = linear_path(source, target, t)
                velocity = model(state, t, coarse, rgb)
                endpoint = state + (1-t[:,None,None,None])*velocity
                loss, parts = objective(velocity, target_velocity, endpoint, truth, torch.ones_like(truth))
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                sums += np.asarray([loss.item(), *[p.item() for p in parts]])*len(coarse)
                count += len(coarse)
            scheduler.step()
            values = sums/count
            writer.writerow([epoch,*values.tolist(),lr])
            log.flush()
            if epoch % eval_every == 0 or epoch == epochs:
                grid = validation()
                # Iteration order breaks ties in favor of the smaller alpha.
                chosen_alpha = max(ALPHAS, key=lambda a:grid[a])
                if grid[chosen_alpha] > best["validation_dice"] + 1e-12:
                    best = dict(epoch=epoch, alpha=chosen_alpha, validation_dice=grid[chosen_alpha])
                    torch.save(dict(state=model.state_dict(), selection=best.copy()), selected_path)
                for alpha in ALPHAS:
                    val_writer.writerow([epoch,alpha,grid[alpha],best["epoch"],best["alpha"]])
                val_log.flush()
                print(f"seed={seed} epoch={epoch}/{epochs} loss={values[0]:.5f} "
                      f"val_raw={grid[1.]:.6f} selected={best}", flush=True)
    final_path = job/f"final_raw_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(), epoch=epochs), final_path)
    decision = dict(
        refiner_seed=seed, baseline_seed=42, selected=best, baseline_validation_dice=baseline_val,
        selected_model_sha256=sha256_file(selected_path), final_raw_model_sha256=sha256_file(final_path),
        raw_reporting="alpha1 from fixed final epoch, independently of the selected deployment candidate",
        parameters=sum(p.numel() for p in model.parameters()),
        selection_locked_at=stamp(), no_test_metrics_used=True,
    )
    write_json(job/f"selection_locked_{tag}.json", decision)
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return decision


def evaluate_one(reference, output, tag, decision, steps):
    import numpy as np
    import torch
    from flow_matching_refiner.model import integrate
    from flow_matching_refiner.pixel_model import PixelVelocity, blend_probabilities, probabilities, source_state

    seed = decision["refiner_seed"]
    job = output/f"refiner_seed_{seed}"
    device, data = reference.device, reference.data
    model = PixelVelocity().to(device)
    raw_path, selected_path = job/f"final_raw_model_{tag}.pt", job/f"selected_model_{tag}.pt"
    assert sha256_file(raw_path) == decision["final_raw_model_sha256"]
    assert sha256_file(selected_path) == decision["selected_model_sha256"]
    rows=[]
    # Compute final-epoch raw outputs first; selection has already been locked.
    model.load_state_dict(torch.load(raw_path, map_location=device, weights_only=True)["state"])
    model.eval().requires_grad_(False)
    raw_predictions={}
    with torch.inference_mode():
        for name in reference.manifest["test"]:
            sample=data[name]
            coarse, rgb = sample["coarse"][None].to(device), sample["rgb"][None].to(device)
            endpoint,nfe=integrate(model, source_state(coarse), coarse, rgb, steps)
            raw_predictions[name]=probabilities(endpoint).cpu()[0]
    selected=decision["selected"]
    if selected["alpha"] != 0:
        model.load_state_dict(torch.load(selected_path, map_location=device, weights_only=True)["state"])
    for name in reference.manifest["test"]:
        sample=data[name]
        with torch.inference_mode():
            if selected["alpha"] == 0:
                p=sample["coarse"].clone()
            else:
                coarse,rgb=sample["coarse"][None].to(device),sample["rgb"][None].to(device)
                endpoint,nfe=integrate(model,source_state(coarse),coarse,rgb,steps)
                p=blend_probabilities(coarse,probabilities(endpoint),selected["alpha"]).cpu()[0]
        raw=raw_predictions[name]
        if not bool(torch.isfinite(p).all() and torch.isfinite(raw).all()):
            raise RuntimeError("Non-finite probability")
        base_metrics,raw_metrics,selected_metrics=[reference.score(x,sample["g"])
                                                  for x in (sample["coarse"],raw,p)]
        row=dict(image=name,subject=sample["subject"],baseline_seed=42,refiner_seed=seed,
                 selected_epoch=selected["epoch"],selected_alpha=selected["alpha"],
                 selected_is_identity=selected["alpha"]==0,steps=steps,
                 raw_nfe=steps,selected_nfe=steps if selected["alpha"] else 0,
                 **{"baseline_"+k:v for k,v in base_metrics.items()},
                 **{"raw_"+k:v for k,v in raw_metrics.items()},
                 **{"selected_"+k:v for k,v in selected_metrics.items()},
                 raw_delta=raw_metrics["dice"]-base_metrics["dice"],
                 selected_delta=selected_metrics["dice"]-base_metrics["dice"])
        path=job/f"{name}_predictions_{tag}.npz"
        np.savez_compressed(path,baseline=sample["coarse"].numpy(),raw=raw.numpy(),
                            selected=p.numpy(),ground_truth=sample["g"].numpy(),
                            fov=np.ones_like(sample["g"].numpy(),dtype=bool))
        with np.load(path,allow_pickle=False) as saved:
            for method in ("baseline","raw","selected"):
                metrics=reference.score(torch.from_numpy(saved[method]),torch.from_numpy(saved["ground_truth"]))
                assert all(metrics[k]==row[method+"_"+k] for k in metrics)
        rows.append(row)
    path=job/f"per_image_test_{tag}.csv"
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    result=dict(refiner_seed=seed,baseline_seed=42,selected=selected,
                baseline_dice=float(np.mean([r["baseline_dice"] for r in rows])),
                final_raw_dice=float(np.mean([r["raw_dice"] for r in rows])),
                selected_dice=float(np.mean([r["selected_dice"] for r in rows])),
                verified_records=len(rows))
    write_json(job/f"completed_{tag}.json",result)
    print("TEST "+json.dumps(result),flush=True)
    return rows,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--training-seeds",nargs="+",type=int,default=SEEDS)
    parser.add_argument("--epochs",type=int,default=150)
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--eval-every",type=int,default=10)
    args=parser.parse_args()
    if not args.execute:
        print("No experiment run; --execute required.")
        return
    if min(args.epochs,args.steps,args.eval_every)<1 or len(set(args.training_seeds))!=len(args.training_seeds):
        raise ValueError("Invalid run arguments")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    tag=output.name[-19:]
    if (output/f"plan_{tag}.json").exists():
        raise FileExistsError("Experiment already initialized; do not overwrite")
    protected=protect_inventory()
    write_json(output/f"protected_files_before_{tag}.json",protected)
    sources=output/"source";sources.mkdir()
    source_hashes={}
    paths=list((ROOT/"flow_matching_refiner").glob("*.py"))
    paths += [ROOT/p for p in ("diffusion_refiner/models.py","diffusion_refiner/utils.py",
                              "paired_reproduction/official_model.py","paired_reproduction/run.py",
                              "paired_reproduction/run_all_datasets_five_full_runs.py")]
    for path in paths:
        relative=path.relative_to(ROOT)
        dest=sources/relative.parent/f"{path.stem}_{tag}{path.suffix}"
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        source_hashes[relative.as_posix()]=sha256_file(path)
    plan=dict(
        created_at=stamp(),dataset="CHASE",baseline_seed=42,refiner_seeds=args.training_seeds,
        epochs=args.epochs,solver="Euler",steps=args.steps,evaluation_every=args.eval_every,
        alpha_grid=ALPHAS,baseline_frozen=True,autoencoder_used=False,
        source="normalized logit(coarse); deterministic, no initial noise",
        target="velocity path endpoint2*GT-1 maps to sigmoid(+/-6); decoded BCE uses hard binaryGT",
        flow_path="linear bridge; velocity=target-source; t25%atom0 and75%uniform",
        architecture="spatial RGB/coarse/state; time-FiLM skip-connected U-Net12/24/48/96; zero velocity head",
        objective="vessel-weighted velocity MSE +0.5 BCE +0.5 Dice on decoded endpoint",
        augmentation="aligned flips and90-degree rotations on RGB/coarse/GT",
        optimizer="AdamW3e-4 weight_decay1e-4; cosine; clip1; batch4; FP32",
        selection="validation only; initial identity + epoch/alpha grid; strict improvements; all choices locked before all test scoring",
        raw_reporting="fixed final epoch,alpha1 regardless of deployment selection",
        thresholds="fixed0.5; never tuned",
        independence="five refiner trainings conditional on ONE frozen baseline; not full-pipeline repetitions",
        test_history="previously examined small cohort; exploratory",
        uncertainty="refiner-seed sample SD only; not patient/generalization CI; no p-values",
        attribution="AE removal,conditioning,architecture,augmentation,loss,andselection change together; not isolated causalablation",
        manuscript_unchanged=True,prior_results_unchanged=True,github_unchanged=True,
        source_hashes=source_hashes,
    )
    write_json(output/f"plan_{tag}.json",plan)
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        reference=prepare(output,tag)
        write_json(output/f"reference_{tag}.json",dict(
            source=reference.source_metadata,split=reference.manifest,test_arrays_exact=True))
        decisions=[]
        for seed in args.training_seeds:
            decisions.append(train_one(reference,output,tag,seed,args.epochs,args.steps,args.eval_every))
        lock_path=output/f"all_selections_locked_before_test_{tag}.json"
        write_json(lock_path,dict(locked_at=stamp(),decisions=decisions))
        locked_digest=sha256_file(lock_path)
        all_rows=[];results=[]
        for decision in decisions:
            write_json(output/f"status_{tag}.json",dict(stage="evaluating_locked_candidates",seed=decision["refiner_seed"]))
            rows,result=evaluate_one(reference,output,tag,decision,args.steps)
            all_rows.extend(rows);results.append(result)
        assert sha256_file(lock_path)==locked_digest
        assert len(all_rows)==8*len(args.training_seeds)
        with (output/f"all_per_image_test_{tag}.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        summary=dict(
            baseline_dice=results[0]["baseline_dice"],
            raw_mean=float(np.mean([r["final_raw_dice"] for r in results])),
            selected_mean=float(np.mean([r["selected_dice"] for r in results])),
            raw_sd=float(np.std([r["final_raw_dice"] for r in results],ddof=1)) if len(results)>1 else None,
            selected_sd=float(np.std([r["selected_dice"] for r in results],ddof=1)) if len(results)>1 else None,
            refiner_runs=results,selection_lock_sha256=locked_digest,
            definition="mean/sample SD across refiner-training means; fixed one baseline;8testimages4subjects",
            no_sampling_repeats=True,no_significance_claim=True,
        )
        write_json(output/f"summary_{tag}.json",summary)
        after={p:sha256_file(p) for p in protected}
        assert after==protected,"Protected prior files changed"
        assert sha256_file(reference.baseline_checkpoint)==reference.source_metadata["baseline_sha256"]
        assert sha256_file(reference.autoencoder_checkpoint)==reference.source_metadata["autoencoder_sha256"]
        write_json(output/f"protected_files_after_{tag}.json",after)
        write_json(output/f"completed_{tag}.json",dict(
            completed_at=stamp(),independent_refiner_trainings=len(results),fixed_baseline_trainings=1,
            verified_image_seed_records=len(all_rows),all_choices_locked_before_test=True,
            previous_files_byte_identical=True,raw_and_selected_both_reported=True))
        write_json(output/f"status_{tag}.json",dict(stage="complete",updated_at=stamp()))
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        write_json(output/f"failed_{tag}.json",dict(failed_at=stamp(),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":
    main()
