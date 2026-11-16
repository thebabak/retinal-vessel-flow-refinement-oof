"""Audited retained-OOF extension to DRIVE/HRF: five paired seeds, five learned arms.

Run explicitly with --execute. All selection locks precede all test scoring.
Old articles, checkpoints and results are never written by this program.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import protected_inputs_and_code, assert_output_path
import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import traceback

from flow_matching_refiner.data import ROOT, sha256_file
import os
import time
from flow_matching_refiner.run import write_json as _legacy_write_json


def write_json(path,value):
    """Retry a transient Windows status-file lock; never overwrite old experiments."""
    for attempt in range(20):
        try:
            return _legacy_write_json(path,value)
        except PermissionError:
            if attempt==19: raise
            time.sleep(.05*(attempt+1))


def image_dice(ref,probability,name):
    d=ref.data[name]
    return ref.score(probability,d["g"],d["fov"]>.5)["dice"]
from flow_matching_refiner.run_pixel import protect_inventory, stamp

SEEDS=[42,123,2026,3000,3001]
ARMS=["direct", "endpoint_iterative", "full_fm", "selective_no_gate", "esfm"]
ALPHAS=[0.,.25,.5,.75,1.]


def protected_files():
    return protected_inputs_and_code()


def train_one(ref,out,tag,arm,seed,epochs,steps):
    import numpy as np
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel,make_target,predict,source_logits
    from flow_matching_refiner.esfm_objective_all import losses
    from flow_matching_refiner.model import linear_path
    from flow_matching_refiner.structure_metrics import soft_cldice_loss

    dest=out/arm/f"seed_{seed}"
    dest.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ErrorSelectiveModel(arm=="esfm").to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    rng=random.Random(seed+91001)
    time_rng=torch.Generator(device=ref.device).manual_seed(seed+91002)
    selective=arm in ("selective_no_gate","esfm")
    use_fm=arm in ("full_fm","selective_no_gate","esfm")
    direct=arm=="direct"
    data,split=ref.data,ref.manifest

    def val_scores():
        model.eval();scores={a:[] for a in ALPHAS}
        with torch.inference_mode():
            for start in range(0,len(split["validation"]),2):
                names=split["validation"][start:start+2]
                coarse=torch.stack([data[n]["coarse"] for n in names]).to(ref.device)
                rgb=torch.stack([data[n]["rgb"] for n in names]).to(ref.device)
                p,_=predict(model,coarse,rgb,steps,direct)
                for alpha in ALPHAS:
                    blended=coarse if alpha==0 else (1-alpha)*coarse+alpha*p
                    for i,n in enumerate(names):
                        scores[alpha].append(image_dice(ref,blended.cpu()[i],n))
        return {a:float(np.mean(v)) for a,v in scores.items()}

    baseline_val=float(np.mean([image_dice(ref,data[n]["coarse"],n) for n in split["validation"]]))
    # Raw model selection excludes epoch0; every raw outcome comes from real fitting.
    best_raw=dict(epoch=None,validation_dice=-1.)
    best_blend=dict(epoch=0,alpha=0.,validation_dice=baseline_val)
    blended_path=dest/f"selected_blend_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(),selection=best_blend),blended_path)
    with (dest/f"training_{tag}.csv").open("w",newline="",encoding="utf-8") as handle,(dest/f"validation_{tag}.csv").open("w",newline="",encoding="utf-8") as vh:
        writer=csv.writer(handle);vw=csv.writer(vh)
        writer.writerow(["epoch","loss","fm","bce","dice","soft_cldice","gate","lr"])
        vw.writerow(["epoch","alpha","validation_dice"]);vw.writerow([0,0.,baseline_val])
        for epoch in range(1,epochs+1):
            write_json(out/f"status_{tag}.json",dict(stage="training",arm=arm,seed=seed,epoch=epoch,epochs=epochs,updated=stamp()))
            model.train();names=split["train"].copy();rng.shuffle(names)
            aggregate=np.zeros(6);seen=0;lr=opt.param_groups[0]["lr"]
            for start in range(0,len(names),4):
                samples=[]
                for n in names[start:start+4]:
                    d=data[n];fields=[d["coarse"],d["rgb"],d["g"],d["thin"],d["fov"]]
                    k=rng.randrange(4);fields=[torch.rot90(x,k,(-2,-1)) for x in fields]
                    for dim in (-1,-2):
                        if rng.random()<.5:fields=[x.flip(dim) for x in fields]
                    samples.append(fields)
                coarse,rgb,truth,thin,fov=[torch.stack([s[j] for s in samples]).to(ref.device).clone() for j in range(5)]
                source=source_logits(coarse)
                target,errors=make_target(source,coarse,truth,selective)
                t=torch.rand(len(coarse),device=ref.device,generator=time_rng)
                t=torch.where(torch.rand(len(coarse),device=ref.device,generator=time_rng)<.25,torch.zeros_like(t),t)
                if direct:t=torch.zeros_like(t)
                state,target_velocity=linear_path(source,target,t)
                gate=model.error_logits(coarse,rgb)
                velocity=model(state,t,coarse,rgb,gate)
                endpoint=state+(1-t[:,None,None,None])*velocity
                loss,parts=losses(velocity,target_velocity,endpoint,coarse,truth,errors,thin,gate,use_fm,selective,soft_cldice_loss,fov)
                if not bool(torch.isfinite(loss)):raise RuntimeError("Non-finite loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                # Independent clipping prevents one auxiliary branch scaling the other.
                torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(model.error_net.parameters(),1.)
                opt.step()
                aggregate+=np.asarray([loss.item(),parts["fm"].item(),parts["bce"].item(),parts["dice"].item(),parts["cldice"].item(),parts["gate"].item()])*len(coarse)
                seen+=len(coarse)
            schedule.step();means=aggregate/seen
            writer.writerow([epoch,*means.tolist(),lr]);handle.flush()
            if epoch%10==0 or epoch==epochs:
                grid=val_scores()
                if grid[1.]>best_raw["validation_dice"]+1e-12:
                    best_raw=dict(epoch=epoch,validation_dice=grid[1.])
                    torch.save(dict(state=model.state_dict(),selection=best_raw),dest/f"best_raw_model_{tag}.pt")
                alpha=max(ALPHAS,key=lambda a:grid[a])
                if grid[alpha]>best_blend["validation_dice"]+1e-12:
                    best_blend=dict(epoch=epoch,alpha=alpha,validation_dice=grid[alpha])
                    torch.save(dict(state=model.state_dict(),selection=best_blend),blended_path)
                for a in ALPHAS:vw.writerow([epoch,a,grid[a]])
                vh.flush()
                print(f"{arm} seed={seed} epoch={epoch} loss={means[0]:.5f} rawval={grid[1.]:.6f} best_raw={best_raw} blend={best_blend}",flush=True)
    torch.save(dict(state=model.state_dict(),epoch=epochs),dest/f"final_model_{tag}.pt")
    decision=dict(arm=arm,refiner_seed=seed,baseline_seed=42,raw=best_raw,blend=best_blend,locked_at=stamp(),
                  raw_sha256=sha256_file(dest/f"best_raw_model_{tag}.pt"),blend_sha256=sha256_file(blended_path),
                  final_sha256=sha256_file(dest/f"final_model_{tag}.pt"),
                  training_parameters=sum(p.numel() for p in model.parameters()),
                  inference_velocity_parameters=sum(p.numel() for p in model.velocity_net.parameters()),
                  inference_gate_parameters=sum(p.numel() for p in model.error_net.parameters()) if model.use_gate else 0)
    write_json(dest/f"selection_locked_{tag}.json",decision)
    del model,opt,schedule;torch.cuda.empty_cache()
    return decision


def evaluate(ref,out,tag,decision,steps,cutoff):
    import numpy as np
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel,predict
    from flow_matching_refiner.structure_metrics_all import metrics
    arm,seed=decision["arm"],decision["refiner_seed"]
    dest=out/arm/f"seed_{seed}";rows=[]
    model=ErrorSelectiveModel(arm=="esfm").to(ref.device)
    def infer(checkpoint):
        model.load_state_dict(torch.load(checkpoint,map_location=ref.device,weights_only=True)["state"])
        model.eval().requires_grad_(False);result={}
        with torch.inference_mode():
            for n in ref.manifest["test"]:
                d=ref.data[n];p,calls=predict(model,d["coarse"][None].to(ref.device),d["rgb"][None].to(ref.device),steps,arm=="direct")
                result[n]=(p.cpu()[0].numpy(),calls)
        return result
    assert sha256_file(dest/f"best_raw_model_{tag}.pt")==decision["raw_sha256"]
    raw=infer(dest/f"best_raw_model_{tag}.pt")
    assert sha256_file(dest/f"selected_blend_model_{tag}.pt")==decision["blend_sha256"]
    alpha=decision["blend"]["alpha"]
    blend_predictions=infer(dest/f"selected_blend_model_{tag}.pt") if alpha else None
    for n in ref.manifest["test"]:
        d=ref.data[n];base=d["coarse"].numpy();gt=d["g"].numpy()
        raw_p,calls=raw[n]
        blended=base.copy() if alpha==0 else (1-alpha)*base+alpha*blend_predictions[n][0]
        path=dest/f"{n}_predictions_{tag}.npz"
        np.savez_compressed(path,baseline=base,raw=raw_p,blend=blended,ground_truth=gt,
                            fov=d["fov"].numpy().astype(bool))
        with np.load(path,allow_pickle=False) as saved:
            for kind in ("raw","blend"):
                mm=metrics(saved[kind][0],saved["ground_truth"][0],saved["baseline"][0],cutoff,fov=saved["fov"][0])
                assert np.isfinite(saved[kind]).all()
                rows.append(dict(arm=arm,refiner_seed=seed,image=n,subject=ref.manifest["test"].index(n),output=kind,
                                 selected_epoch=decision[kind]["epoch"] if kind=="raw" else decision["blend"]["epoch"],
                                 alpha=1. if kind=="raw" else alpha,velocity_calls=calls["velocity_calls"] if kind=="raw" or alpha else 0,
                                 gate_calls=calls["error_gate_calls"] if kind=="raw" or alpha else 0,**mm))
    with (dest/f"test_records_{tag}.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(dest/f"completed_{tag}.json",dict(records=len(rows),test_images=len(ref.manifest["test"]),raw_and_blend_stored=True))
    print("TEST_COMPLETE",arm,seed,flush=True)
    del model;torch.cuda.empty_cache()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=["DRIVE","HRF"],required=True)
    p.add_argument("--execute",action="store_true");p.add_argument("--output",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=120);p.add_argument("--steps",type=int,default=8)
    a=p.parse_args()
    if not a.execute:print("No experiment run; --execute required.");return
    if min(a.epochs,a.steps)<1:raise ValueError("Invalid epoch/step count")
    out=a.output.resolve();assert_output_path(out);out.mkdir(parents=True,exist_ok=True);tag=out.name[-19:]
    if (out/f"plan_{tag}.json").exists():raise FileExistsError("Prior experiment must not be overwritten")
    old=protected_files();write_json(out/f"protected_before_{tag}.json",old)
    source_hashes={}
    for source in (ROOT/"flow_matching_refiner").glob("*.py"):
        if source.name.startswith("report_esfm_"):continue
        dest=out/"source"/f"{source.stem}_{tag}.py";dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest);source_hashes[source.name]=sha256_file(source)
    plan=dict(version=tag,created=stamp(),dataset=a.dataset,fixed_baseline_seed=42,refiner_training_seeds=SEEDS,
              arms=ARMS,learned_fits=25,epochs=a.epochs,batch=4,steps=a.steps,solver="Euler",threshold=.5,
              validation_every=10,selection="highest validation mean image Dice; raw best learned epoch excludes epoch0; blend includes identity; all25 choices locked before test",
              blend_grid=ALPHAS,primary_output="raw validation-selected model alpha1",secondary_output="identical validation blend policy for every arm",
              inputs="spatial RGB, coarse and current logits; retained cross-fitted OOF trainmaps; full baseline val/testmaps",
              training_loss_region="valid FOV for FM/BCE/Dice/gate; softclDice on probability and GT intersect FOV; identical coefficients, all-valid FOV recovers original CHASE objective",
              evaluation_region="retained official/source FOV resized nearest",
              inference_data_unchanged=True,
              test_unit="image; subject/patient independence not verified; CSV subject is a stable image-index alias",
              source="unscaled logit(clamp(coarse,1e-4,1-1e-4))",margin=6.,gate_floor=.05,initial_noise=False,
              selective_target="retain source logits where coarse binary prediction correct; +/-6 only on errors",
              time_sampling="25percent t0,75percentuniform; directalways t0 but consumes same RNG draws",
              velocity_parameterization="6times shared PixelVelocity12/24/48/96; independent 4to12to12to1 auxiliary error classifier",
              gate_use="allarms fit balanced auxiliaryerrorclassifier; only ESFM uses its score invelocity, withfloor.05",
              losses="FM arms: thinweighted velocity MSE/36 +.5BCE+.5Dice+.1softclDice+.1balancedgateBCE; otherarms omitFMterm",
              selective_auxiliary="(1-E)coarse+E sigmoid(endpoint) onlyselectivearms; E trainingonly",
              iterative_control="endpoint-supervised continuous-time control on same fulltargetbridge, noFM MSE; no unrolledtraining",
              ordinary_refiner="direct source-logit residual supervised by FOV-masked segmentationloss, onecall",
              thin_weight=4.,soft_skeleton_iterations=10,thin_cutoff="trainingGT skeleton radius quantile1/3 fixedbeforevaltest",
              optimizer="AdamW3e-4 weight_decay1e-4; cosine; separatebranchgradclip1; FP32",
              augmentation="aligned flips and90degree rotations on RGB/coarse/GT/thinmap/FOV",
              primary_comparison="raw ESFM minus direct ordinaryrefiner pairedimageDice; baseline mandatory secondary",
              secondary_comparisons="ESFM vs endpoint iterative,fullFM,selective no-gate; structural outcomes secondary",
              uncertainty="imagecluster percentile bootstrap50000 draws fixedseed20261115; seedSDseparate; patientindependenceunverified; no seedtestpvalues",
              test_history="previously inspected cohort, exploratory; no external validation",
              previous_results_unchanged=True,published_article_unchanged=True,source_hashes=source_hashes)
    write_json(out/f"plan_{tag}.json",plan)
    try:
        import numpy as np
        import torch
        from skimage.morphology import skeletonize
        from scipy.ndimage import distance_transform_edt
        from flow_matching_refiner.oof_data_all import load_dataset_oof
        from flow_matching_refiner.structure_metrics_all import derive_thin_cutoff,metrics,dependency_versions
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ref,audit=load_dataset_oof(a.dataset,out/"input_audit","cuda")
        for path,digest in audit["source_sha256_before"].items():
            assert path not in old or old[path]==digest
            old[path]=digest
        write_json(out/f"all_sources_protected_before_{tag}.json",old)
        thin_info=derive_thin_cutoff([ref.data[n]["g"].numpy()[0] for n in ref.manifest["train"]],
                                     [ref.data[n]["fov"].numpy()[0] for n in ref.manifest["train"]])
        cutoff=thin_info["cutoff"]
        write_json(out/f"thin_definition_{tag}.json",{**thin_info,"training_image_ids":ref.manifest["train"]})
        for n in ref.manifest["train"]:
            gt=(ref.data[n]["g"].numpy()[0]>=.5)&ref.data[n]["fov"].numpy()[0].astype(bool)
            radius=distance_transform_edt(np.pad(gt,1))[1:-1,1:-1]
            thin=skeletonize(gt)&(radius<=cutoff)
            ref.data[n]["thin"]=torch.from_numpy(thin.astype(np.float32))[None]
        write_json(out/f"environment_{tag}.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0),structural_libraries=dependency_versions(),input_audit=audit["output_audit"]))
        decisions=[]
        for seed in SEEDS:
            for arm in ARMS:
                decisions.append(train_one(ref,out,tag,arm,seed,a.epochs,a.steps))
        locks=out/f"all_selections_before_test_{tag}.json";write_json(locks,dict(locked_at=stamp(),decisions=decisions))
        lock_hash=sha256_file(locks)
        baseline=[]
        for n in ref.manifest["test"]:
            d=ref.data[n];base=d["coarse"].numpy()[0];gt=d["g"].numpy()[0]
            baseline.append(dict(arm="baseline",refiner_seed=-1,image=n,subject=ref.manifest["test"].index(n),output="raw",
                                 selected_epoch=0,alpha=0.,velocity_calls=0,gate_calls=0,**metrics(base,gt,base,cutoff,fov=d["fov"].numpy()[0])))
        all_rows=baseline.copy()
        for decision in decisions:
            write_json(out/f"status_{tag}.json",dict(stage="locked_test_evaluation",arm=decision["arm"],seed=decision["refiner_seed"]))
            all_rows.extend(evaluate(ref,out,tag,decision,a.steps,cutoff))
        expected_records=len(ref.manifest["test"])*51
        assert len(all_rows)==expected_records
        with (out/f"all_test_records_{tag}.csv").open("w",newline="",encoding="utf-8") as h:
            w=csv.DictWriter(h,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
        assert sha256_file(locks)==lock_hash
        after={path:sha256_file(path) for path in old};assert after==old,"Prior files changed"
        write_json(out/f"protected_after_{tag}.json",after)
        write_json(out/f"completed_training_evaluation_{tag}.json",dict(completed_at=stamp(),fits=25,baseline_trainings=0,
                   newly_trained_refiners=25,independent_baseline_checkpoints=1,records=expected_records,test_images=len(ref.manifest["test"]),
                   image_clusters=len(ref.manifest["test"]),patient_independence_verified=False,
                   choices_locked_before_test=True,lock_sha256=lock_hash,previous_files_byte_identical=True,
                   raw_and_blend_reported=True,OOF_maps_audited=True))
        write_json(out/f"status_{tag}.json",dict(stage="training_evaluation_complete",dataset=a.dataset,updated_at=stamp()))
        print("EXPERIMENT_COMPLETE",str(out),flush=True)
    except BaseException:
        write_json(out/f"failed_{tag}.json",dict(failed_at=stamp(),traceback=traceback.format_exc()))
        raise


if __name__=="__main__":main()
