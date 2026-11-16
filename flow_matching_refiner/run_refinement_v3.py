"""Isolated two-objective follow-up: pilot validation tuning, fresh final fits.

No legacy source or result is edited. All hyperparameters are globally locked
before final training; all final checkpoints globally locked before test scoring.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import current, output_root, protected_inputs_and_code, assert_output_path
import argparse,csv,json,random,shutil,traceback,time
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run import write_json as _write_json
from flow_matching_refiner.run_pixel import stamp
SEEDS=[42,123,2026,3000,3001]
ARMS=["direct_thin_anchor","esfm_endpoint_anchor"]
DATASETS=["CHASE","DRIVE","HRF"]
ALPHAS=[0.,.25,.5,.75,1.]
ANCHORS=[.1,1.]
PILOT_SEED=91011
OLD={d:current().control(d) for d in DATASETS}

def write_json(path,value):
    for attempt in range(20):
        try:return _write_json(path,value)
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05*(attempt+1))

def subject(ref,name):
    return int(ref.data[name]["subject"]) if ref.dataset=="CHASE" else ref.manifest["test"].index(name)

def image_dice(ref,probability,name):
    d=ref.data[name]
    args=(probability,d["g"]) if ref.dataset=="CHASE" else (probability,d["g"],d["fov"]>.5)
    return ref.score(*args)["dice"]

def train_one(ref,out,tag,arm,seed,epochs,steps,anchor_weight,thin_weight=.5):
    import numpy as np
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel,make_target,predict,source_logits
    from flow_matching_refiner.refinement_objectives_v3 import losses
    from flow_matching_refiner.model import linear_path
    from flow_matching_refiner.structure_metrics import soft_cldice_loss

    dest=out/arm/f"seed_{seed}"
    dest.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ErrorSelectiveModel(arm=="esfm_endpoint_anchor").to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    rng=random.Random(seed+91001)
    time_rng=torch.Generator(device=ref.device).manual_seed(seed+91002)
    selective=arm=="esfm_endpoint_anchor"
    option="option2" if selective else "option1"
    direct=arm=="direct_thin_anchor"
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
        writer.writerow(["epoch","loss","fm","bce","dice","soft_cldice","gate","thin_recovery","identity","lr"])
        vw.writerow(["epoch","alpha","validation_dice"]);vw.writerow([0,0.,baseline_val])
        for epoch in range(1,epochs+1):
            write_json(out/f"status_{tag}.json",dict(stage="training",arm=arm,seed=seed,epoch=epoch,epochs=epochs,updated=stamp()))
            model.train();names=split["train"].copy();rng.shuffle(names)
            aggregate=np.zeros(8);seen=0;lr=opt.param_groups[0]["lr"]
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
                loss,parts=losses(velocity,target_velocity,endpoint,coarse,truth,errors,thin,gate,option,anchor_weight,soft_cldice_loss,fov,thin_weight=thin_weight)
                if not bool(torch.isfinite(loss)):raise RuntimeError("Non-finite loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                # Independent clipping prevents one auxiliary branch scaling the other.
                torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(model.error_net.parameters(),1.)
                opt.step()
                aggregate+=np.asarray([loss.item(),parts["fm"].item(),parts["bce"].item(),parts["dice"].item(),parts["cldice"].item(),parts["gate"].item(),parts["thin_recovery"].item(),parts["identity"].item()])*len(coarse)
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
    decision=dict(arm=arm,refiner_seed=seed,baseline_seed=42,anchor_weight=anchor_weight,thin_weight=thin_weight if option=="option1" else 0.,raw=best_raw,blend=best_blend,locked_at=stamp(),
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
    model=ErrorSelectiveModel(arm=="esfm_endpoint_anchor").to(ref.device)
    def infer(checkpoint):
        model.load_state_dict(torch.load(checkpoint,map_location=ref.device,weights_only=True)["state"])
        model.eval().requires_grad_(False);result={}
        with torch.inference_mode():
            for n in ref.manifest["test"]:
                d=ref.data[n];p,calls=predict(model,d["coarse"][None].to(ref.device),d["rgb"][None].to(ref.device),steps,arm=="direct_thin_anchor")
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
                rows.append(dict(arm=arm,refiner_seed=seed,image=n,subject=subject(ref,n),output=kind,
                                 selected_epoch=decision[kind]["epoch"] if kind=="raw" else decision["blend"]["epoch"],
                                 alpha=1. if kind=="raw" else alpha,velocity_calls=calls["velocity_calls"] if kind=="raw" or alpha else 0,
                                 gate_calls=calls["error_gate_calls"] if kind=="raw" or alpha else 0,**mm))
    with (dest/f"test_records_{tag}.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(dest/f"completed_{tag}.json",dict(records=len(rows),test_images=len(ref.manifest["test"]),raw_and_blend_stored=True))
    print("TEST_COMPLETE",arm,seed,flush=True)
    del model;torch.cuda.empty_cache()
    return rows


def protect_previous():
    return protected_inputs_and_code()


def prepare(dataset,out,tag):
    import numpy as np
    import torch
    from flow_matching_refiner.oof_data_all import load_dataset_oof
    from flow_matching_refiner.structure_metrics_all import derive_thin_cutoff,thin_skeleton_mask
    ref,audit=load_dataset_oof(dataset,out/"input_audit","cuda")
    for d in ref.data.values():
        if "fov" not in d:d["fov"]=torch.ones_like(d["g"],dtype=torch.bool)
    info=derive_thin_cutoff([ref.data[n]["g"].numpy()[0] for n in ref.manifest["train"]],
                          [ref.data[n]["fov"].numpy()[0] for n in ref.manifest["train"]])
    old_tag=OLD[dataset].name[-19:]
    old_info=json.loads((OLD[dataset]/f"thin_definition_{old_tag}.json").read_text())
    assert info["cutoff"]==old_info["cutoff"],"Thin definition differs from current comparator"
    info["training_image_ids"]=ref.manifest["train"]
    write_json(out/f"thin_definition_{tag}.json",info)
    for n in ref.manifest["train"]:
        d=ref.data[n]
        thin=thin_skeleton_mask(d["g"].numpy()[0],info["cutoff"],d["fov"].numpy()[0])
        d["thin"]=torch.from_numpy(thin.astype(np.float32))[None]
    for n in ref.manifest["test"]:
        with np.load(OLD[dataset]/"direct/seed_42"/f"{n}_predictions_{old_tag}.npz",allow_pickle=False) as saved:
            for key,field in (("baseline","coarse"),("ground_truth","g"),("fov","fov")):
                assert np.array_equal(saved[key],ref.data[n][field].numpy()),f"Existing comparator {key} mismatch"
    return ref,audit,info


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not args.execute:
        print("No experiment run; --execute required.");return
    out=(args.output or output_root()/f"refinement_v3_{stamp()}").resolve()
    assert_output_path(out);out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,stage="preparing",new_pilot_fits=12,new_final_fits=30,
               pilot_seed=PILOT_SEED,final_seeds=SEEDS,datasets=DATASETS,arms=ARMS,
               epochs=120,steps=8,anchor_candidates=ANCHORS,fixed_option1_thin_weight=.5,
               old_manuscripts_and_results_unchanged=True)
    def status(stage,**extra):
        state.update(stage=stage,updated_at=stamp(),**extra)
        write_json(out/f"controller_status_{tag}.json",state)
        write_json(output_root()/"current_refinement_v3_work.json",state)
    status("hashing_previous_artifacts")
    protected=protect_previous()
    write_json(out/f"protected_before_{tag}.json",protected)
    sources={}
    for source in (ROOT/"flow_matching_refiner").glob("*.py"):
        if source.name=="report_refinement_v3.py":continue
        dest=out/"source"/f"{source.stem}_{tag}.py";dest.parent.mkdir(exist_ok=True)
        shutil.copy2(source,dest);sources[source.name]=sha256_file(source)
    study_plan=dict(**state,source_sha256=sources,previous_files_count=len(protected),
        baseline_seed=42,new_baseline_fits=0,OOF="retained audited OOF maps; baseline seed42 for all runs",
        pilot_selection="best learned raw validation Dice per candidate; improvement tolerance1e-12; ties smaller anchor then earlier epoch",
        fixed_recipe="AdamW3e-4 wd1e-4 cosine120epochs batch4 FP32; unchanged architecture/augmentations/eight-step inference",
        option1="direct plus fixed0.5 missed-thin positive loss and class-balanced correct-pixel probability anchor",
        option2="selective-target ESFM with full-support local auxiliary endpoint supervision and correct-pixel anchor; not unrolled ODE training",
        anchors="mean squared probability displacement from detached coarse, equal average over nonempty correct foreground/background classes inside FOV",
        thin_loss="normalized softplus(-local endpoint) only on training GT thin-skeleton pixels missed by coarse; empty support zero",
        no_oracle_at_inference=True,config_lock_before_all_final_fits=True,global_checkpoint_lock_before_any_test=True,
        old_comparison_scope="existing five-arm models reused, not retrained; new recipes receive additional pilot validation search",
        test_history="previously examined cohorts; exploratory; five refiner fits share one fixed baseline per dataset")
    write_json(out/f"study_plan_{tag}.json",study_plan)
    print(json.dumps(state),flush=True)
    try:
        import numpy as np
        import torch
        from flow_matching_refiner.structure_metrics_all import metrics,dependency_versions
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write_json(out/f"environment_{tag}.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
                   GPU=torch.cuda.get_device_name(0),structural_libraries=dependency_versions()))
        pilot_decisions=[];configs=[]
        for dataset in DATASETS:
            prep=out/"pilot_preparation"/dataset;prep.mkdir(parents=True)
            ref,audit,thin=prepare(dataset,prep,tag)
            for path,digest in audit["source_sha256_before"].items():
                assert path not in protected or protected[path]==digest;protected[path]=digest
            for arm in ARMS:
                candidates=[]
                for anchor in ANCHORS:
                    pilot=out/"validation_pilots"/dataset/f"anchor_{anchor:g}";pilot.mkdir(parents=True,exist_ok=True)
                    status("validation_pilots",dataset=dataset,arm=arm,anchor=anchor,seed=PILOT_SEED,current_output=str(pilot))
                    d=train_one(ref,pilot,tag,arm,PILOT_SEED,120,8,anchor)
                    d=dict(dataset=dataset,**d);candidates.append(d);pilot_decisions.append(d)
                chosen=candidates[0]
                for d in candidates[1:]:
                    if d["raw"]["validation_dice"]>chosen["raw"]["validation_dice"]+1e-12:chosen=d
                configs.append(dict(dataset=dataset,arm=arm,anchor_weight=chosen["anchor_weight"],
                    thin_weight=.5 if arm=="direct_thin_anchor" else 0.,pilot_seed=PILOT_SEED,
                    pilot_raw_validation_dice=chosen["raw"]["validation_dice"],pilot_raw_epoch=chosen["raw"]["epoch"]))
            del ref;torch.cuda.empty_cache()
        config_path=out/f"configurations_locked_before_final_training_{tag}.json"
        write_json(config_path,dict(locked_at=stamp(),selected=configs,pilot_decisions=pilot_decisions,
                   selection=study_plan["pilot_selection"],test_scoring_performed=False))
        config_hash=sha256_file(config_path)
        write_json(out/f"all_sources_protected_before_{tag}.json",protected)
        decisions=[]
        for dataset in DATASETS:
            job=out/f"{dataset}_{tag}";job.mkdir()
            ref,audit,thin=prepare(dataset,job,tag)
            for path,digest in audit["source_sha256_before"].items():assert protected[path]==digest
            plan=dict(dataset=dataset,arms=ARMS,refiner_training_seeds=SEEDS,fixed_baseline_seed=42,
                epochs=120,steps=8,batch=4,old_experiment=str(OLD[dataset]),
                primary_output="raw validation-selected model alpha1",secondary_output="identical validation blend policy for both new recipes",
                config_lock_path=str(config_path),config_lock_sha256=config_hash,
                selected_configurations=[c for c in configs if c["dataset"]==dataset],study_plan=str(out/f"study_plan_{tag}.json"),
                inference="unchanged old architecture and predict; direct one call; ESFM eight velocity calls plus one learned gate call",
                option2_scope=study_plan["option2"],training_loss_region="valid FOV; all-valid FOV for CHASE")
            write_json(job/f"plan_{tag}.json",plan)
            write_json(job/f"environment_{tag}.json",dict(torch=torch.__version__,cuda=torch.version.cuda,
                 structural_libraries=dependency_versions(),input_audit=audit["output_audit"]))
            for seed in SEEDS:
                for arm in ARMS:
                    c=next(c for c in configs if c["dataset"]==dataset and c["arm"]==arm)
                    status("final_training",dataset=dataset,arm=arm,anchor=c["anchor_weight"],seed=seed,current_output=str(job))
                    d=train_one(ref,job,tag,arm,seed,120,8,c["anchor_weight"])
                    decisions.append(dict(dataset=dataset,**d))
            del ref;torch.cuda.empty_cache()
        assert sha256_file(config_path)==config_hash
        global_path=out/f"all_final_choices_locked_before_any_test_{tag}.json"
        write_json(global_path,dict(locked_at=stamp(),decisions=decisions,config_lock_path=str(config_path),config_lock_sha256=config_hash))
        global_hash=sha256_file(global_path)
        for dataset in DATASETS:
            job=out/f"{dataset}_{tag}";local=[{k:v for k,v in d.items() if k!="dataset"} for d in decisions if d["dataset"]==dataset]
            write_json(job/f"all_selections_before_test_{tag}.json",dict(locked_at=stamp(),decisions=local,
                       global_lock_path=str(global_path),global_lock_sha256=global_hash))
        for dataset in DATASETS:
            job=out/f"{dataset}_{tag}";evaluation=job/"evaluation_inputs";evaluation.mkdir()
            ref,audit,thin=prepare(dataset,evaluation,tag);cutoff=thin["cutoff"]
            rows=[]
            for n in ref.manifest["test"]:
                d=ref.data[n];base=d["coarse"].numpy()[0];gt=d["g"].numpy()[0]
                rows.append(dict(arm="baseline",refiner_seed=-1,image=n,subject=subject(ref,n),output="raw",
                    selected_epoch=0,alpha=0.,velocity_calls=0,gate_calls=0,**metrics(base,gt,base,cutoff,fov=d["fov"].numpy()[0])))
            for d in decisions:
                if d["dataset"]!=dataset:continue
                status("locked_test_evaluation",dataset=dataset,arm=d["arm"],seed=d["refiner_seed"],current_output=str(job))
                rows.extend(evaluate(ref,job,tag,d,8,cutoff))
            assert len(rows)==len(ref.manifest["test"])*21
            with (job/f"all_test_records_{tag}.csv").open("x",newline="",encoding="utf-8") as h:
                w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            assert sha256_file(config_path)==config_hash and sha256_file(global_path)==global_hash
            assert all(sha256_file(path)==digest for path,digest in protected.items()),"Historical/source bytes changed"
            local_path=job/f"all_selections_before_test_{tag}.json"
            write_json(job/f"completed_training_evaluation_{tag}.json",dict(completed_at=stamp(),fits=10,
                records=len(rows),test_images=len(ref.manifest["test"]),baseline_trainings=0,independent_baseline_checkpoints=1,
                configs_locked_before_final_training=True,choices_locked_before_test=True,global_choices_locked_before_test=True,
                previous_files_byte_identical=True,OOF_maps_audited=True,lock_sha256=sha256_file(local_path),
                config_lock_path=str(config_path),config_lock_sha256=config_hash,
                global_lock_path=str(global_path),global_lock_sha256=global_hash))
            del ref;torch.cuda.empty_cache()
            print("DATASET_COMPLETE",dataset,str(job),flush=True)
        after={p:sha256_file(p) for p in protected};assert after==protected
        write_json(out/f"protected_after_{tag}.json",after)
        write_json(out/f"all_training_completed_{tag}.json",dict(completed_at=stamp(),pilot_fits=12,final_fits=30,
            datasets=DATASETS,total_new_test_records=840,previous_files_byte_identical=True,
            config_lock_path=str(config_path),global_lock_path=str(global_path)))
        status("all_training_evaluation_complete",current_output=str(out))
        print("ALL_V3_EXPERIMENTS_COMPLETE",str(out),flush=True)
    except BaseException:
        write_json(out/f"failed_{tag}.json",dict(failed_at=stamp(),traceback=traceback.format_exc()))
        status("failed");raise


if __name__=="__main__":main()

