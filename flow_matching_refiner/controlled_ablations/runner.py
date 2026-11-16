"""Execute the registered 240-fit plan, protecting every retained input.

No data load, model fitting or evaluation occurs at import. Training and test
evaluation are separate phases. All final choices are globally locked before
any new test prediction; resumed incomplete fits restart from initialization.
"""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,random,shutil,sys,time,traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .specs import ARMS,BY_NAME,SEEDS,DATASETS,EPOCHS,STEPS,ALPHAS,plan

from flow_matching_refiner.portable_paths import CODE_ROOT, protected_inputs_and_code, assert_output_path
ROOT=CODE_ROOT

def stamp(): return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d_%H-%M-%S")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for retry in range(15):
        temporary=path.with_suffix(path.suffix+".tmp")
        try:
            temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
            temporary.replace(path);return
        except PermissionError:
            if retry==14:raise
            time.sleep(.05*(retry+1))

def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def image_dice(ref,p,name):
    sample=ref.data[name]
    args=(p,sample["g"]) if ref.dataset=="CHASE" else (p,sample["g"],sample["fov"]>.5)
    return ref.score(*args)["dice"]

def subject(ref,name):
    return int(ref.data[name]["subject"]) if ref.dataset=="CHASE" else ref.manifest["test"].index(name)

def protected_inventory():
    """Protect explicitly configured retained inputs and portable public code."""
    result=protected_inputs_and_code()
    for path in Path(__file__).resolve().parent.glob("*.py"):
        result[str(path.resolve())]=sha(path)
    return result

def prepare(dataset,job,tag,phase):
    import numpy as np
    import torch
    from flow_matching_refiner.run_native_patch import prepare as original_prepare
    from flow_matching_refiner.structure_metrics_all import thin_skeleton_mask
    destination=job/(phase+"_inputs")/stamp();destination.mkdir(parents=True,exist_ok=False)
    ref,audit,thin=original_prepare(dataset,destination,tag)
    for name in ref.manifest["train"]:
        sample=ref.data[name]
        support=thin_skeleton_mask(sample["g"].numpy()[0],thin["cutoff"],sample["fov"].numpy()[0])
        sample["thin"]=torch.from_numpy(support.astype(np.float32))[None]
    info_path=job/f"thin_definition_{tag}.json"
    info=read(destination/f"thin_definition_{tag}.json")
    if info_path.exists(): assert read(info_path)==info
    else:write(info_path,info)
    return ref,audit,thin

def create_views(ref,spec):
    import torch
    from torch.nn import functional as F
    from flow_matching_refiner.native_patch_model import training_center_pools
    from .operations import patch_fields
    if spec.family!="patch":return {},{},{}
    views={n:patch_fields(ref.data[n],spec,ref.dataset) for n in ref.manifest["train"]}
    # Information controls always sample using original native GT/FOV, never the degraded targets.
    pools={}
    for name in ref.manifest["train"]:
        d=ref.data[name]
        gt,fov=(d["g"],d["fov"]) if spec.patch_grid=="512" else (d["n_g"],d["n_fov"])
        pools[name]=training_center_pools(gt,fov)
    validation_rgb={}
    if spec.degrade_rgb:
        for n in ref.manifest["validation"]:
            d=ref.data[n]
            validation_rgb[n]=F.interpolate(d["rgb"].cpu()[None],size=d["n_coarse"].shape[-2:],mode="bilinear",align_corners=False)[0].contiguous()
    return views,pools,validation_rgb

def train_one(ref,job,tag,spec,seed,views,pools,validation_rgb,status):
    import numpy as np
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel,make_target,source_logits
    from flow_matching_refiner.model import linear_path
    from .operations import aligned_augmentation,condition_rgb,objective,infer,sample_patch
    dest=job/spec.name/f"seed_{seed}";dest.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ErrorSelectiveModel(spec.use_gate).to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    rng=random.Random(seed+(141001 if spec.family=="patch" else 91001))
    time_rng=torch.Generator(device=ref.device).manual_seed(seed+91002)
    baseline_val=float(np.mean([image_dice(ref,ref.data[n]["coarse"],n) for n in ref.manifest["validation"]]))
    best_raw=dict(epoch=None,validation_dice=-1.)
    best_blend=dict(epoch=0,alpha=0.,validation_dice=baseline_val)
    blend_path=dest/f"selected_blend_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(),selection=best_blend,spec=spec.record()),blend_path)

    def validation():
        model.eval();scores={alpha:[] for alpha in ALPHAS}
        with torch.inference_mode():
            # Same fixed images/mean weighting as retained runs. Patch inference remains imagewise.
            if spec.family=="grid":
                from flow_matching_refiner.esfm_model import predict
                for start in range(0,len(ref.manifest["validation"]),2):
                    names=ref.manifest["validation"][start:start+2]
                    coarse=torch.stack([ref.data[n]["coarse"] for n in names]).to(ref.device)
                    rgb=condition_rgb(torch.stack([ref.data[n]["rgb"] for n in names]).to(ref.device),spec)
                    probability,_=predict(model,coarse,rgb,STEPS,spec.direct)
                    for alpha in ALPHAS:
                        blended=coarse if alpha==0 else (1-alpha)*coarse+alpha*probability
                        for i,n in enumerate(names):scores[alpha].append(image_dice(ref,blended.cpu()[i],n))
            else:
                for n in ref.manifest["validation"]:
                    p,_,_=infer(model,ref.data[n],spec,ref.dataset,ref.device,STEPS,validation_rgb.get(n))
                    for alpha in ALPHAS:
                        blended=ref.data[n]["coarse"] if alpha==0 else (1-alpha)*ref.data[n]["coarse"]+alpha*p
                        scores[alpha].append(image_dice(ref,blended,n))
        return {alpha:float(np.mean(values)) for alpha,values in scores.items()}

    with (dest/f"training_{tag}.csv").open("x",newline="",encoding="utf-8") as th, \
         (dest/f"validation_{tag}.csv").open("x",newline="",encoding="utf-8") as vh, \
         (dest/f"sample_draws_{tag}.csv").open("x",newline="",encoding="utf-8") as ph:
        tw,vw,pw=csv.writer(th),csv.writer(vh),csv.writer(ph)
        tw.writerow(["epoch","loss","fm","bce","dice","cldice","gate","thin_recovery","identity","lr","samples_seen","optimizer_updates"])
        vw.writerow(["epoch","alpha","validation_dice"]);vw.writerow([0,0.,baseline_val])
        pw.writerow(["epoch","sample_index","image","centre_y","centre_x","top","left","vessel_centre","rotation90","flip_horizontal","flip_vertical"])
        for epoch in range(1,EPOCHS+1):
            status("training",dataset=ref.dataset,arm=spec.name,seed=seed,epoch=epoch)
            model.train();names=ref.manifest["train"].copy()
            if spec.family=="patch":names*=4
            rng.shuffle(names)
            batch_size=16 if spec.family=="patch" else 4
            aggregate=np.zeros(8);seen=0;updates=0;lr=opt.param_groups[0]["lr"]
            for start in range(0,len(names),batch_size):
                samples=[]
                for index,n in enumerate(names[start:start+batch_size],start=start):
                    d=ref.data[n]
                    if spec.family=="patch":
                        view=views[n]
                        sampling_truth=d["g"] if spec.patch_grid=="512" else d["n_g"]
                        crop,meta=sample_patch(view,sampling_truth,pools[n],rng,spec)
                        fields=[crop[k] for k in ("coarse","rgb","gt","fov")]
                        fields,aug=aligned_augmentation(fields,rng,spec.augmentation)
                        fields=[*fields[:3],torch.zeros_like(fields[2]),fields[3]]
                        pw.writerow([epoch,index,n,meta["centre_y"],meta["centre_x"],meta["top"],meta["left"],meta["used_vessel_centre"],aug["rotation90"],aug["flip_horizontal"],aug["flip_vertical"]])
                    else:
                        fields=[d[k] for k in ("coarse","rgb","g","thin","fov")]
                        fields,aug=aligned_augmentation(fields,rng,spec.augmentation)
                        pw.writerow([epoch,index,n,"","","","","",aug["rotation90"],aug["flip_horizontal"],aug["flip_vertical"]])
                    samples.append(fields)
                coarse,rgb,truth,thin,fov=[torch.stack([x[j] for x in samples]).to(ref.device).clone() for j in range(5)]
                rgb=condition_rgb(rgb,spec)
                source=source_logits(coarse);target,errors=make_target(source,coarse,truth,spec.selective_target)
                t=torch.rand(len(coarse),device=ref.device,generator=time_rng)
                t=torch.where(torch.rand(len(coarse),device=ref.device,generator=time_rng)<.25,torch.zeros_like(t),t)
                if spec.direct:t=torch.zeros_like(t)
                state,target_velocity=linear_path(source,target,t)
                gate=model.error_logits(coarse,rgb);velocity=model(state,t,coarse,rgb,gate)
                endpoint=state+(1-t[:,None,None,None])*velocity
                loss,parts=objective(spec,velocity,target_velocity,endpoint,coarse,truth,errors,thin,gate,fov)
                if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite ablation training loss")
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(model.error_net.parameters(),1.)
                opt.step();seen+=len(coarse);updates+=1
                aggregate+=np.array([loss.item(),*[parts[k].item() for k in ("fm","bce","dice","cldice","gate","thin_recovery","identity")]])*len(coarse)
            scheduler.step()
            assert seen==len(ref.manifest["train"])*(4 if spec.family=="patch" else 1)
            assert updates==(len(ref.manifest["train"])+3)//4
            tw.writerow([epoch,*(aggregate/seen).tolist(),lr,seen,updates]);th.flush();ph.flush()
            if epoch%10==0:
                values=validation()
                if values[1.]>best_raw["validation_dice"]+1e-12:
                    best_raw=dict(epoch=epoch,validation_dice=values[1.])
                    torch.save(dict(state=model.state_dict(),selection=best_raw,spec=spec.record()),dest/f"best_raw_model_{tag}.pt")
                alpha=max(ALPHAS,key=lambda a:values[a])
                if values[alpha]>best_blend["validation_dice"]+1e-12:
                    best_blend=dict(epoch=epoch,alpha=alpha,validation_dice=values[alpha])
                    torch.save(dict(state=model.state_dict(),selection=best_blend,spec=spec.record()),blend_path)
                for alpha in ALPHAS:vw.writerow([epoch,alpha,values[alpha]])
                vh.flush();print(ref.dataset,spec.name,seed,epoch,"rawval",values[1.],"best",best_raw,flush=True)
    final_path=dest/f"final_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(),epoch=EPOCHS,spec=spec.record()),final_path)
    decision=dict(arm=spec.name,refiner_seed=seed,baseline_seed=42,raw=best_raw,blend=best_blend,
        locked_at=stamp(),spec=spec.record(),raw_sha256=sha(dest/f"best_raw_model_{tag}.pt"),
        blend_sha256=sha(blend_path),final_sha256=sha(final_path),
        training_parameters=sum(p.numel() for p in model.parameters()),
        inference_velocity_parameters=sum(p.numel() for p in model.velocity_net.parameters()),
        inference_gate_parameters=sum(p.numel() for p in model.error_net.parameters()) if spec.use_gate else 0)
    write(dest/f"selection_locked_{tag}.json",decision)
    del model,opt,scheduler;torch.cuda.empty_cache()
    return decision

def verify_decision(job,tag,decision):
    dest=job/decision["arm"]/f'seed_{decision["refiner_seed"]}'
    for stem,key in (("best_raw_model","raw_sha256"),("selected_blend_model","blend_sha256"),("final_model","final_sha256")):
        assert sha(dest/f"{stem}_{tag}.pt")==decision[key]
    assert decision["spec"]==BY_NAME[decision["arm"]].record()

def evaluate_one(ref,job,tag,decision,cutoff,status):
    import numpy as np
    import torch
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    from flow_matching_refiner.structure_metrics_all import metrics
    from .operations import infer
    spec=BY_NAME[decision["arm"]];seed=decision["refiner_seed"]
    dest=job/spec.name/f"seed_{seed}";verify_decision(job,tag,decision)
    model=ErrorSelectiveModel(spec.use_gate).to(ref.device)
    raw_state=torch.load(dest/f"best_raw_model_{tag}.pt",map_location=ref.device,weights_only=True)["state"]
    alpha=decision["blend"]["alpha"]
    blend_state=torch.load(dest/f"selected_blend_model_{tag}.pt",map_location=ref.device,weights_only=True)["state"] if alpha else None
    rows=[];inventory=[]
    for name in ref.manifest["test"]:
        status("locked_test_evaluation",dataset=ref.dataset,arm=spec.name,seed=seed,image=name)
        d=ref.data[name];model.load_state_dict(raw_state);model.eval().requires_grad_(False)
        raw,residual,meta=infer(model,d,spec,ref.dataset,ref.device,STEPS)
        native_arrays={}
        if residual is not None:native_arrays["raw_residual_native"]=residual.numpy()
        blend_meta=None
        if alpha:
            if decision["raw"]["epoch"]==decision["blend"]["epoch"]:
                candidate=raw;blend_meta=dict(meta,reused_raw=True)
            else:
                model.load_state_dict(blend_state)
                candidate,blend_residual,blend_meta=infer(model,d,spec,ref.dataset,ref.device,STEPS)
                if blend_residual is not None:native_arrays["blend_candidate_residual_native"]=blend_residual.numpy()
            blended=(1-alpha)*d["coarse"]+alpha*candidate
        else:blended=d["coarse"].clone()
        prediction=dest/f"{name}_predictions_{tag}.npz"
        np.savez_compressed(prediction,baseline=d["coarse"].numpy(),raw=raw.numpy(),blend=blended.numpy(),ground_truth=d["g"].numpy(),fov=d["fov"].numpy().astype(bool))
        native_path=None
        if native_arrays:
            native_path=dest/f"{name}_native_residuals_{tag}.npz";np.savez_compressed(native_path,**native_arrays)
        with np.load(prediction,allow_pickle=False) as saved:
            for kind in ("raw","blend"):
                assert saved[kind].dtype==np.float32 and np.isfinite(saved[kind]).all()
                calls=meta if kind=="raw" else blend_meta
                rows.append(dict(arm=spec.name,refiner_seed=seed,image=name,subject=subject(ref,name),output=kind,
                    selected_epoch=decision[kind]["epoch"],alpha=1. if kind=="raw" else alpha,
                    velocity_calls=calls["velocity_calls"] if calls else 0,gate_calls=calls["gate_calls"] if calls else 0,
                    tile_forward_batches=calls["tile_forward_batches"] if calls else 0,
                    **metrics(saved[kind][0],saved["ground_truth"][0],saved["baseline"][0],cutoff,saved["fov"][0])))
        inventory.append(dict(image=name,prediction_path=str(prediction),prediction_sha256=sha(prediction),
            residual_path=str(native_path) if native_path else None,residual_sha256=sha(native_path) if native_path else None,
            raw=meta,blend=blend_meta,native_probabilities_stored=False,full512_FP32_predictions_stored=True))
    with (dest/f"test_records_{tag}.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(dest/f"prediction_inventory_{tag}.json",inventory)
    write(dest/f"evaluation_completed_{tag}.json",dict(records=len(rows),test_images=len(ref.manifest["test"]),FP32_probabilities_verified=True))
    del model,raw_state,blend_state;torch.cuda.empty_cache();return rows

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=["train","evaluate","all"],default="all")
    parser.add_argument("--max-fits",type=int)
    parser.add_argument("--stop-after-dataset",choices=DATASETS)
    parser.add_argument("--cuda-device",type=int,default=0)
    args=parser.parse_args(argv)
    if not args.execute:print("No data/model execution; add --execute for the registered study.");return 0
    out=args.output.resolve();tag=out.name[-19:]
    out=assert_output_path(out,must_exist=True)
    active=read(out/"active_design.json");design_path=out/active["file"]
    assert design_path.resolve().parent==out and sha(design_path)==active["sha256"]
    registered=read(design_path)
    assert {k:v for k,v in registered.items() if k not in {"registered_at","supersedes"}}==plan()
    source_lock=read(out/f"source_lock_{tag}.json")
    assert all(sha(Path(p))==h for p,h in source_lock.items()),"Registered source changed"
    previous_path=out/f"protected_before_{tag}.json"
    previous=read(previous_path) if previous_path.exists() else protected_inventory()
    for p,h in source_lock.items():
        assert p not in previous or previous[p]==h;previous[p]=h
    for fixed in (design_path,out/"active_design.json",out/f"source_lock_{tag}.json"):
        previous[str(fixed.resolve())]=sha(fixed)
    if not previous_path.exists():write(previous_path,previous)
    assert all(sha(Path(p))==h for p,h in previous.items()),"Historical source changed before execution"
    state=dict(output=str(out),tag=tag,stage="starting",expected_final_fits=240,epochs=EPOCHS)
    def status(stage,**extra):
        state.update(stage=stage,updated_at=stamp(),**extra);write(out/f"controller_status_{tag}.json",state)
    import numpy as np
    import torch
    from flow_matching_refiner.structure_metrics_all import metrics,dependency_versions
    torch.cuda.set_device(args.cuda_device)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    environment=dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(args.cuda_device),cuda_device=args.cuda_device,libraries=dependency_versions())
    environment_path=out/f"environment_{tag}.json"
    if environment_path.exists():assert read(environment_path)==environment
    else:write(environment_path,environment)
    try:
        decisions=[];new_fits=0
        if args.phase in {"train","all"}:
            for dataset in DATASETS:
                job=out/f"{dataset}_{tag}";job.mkdir(exist_ok=True)
                ref,audit,thin=prepare(dataset,job,tag,"training")
                for p,h in audit["source_sha256_before"].items():
                    assert p not in previous or previous[p]==h;previous[p]=h
                write(out/f"all_sources_protected_{tag}.json",previous)
                for spec in ARMS:
                    views,pools,validation_rgb=create_views(ref,spec)
                    for seed in SEEDS:
                        dest=job/spec.name/f"seed_{seed}";locked=dest/f"selection_locked_{tag}.json"
                        if locked.exists():
                            decision=read(locked);verify_decision(job,tag,decision)
                        else:
                            if args.max_fits is not None and new_fits>=args.max_fits:
                                status("training_pending",new_fits_this_invocation=new_fits,completed_choices=len(decisions));return 0
                            if dest.exists():
                                archived=out/"interrupted_fits"/f"{dataset}_{spec.name}_{seed}_{stamp()}"
                                archived.parent.mkdir(exist_ok=True);assert not archived.exists()
                                assert dest.resolve().is_relative_to(out) and archived.resolve().is_relative_to(out)
                                dest.rename(archived)
                                write(archived/f"restart_note_{stamp()}.json",dict(reason="Incomplete fit retained; fresh same-seed initialization for final result",new_destination=str(dest)))
                            decision=train_one(ref,job,tag,spec,seed,views,pools,validation_rgb,status);new_fits+=1
                        decisions.append(dict(dataset=dataset,**decision))
                    del views,pools,validation_rgb;gc.collect()
                del ref;gc.collect();torch.cuda.empty_cache()
                if args.stop_after_dataset==dataset and dataset!=DATASETS[-1]:
                    status("training_pending_dataset_gate",completed_choices=len(decisions),stopped_after_dataset=dataset)
                    return 0
            assert len(decisions)==240
            lock=out/f"all_choices_locked_before_any_test_{tag}.json"
            if not lock.exists():write(lock,dict(locked_at=stamp(),decisions=decisions,design_sha256=sha(design_path),source_lock_sha256=sha(out/f"source_lock_{tag}.json")))
            else:assert read(lock)["decisions"]==decisions
            status("all_training_locked",completed_choices=240)
        lock=out/f"all_choices_locked_before_any_test_{tag}.json"
        if args.phase in {"evaluate","all"}:
            assert lock.exists(),"All240choices must be locked before test evaluation"
            decisions=read(lock)["decisions"];assert len(decisions)==240;lock_hash=sha(lock)
            if (out/f"all_sources_protected_{tag}.json").exists():previous=read(out/f"all_sources_protected_{tag}.json")
            for dataset in DATASETS:
                job=out/f"{dataset}_{tag}"
                finished=job/f"completed_training_evaluation_{tag}.json"
                if finished.exists():continue
                ref,audit,thin=prepare(dataset,job,tag,"evaluation");cutoff=thin["cutoff"]
                for p,h in audit["source_sha256_before"].items():assert previous[p]==h
                rows=[]
                for name in ref.manifest["test"]:
                    d=ref.data[name];base=d["coarse"].numpy()[0]
                    rows.append(dict(arm="baseline",refiner_seed=-1,image=name,subject=subject(ref,name),output="raw",selected_epoch=0,alpha=0.,velocity_calls=0,gate_calls=0,tile_forward_batches=0,
                        **metrics(base,d["g"].numpy()[0],base,cutoff,d["fov"].numpy()[0])))
                for decision in decisions:
                    if decision["dataset"]!=dataset:continue
                    dest=job/decision["arm"]/f'seed_{decision["refiner_seed"]}'
                    records=dest/f"test_records_{tag}.csv"
                    if (dest/f"evaluation_completed_{tag}.json").exists():
                        with records.open(newline="",encoding="utf-8") as f:rows.extend(csv.DictReader(f))
                    else:
                        partial=list(dest.glob(f"*_predictions_{tag}.npz"))+list(dest.glob(f"*_native_residuals_{tag}.npz"))
                        partial += [p for p in (records,dest/f"prediction_inventory_{tag}.json") if p.exists()]
                        if partial:
                            archive=out/"interrupted_evaluations"/f'{dataset}_{decision["arm"]}_{decision["refiner_seed"]}_{stamp()}'
                            archive.mkdir(parents=True,exist_ok=False)
                            assert archive.resolve().is_relative_to(out)
                            for p in partial:
                                target=archive/p.name
                                assert p.resolve().is_relative_to(out) and target.resolve().is_relative_to(out)
                                p.rename(target)
                        rows.extend(evaluate_one(ref,job,tag,decision,cutoff,status))
                assert len(rows)==161*len(ref.manifest["test"])
                with (job/f"all_test_records_{tag}.csv").open("w",newline="",encoding="utf-8") as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                assert sha(lock)==lock_hash
                write(finished,dict(completed_at=stamp(),fits=80,records=len(rows),test_images=len(ref.manifest["test"]),global240choices_locked_before_test=True,global_lock_sha256=lock_hash,FP32_predictions_verified=True,OOF_inputs_audited=True))
                del ref;gc.collect();torch.cuda.empty_cache()
            assert all(sha(p)==h for p,h in previous.items()),"Historical/source bytes changed"
            write(out/f"protected_after_{tag}.json",{p:sha(p) for p in previous})
            write(out/f"verified_training_evaluation_completed_{tag}.json",dict(completed_at=stamp(),new_final_fits=240,test_records=6440,global_lock_sha256=lock_hash,original_files_unchanged=True))
            status("complete",completed_choices=240,test_records=6440)
        return 0
    except BaseException:
        write(out/f"failed_{stamp()}.json",dict(traceback=traceback.format_exc()));status("failed");raise

if __name__=="__main__":raise SystemExit(main())
