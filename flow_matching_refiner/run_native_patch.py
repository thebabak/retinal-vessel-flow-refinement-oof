"""Explicit isolated native-patch experiment; old articles/results stay read-only.

Fifteen fresh Direct refiners use audited retained OOF coarse inputs and native
RGB/annotations. All selections across three datasets precede any test scoring.
"""
from __future__ import annotations
from flow_matching_refiner.portable_paths import current, output_root, protected_inputs_and_code, assert_output_path
import argparse,csv,json,random,shutil,traceback,gc
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import write_json,stamp,OLD

SEEDS=[42,123,2026,3000,3001]
DATASETS=["CHASE","DRIVE","HRF"]
ARM="native_patch_direct"
ALPHAS=[0.,.25,.5,.75,1.]
TILE=256
STRIDE=128
TRAIN_BATCH=16
INFER_BATCH=16
PATCHES_PER_IMAGE=4
EPOCHS=120
V3_ROOT=current().v3_root


def subject(ref,name):
    return int(ref.data[name]["subject"]) if ref.dataset=="CHASE" else ref.manifest["test"].index(name)


def score_dice(ref,p,name):
    d=ref.data[name]
    args=(p,d["g"]) if ref.dataset=="CHASE" else (p,d["g"],d["fov"]>.5)
    return ref.score(*args)["dice"]


def protected_files():
    return protected_inputs_and_code()


def prepare(dataset,out,tag):
    import numpy as np
    from flow_matching_refiner.native_patch_data import load_native_dataset
    from flow_matching_refiner.structure_metrics_all import derive_thin_cutoff
    ref,audit=load_native_dataset(dataset,out/"input_audit","cuda")
    thin=derive_thin_cutoff([ref.data[n]["g"].numpy()[0] for n in ref.manifest["train"]],
                          [ref.data[n]["fov"].numpy()[0] for n in ref.manifest["train"]])
    old_tag=OLD[dataset].name[-19:]
    prior=json.loads((OLD[dataset]/f"thin_definition_{old_tag}.json").read_text())
    assert thin["cutoff"]==prior["cutoff"]
    write_json(out/f"thin_definition_{tag}.json",dict(**thin,training_image_ids=ref.manifest["train"],
               comparison_grid_only=True,native_cutoff_derived=False))
    for n in ref.manifest["test"]:
        d=ref.data[n]
        with np.load(OLD[dataset]/"direct/seed_42"/f"{n}_predictions_{old_tag}.npz",allow_pickle=False) as a:
            for key,field in (("baseline","coarse"),("ground_truth","g"),("fov","fov")):
                assert a[key].dtype==d[field].numpy().dtype and np.array_equal(a[key],d[field].numpy())
    return ref,audit,thin


def train_one(ref,out,tag,seed,pools,status_callback):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_model import (ErrorSelectiveModel,direct_loss,
          sample_training_patch,predict_tiled,project_residual_512,apply_residual)
    dest=out/ARM/f"seed_{seed}";dest.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ErrorSelectiveModel(False).to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    rng=random.Random(seed+141001)
    best_raw=dict(epoch=None,validation_dice=-1.)
    baseline_val=float(np.mean([score_dice(ref,ref.data[n]["coarse"],n) for n in ref.manifest["validation"]]))
    best_blend=dict(epoch=0,alpha=0.,validation_dice=baseline_val)
    blend_path=dest/f"selected_blend_model_{tag}.pt"
    torch.save(dict(state=model.state_dict(),selection=best_blend),blend_path)
    validation_metadata=[]

    def validation(epoch):
        scores={alpha:[] for alpha in ALPHAS}
        for n in ref.manifest["validation"]:
            d=ref.data[n]
            _,delta,meta=predict_tiled(model,d["n_coarse"],d["n_rgb"],ref.device,
                           tile=TILE,stride=STRIDE,batch=INFER_BATCH,return_residual=True)
            raw=apply_residual(d["coarse"],project_residual_512(delta,ref.dataset))
            for alpha in ALPHAS:
                p=d["coarse"] if alpha==0 else (1-alpha)*d["coarse"]+alpha*raw
                scores[alpha].append(score_dice(ref,p,n))
            validation_metadata.append(dict(epoch=epoch,image=n,**meta))
            del delta,raw
        return {alpha:float(np.mean(values)) for alpha,values in scores.items()}

    with (dest/f"training_{tag}.csv").open("x",newline="",encoding="utf-8") as th, \
         (dest/f"validation_{tag}.csv").open("x",newline="",encoding="utf-8") as vh, \
         (dest/f"patch_draws_{tag}.csv").open("x",newline="",encoding="utf-8") as ph:
        tw,vw,pw=csv.writer(th),csv.writer(vh),csv.writer(ph)
        tw.writerow(["epoch","loss","bce","dice","soft_cldice","gate","lr","patches_seen","optimizer_updates"])
        vw.writerow(["epoch","alpha","validation_dice"]);vw.writerow([0,0.,baseline_val])
        pw.writerow(["epoch","patch_index","image","centre_y","centre_x","top","left","used_vessel_centre","rotation90","flip_horizontal","flip_vertical"])
        for epoch in range(1,EPOCHS+1):
            status_callback("training",dataset=ref.dataset,seed=seed,epoch=epoch,epochs=EPOCHS,current_output=str(out))
            model.train();names=ref.manifest["train"]*PATCHES_PER_IMAGE;rng.shuffle(names)
            aggregate=np.zeros(5);seen=0;updates=0;lr=opt.param_groups[0]["lr"]
            for start in range(0,len(names),TRAIN_BATCH):
                samples=[]
                for index,n in enumerate(names[start:start+TRAIN_BATCH],start=start):
                    d=ref.data[n]
                    c,meta=sample_training_patch(d["n_coarse"],d["n_rgb"],d["n_g"],d["n_fov"],
                           tile=TILE,rng=rng,vessel_probability=.5,center_pools=pools[n])
                    k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5
                    fields=[c[key] for key in ("coarse","rgb","gt","fov")]
                    fields=[torch.rot90(t,k,(-2,-1)) for t in fields]
                    if fh:fields=[t.flip(-1) for t in fields]
                    if fv:fields=[t.flip(-2) for t in fields]
                    samples.append(fields)
                    pw.writerow([epoch,index,n,meta["centre_y"],meta["centre_x"],meta["top"],meta["left"],meta["used_vessel_centre"],k,fh,fv])
                coarse,rgb,gt,fov=[torch.stack([a[j] for a in samples]).to(ref.device).clone() for j in range(4)]
                loss,parts=direct_loss(model,coarse,rgb,gt,fov)
                assert bool(torch.isfinite(loss)),"Nonfinite native-patch training loss"
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(model.error_net.parameters(),1.)
                opt.step();updates+=1;seen+=len(coarse)
                aggregate+=np.array([loss.item(),parts["bce"].item(),parts["dice"].item(),parts["cldice"].item(),parts["gate"].item()])*len(coarse)
            scheduler.step();assert seen==len(ref.manifest["train"])*PATCHES_PER_IMAGE
            assert updates==(len(ref.manifest["train"])+3)//4
            tw.writerow([epoch,*(aggregate/seen).tolist(),lr,seen,updates]);th.flush();ph.flush()
            if epoch%10==0:
                status_callback("validation_tiling",dataset=ref.dataset,seed=seed,epoch=epoch,epochs=EPOCHS,current_output=str(out))
                values=validation(epoch)
                if values[1.]>best_raw["validation_dice"]+1e-12:
                    best_raw=dict(epoch=epoch,validation_dice=values[1.])
                    torch.save(dict(state=model.state_dict(),selection=best_raw),dest/f"best_raw_model_{tag}.pt")
                alpha=max(ALPHAS,key=lambda a:values[a])
                if values[alpha]>best_blend["validation_dice"]+1e-12:
                    best_blend=dict(epoch=epoch,alpha=alpha,validation_dice=values[alpha])
                    torch.save(dict(state=model.state_dict(),selection=best_blend),blend_path)
                for alpha in ALPHAS:vw.writerow([epoch,alpha,values[alpha]])
                vh.flush()
                print(ref.dataset,seed,epoch,"raw_val",values[1.],"best",best_raw,"blend",best_blend,flush=True)
    final_path=dest/f"final_model_{tag}.pt";torch.save(dict(state=model.state_dict(),epoch=EPOCHS),final_path)
    write_json(dest/f"validation_tiling_{tag}.json",validation_metadata)
    decision=dict(arm=ARM,refiner_seed=seed,baseline_seed=42,raw=best_raw,blend=best_blend,locked_at=stamp(),
       raw_sha256=sha256_file(dest/f"best_raw_model_{tag}.pt"),blend_sha256=sha256_file(blend_path),final_sha256=sha256_file(final_path),
       training_parameters=sum(p.numel() for p in model.parameters()),
       inference_velocity_parameters=sum(p.numel() for p in model.velocity_net.parameters()),inference_gate_parameters=0,
       tile=TILE,stride=STRIDE,training_patches_per_image_per_epoch=PATCHES_PER_IMAGE,train_batch=TRAIN_BATCH,infer_batch=INFER_BATCH)
    write_json(dest/f"selection_locked_{tag}.json",decision)
    del model,opt,scheduler;torch.cuda.empty_cache()
    return decision


def evaluate_one(ref,out,tag,decision,cutoff,status_callback):
    import numpy as np
    import torch
    from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,predict_tiled,project_residual_512,apply_residual
    from flow_matching_refiner.structure_metrics_all import metrics
    seed=decision["refiner_seed"];dest=out/ARM/f"seed_{seed}";model=ErrorSelectiveModel(False).to(ref.device)
    raw_path=dest/f"best_raw_model_{tag}.pt";blend_path=dest/f"selected_blend_model_{tag}.pt"
    assert sha256_file(raw_path)==decision["raw_sha256"] and sha256_file(blend_path)==decision["blend_sha256"]
    raw_state=torch.load(raw_path,map_location=ref.device,weights_only=True)["state"]
    alpha=decision["blend"]["alpha"]
    blend_state=torch.load(blend_path,map_location=ref.device,weights_only=True)["state"] if alpha else None
    rows=[];inference=[]
    for n in ref.manifest["test"]:
        status_callback("locked_test_evaluation",dataset=ref.dataset,seed=seed,image=n,current_output=str(out))
        d=ref.data[n];model.load_state_dict(raw_state)
        p_native,delta_native,meta=predict_tiled(model,d["n_coarse"],d["n_rgb"],ref.device,
                  tile=TILE,stride=STRIDE,batch=INFER_BATCH,return_residual=True)
        raw=apply_residual(d["coarse"],project_residual_512(delta_native,ref.dataset))
        arrays=dict(raw_residual_native=delta_native.numpy(),raw_probability_native=p_native.numpy())
        blend_meta=None
        if alpha:
            if decision["raw"]["epoch"]==decision["blend"]["epoch"]:
                blend_delta=delta_native;blend_meta=dict(meta,reused_raw=True)
            else:
                model.load_state_dict(blend_state)
                _,blend_delta,blend_meta=predict_tiled(model,d["n_coarse"],d["n_rgb"],ref.device,
                        tile=TILE,stride=STRIDE,batch=INFER_BATCH,return_residual=True)
            candidate=apply_residual(d["coarse"],project_residual_512(blend_delta,ref.dataset))
            blended=(1-alpha)*d["coarse"]+alpha*candidate
            arrays["blend_candidate_residual_native"]=blend_delta.numpy()
        else:blended=d["coarse"].clone()
        native_path=dest/f"{n}_native_predictions_{tag}.npz";np.savez_compressed(native_path,**arrays)
        prediction_path=dest/f"{n}_predictions_{tag}.npz"
        np.savez_compressed(prediction_path,baseline=d["coarse"].numpy(),raw=raw.numpy(),blend=blended.numpy(),
                            ground_truth=d["g"].numpy(),fov=d["fov"].numpy().astype(bool))
        with np.load(prediction_path,allow_pickle=False) as saved:
            for kind in ("raw","blend"):
                assert np.isfinite(saved[kind]).all()
                calls=meta if kind=="raw" else blend_meta
                rows.append(dict(arm=ARM,refiner_seed=seed,image=n,subject=subject(ref,n),output=kind,
                    selected_epoch=decision[kind]["epoch"],alpha=1. if kind=="raw" else alpha,
                    velocity_calls=calls["velocity_calls"] if calls else 0,gate_calls=0,
                    tile_forward_batches=calls["velocity_forward_batches"] if calls else 0,
                    native_height=int(d["n_coarse"].shape[-2]),native_width=int(d["n_coarse"].shape[-1]),
                    **metrics(saved[kind][0],saved["ground_truth"][0],saved["baseline"][0],cutoff,saved["fov"][0])))
        inference.append(dict(image=n,native_path=str(native_path),native_sha256=sha256_file(native_path),
            prediction_path=str(prediction_path),prediction_sha256=sha256_file(prediction_path),raw=meta,blend=blend_meta,
            alpha=alpha,native_reference=str(out/"native_references"/f"{n}_native_reference_{tag}.npz")))
        del p_native,delta_native,raw,blended,arrays
        if alpha:del blend_delta,candidate
    with (dest/f"test_records_{tag}.csv").open("x",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(dest/f"inference_{tag}.json",inference)
    write_json(dest/f"completed_{tag}.json",dict(records=len(rows),test_images=len(ref.manifest["test"]),raw_and_blend_stored=True,
               native_outputs_stored=True,primary_evaluation_grid=512))
    del model,raw_state,blend_state;torch.cuda.empty_cache()
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--execute",action="store_true");parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not args.execute:print("No training or inference; --execute required.");return
    out=(args.output or output_root()/f"native_patch_study_{stamp()}").resolve();assert_output_path(out);out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,stage="hashing_previous_artifacts",new_final_fits=15,pilot_fits=0,
        seeds=SEEDS,datasets=DATASETS,arm=ARM,epochs=EPOCHS,tile=TILE,stride=STRIDE,train_batch=TRAIN_BATCH,infer_batch=INFER_BATCH,
        patches_per_training_image_per_epoch=PATCHES_PER_IMAGE)
    def status(stage,**extra):
        state.update(stage=stage,updated_at=stamp(),**extra);write_json(out/f"controller_status_{tag}.json",state)
        write_json(output_root()/"current_native_patch_work.json",state)
    status("hashing_previous_artifacts");old=protected_files();write_json(out/f"protected_before_{tag}.json",old)
    sources={}
    for path in (ROOT/"flow_matching_refiner").glob("*.py"):
        if path.name=="report_native_patch.py":continue
        dest=out/"source"/f"{path.stem}_{tag}.py";dest.parent.mkdir(exist_ok=True);shutil.copy2(path,dest);sources[path.name]=sha256_file(path)
    study_plan=dict(**state,source_sha256=sources,new_baseline_fits=0,baseline_seed=42,
       loss="unchanged old Direct:0.5BCE+0.5Dice+0.1softclDice+0.1balancedauxgate; noFM,noO1/O2extra losses",
       optimizer="AdamW3e-4 weight_decay1e-4 cosine120epochs FP32 separatebranchgradclip1",
       sampling="four native256patches per trainimage/epoch; eachcentre50percentGTvesselinsideFOV/50percentuniformFOV; labels trainonly",
       augmentation="aligned90degree rotations andhorizontal/verticalflips oncoarse/RGB/GT/FOV",
       budget="same sampled training pixels and optimizer updates per image-epoch as old four512image batches; nativepatch batch16",
       fusion="positiveHann weighted logit-residual fusion tile256stride128; fixedlastedge; nohalo; GroupNormtile statistics",
       projection="native logitresidual projected to512 with PillowfloatnearestCHASE/DRIVE orcv2INTER_NEARESTHRF; reapplied tooriginal512coarselogit",
       comparison_grid="original512GT/FOV/cutoff/threshold0.5; nativeprobabilities andreferences archived separately; no native-grid score claims",
       secondary_blend="0,.25,.5,.75,1 on512comparison grid with exactoriginalbaseline; rawlearnedmodelprimary",
       selection="rawhighest mean validationimageDice at epochs10..120; blendbest validationalpha/epochincludingidentity; all15choiceslocked beforeANYtest",
       hypothesis_scope="native-patch recipe comparison changes resolution/sampling/tiling/normalization; no matched512patchcontrol or purecausalresolutionclaim",
       test_history="previously examined cohorts; exploratory conditional comparisons; one frozen baseline per dataset",
       old_manuscript_and_results_unchanged=True)
    write_json(out/f"study_plan_{tag}.json",study_plan);print(json.dumps(state),flush=True)
    try:
        import numpy as np
        import torch
        from flow_matching_refiner.native_patch_model import training_center_pools
        from flow_matching_refiner.structure_metrics_all import metrics,dependency_versions
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        decisions=[]
        for dataset in DATASETS:
            job=out/f"{dataset}_{tag}";job.mkdir();status("loading_native_inputs",dataset=dataset,current_output=str(job))
            ref,audit,thin=prepare(dataset,job,tag)
            for p,h in audit["source_sha256_before"].items():assert p not in old or old[p]==h;old[p]=h
            write_json(out/f"all_sources_protected_before_{tag}.json",old)
            plan=dict(dataset=dataset,arms=[ARM],refiner_training_seeds=SEEDS,fixed_baseline_seed=42,epochs=EPOCHS,
               batch=TRAIN_BATCH,tile=TILE,stride=STRIDE,infer_batch=INFER_BATCH,patches_per_image=PATCHES_PER_IMAGE,
               old_experiment=str(OLD[dataset]),v3_experiment=str(V3_ROOT/f"{dataset}_{V3_ROOT.name[-19:]}"),
               primary_output="raw validation-selected learned nativepatch model alpha1",secondary_output="validation-selected baseline blend on512grid",
               fixed_settings_before_training=True,study_plan=str(out/f"study_plan_{tag}.json"),native_source_audit=audit["output_audit"],
               projection=study_plan["projection"],fusion=study_plan["fusion"],comparison_grid=512,threshold=.5)
            write_json(job/f"plan_{tag}.json",plan)
            write_json(job/f"environment_{tag}.json",dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
                structural_libraries=dependency_versions(),input_audit=audit["output_audit"]))
            pools={n:training_center_pools(ref.data[n]["n_g"],ref.data[n]["n_fov"]) for n in ref.manifest["train"]}
            for seed in SEEDS:
                decision=train_one(ref,job,tag,seed,pools,status);decisions.append(dict(dataset=dataset,**decision))
            del pools,ref;gc.collect();torch.cuda.empty_cache()
        global_path=out/f"all_choices_locked_before_any_test_{tag}.json"
        write_json(global_path,dict(locked_at=stamp(),decisions=decisions,fixed_study_plan_sha256=sha256_file(out/f"study_plan_{tag}.json")))
        global_hash=sha256_file(global_path)
        for dataset in DATASETS:
            job=out/f"{dataset}_{tag}";local=[{k:v for k,v in d.items() if k!="dataset"} for d in decisions if d["dataset"]==dataset]
            write_json(job/f"all_selections_before_test_{tag}.json",dict(locked_at=stamp(),decisions=local,global_lock_path=str(global_path),global_lock_sha256=global_hash))
        for dataset in DATASETS:
            job=out/f"{dataset}_{tag}";prep=job/"evaluation_inputs";prep.mkdir();status("loading_evaluation_inputs",dataset=dataset,current_output=str(job))
            ref,audit,thin=prepare(dataset,prep,tag);cutoff=thin["cutoff"]
            for p,h in audit["source_sha256_before"].items():assert old[p]==h
            refs=job/"native_references";refs.mkdir();rows=[]
            for n in ref.manifest["test"]:
                d=ref.data[n]
                np.savez_compressed(refs/f"{n}_native_reference_{tag}.npz",coarse_native=d["n_coarse"].numpy(),
                    ground_truth_native=d["n_g"].numpy().astype(bool),fov_native=d["n_fov"].numpy().astype(bool))
                base=d["coarse"].numpy()[0];gt=d["g"].numpy()[0]
                rows.append(dict(arm="baseline",refiner_seed=-1,image=n,subject=subject(ref,n),output="raw",selected_epoch=0,alpha=0.,
                    velocity_calls=0,gate_calls=0,tile_forward_batches=0,native_height=int(d["n_coarse"].shape[-2]),native_width=int(d["n_coarse"].shape[-1]),
                    **metrics(base,gt,base,cutoff,d["fov"].numpy()[0])))
            for decision in decisions:
                if decision["dataset"]==dataset:rows.extend(evaluate_one(ref,job,tag,decision,cutoff,status))
            assert len(rows)==11*len(ref.manifest["test"])
            with (job/f"all_test_records_{tag}.csv").open("x",newline="",encoding="utf-8") as h:
                w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            assert sha256_file(global_path)==global_hash
            assert all(sha256_file(p)==h for p,h in old.items()),"Historical/source files changed"
            write_json(job/f"completed_training_evaluation_{tag}.json",dict(completed_at=stamp(),fits=5,records=len(rows),test_images=len(ref.manifest["test"]),
                baseline_trainings=0,choices_locked_before_test=True,global_choices_locked_before_test=True,fixed_settings_before_training=True,
                previous_files_byte_identical=True,OOF_maps_audited=True,native_source_audit_passed=True,original512_fields_unchanged=True,
                lock_sha256=sha256_file(job/f"all_selections_before_test_{tag}.json"),global_lock_path=str(global_path),global_lock_sha256=global_hash,
                native_prediction_archives=5*len(ref.manifest["test"]),primary_evaluation_grid=512))
            del ref;gc.collect();torch.cuda.empty_cache();print("DATASET_COMPLETE",dataset,str(job),flush=True)
        after={p:sha256_file(p) for p in old};assert after==old
        write_json(out/f"protected_after_{tag}.json",after)
        write_json(out/f"all_training_completed_{tag}.json",dict(completed_at=stamp(),new_fits=15,records=440,
                   datasets=DATASETS,previous_files_byte_identical=True,global_lock_path=str(global_path),global_lock_sha256=global_hash))
        status("all_training_evaluation_complete",current_output=str(out));print("NATIVE_PATCH_COMPLETE",str(out),flush=True)
    except BaseException:
        write_json(out/f"failed_{tag}.json",dict(failed_at=stamp(),traceback=traceback.format_exc()));status("failed");raise


if __name__=="__main__":main()
