"""Single-factor objective/input/augmentation operations, without legacy mutations."""
from dataclasses import replace
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import source_logits, make_target, predict
from flow_matching_refiner.esfm_objective_all import losses as original_losses
from flow_matching_refiner.refinement_objectives_v3 import correct_pixel_identity, missed_thin_recovery
from flow_matching_refiner.structure_metrics import soft_cldice_loss
from flow_matching_refiner.native_patch_model import predict_tiled, project_residual_512, apply_residual

def condition_rgb(rgb, spec):
    return rgb if spec.use_rgb else torch.full_like(rgb, .5)

def objective(spec, velocity, target_velocity, endpoint, coarse, truth, errors, thin, gate, fov):
    total, parts = original_losses(velocity, target_velocity, endpoint, coarse, truth, errors, thin,
        gate, spec.use_fm, spec.selective_segmentation, soft_cldice_loss, fov)
    anchor = correct_pixel_identity(endpoint.sigmoid(), coarse, truth, fov) if spec.anchor_weight else endpoint.sum()*0.
    recovery = missed_thin_recovery(endpoint, coarse, truth, thin, fov) if spec.thin_weight else endpoint.sum()*0.
    total = total + (spec.dice_weight-.5)*parts["dice"] + spec.anchor_weight*anchor + spec.thin_weight*recovery
    return total, {**parts, "thin_recovery": recovery, "identity": anchor}

def aligned_augmentation(fields, rng, enabled):
    """Always consume exactly k/flip/flip draws, preserving paired later sampling."""
    k = rng.randrange(4); horizontal = rng.random()<.5; vertical = rng.random()<.5
    if not enabled:
        return fields, dict(rotation90=0, flip_horizontal=False, flip_vertical=False,
                            drawn_rotation90=k, drawn_flip_horizontal=horizontal, drawn_flip_vertical=vertical)
    values = [torch.rot90(value, k, (-2,-1)) for value in fields]
    if horizontal: values = [value.flip(-1) for value in values]
    if vertical: values = [value.flip(-2) for value in values]
    return values, dict(rotation90=k, flip_horizontal=horizontal, flip_vertical=vertical,
                        drawn_rotation90=k, drawn_flip_horizontal=horizontal, drawn_flip_vertical=vertical)

def patch_fields(sample, spec, dataset):
    """No mutation: source data remain untouched, and original-native pools stay external."""
    if spec.patch_grid == "512":
        return dict(coarse=sample["coarse"], rgb=sample["rgb"], gt=sample["g"], fov=sample["fov"])
    result = dict(coarse=sample["n_coarse"], rgb=sample["n_rgb"], gt=sample["n_g"], fov=sample["n_fov"])
    shape = sample["n_coarse"].shape[-2:]
    if spec.degrade_rgb:
        result["rgb"] = F.interpolate(sample["rgb"].cpu()[None], size=shape, mode="bilinear", align_corners=False)[0].contiguous()
    if spec.degrade_gt:
        if sample["g"] is None: raise ValueError("Retained training/validation512GT is required")
        mode = "nearest" if dataset=="HRF" else "nearest-exact"
        result["gt"] = F.interpolate(sample["g"].cpu().float()[None], size=shape, mode=mode)[0].contiguous()
    return result

def sample_patch(view, original_truth, pools, rng, spec):
    """Keep sampling evidence original even when its loss target is degraded."""
    from flow_matching_refiner.native_patch_model import sample_training_patch
    crops,meta=sample_training_patch(view["coarse"],view["rgb"],original_truth,view["fov"],
        tile=256,rng=rng,vessel_probability=spec.vessel_probability,center_pools=pools)
    if spec.degrade_gt:
        top,left=meta["top"],meta["left"]
        height,width=view["gt"].shape[-2:]
        crop=view["gt"][:,top:min(top+256,height),left:min(left+256,width)]
        crops["gt"]=F.pad(crop.float(),(0,256-crop.shape[-1],0,256-crop.shape[-2]),value=0.)
    return crops,meta

@torch.no_grad()
def infer(model, sample, spec, dataset, device, steps=8, rgb_override=None):
    """Inference references coarse/RGB only; degradation of GT is training-only."""
    if spec.family == "grid":
        coarse=sample["coarse"][None].to(device)
        rgb=condition_rgb(sample["rgb"][None].to(device),spec)
        p,meta=predict(model,coarse,rgb,steps,spec.direct)
        return p.cpu()[0], None, dict(velocity_calls=meta["velocity_calls"],gate_calls=meta["error_gate_calls"],tile_forward_batches=0)
    if spec.patch_grid=="512":
        coarse,rgb=sample["coarse"],sample["rgb"]
    else:
        coarse,rgb=sample["n_coarse"],sample["n_rgb"]
        if spec.degrade_rgb and rgb_override is None:
            rgb=F.interpolate(sample["rgb"].cpu()[None],size=coarse.shape[-2:],mode="bilinear",align_corners=False)[0]
    if rgb_override is not None: rgb=rgb_override
    _,residual,meta=predict_tiled(model,coarse,condition_rgb(rgb,spec),device,
                                tile=256,stride=128,batch=16,return_residual=True)
    result=apply_residual(sample["coarse"],project_residual_512(residual,dataset))
    return result,residual,dict(velocity_calls=meta["velocity_calls"],gate_calls=0,
        tile_forward_batches=meta["velocity_forward_batches"],patch_grid=spec.patch_grid,
        inference_height=meta["native_height"],inference_width=meta["native_width"])
