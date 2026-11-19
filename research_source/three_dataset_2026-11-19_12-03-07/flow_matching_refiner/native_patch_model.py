"""Native-resolution Direct patch utilities, isolated from prior experiments.

The unchanged ErrorSelectiveModel(use_gate=False) supplies one residual call.
Training retains the original Direct FOV objective without FM or V3 extras.
Sliding-window inference never accepts GT/FOV labels: coarse and RGB only.

Tiling uses overlapping logit-residual averaging with a positive separable Hann
window, a fixed final edge anchor and replication only when an input image is
smaller than a tile. No context halo is used. GroupNorm computes tile-specific
statistics, so this operation is not equivalent to a full-image forward pass.
"""
from __future__ import annotations

import math
import random

import torch
from torch.nn import functional as F

from flow_matching_refiner.esfm_model import ErrorSelectiveModel,MARGIN,source_logits
from flow_matching_refiner.esfm_objective_all import losses as _direct_losses
from flow_matching_refiner.structure_metrics import soft_cldice_loss


def _positive_integer(value,name,minimum=1):
    if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _image(value,channels,name):
    tensor=torch.as_tensor(value)
    if tensor.ndim!=3 or tensor.shape[0]!=channels or min(tensor.shape[-2:])<1:
        raise ValueError(f"{name} must have shape [{channels},H,W]")
    return tensor


def tile_starts(length,tile,stride):
    """Anchored starts guarantee coverage, including an irregular final step."""
    _positive_integer(length,"length")
    _positive_integer(tile,"tile",8)
    _positive_integer(stride,"stride")
    if stride>tile:raise ValueError("stride cannot exceed tile because that leaves gaps")
    if length<=tile:return [0]
    starts=list(range(0,length-tile+1,stride))
    if starts[-1]!=length-tile:starts.append(length-tile)
    return starts


def positive_hann(tile):
    """CPU FP32 separable Hann; 1-D floor 1e-3 gives positive 2-D corners."""
    _positive_integer(tile,"tile",8)
    axis=torch.hann_window(tile,periodic=False,dtype=torch.float32).clamp_min(1e-3)
    return (axis[:,None]*axis[None,:])[None]


def apply_residual(coarse,delta):
    """Apply a same-shape logit residual to the original coarse probabilities.

    Zero residual preserves the >=.5 binary segmentation exactly, including
    coarse values 0, .5 and 1. Extreme probabilities inherit the unchanged
    source_logits clamp [1e-4,1-1e-4]; they are not silently treated as logits.
    """
    probability=torch.as_tensor(coarse).float()
    residual=torch.as_tensor(delta,device=probability.device).float()
    if probability.shape!=residual.shape:raise ValueError("Coarse/residual shapes must match")
    updated=(source_logits(probability)+residual).sigmoid()
    # Inference-only exact identity: logit/sigmoid can round the FP32 number
    # immediately below .5 to .5. Training uses its unchanged local endpoint
    # directly, so this zero-delta branch does not suppress training gradients.
    return torch.where(residual==0,probability,updated)


def project_residual_512(delta,dataset):
    """Project only the residual using the dataset's exact GT nearest rule.

    CHASE/DRIVE: Pillow float32-image nearest; HRF: OpenCV INTER_NEAREST.
    These rules can select different source indices and must not be unified.
    Apply the result to the ORIGINAL frozen 512 coarse map with apply_residual.
    Downsampling native probabilities would alter the coarse vessel coverage.
    """
    import numpy as np
    residual=_image(delta,1,"delta").detach().to(device="cpu",dtype=torch.float32)
    if not bool(torch.isfinite(residual).all()):raise ValueError("Residual contains nonfinite values")
    name=str(dataset).upper().replace("-","").replace("_","")
    array=residual[0].contiguous().numpy()
    if name in ("CHASE","CHASEDB1","DRIVE"):
        from PIL import Image
        projected=np.asarray(Image.fromarray(array).resize((512,512),Image.Resampling.NEAREST),dtype=np.float32).copy()
    elif name=="HRF":
        import cv2
        projected=cv2.resize(array,(512,512),interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError("dataset must be CHASE, DRIVE or HRF")
    return torch.from_numpy(np.ascontiguousarray(projected))[None]


def direct_loss(model,coarse,rgb,gt,fov):
    """Differentiable original Direct loss; unchanged coefficients/architecture.

    Returned parts are the original fm/bce/dice/cldice/gate diagnostics. The fm
    diagnostic is computed but is not included in the objective. There are no
    missed-thin or preservation-anchor terms and no unrolled flow training.
    """
    if getattr(model,"use_gate",False):raise ValueError("Native option3 uses the unchanged ungated Direct model")
    if coarse.ndim!=4 or coarse.shape[1]!=1 or gt.shape!=coarse.shape or fov.shape!=coarse.shape:
        raise ValueError("coarse/gt/fov must match [B,1,H,W]")
    if rgb.shape!=(coarse.shape[0],3,*coarse.shape[-2:]):
        raise ValueError("RGB must match [B,3,H,W]")
    source=source_logits(coarse)
    time=torch.zeros(len(coarse),device=coarse.device,dtype=coarse.dtype)
    gate=model.error_logits(coarse,rgb)
    velocity=model(source,time,coarse,rgb,gate)
    endpoint=source+velocity
    errors=((coarse>=.5)!=(gt>=.5)).to(gt.dtype)
    full_target=MARGIN*(2*gt-1)
    return _direct_losses(velocity,full_target-source,endpoint,coarse,gt,errors,
                          torch.zeros_like(gt),gate,use_fm=False,selective=False,
                          soft_cldice=soft_cldice_loss,fov=fov)


def training_center_pools(gt,fov):
    """Precompute CPU flat centre indices from TRAINING labels only.

    The optional cache avoids repeatedly scanning large HRF masks. Valid-FOV
    and positive-vessel indices can be large; keep them on CPU, never GPU.
    """
    truth=_image(gt,1,"gt").detach().to("cpu")>=.5
    valid=_image(fov,1,"fov").detach().to("cpu")>.5
    if truth.shape!=valid.shape:raise ValueError("GT/FOV shapes must match")
    fov_indices=torch.nonzero(valid.reshape(-1),as_tuple=False).flatten()
    if not len(fov_indices):raise ValueError("A training image needs nonempty FOV")
    vessel_indices=torch.nonzero((truth&valid).reshape(-1),as_tuple=False).flatten()
    return {"shape":tuple(truth.shape[-2:]),"fov_indices":fov_indices,"vessel_indices":vessel_indices}


def sample_training_patch(coarse,rgb,gt,fov,tile=256,rng=None,
                          vessel_probability=.5,center_pools=None):
    """Return aligned native crops and centre metadata, using training GT only.

    With probability .5, draw a centre uniformly from GT vessel pixels inside
    FOV; otherwise draw uniformly from FOV. If no vessel exists, use FOV. Crop
    origins are clamped at image edges. For undersized images RGB/coarse are
    replication padded, whereas GT/FOV are zero padded so artificial pixels do
    not supervise the loss. Supply a seeded random.Random for reproducibility.
    """
    _positive_integer(tile,"tile",8)
    probability=float(vessel_probability)
    if not math.isfinite(probability) or not 0<=probability<=1:
        raise ValueError("vessel_probability must lie in [0,1]")
    fields={"coarse":_image(coarse,1,"coarse"),"rgb":_image(rgb,3,"rgb"),
            "gt":_image(gt,1,"gt"),"fov":_image(fov,1,"fov")}
    height,width=fields["coarse"].shape[-2:]
    if any(tuple(value.shape[-2:])!=(height,width) for value in fields.values()):
        raise ValueError("All native images must have matching spatial shapes")
    pools=center_pools if center_pools is not None else training_center_pools(fields["gt"],fields["fov"])
    if tuple(pools["shape"])!=(height,width):raise ValueError("Centre cache belongs to a different image shape")
    generator=rng if rng is not None else random.Random()
    requested_vessel=generator.random()<probability
    choose_vessel=requested_vessel and len(pools["vessel_indices"])>0
    candidates=pools["vessel_indices"] if choose_vessel else pools["fov_indices"]
    if not len(candidates):raise ValueError("Centre cache contains no eligible FOV pixels")
    flat=int(candidates[generator.randrange(len(candidates))])
    centre_y,centre_x=divmod(flat,width)
    if not (0<=centre_y<height and 0<=centre_x<width):raise ValueError("Invalid cached centre")
    if not bool(fields["fov"][0,centre_y,centre_x]>.5):raise ValueError("Cached centre is outside FOV")
    if choose_vessel and not bool(fields["gt"][0,centre_y,centre_x]>=.5):raise ValueError("Cached vessel centre is not GT vessel")
    top=min(max(centre_y-tile//2,0),max(height-tile,0))
    left=min(max(centre_x-tile//2,0),max(width-tile,0))
    crops={}
    for name,value in fields.items():
        crop=value[:,top:min(top+tile,height),left:min(left+tile,width)]
        pad=(0,tile-crop.shape[-1],0,tile-crop.shape[-2])
        if pad[1] or pad[3]:
            if name in ("coarse","rgb"):
                crop=F.pad(crop[None].float(),pad,mode="replicate")[0]
            else:
                crop=F.pad(crop.float(),pad,mode="constant",value=0.)
        crops[name]=crop.clone()
    metadata={"centre_y":centre_y,"centre_x":centre_x,"top":top,"left":left,
              "native_height":height,"native_width":width,"tile":tile,
              "requested_vessel_centre":requested_vessel,"used_vessel_centre":choose_vessel,
              "fallback_no_vessels":requested_vessel and not choose_vessel,
              "rng_supplied":rng is not None,
              "padding":"RGB/coarse replicate; GT/FOV zero so padded targets are excluded"}
    return crops,metadata


@torch.no_grad()
def predict_tiled(model,coarse,rgb,device,tile=256,stride=128,batch=4,return_residual=False):
    """Stitch GT-free direct predictions with bounded accelerator tile batches.

    Inputs/accumulators stay on CPU; only <=batch tiles go to the given device.
    Returns (CPU FP32 probability [1,H,W], metadata), or (probability, residual,
    metadata) when return_residual=True. velocity_calls counts per-tile evaluations;
    velocity_forward_batches separately records actual batched forward calls.
    Mode is restored after inference. No weights are moved or altered here.
    """
    _positive_integer(tile,"tile",8);_positive_integer(stride,"stride");_positive_integer(batch,"batch")
    if stride>tile:raise ValueError("stride cannot exceed tile")
    if getattr(model,"use_gate",False):raise ValueError("Native patch inference uses the ungated Direct model")
    coarse_cpu=_image(coarse,1,"coarse").detach().to(device="cpu",dtype=torch.float32)
    rgb_cpu=_image(rgb,3,"rgb").detach().to(device="cpu",dtype=torch.float32)
    if coarse_cpu.shape[-2:]!=rgb_cpu.shape[-2:]:raise ValueError("Coarse/RGB spatial shapes must match")
    if not bool(torch.isfinite(coarse_cpu).all()) or not bool(torch.isfinite(rgb_cpu).all()):
        raise ValueError("Native input contains nonfinite values")
    if not bool(((coarse_cpu>=0)&(coarse_cpu<=1)).all()):raise ValueError("Coarse probabilities must lie in [0,1]")
    height,width=coarse_cpu.shape[-2:]
    padded_height,padded_width=max(height,tile),max(width,tile)
    padding=(0,padded_width-width,0,padded_height-height)
    if padding[1] or padding[3]:
        coarse_cpu=F.pad(coarse_cpu[None],padding,mode="replicate")[0]
        rgb_cpu=F.pad(rgb_cpu[None],padding,mode="replicate")[0]
    ys,xs=tile_starts(padded_height,tile,stride),tile_starts(padded_width,tile,stride)
    positions=[(y,x) for y in ys for x in xs]
    window=positive_hann(tile)
    numerator=torch.zeros((1,padded_height,padded_width),dtype=torch.float32)
    coverage=torch.zeros_like(numerator)
    old_training=getattr(model,"training",False)
    model.eval();forward_batches=0
    try:
        for start in range(0,len(positions),batch):
            current=positions[start:start+batch]
            c=torch.stack([coarse_cpu[:,y:y+tile,x:x+tile] for y,x in current]).to(device)
            image=torch.stack([rgb_cpu[:,y:y+tile,x:x+tile] for y,x in current]).to(device)
            source=source_logits(c)
            time=torch.zeros(len(current),device=c.device,dtype=c.dtype)
            residual=model(source,time,c,image,None)
            if residual.shape!=(len(current),1,tile,tile) or not bool(torch.isfinite(residual).all()):
                raise RuntimeError("Tile model returned invalid logit residual")
            residual=residual.detach().to(device="cpu",dtype=torch.float32)
            for index,(y,x) in enumerate(current):
                numerator[:,y:y+tile,x:x+tile].add_(residual[index]*window)
                coverage[:,y:y+tile,x:x+tile].add_(window)
            forward_batches+=1
    finally:
        model.train(old_training)
    valid_coverage=coverage[:,:height,:width]
    minimum=float(valid_coverage.min())
    if minimum<=0:raise RuntimeError("Sliding-window stitching left uncovered native pixels")
    stitched_residual=numerator[:,:height,:width]/valid_coverage
    result=apply_residual(coarse_cpu[:,:height,:width],stitched_residual)
    metadata={"native_height":height,"native_width":width,"tile":tile,"stride":stride,"batch":batch,
              "totaltiles":len(positions),"total_tiles":len(positions),"velocity_calls":len(positions),
              "velocity_forward_batches":forward_batches,"gate_calls":0,"coverage_min":minimum,
              "coverage_max":float(valid_coverage.max()),"padded_height":padded_height,"padded_width":padded_width,
              "hann_1d_floor":1e-3,"overlaprule":"FP32 logit-residual average weighted by positive separable nonperiodic Hann; sigmoid only after fusion",
              "halo":0,"edge_rule":"last tile anchored to final native edge; replicate input padding only for images smaller than tile",
              "normalization_note":"GroupNorm is computed within each tile; output need not equal full-image inference",
              "output_device":"cpu","output_dtype":"float32",
              "residual_projection":"project delta only with dataset GT nearest rule; reapply to original frozen coarse512"}
    if return_residual:return result.contiguous(),stitched_residual.contiguous(),metadata
    return result.contiguous(),metadata
