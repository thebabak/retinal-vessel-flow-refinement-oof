"""Two separately specified FOV-aware refinement objectives.

Option 1: unchanged Direct segmentation objective plus targeted recovery of
training-GT thin skeleton pixels missed by coarse, and correct-pixel anchoring.
Option 2: unchanged selective ESFM velocity objective, but segmentation losses
observe the full actual endpoint instead of a GT-oracle composite, plus the
same correct-pixel anchor. There is no extra thin-recovery term in option 2.

Only objectives change. Architectures, inference, targets, existing coefficients,
gate floor and historical files are not modified by this module. GT-derived
supports are training-only supervision; they are never inference inputs.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from flow_matching_refiner.esfm_objective_all import losses as _original_losses


def _option(value):
    if isinstance(value,bool):
        raise ValueError("option must identify option1 or option2")
    if value in (1,"option1","direct_thin"):
        return 1
    if value in (2,"option2","esfm_full_endpoint"):
        return 2
    raise ValueError("option must be 1/option1/direct_thin or 2/option2/esfm_full_endpoint")


def _weight(value,name):
    result=float(value)
    if not math.isfinite(result) or result<0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def correct_pixel_identity(probability,coarse,truth,fov):
    """Unweighted class-balanced squared probability deviation on correct pixels.

    Correct foreground and correct background supports each contribute one
    class mean, averaged equally over nonempty classes in the batch. This
    prevents a large correctly negative background drowning out vessels.
    With no correct pixels the returned differentiable scalar is exactly zero.
    The frozen coarse probability is a detached reference, not an update target.
    """
    valid=fov>.5
    foreground=truth>=.5
    correct=(coarse>=.5)==foreground
    square=(probability-coarse.detach()).square()
    terms=[]
    for support in (valid & correct & foreground,valid & correct & ~foreground):
        count=support.sum()
        if bool(count>0):
            terms.append((square*support.to(square.dtype)).sum()/count)
    return torch.stack(terms).mean() if terms else probability.sum()*0.


def missed_thin_recovery(endpoint,coarse,truth,thin,fov):
    """Unweighted normalized -log(p) on missed positive training thin skeleton.

    softplus(-endpoint) is the numerically stable -log(sigmoid(endpoint)).
    Correct thin pixels, non-thin missed pixels and all outside-FOV positions
    do not enter this additional term. Empty support returns exact zero.
    """
    support=((thin>.5)&(truth>=.5)&(coarse<.5)&(fov>.5)).to(endpoint.dtype)
    denominator=support.sum().clamp_min(1.)
    return (F.softplus(-endpoint)*support).sum()/denominator


def losses(velocity,target_velocity,endpoint,coarse,truth,errors,thin,
           gate_logits,option,anchor_weight,soft_cldice,fov,thin_weight=.5):
    """Return total and unweighted diagnostic parts, using the real endpoint.

    Existing coefficients remain BCE=.5, Dice=.5, soft-clDice=.1 and gate=.1.
    Option 2 additionally retains the original thin-weighted velocity MSE/36.
    ``parts['thin_recovery']`` and ``parts['identity']`` are unweighted terms;
    total adds thin_weight*thin_recovery (option 1 only) + anchor_weight*identity.
    Setting both new weights to zero in option 1 exactly recovers old Direct.
    """
    selected=_option(option)
    anchor=_weight(anchor_weight,"anchor_weight")
    thin_coefficient=_weight(thin_weight,"thin_weight")
    # selective=False affects only segmentation probabilities here. The caller
    # supplies option 2's unchanged selective target velocity and gate labels.
    total,parts=_original_losses(
        velocity,target_velocity,endpoint,coarse,truth,errors,thin,gate_logits,
        use_fm=selected==2,selective=False,soft_cldice=soft_cldice,fov=fov,
    )
    probability=endpoint.sigmoid()
    identity=correct_pixel_identity(probability,coarse,truth,fov)
    recovery=missed_thin_recovery(endpoint,coarse,truth,thin,fov) if selected==1 else endpoint.sum()*0.
    total=total+anchor*identity+(thin_coefficient*recovery if selected==1 else 0.)
    return total,{**parts,"thin_recovery":recovery,"identity":identity}
