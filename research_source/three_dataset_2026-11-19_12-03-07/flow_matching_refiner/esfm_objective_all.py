"""FOV-aware extension of the frozen CHASE ESFM objective.

Coefficients and parameterization are unchanged. An all-valid FOV is exactly the
original objective; invalid pixels do not supervise any loss. Soft skeletons
operate on probability/reference fields intersected with the field of view.
"""
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import MARGIN, auxiliary_probability


def losses(velocity, target_velocity, endpoint, coarse, truth, errors, thin,
           gate_logits, use_fm, selective, soft_cldice, fov):
    valid=(fov>.5).to(truth.dtype)
    if valid.shape!=truth.shape or not bool((valid.flatten(1).sum(1)>0).all()):
        raise ValueError("Each training image requires a nonempty matching FOV")
    weights=(1+4*thin)*valid
    fm=(((velocity-target_velocity)/MARGIN).square()*weights).sum()/weights.sum()
    p=auxiliary_probability(endpoint,coarse,errors,selective)
    bce=(F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),truth,reduction="none")*valid).sum()/valid.sum()
    pp,gg=(p*valid).flatten(1),(truth*valid).flatten(1)
    dice=1-((2*(pp*gg).sum(1)+1e-6)/(pp.sum(1)+gg.sum(1)+1e-6)).mean()
    topology=soft_cldice(p*valid,truth*valid,iterations=10,smooth=1.)
    per_pixel=F.binary_cross_entropy_with_logits(gate_logits,errors,reduction="none")
    positive,negative=errors*valid,(1-errors)*valid
    terms=[]
    for support in (positive,negative):
        count=support.sum()
        if bool(count>0):terms.append((per_pixel*support).sum()/count)
    gate=torch.stack(terms).mean()
    total=(fm if use_fm else 0)+.5*bce+.5*dice+.1*topology+.1*gate
    return total,dict(fm=fm,bce=bce,dice=dice,cldice=topology,gate=gate)
