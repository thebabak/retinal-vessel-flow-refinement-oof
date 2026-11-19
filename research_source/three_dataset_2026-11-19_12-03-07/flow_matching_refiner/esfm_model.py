"""Error-selective flow, isolated from all previously executed model files."""
import torch
from torch import nn
from torch.nn import functional as F

from flow_matching_refiner.pixel_model import PixelVelocity

MARGIN = 6.0
EPSILON = 1e-4
GATE_FLOOR = .05


def source_logits(coarse):
    return torch.logit(coarse.clamp(EPSILON, 1-EPSILON))


def make_target(source, coarse, truth, selective=True):
    errors = ((coarse >= .5) != (truth >= .5)).to(truth.dtype)
    full = MARGIN * (2*truth-1)
    target = torch.where(errors.bool(), full, source) if selective else full
    return target, errors


def auxiliary_probability(endpoint, coarse, errors, selective):
    p = endpoint.sigmoid()
    return (1-errors)*coarse + errors*p if selective else p


class ErrorSelectiveModel(nn.Module):
    def __init__(self, use_gate=False):
        super().__init__()
        self.use_gate = use_gate
        self.velocity_net = PixelVelocity(base=12)
        self.error_net = nn.Sequential(
            nn.Conv2d(4,12,3,padding=1,bias=False), nn.GroupNorm(4,12), nn.SiLU(),
            nn.Conv2d(12,12,3,padding=1,bias=False), nn.GroupNorm(4,12), nn.SiLU(),
            nn.Conv2d(12,1,1),
        )

    def error_logits(self, coarse, rgb):
        return self.error_net(torch.cat((coarse, 2*rgb-1),1))

    def forward(self, state, t, coarse, rgb, gate_logits=None):
        velocity = MARGIN * self.velocity_net(state, t, coarse, rgb)
        if self.use_gate:
            if gate_logits is None:
                gate_logits = self.error_logits(coarse,rgb)
            velocity = velocity * (GATE_FLOOR+(1-GATE_FLOOR)*gate_logits.sigmoid())
        return velocity


def balanced_gate_loss(logits, errors):
    per_pixel=F.binary_cross_entropy_with_logits(logits, errors, reduction="none")
    positive,negative=errors.sum(),(1-errors).sum()
    terms=[]
    if bool(positive>0):
        terms.append((per_pixel*errors).sum()/positive)
    if bool(negative>0):
        terms.append((per_pixel*(1-errors)).sum()/negative)
    return torch.stack(terms).mean()


def losses(velocity, target_velocity, endpoint, coarse, truth, errors, thin,
           gate_logits, use_fm, selective, soft_cldice):
    """All coefficients fixed in the run plan before validation/test."""
    weights=1+4*thin
    fm=(((velocity-target_velocity)/MARGIN).square()*weights).sum()/weights.sum()
    p=auxiliary_probability(endpoint,coarse,errors,selective)
    bce=F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),truth)
    pp,gg=p.flatten(1),truth.flatten(1)
    dice=1-((2*(pp*gg).sum(1)+1e-6)/(pp.sum(1)+gg.sum(1)+1e-6)).mean()
    topology=soft_cldice(p,truth,iterations=10,smooth=1.)
    gate=balanced_gate_loss(gate_logits,errors)
    total=(fm if use_fm else 0)+.5*bce+.5*dice+.1*topology+.1*gate
    return total,dict(fm=fm,bce=bce,dice=dice,cldice=topology,gate=gate)


@torch.no_grad()
def predict(model, coarse, rgb, steps=8, direct=False):
    """Ground-truth-free inference. Return probabilities and explicit call counts."""
    if not isinstance(steps,int) or isinstance(steps,bool) or steps<1:
        raise ValueError("steps must be a positive integer")
    state=source_logits(coarse)
    gate=model.error_logits(coarse,rgb) if model.use_gate else None
    if direct:
        t=torch.zeros(len(coarse),device=coarse.device)
        state=state+model(state,t,coarse,rgb,gate)
        nfe=1
    else:
        for index in range(steps):
            t=torch.full((len(coarse),),index/steps,device=coarse.device)
            state=state+model(state,t,coarse,rgb,gate)/steps
        nfe=steps
    return state.sigmoid(),dict(velocity_calls=nfe,error_gate_calls=int(model.use_gate))
