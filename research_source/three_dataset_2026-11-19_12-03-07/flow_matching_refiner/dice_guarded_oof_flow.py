"""Strict-OOF bidirectional Flow with Dice regret and gain-retention selection.

Training GT defines supports and targets only. Deployment uses the frozen O3
probability and RGB. New source files leave every preceding method unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from flow_matching_refiner.topology_oof_flow import (TopologyFlow,aligned_batch,
    mean_support,dice_loss,structural_losses,selection_score)
from flow_matching_refiner.esfm_model import MARGIN,source_logits

class DiceGuardedFlow(TopologyFlow):
    def __init__(self,anchor,cap=2.):
        super().__init__(anchor,cap)
        self.field.error_net[-1]=nn.Conv2d(12,2,1)
        self.gate_floor=.1
    def velocity(self,state,time,condition,rgb):
        raw=self.cap*torch.tanh(self.field(state,time,condition,rgb,None)/self.cap)
        gates=self.gate_floor+(1-self.gate_floor)*self.field.error_logits(condition,rgb).sigmoid()
        # where, rather than relu(raw), keeps a nonzero head gradient at the
        # exact zero initialization. Channel0 permits growth; channel1 removal.
        return raw*torch.where(raw>=0,gates[:,:1],gates[:,1:])

def soft_dice_values(p,g,valid):
    pp=(p*valid).flatten(1);gg=(g*valid).flatten(1)
    return (2*(pp*gg).sum(1)+1e-6)/(pp.sum(1)+gg.sum(1)+1e-6)

def dice_regret(p,source,g,valid):
    return F.relu(soft_dice_values(source.detach(),g,valid)-soft_dice_values(p,g,valid)).mean()

def balanced_gate(logits,target,valid):
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    terms=[mean_support(loss,s) for s in (valid*target,valid*(1-target)) if bool(s.sum()>0)]
    return torch.stack(terms).mean() if terms else loss.sum()*0.

def objective(model,c,rgb,g,fov,time,anchor_delta,native_skeleton,alignment,cfg):
    if anchor_delta.requires_grad or anchor_delta.shape!=c.shape:raise ValueError('Detached strict-OOF source required')
    valid=(fov>.5).to(g.dtype)
    with torch.no_grad():
        origin=source_logits(c)+anchor_delta;source=origin.sigmoid();sk=(native_skeleton>.5).to(g.dtype)
        errors=((source>=.5)!=(g>=.5)).to(g.dtype);assert not bool((sk*(1-g)).any())
        full=(MARGIN*(2*g-1)-origin).clamp(-model.cap,model.cap)
        target=torch.where(sk>.5,(MARGIN-origin).clamp(0,model.cap),full*errors)
        grow=((sk>.5)|((source<.5)&(g>=.5))).to(g.dtype)
        remove=((source>=.5)&(g<.5)).to(g.dtype)
    teacher=origin+time[:,None,None,None]*target
    squared=((model.velocity(teacher,time,source,rgb)-target)/model.cap).square()
    supports=[valid*errors*g*(1-sk),valid*errors*(1-g),valid*sk,valid*(1-errors)*g*(1-sk),valid*(1-errors)*(1-g)]
    fm=torch.stack([mean_support(squared,s) for s in supports if bool(s.sum()>0)]).mean()
    endpoint=origin+model.rollout(origin,c,rgb,steps=2);p=endpoint.sigmoid()
    bce=mean_support(F.binary_cross_entropy_with_logits(endpoint,g,reduction='none'),valid);dice=dice_loss(p,g,valid)
    z=aligned_batch(endpoint,alignment);z0=aligned_batch(origin,alignment)
    ag,av,ask,ath=[torch.stack([r[k] for r in alignment]).to(z.device) for k in ('gt','fov','skeleton','thin')]
    ebce=mean_support(F.binary_cross_entropy_with_logits(z,ag,reduction='none'),av);edice=dice_loss(z.sigmoid(),ag,av)
    topology,thin,skloss=structural_losses(z,ag,av,ask,ath)
    regret=.5*(dice_regret(p,source,g,valid)+dice_regret(z.sigmoid(),z0.sigmoid(),ag,av))
    fp_native=mean_support(F.softplus(endpoint),valid*remove)
    fp_eval=mean_support(F.softplus(z),av*((z0.sigmoid()>=.5)&(ag<.5)).to(z.dtype));fp_repair=.5*(fp_native+fp_eval)
    square=(p-source).square();preservation=torch.stack([mean_support(square,s) for s in (valid*(1-errors)*g*(1-sk),valid*(1-errors)*(1-g))]).mean()
    centerline_drop=mean_support(F.relu(origin-endpoint),valid*sk*(1-errors))
    gate_logits=model.field.error_logits(source,rgb)
    gate=.5*(balanced_gate(gate_logits[:,:1],grow,valid)+balanced_gate(gate_logits[:,1:],remove,valid))
    total=.01*fm+.25*(bce+dice+ebce+edice)+cfg['structure_weight']*(topology+.5*thin+.25*skloss)+cfg['regret_weight']*regret+.05*fp_repair+.25*preservation+.25*centerline_drop+.25*gate
    return total,dict(fm=fm,bce=bce,dice=dice,eval_bce=ebce,eval_dice=edice,cldice=topology,thin=thin,skeleton=skloss,dice_regret=regret,fp_repair=fp_repair,preservation=preservation,centerline_drop=centerline_drop,gate=gate)

def feasible(candidate,base,require_structure=True):
    return (candidate['dice']>=base['dice']-1e-12 and candidate['precision']>=base['precision']-.002-1e-12
       and (not require_structure or candidate['cldice']>=base['cldice']-1e-12 and candidate['thin_skeleton_recall']>=base['thin_skeleton_recall']-1e-12))

def retained(candidate,base,previous):
    return feasible(candidate,base) and candidate['cldice']>=max(base['cldice'],previous['cldice'])-1e-12 and candidate['thin_skeleton_recall']>=max(base['thin_skeleton_recall'],previous['thin_skeleton_recall'])-1e-12

def primary_key(choice):return (choice['full_gain_retained'],choice['structural_score'])
def raw_key(choice):return (choice['full_gain_retained'],choice['structure_feasible'],choice['quality_feasible'],choice['structural_score'],choice['validation_dice'])

def self_test():
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    torch.set_num_threads(2);torch.manual_seed(147);m=DiceGuardedFlow(ErrorSelectiveModel(False),2.)
    c=torch.full((2,1,32,32),.15);g=torch.zeros_like(c);g[:,:,4:28,15]=1.;f=torch.ones_like(c);rgb=torch.rand(2,3,32,32);sk=g.clone();a=torch.zeros_like(c)
    align=[dict(iy=torch.arange(32),ix=torch.arange(32),rotation90=0,flip_horizontal=False,flip_vertical=False,gt=g[j],fov=f[j],skeleton=sk[j],thin=sk[j]) for j in range(2)]
    loss,parts=objective(m,c,rgb,g,f,torch.tensor([0.,.7]),a,sk,align,dict(structure_weight=.5,regret_weight=2.))
    assert torch.isfinite(loss);loss.backward();assert m.field.velocity_net.head.weight.grad.abs().sum()>0 and m.field.error_net[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in m.anchor.parameters())
    base=dict(dice=.8,precision=.8,cldice=.8,thin_skeleton_recall=.6);previous=dict(base,cldice=.801,thin_skeleton_recall=.604)
    assert feasible(base,base) and not feasible(dict(base,dice=.8-1e-8),base)
    assert not retained(base,base,previous) and retained(dict(previous,dice=.801),base,previous)
    s=torch.full_like(g,.1);s[g>.5]=.8;w=s.clone();w[:,:,0,:]=.9
    assert dice_regret(w,s,g,f)>0 and dice_regret(s,s,g,f)==0
    print('Zero-initial-head gradient, directional gates, frozen source, strict Dice and retention checks passed; synthetic only.')

if __name__=='__main__':self_test()
