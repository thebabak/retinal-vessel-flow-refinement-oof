"""Quality fine-tuning of the prior structural field; O3 inputs remain OOF.

The frozen copy of the initial field is used only in training losses, never as
an additional source/input feature. All deployed field weights are trainable.
"""
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import source_logits,MARGIN
from flow_matching_refiner.topology_oof_flow import (aligned_batch,mean_support,dice_loss,structural_losses)
from flow_matching_refiner.dice_guarded_oof_flow import dice_regret,balanced_gate

def objective(model,reference,c,rgb,g,fov,time,anchor_delta,native_skeleton,alignment,cfg):
    if anchor_delta.requires_grad:raise ValueError('Detached strict-OOF O3 input required')
    valid=(fov>.5).to(g.dtype);beta=cfg['train_beta']
    with torch.no_grad():
        origin=source_logits(c)+anchor_delta;source=origin.sigmoid();sk=(native_skeleton>.5).to(g.dtype)
        errors=((source>=.5)!=(g>=.5)).to(g.dtype)
        target=torch.where(sk>.5,(MARGIN-origin).clamp(0,model.cap),(MARGIN*(2*g-1)-origin).clamp(-model.cap,model.cap)*errors)
        reference_endpoint=origin+beta*reference.rollout(origin,c,rgb,steps=2);reference_probability=reference_endpoint.sigmoid()
    teacher=origin+time[:,None,None,None]*target
    squared=((model.velocity(teacher,time,source,rgb)-target)/model.cap).square()
    supports=[valid*errors*g*(1-sk),valid*errors*(1-g),valid*sk,valid*(1-errors)*g*(1-sk),valid*(1-errors)*(1-g)]
    fm=torch.stack([mean_support(squared,s) for s in supports if bool(s.sum()>0)]).mean()
    endpoint=origin+beta*model.rollout(origin,c,rgb,steps=2);p=endpoint.sigmoid()
    bce=mean_support(F.binary_cross_entropy_with_logits(endpoint,g,reduction='none'),valid);dice=dice_loss(p,g,valid)
    z=aligned_batch(endpoint,alignment);z0=aligned_batch(origin,alignment);zr=aligned_batch(reference_endpoint,alignment)
    ag,av,ask,ath=[torch.stack([r[k] for r in alignment]).to(z.device) for k in ('gt','fov','skeleton','thin')]
    ebce=mean_support(F.binary_cross_entropy_with_logits(z,ag,reduction='none'),av);edice=dice_loss(z.sigmoid(),ag,av)
    topology,thin,skeleton=structural_losses(z,ag,av,ask,ath)
    with torch.no_grad():rt,rthin,_=structural_losses(zr,ag,av,ask,ath)
    reference_structure=F.relu(topology-rt)+.5*F.relu(thin-rthin)
    regret=.5*(dice_regret(p,source,g,valid)+dice_regret(z.sigmoid(),z0.sigmoid(),ag,av))
    fp_native=valid*((reference_probability>=.5)&(g<.5)).to(g.dtype)
    fp_eval=av*((zr.sigmoid()>=.5)&(ag<.5)).to(g.dtype)
    repair=.5*(mean_support(F.softplus(endpoint),fp_native)+mean_support(F.softplus(z),fp_eval))
    correct=(reference_probability>=.5)==(g>=.5)
    preservation=torch.stack([mean_support((p-reference_probability).square(),s) for s in (valid*correct*g*(1-sk),valid*correct*(1-g))]).mean()
    centerline_drop=mean_support(F.relu(reference_endpoint-endpoint),valid*sk*(reference_probability>=.5))
    gate=balanced_gate(model.field.error_logits(source,rgb),errors,valid)
    total=.01*fm+.5*(bce+dice+ebce+edice)+topology+.5*thin+.25*skeleton+8*regret+cfg['fp_weight']*repair+.25*preservation+.5*centerline_drop+2*reference_structure+.1*gate
    return total,dict(fm=fm,bce=bce,dice=dice,eval_bce=ebce,eval_dice=edice,cldice=topology,thin=thin,skeleton=skeleton,dice_regret=regret,fp_repair=repair,preservation=preservation,centerline_drop=centerline_drop,reference_structure=reference_structure,gate=gate)

def self_test():
    import copy
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    from flow_matching_refiner.topology_oof_flow import TopologyFlow
    torch.set_num_threads(2);torch.manual_seed(71);m=TopologyFlow(ErrorSelectiveModel(False),2.);reference=copy.deepcopy(m).eval().requires_grad_(False)
    c=torch.rand(2,1,32,32)*.8+.1;g=(torch.rand_like(c)>.85).float();f=torch.ones_like(g);rgb=torch.rand(2,3,32,32);sk=g.clone();a=torch.zeros_like(c)
    align=[dict(iy=torch.arange(32),ix=torch.arange(32),rotation90=0,flip_horizontal=False,flip_vertical=False,gt=g[j],fov=f[j],skeleton=sk[j],thin=sk[j]) for j in range(2)]
    loss,parts=objective(m,reference,c,rgb,g,f,torch.tensor([0.,.5]),a,sk,align,dict(train_beta=.25,fp_weight=.05))
    assert torch.isfinite(loss) and parts['reference_structure']==0 and parts['centerline_drop']==0
    loss.backward();assert m.field.velocity_net.head.weight.grad.abs().sum()>0 and all(p.grad is None for p in reference.parameters()) and all(p.grad is None for p in m.anchor.parameters())
    print('Warm-field gradients, loss-only reference, frozen O3 and initial retention checks passed; synthetic only.')

if __name__=='__main__':self_test()
