"""Same bounded v2 field objective, with an explicit strict-OOF anchor input."""
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import MARGIN,source_logits
from flow_matching_refiner.structure_metrics import soft_cldice_loss
from flow_matching_refiner.refinement_objectives_v3 import correct_pixel_identity

@torch.no_grad()
def routed_anchor(models,coarse,rgb,fold_ids):
    """GT-free patch predictions from the checkpoint excluding each image fold."""
    if len(fold_ids)!=len(coarse):raise ValueError('One source fold per patch required')
    result=torch.empty_like(coarse)
    for fold in sorted(set(fold_ids)):
        if fold not in models:raise ValueError('Missing OOF model; no full-model fallback')
        idx=[i for i,f in enumerate(fold_ids) if f==fold]
        c=coarse[idx];image=rgb[idx]
        result[idx]=models[fold](source_logits(c),torch.zeros(len(idx),device=c.device),c,image,None)
    if not bool(torch.isfinite(result).all()):raise RuntimeError('Nonfinite OOF anchor')
    return result

def objective(model,coarse,rgb,gt,fov,time,fm_weight,anchor_delta):
    """Only anchor provenance changes relative to v2's existing objective."""
    if anchor_delta.shape!=coarse.shape or anchor_delta.requires_grad:raise ValueError('Expected detached OOF anchor residual')
    valid=(fov>.5).to(gt.dtype)
    with torch.no_grad():
        origin=source_logits(coarse)+anchor_delta
        anchor_p=origin.sigmoid();errors=((anchor_p>=.5)!=(gt>=.5)).to(gt.dtype)
        target_delta=(MARGIN*(2*gt-1)-origin).clamp(-model.cap,model.cap)*errors
    state=origin+time[:,None,None,None]*target_delta
    squared=((model.velocity(state,time,coarse,rgb)-target_delta)/model.cap).square()
    supports=[valid*errors,valid*(1-errors)*gt,valid*(1-errors)*(1-gt)]
    fm=torch.stack([(squared*s).sum()/s.sum() for s in supports if bool(s.sum()>0)]).mean()
    endpoint=origin+model.rollout(origin,coarse,rgb,steps=2);p=endpoint.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(endpoint,gt,reduction='none')*valid).sum()/valid.sum()
    pp,gg=(p*valid).flatten(1),(gt*valid).flatten(1)
    dice=1-((2*(pp*gg).sum(1)+1e-6)/(pp.sum(1)+gg.sum(1)+1e-6)).mean()
    cldice=soft_cldice_loss(p*valid,gt*valid,iterations=10,smooth=1.)
    preservation=correct_pixel_identity(p,anchor_p,gt,valid)
    pixel=F.binary_cross_entropy_with_logits(model.field.error_logits(coarse,rgb),errors,reduction='none')
    gate=torch.stack([(pixel*s).sum()/s.sum() for s in (valid*errors,valid*(1-errors)) if bool(s.sum()>0)]).mean()
    total=float(fm_weight)*fm+.5*bce+.5*dice+.1*cldice+.1*gate+preservation
    return total,dict(fm=fm,bce=bce,dice=dice,cldice=cldice,gate=gate,preservation=preservation)

def self_test():
    from flow_matching_refiner.native_patch_flow_v2 import AnchoredFlow,objective as previous_objective
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    torch.set_num_threads(2);torch.manual_seed(13)
    model=AnchoredFlow(ErrorSelectiveModel(False),2.)
    with torch.no_grad():model.field.velocity_net.head.bias.fill_(.04)
    c=torch.rand(2,1,16,16)*.8+.1;rgb=torch.rand(2,3,16,16);g=(torch.rand_like(c)>.7).float();f=torch.ones_like(g);t=torch.tensor([0.,.6])
    a=model.anchor_delta(c,rgb)
    first,parts=objective(model,c,rgb,g,f,t,.1,a)
    second,reference=previous_objective(model,c,rgb,g,f,t,.1)
    assert torch.equal(first,second) and all(torch.equal(parts[k],reference[k]) for k in parts)
    first.backward();assert model.field.velocity_net.head.weight.grad.abs().sum()>0
    class Constant(torch.nn.Module):
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,state,time,coarse,rgb,gate):return torch.ones_like(state)*self.value
    r=routed_anchor({1:Constant(.2),2:Constant(-.3)},c,rgb,[2,1])
    assert torch.allclose(r[0],torch.full_like(r[0],-.3)) and torch.allclose(r[1],torch.full_like(r[1],.2))
    try:routed_anchor({1:Constant(.2)},c,rgb,[2,1])
    except ValueError:pass
    else:raise AssertionError('OOF source was silently replaced')
    print('Fully-OOF objective equivalence, source routing, gradients and missing-fold rejection passed; synthetic inputs only.')

if __name__=='__main__':self_test()
