"""O3-anchored bounded flow with supervision on its actual rollout.

Frozen option3 supplies the native starting logits. A separately fitted zero-
initialized field learns bounded corrections; identity is an explicit validation
candidate. No ground truth, error mask or FOV is an inference input.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import ErrorSelectiveModel,MARGIN,source_logits
from flow_matching_refiner.native_patch_model import (_image,_positive_integer,tile_starts,positive_hann,apply_residual)
from flow_matching_refiner.structure_metrics import soft_cldice_loss
from flow_matching_refiner.refinement_objectives_v3 import correct_pixel_identity

class AnchoredFlow(torch.nn.Module):
    def __init__(self,anchor,cap):
        super().__init__()
        if not 0<float(cap)<=4:raise ValueError('Expected a fixed positive cap <=4')
        self.anchor=anchor.eval().requires_grad_(False)
        self.field=ErrorSelectiveModel(False)
        self.cap=float(cap)
    def train(self,mode=True):
        super().train(mode);self.anchor.eval();return self
    def velocity(self,state,time,coarse,rgb):
        return self.cap*torch.tanh(self.field(state,time,coarse,rgb,None)/self.cap)
    @torch.no_grad()
    def anchor_delta(self,coarse,rgb):
        return self.anchor(source_logits(coarse),torch.zeros(len(coarse),device=coarse.device),coarse,rgb,None)
    def rollout(self,origin,coarse,rgb,steps=2):
        _positive_integer(steps,'steps')
        delta=torch.zeros_like(origin)
        for k in range(steps):
            t=torch.full((len(coarse),),k/steps,device=coarse.device,dtype=coarse.dtype)
            delta=delta+self.velocity(origin+delta,t,coarse,rgb)/steps
        return delta

def objective(model,coarse,rgb,gt,fov,time,fm_weight):
    valid=(fov>.5).to(gt.dtype)
    with torch.no_grad():
        origin=source_logits(coarse)+model.anchor_delta(coarse,rgb)
        anchor_p=origin.sigmoid()
        errors=((anchor_p>=.5)!=(gt>=.5)).to(gt.dtype)
        target_delta=(MARGIN*(2*gt-1)-origin).clamp(-model.cap,model.cap)*errors
    state=origin+time[:,None,None,None]*target_delta
    predicted=model.velocity(state,time,coarse,rgb)
    squared=((predicted-target_delta)/model.cap).square()
    supports=[valid*errors,valid*(1-errors)*gt,valid*(1-errors)*(1-gt)]
    terms=[(squared*s).sum()/s.sum() for s in supports if bool(s.sum()>0)]
    fm=torch.stack(terms).mean()
    delta=model.rollout(origin,coarse,rgb,steps=2)
    endpoint=origin+delta;p=endpoint.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(endpoint,gt,reduction='none')*valid).sum()/valid.sum()
    pp,gg=(p*valid).flatten(1),(gt*valid).flatten(1)
    dice=1-((2*(pp*gg).sum(1)+1e-6)/(pp.sum(1)+gg.sum(1)+1e-6)).mean()
    cldice=soft_cldice_loss(p*valid,gt*valid,iterations=10,smooth=1.)
    preservation=correct_pixel_identity(p,anchor_p,gt,valid)
    gate_logits=model.field.error_logits(coarse,rgb)
    gate_pixel=F.binary_cross_entropy_with_logits(gate_logits,errors,reduction='none')
    gate_terms=[(gate_pixel*s).sum()/s.sum() for s in (valid*errors,valid*(1-errors)) if bool(s.sum()>0)]
    gate=torch.stack(gate_terms).mean()
    total=float(fm_weight)*fm+.5*bce+.5*dice+.1*cldice+.1*gate+preservation
    return total,dict(fm=fm,bce=bce,dice=dice,cldice=cldice,gate=gate,preservation=preservation)

@torch.no_grad()
def predict_tiled(model,coarse,rgb,device,tile=256,stride=128,batch=16,steps=2):
    """Return separately stitched frozen-anchor and learned-flow residuals."""
    for v,n in ((tile,'tile'),(stride,'stride'),(batch,'batch'),(steps,'steps')):_positive_integer(v,n)
    if stride>tile:raise ValueError('Stride leaves gaps')
    c=_image(coarse,1,'coarse').detach().to('cpu',torch.float32)
    image=_image(rgb,3,'rgb').detach().to('cpu',torch.float32)
    if c.shape[-2:]!=image.shape[-2:]:raise ValueError('Shape mismatch')
    if not bool(torch.isfinite(c).all() and torch.isfinite(image).all()):raise ValueError('Nonfinite inputs')
    h,w=c.shape[-2:];ph,pw=max(h,tile),max(w,tile);padding=(0,pw-w,0,ph-h)
    if padding[1] or padding[3]:c=F.pad(c[None],padding,mode='replicate')[0];image=F.pad(image[None],padding,mode='replicate')[0]
    positions=[(y,x) for y in tile_starts(ph,tile,stride) for x in tile_starts(pw,tile,stride)]
    anchor_sum=torch.zeros((1,ph,pw));flow_sum=torch.zeros_like(anchor_sum);coverage=torch.zeros_like(anchor_sum)
    window=positive_hann(tile);was_training=model.training;model.eval()
    try:
        for start in range(0,len(positions),batch):
            locs=positions[start:start+batch]
            bc=torch.stack([c[:,y:y+tile,x:x+tile] for y,x in locs]).to(device)
            bi=torch.stack([image[:,y:y+tile,x:x+tile] for y,x in locs]).to(device)
            a=model.anchor_delta(bc,bi);origin=source_logits(bc)+a
            f=model.rollout(origin,bc,bi,steps=steps)
            if not bool(torch.isfinite(a).all() and torch.isfinite(f).all()):raise RuntimeError('Nonfinite residual')
            if float(f.abs().max())>model.cap+1e-5:raise RuntimeError('Flow correction exceeded its cap')
            a=a.to('cpu',torch.float32);f=f.to('cpu',torch.float32)
            for j,(y,x) in enumerate(locs):
                anchor_sum[:,y:y+tile,x:x+tile].add_(a[j]*window)
                flow_sum[:,y:y+tile,x:x+tile].add_(f[j]*window)
                coverage[:,y:y+tile,x:x+tile].add_(window)
    finally:model.train(was_training)
    cov=coverage[:,:h,:w]
    if float(cov.min())<=0:raise RuntimeError('Incomplete coverage')
    a=(anchor_sum[:,:h,:w]/cov).contiguous();f=(flow_sum[:,:h,:w]/cov).contiguous()
    return a,f,dict(tile=tile,stride=stride,steps=steps,total_tiles=len(positions),anchor_calls=len(positions),flow_calls=len(positions)*steps,
                   cap=model.cap,flow_abs_max=float(f.abs().max()),coverage_min=float(cov.min()),label_free_inference=True,
                   fusion='separate FP32 positive-Hann anchor/flow logit-residual averages')

def self_test():
    torch.set_num_threads(2);torch.manual_seed(17)
    anchor=ErrorSelectiveModel(False);model=AnchoredFlow(anchor,1.)
    c=torch.rand(2,1,16,16)*.8+.1;rgb=torch.rand(2,3,16,16);gt=(torch.rand_like(c)>.8).float();fov=torch.ones_like(gt)
    a,f,meta=predict_tiled(model,c[0],rgb[0],'cpu',tile=16,stride=8,batch=2)
    assert torch.equal(f,torch.zeros_like(f)) and model.training and not model.anchor.training
    time=torch.tensor([0.,.6]);loss,parts=objective(model,c,rgb,gt,fov,time,.1)
    assert torch.isfinite(loss);loss.backward()
    assert all(v.grad is None for v in anchor.parameters())
    assert model.field.velocity_net.head.weight.grad.abs().sum()>0
    with torch.no_grad():model.field.velocity_net.head.bias.fill_(100.)
    origin=source_logits(c)+model.anchor_delta(c,rgb)
    correction=model.rollout(origin,c,rgb,2)
    assert correction.abs().max()<=1.+1e-6
    assert not any('gt' in n or 'fov' in n for n in __import__('inspect').signature(predict_tiled).parameters)
    print('Synthetic v2 checks passed: exact zero-flow identity, bounded rollout, frozen anchor, real-rollout gradients and label-free inference.',flush=True)

if __name__=='__main__':self_test()
