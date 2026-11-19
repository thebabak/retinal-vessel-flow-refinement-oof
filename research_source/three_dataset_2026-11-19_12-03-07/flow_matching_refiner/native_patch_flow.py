"""Native-patch conditional flow matching, separate from all earlier models.

Same ungated ErrorSelectiveModel and patch/stitch conventions as option 3.
Flow trains a straight path from coarse logits to full-GT +/-6 logits. Test
inference integrates an eight-step Euler ODE per tile without GT or FOV inputs.
The stitched native logit residual is projected onto the original 512 baseline.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import ErrorSelectiveModel,MARGIN,source_logits
from flow_matching_refiner.esfm_objective_all import losses
from flow_matching_refiner.model import linear_path
from flow_matching_refiner.structure_metrics import soft_cldice_loss
from flow_matching_refiner.native_patch_model import (
    _positive_integer,_image,tile_starts,positive_hann,apply_residual,
    project_residual_512,sample_training_patch,training_center_pools)


def flow_loss(model,coarse,rgb,gt,fov,time):
    """Full-target FM + option-3 auxiliary losses, with FOV-only supervision."""
    if model.use_gate:raise ValueError('This matched architecture has no inference gate')
    if time.shape!=(len(coarse),) or not bool(((time>=0)&(time<=1)).all()):
        raise ValueError('Time must be one value in [0,1] per patch')
    source=source_logits(coarse)
    target=MARGIN*(2*gt-1)
    state,target_velocity=linear_path(source,target,time)
    gate=model.error_logits(coarse,rgb)
    velocity=model(state,time,coarse,rgb,None)
    endpoint=state+(1-time[:,None,None,None])*velocity
    errors=((coarse>=.5)!=(gt>=.5)).to(gt.dtype)
    return losses(velocity,target_velocity,endpoint,coarse,gt,errors,torch.zeros_like(gt),
                  gate,use_fm=True,selective=False,soft_cldice=soft_cldice_loss,fov=fov)


@torch.no_grad()
def predict_flow_tiled(model,coarse,rgb,device,tile=256,stride=128,batch=16,steps=8,return_residual=False):
    """Label-free per-tile Euler integration and positive-Hann residual fusion."""
    for v,name,minimum in ((tile,'tile',8),(stride,'stride',1),(batch,'batch',1),(steps,'steps',1)):
        _positive_integer(v,name,minimum)
    if stride>tile:raise ValueError('Stride leaves gaps')
    if getattr(model,'use_gate',False):raise ValueError('Ungated model required')
    c=_image(coarse,1,'coarse').detach().to('cpu',torch.float32)
    image=_image(rgb,3,'rgb').detach().to('cpu',torch.float32)
    if c.shape[-2:]!=image.shape[-2:]:raise ValueError('Coarse/RGB shapes differ')
    if not bool(torch.isfinite(c).all() and torch.isfinite(image).all()):raise ValueError('Nonfinite inputs')
    if not bool(((c>=0)&(c<=1)).all()):raise ValueError('Invalid coarse probability')
    h,w=c.shape[-2:];ph,pw=max(h,tile),max(w,tile)
    padding=(0,pw-w,0,ph-h)
    if padding[1] or padding[3]:
        c=F.pad(c[None],padding,mode='replicate')[0]
        image=F.pad(image[None],padding,mode='replicate')[0]
    positions=[(y,x) for y in tile_starts(ph,tile,stride) for x in tile_starts(pw,tile,stride)]
    numerator=torch.zeros((1,ph,pw),dtype=torch.float32);coverage=torch.zeros_like(numerator)
    window=positive_hann(tile);was_training=model.training;model.eval();forward_batches=0
    try:
        for start in range(0,len(positions),batch):
            locs=positions[start:start+batch]
            bc=torch.stack([c[:,y:y+tile,x:x+tile] for y,x in locs]).to(device)
            bi=torch.stack([image[:,y:y+tile,x:x+tile] for y,x in locs]).to(device)
            origin=source_logits(bc);state=origin.clone()
            for index in range(steps):
                time=torch.full((len(locs),),index/steps,device=device,dtype=state.dtype)
                velocity=model(state,time,bc,bi,None)
                if velocity.shape!=state.shape or not bool(torch.isfinite(velocity).all()):
                    raise RuntimeError('Invalid flow velocity')
                state=state+velocity/steps
                forward_batches+=1
            delta=(state-origin).detach().to('cpu',torch.float32)
            if not bool(torch.isfinite(delta).all()):raise RuntimeError('Nonfinite integrated residual')
            for j,(y,x) in enumerate(locs):
                numerator[:,y:y+tile,x:x+tile].add_(delta[j]*window)
                coverage[:,y:y+tile,x:x+tile].add_(window)
    finally:model.train(was_training)
    valid=coverage[:,:h,:w]
    if float(valid.min())<=0:raise RuntimeError('Uncovered native pixels')
    residual=numerator[:,:h,:w]/valid
    probability=apply_residual(c[:,:h,:w],residual)
    metadata=dict(native_height=h,native_width=w,tile=tile,stride=stride,batch=batch,steps=steps,
                  total_tiles=len(positions),velocity_calls=len(positions)*steps,
                  velocity_forward_batches=forward_batches,gate_calls=0,
                  coverage_min=float(valid.min()),coverage_max=float(valid.max()),
                  fusion='FP32 positive-Hann average of integrated tile logit residuals',
                  ODE='explicit Euler t=0..(S-1)/S; dt=1/S; deterministic initial coarse logits',
                  label_free_inference=True,halo=0,output_dtype='float32',
                  normalization='GroupNorm within tiles; no full-image equivalence claim')
    if return_residual:return probability.contiguous(),residual.contiguous(),metadata
    return probability.contiguous(),metadata


def self_test():
    """Synthetic analytic ODE, exact identity, fusion, FOV loss and gradients."""
    torch.set_num_threads(2)
    class Analytic(torch.nn.Module):
        use_gate=False
        def __init__(self,constant=0.,slope=0.):super().__init__();self.constant=constant;self.slope=slope
        def forward(self,state,time,coarse,rgb,gate=None):return torch.ones_like(state)*(self.constant+self.slope*time[:,None,None,None])
    base=torch.full((1,21,27),.4);rgb=torch.zeros((3,21,27))
    base[0,0,:3]=torch.tensor([0.,.5,1.])
    zero=Analytic();p,delta,meta=predict_flow_tiled(zero,base,rgb,'cpu',tile=16,stride=8,batch=3,steps=8,return_residual=True)
    assert torch.equal(p,base) and torch.equal(delta,torch.zeros_like(delta)) and zero.training
    for steps in (1,4,8):
        model=Analytic(.2,.4)
        _,d,m=predict_flow_tiled(model,base,rgb,'cpu',tile=16,stride=8,batch=2,steps=steps,return_residual=True)
        expected=.2+.4*(steps-1)/(2*steps)
        assert torch.allclose(d,torch.full_like(d,expected),atol=3e-6)
        assert m['velocity_calls']==m['total_tiles']*steps
    torch.manual_seed(9);net=ErrorSelectiveModel(False)
    c=torch.rand((2,1,16,16))*.8+.1;image=torch.rand((2,3,16,16));g=(torch.rand_like(c)>.7).float();fov=torch.ones_like(g);fov[:,:,:2]=0
    loss,parts=flow_loss(net,c,image,g,fov,torch.tensor([0.,.7]))
    assert torch.isfinite(loss) and parts['fm']>0
    loss.backward();assert net.velocity_net.head.weight.grad.abs().sum()>0
    assert net.error_net[-1].weight.grad.abs().sum()>0
    print('Synthetic native flow: analytic Euler, exact zero-flow identity, full coverage and FM gradients passed. No real data/training run.',flush=True)

if __name__=='__main__':self_test()
