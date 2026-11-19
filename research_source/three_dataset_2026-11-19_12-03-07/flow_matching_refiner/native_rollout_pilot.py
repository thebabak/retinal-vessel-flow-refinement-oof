"""Native CHASE correction Flow with matched direct control and rollout supervision.

No ground truth is an inference input. Training-only error supports change crop
sampling and losses. The full source chain is constructed by the separate runner.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from flow_matching_refiner.pixel_model import PixelVelocity
from flow_matching_refiner.esfm_model import source_logits
from flow_matching_refiner.native_patch_model import tile_starts,positive_hann
from flow_matching_refiner.topology_oof_flow import mean_support,dice_loss,structural_losses,aligned_batch

CAP=6.;TILE=256;STRIDE=128;TRAIN_STEPS=4

class CorrectionField(nn.Module):
    def __init__(self,mode):
        super().__init__();assert mode in ('flow','direct');self.mode=mode;self.field=PixelVelocity(base=8)
    def velocity(self,state,time,source,rgb):return CAP*torch.tanh(self.field(state,time,source,rgb)/CAP)
    def delta(self,source,rgb,steps=4,stop=None):
        assert steps in (1,2,4,8);state=torch.zeros_like(source)
        if self.mode=='direct':return self.velocity(state,torch.zeros(len(source),device=source.device),source,rgb)
        for k in range(steps if stop is None else stop):
            time=torch.full((len(source),),k/steps,device=source.device,dtype=source.dtype);state=state+self.velocity(state,time,source,rgb)/steps
        return state

def _balanced(value,supports):
    terms=[mean_support(value,s) for s in supports if bool(s.sum()>0)]
    return torch.stack(terms).mean() if terms else value.sum()*0

def _preserve(endpoint,origin,g,valid):
    correct=((origin>=0)==(g>=.5)).to(g.dtype)*valid;sign=2*g-1
    margin=(sign*origin).clamp(max=1.).detach()
    hinge=F.relu(margin-sign*endpoint).square();change=(endpoint.sigmoid()-origin.sigmoid()).square()
    return _balanced(hinge+.25*change,[correct*g,correct*(1-g)])

def objective(model,p,rgb,g,valid,native_sk,native_thin,alignment,time,cfg):
    origin=source_logits(p);errors=((p>=.5)!=(g>=.5)).to(g.dtype)
    target=((2*g-1)*1.5-origin).clamp(-CAP,CAP)*errors
    supports=[valid*errors*native_thin,valid*errors*g*(1-native_thin),valid*errors*(1-g),valid*(1-errors)*g,valid*(1-errors)*(1-g)]
    fm=origin.sum()*0;onpolicy=fm
    if model.mode=='flow':
        state=time[:,None,None,None]*target
        fm=_balanced(((model.velocity(state,time,p,rgb)-target)/CAP).square(),supports)
        # Rollout-state imitation: detached model states, expert remaining correction.
        # This auxiliary is distinguished from conditional flow matching itself.
        with torch.no_grad():own=model.delta(p,rgb,steps=4,stop=2);expert=((target-own)/.5).clamp(-CAP,CAP)
        onpolicy=_balanced(((model.velocity(own,torch.full_like(time,.5),p,rgb)-expert)/CAP).square(),supports)
    delta=model.delta(p,rgb,steps=TRAIN_STEPS);endpoint=origin+delta
    bce=_balanced(F.binary_cross_entropy_with_logits(endpoint,g,reduction='none'),[valid*g,valid*(1-g)])
    dice=dice_loss(endpoint.sigmoid(),g,valid);base_dice=dice_loss(p,g,valid);regret=F.relu(dice-base_dice)
    ncl,ntl,nsl=structural_losses(endpoint,g,valid,native_sk,native_thin)
    # Comparison-grid logits retain its exact original O3 probabilities.
    projected=aligned_batch(delta,alignment);ag,av,ask,ath,ap=[torch.stack([r[k] for r in alignment]).to(p.device) for k in ('gt','fov','skeleton','thin','source')]
    aorigin=source_logits(ap);aendpoint=aorigin+projected
    abce=_balanced(F.binary_cross_entropy_with_logits(aendpoint,ag,reduction='none'),[av*ag,av*(1-ag)])
    adice=dice_loss(aendpoint.sigmoid(),ag,av);aregret=F.relu(adice-dice_loss(ap,ag,av));cl,tl,sl=structural_losses(aendpoint,ag,av,ask,ath)
    preservation=.5*(_preserve(endpoint,origin,g,valid)+_preserve(aendpoint,aorigin,ag,av))
    structure=.5*((ncl+.5*ntl+.25*nsl)+(cl+.5*tl+.25*sl))
    value=.05*fm+.05*onpolicy+.125*(bce+abce)+.25*(dice+adice)+.5*structure+cfg['preservation_weight']*preservation+2*(regret+aregret)
    return value,dict(fm=fm,onpolicy=onpolicy,bce=bce,eval_bce=abce,dice=dice,eval_dice=adice,native_cldice=ncl,native_thin=ntl,native_skeleton=nsl,cldice=cl,thin=tl,skeleton=sl,preservation=preservation,dice_regret=regret,eval_dice_regret=aregret,structure=structure)

@torch.inference_mode()
def predict_delta(model,source,rgb,device,steps=4,batch=8):
    """Labels/FOV are deliberately absent. Positive-Hann correction fusion."""
    assert source.device.type==rgb.device.type=='cpu' and source.shape[0]==1 and rgb.shape[0]==3
    h,w=source.shape[-2:];assert min(h,w)>=TILE
    coords=[(y,x) for y in tile_starts(h,TILE,STRIDE) for x in tile_starts(w,TILE,STRIDE)];window=positive_hann(TILE);accum=torch.zeros_like(source);coverage=torch.zeros_like(source);was=model.training;model.eval()
    try:
        for start in range(0,len(coords),batch):
            loc=coords[start:start+batch];p=torch.stack([source[:,y:y+TILE,x:x+TILE] for y,x in loc]).to(device);im=torch.stack([rgb[:,y:y+TILE,x:x+TILE] for y,x in loc]).to(device);delta=model.delta(p,im,steps=steps).cpu()
            for j,(y,x) in enumerate(loc):accum[:,y:y+TILE,x:x+TILE]+=delta[j]*window;coverage[:,y:y+TILE,x:x+TILE]+=window
    finally:model.train(was)
    assert coverage.min()>0;delta=accum/coverage;assert torch.isfinite(delta).all() and delta.abs().max()<=CAP+1e-5
    return delta,dict(tile=TILE,stride=STRIDE,steps=steps,velocity_calls=len(coords)*(1 if model.mode=='direct' else steps),tiles=len(coords),coverage_min=float(coverage.min()),cap=CAP,GT_free=True)

def eligible(candidate,base):
    return candidate['dice']>=base['dice']-1e-12 and candidate['cldice']>base['cldice']+1e-12 and candidate['thin_skeleton_recall']>base['thin_skeleton_recall']+1e-12

def self_test():
    import inspect
    torch.set_num_threads(2);torch.manual_seed(901);a=CorrectionField('flow');b=CorrectionField('direct');b.load_state_dict(a.state_dict());assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    p=torch.rand(2,1,32,32)*.6+.2;im=torch.rand(2,3,32,32);g=torch.zeros_like(p);g[:,:,3:29,15]=1;v=torch.ones_like(g)
    records=[dict(iy=torch.arange(32),ix=torch.arange(32),rotation90=0,flip_horizontal=False,flip_vertical=False,gt=g[j],fov=v[j],skeleton=g[j],thin=g[j],source=p[j]) for j in range(2)]
    for m in (a,b):
        assert torch.equal(m.delta(p,im),torch.zeros_like(p));value,parts=objective(m,p,im,g,v,g,g,records,torch.tensor([.2,.8]),dict(preservation_weight=1.));assert torch.isfinite(value);value.backward();assert m.field.head.weight.grad.abs().sum()>0
        if m.mode=='direct':assert parts['fm']==parts['onpolicy']==0
    assert not any('gt' in x or 'truth' in x or 'fov' in x for x in inspect.signature(predict_delta).parameters)
    base=dict(dice=.8,cldice=.8,thin_skeleton_recall=.6);assert not eligible(base,base);assert eligible(dict(dice=.8,cldice=.81,thin_skeleton_recall=.61),base);assert not eligible(dict(dice=.799,cldice=.81,thin_skeleton_recall=.61),base)
    print('Synthetic checks passed: exact initial identity, matched parameter counts, nonzero real-rollout gradients, comparison-grid alignment and strict validation gate.',flush=True)

if __name__=='__main__':self_test()
