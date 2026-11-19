"""Strict-OOF, centerline-supervised Flow on native and exact comparison grids.

Only training labels build skeleton/near-negative supports. Inference accepts
coarse probabilities and RGB only, via the unchanged native tiling routine.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F
from flow_matching_refiner.esfm_model import MARGIN,source_logits
from flow_matching_refiner.native_patch_flow_v2 import AnchoredFlow
from flow_matching_refiner.structure_metrics import soft_skeletonize

class TopologyFlow(AnchoredFlow):
    def rollout(self,origin,coarse,rgb,steps=2):
        if isinstance(steps,bool) or not isinstance(steps,int) or steps<1:raise ValueError('Positive Euler count required')
        condition=origin.detach().sigmoid();delta=torch.zeros_like(origin)
        for k in range(steps):
            t=torch.full((len(coarse),),k/steps,device=coarse.device,dtype=coarse.dtype)
            delta=delta+self.velocity(origin+delta,t,condition,rgb)/steps
        return delta

def exact_nearest_indices(length):
    from PIL import Image
    values=np.arange(length,dtype=np.float32)[:,None]
    return np.asarray(Image.fromarray(values).resize((1,512),Image.Resampling.NEAREST),dtype=np.int64)[:,0].copy()

def prepare_supervision(ref):
    from skimage.morphology import skeletonize
    from flow_matching_refiner.structure_metrics_all import thin_skeleton_mask
    h=min(math.floor(256*512/ref.data[n]['n_g'].shape[-2])-1 for n in ref.manifest['train'])
    w=min(math.floor(256*512/ref.data[n]['n_g'].shape[-1])-1 for n in ref.manifest['train'])
    assert h>6 and w>6
    cache={}
    for n in ref.manifest['train']:
        d=ref.data[n];ng=d['n_g'].numpy()[0]>=.5;nf=d['n_fov'].numpy()[0]>.5
        g=d['g'].numpy()[0]>=.5;f=d['fov'].numpy()[0]>.5
        ys=exact_nearest_indices(ng.shape[0]);xs=exact_nearest_indices(ng.shape[1])
        assert np.array_equal(ng[ys[:,None],xs[None,:]],g),f'Native/512 GT mismatch: {n}'
        cache[n]=dict(native_skeleton=torch.from_numpy(skeletonize(ng&nf).copy())[None],
             gt=d['g'].float(),fov=d['fov'].float(),skeleton=torch.from_numpy(skeletonize(g&f).copy())[None].float(),
             thin=torch.from_numpy(thin_skeleton_mask(g,1.,f).copy())[None].float(),ys=ys,xs=xs,shape=(h,w))
    return cache

def patch_supervision(cache,meta):
    top,left=meta['top'],meta['left'];h,w=cache['shape']
    yi=np.flatnonzero((cache['ys']>=top)&(cache['ys']<top+256));xi=np.flatnonzero((cache['xs']>=left)&(cache['xs']<left+256))
    assert len(yi)>=h and len(xi)>=w
    a=(len(yi)-h)//2;b=(len(xi)-w)//2;yi=yi[a:a+h];xi=xi[b:b+w]
    iy=torch.from_numpy(cache['ys'][yi]-top);ix=torch.from_numpy(cache['xs'][xi]-left)
    fields={k:cache[k][:,yi[:,None],xi[None,:]].contiguous() for k in ('gt','fov','skeleton','thin')}
    fields.update(iy=iy,ix=ix)
    native=cache['native_skeleton'][:,top:top+256,left:left+256].float()
    assert native.shape==(1,256,256)
    return fields,native

def undo_augmentation(x,k,fh,fv):
    if fv:x=x.flip(-2)
    if fh:x=x.flip(-1)
    return torch.rot90(x,-k,(-2,-1))

def aligned_batch(endpoint,records):
    out=[]
    for z,r in zip(endpoint,records):
        z=undo_augmentation(z,r['rotation90'],r['flip_horizontal'],r['flip_vertical'])
        out.append(z[:,r['iy'].to(z.device)[:,None],r['ix'].to(z.device)[None,:]])
    return torch.stack(out)

def mean_support(value,support):
    count=support.sum()
    return (value*support).sum()/count if bool(count>0) else value.sum()*0.

def dice_loss(p,g,valid):
    pp=(p*valid).flatten(1);gg=(g*valid).flatten(1)
    return 1-((2*(pp*gg).sum(1)+1e-6)/(pp.sum(1)+gg.sum(1)+1e-6)).mean()

def structural_losses(z,g,valid,skeleton,thin):
    # Full-image hard GT skeletons are cropped, not re-skeletonized per patch.
    mask=valid.clone();mask[:,:,:3,:]=0;mask[:,:,-3:,:]=0;mask[:,:,:,:3]=0;mask[:,:,:,-3:]=0
    p=z.sigmoid();sp=soft_skeletonize(p*valid,iterations=10)*mask;sg=skeleton*mask
    dims=(1,2,3);precision=((sp*g).sum(dims)+1)/(sp.sum(dims)+1)
    sensitivity=((sg*p).sum(dims)+1)/(sg.sum(dims)+1)
    topology=(1-2*precision*sensitivity/(precision+sensitivity).clamp_min(1e-8)).mean()
    thin_loss=mean_support(F.softplus(-z),thin*mask)
    near=((F.max_pool2d(skeleton,7,1,3)>0)&(g<.5)).to(z.dtype)*mask
    skeleton_bce=.5*(mean_support(F.softplus(-z),sg)+mean_support(F.softplus(z),near))
    return topology,thin_loss,skeleton_bce

def objective(model,c,rgb,g,fov,time,anchor_delta,native_skeleton,alignment,cfg):
    if anchor_delta.requires_grad or anchor_delta.shape!=c.shape:raise ValueError('Detached OOF source required')
    valid=(fov>.5).to(g.dtype)
    with torch.no_grad():
        origin=source_logits(c)+anchor_delta;anchor_p=origin.sigmoid();errors=((anchor_p>=.5)!=(g>=.5)).to(g.dtype)
        sk=(native_skeleton>.5).to(g.dtype);assert not bool((sk*(1-g)).any())
        full=(MARGIN*(2*g-1)-origin).clamp(-model.cap,model.cap)
        target=torch.where(sk>.5,(MARGIN-origin).clamp(0,model.cap),full*errors)
    state=origin+time[:,None,None,None]*target
    squared=((model.velocity(state,time,anchor_p,rgb)-target)/model.cap).square()
    supports=[valid*errors*g*(1-sk),valid*errors*(1-g),valid*sk,valid*(1-errors)*g*(1-sk),valid*(1-errors)*(1-g)]
    terms=[mean_support(squared,s) for s in supports if bool(s.sum()>0)];fm=torch.stack(terms).mean()
    endpoint=origin+model.rollout(origin,c,rgb,steps=2);p=endpoint.sigmoid()
    bce=mean_support(F.binary_cross_entropy_with_logits(endpoint,g,reduction='none'),valid);dice=dice_loss(p,g,valid)
    z=aligned_batch(endpoint,alignment)
    ag,av,ask,ath=[torch.stack([r[k] for r in alignment]).to(z.device) for k in ('gt','fov','skeleton','thin')]
    ebce=mean_support(F.binary_cross_entropy_with_logits(z,ag,reduction='none'),av);edice=dice_loss(z.sigmoid(),ag,av)
    topology,thin_loss,skeleton_bce=structural_losses(z,ag,av,ask,ath)
    square=(p-anchor_p).square();pres_support=[valid*(1-errors)*g*(1-sk),valid*(1-errors)*(1-g)]
    preservation=torch.stack([mean_support(square,s) for s in pres_support]).mean()
    gate_pixel=F.binary_cross_entropy_with_logits(model.field.error_logits(anchor_p,rgb),errors,reduction='none')
    gate=torch.stack([mean_support(gate_pixel,s) for s in (valid*errors,valid*(1-errors)) if bool(s.sum()>0)]).mean()
    total=.01*fm+.25*(bce+dice+ebce+edice)+cfg['structure_weight']*(topology+.5*thin_loss+.25*skeleton_bce)+.25*preservation+.1*gate
    return total,dict(fm=fm,bce=bce,dice=dice,eval_bce=ebce,eval_dice=edice,cldice=topology,thin=thin_loss,skeleton=skeleton_bce,preservation=preservation,gate=gate)

def selection_score(candidate,base):
    return .5*((candidate['cldice']-base['cldice'])+(candidate['thin_skeleton_recall']-base['thin_skeleton_recall']))

def feasible(candidate,base,require_structure=True):
    return (candidate['dice']>=base['dice']-.0005-1e-12 and candidate['precision']>=base['precision']-.002-1e-12
       and (not require_structure or candidate['cldice']>=base['cldice']-1e-12 and candidate['thin_skeleton_recall']>=base['thin_skeleton_recall']-1e-12))

def self_test():
    from flow_matching_refiner.esfm_model import ErrorSelectiveModel
    torch.set_num_threads(2);torch.manual_seed(91);m=TopologyFlow(ErrorSelectiveModel(False),1.)
    c=torch.rand(2,1,32,32)*.8+.1;rgb=torch.rand(2,3,32,32);g=torch.zeros_like(c);g[:,:,4:28,15]=1;f=torch.ones_like(c);sk=g.clone()
    a=torch.zeros_like(c);rs=[dict(iy=torch.arange(32),ix=torch.arange(32),rotation90=k,flip_horizontal=True,flip_vertical=False,gt=g[j],fov=f[j],skeleton=sk[j],thin=sk[j]) for j,k in enumerate((1,3))]
    orig=torch.randn_like(c);aug=torch.stack([torch.rot90(orig[j],r['rotation90'],(-2,-1)).flip(-1) for j,r in enumerate(rs)])
    assert torch.equal(aligned_batch(aug,rs),orig)
    loss,parts=objective(m,c,rgb,g,f,torch.tensor([0.,.6]),a,sk,[dict(r,rotation90=0,flip_horizontal=False) for r in rs],dict(structure_weight=1.))
    assert torch.isfinite(loss);loss.backward();assert m.field.velocity_net.head.weight.grad.abs().sum()>0 and all(p.grad is None for p in m.anchor.parameters())
    z1=torch.zeros(1,1,32,32,requires_grad=True);z2=z1.detach().clone();z2[:,:,4:28,15]=2
    assert structural_losses(z2,g[:1],f[:1],sk[:1],sk[:1])[1]<structural_losses(z1,g[:1],f[:1],sk[:1],sk[:1])[1]
    base=dict(dice=.8,precision=.8,cldice=.8,thin_skeleton_recall=.6)
    assert feasible(base,base) and not feasible(dict(base,cldice=.79),base) and not feasible(dict(base,dice=.79),base)
    print('Synthetic aligned-grid gradients, OOF freeze, inverse augmentation and structural selection guards passed.')

if __name__=='__main__':self_test()
