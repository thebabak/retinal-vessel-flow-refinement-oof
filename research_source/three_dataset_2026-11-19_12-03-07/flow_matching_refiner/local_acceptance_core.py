"""Sparse signed-distance proposals and a GT-free local quality critic.

Model state is an SDF on the unchanged 512 comparison grid. RGB and O3 alone
define regions. Identical classifiers support Flow and the direct control.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from flow_matching_refiner.pixel_model import PixelVelocity
from flow_matching_refiner.topology_oof_flow import mean_support,dice_loss,structural_losses

RADIUS=4.;TEMPERATURE=.25;CAP=2.;TILE=128;STRIDE=64;REGION=32;MAX_REGIONS=24
BETAS=(.25,.5,1.);KAPPAS=(0.,.5,1.)

def sdf(binary):
    b=np.asarray(binary,dtype=bool)
    return np.clip((ndi.distance_transform_edt(b)-ndi.distance_transform_edt(~b))/RADIUS,-1.,1.).astype(np.float32)

def image_geometry(probability,rgb):
    """GT-free locations; even confident missing branches can enter near endpoints."""
    p=np.asarray(probability,dtype=np.float32).reshape(512,512);rgb=np.asarray(rgb,dtype=np.float32)
    green=rgb[1];v=np.zeros_like(green)
    for sigma in (1.,2.):
        xx=ndi.gaussian_filter(green,sigma,order=(0,2))*sigma**2;yy=ndi.gaussian_filter(green,sigma,order=(2,0))*sigma**2;xy=ndi.gaussian_filter(green,sigma,order=(1,1))*sigma**2
        root=np.sqrt((xx-yy)**2+4*xy**2);a=(xx+yy-root)/2;b=(xx+yy+root)/2
        small=np.where(np.abs(a)<=np.abs(b),a,b);large=np.where(np.abs(a)<=np.abs(b),b,a)
        v=np.maximum(v,np.maximum(large,0)*np.exp(-.5*(small/(np.abs(large)+1e-8))**2))
    scale=float(np.quantile(v,.995));v=np.clip(v/max(scale,1e-8),0,1).astype(np.float32)
    foreground=p>=.5;sk=skeletonize(foreground);inside=ndi.distance_transform_edt(foreground)
    ends=sk&(ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')==2)
    thin=sk&(inside<=2.);band=(p>.15)&(p<.85)
    dilation=lambda a,n:ndi.binary_dilation(a,iterations=n)
    locations=(dilation(ends,6)&((v>.1)|(p>.08))) | (dilation(thin,3)&(band|(v>.2))) | (band&(v>.25)&dilation(foreground,3))
    ranked=[]
    score=(v+.5*(1-np.abs(2*p-1))+.5*dilation(ends,3))*locations
    for y in range(0,512,REGION):
        for x in range(0,512,REGION):
            mask=locations[y:y+REGION,x:x+REGION]
            if int(mask.sum())>=4:ranked.append((float(score[y:y+REGION,x:x+REGION].sum()),y,x))
    ranked.sort(key=lambda r:(-r[0],r[1],r[2]));regions=[dict(y=y,x=x,score=s) for s,y,x in ranked[:MAX_REGIONS]]
    roi=np.zeros_like(locations)
    for r in regions:
        y,x=r['y'],r['x'];roi[y:y+REGION,x:x+REGION]=locations[y:y+REGION,x:x+REGION]
    return dict(phi=sdf(foreground)[None],roi=roi[None],vesselness=v[None],regions=regions)

def candidate_probability(source,phi,delta,roi,beta):
    """Exact identity probabilities whenever no class changes, including beta zero."""
    if beta==0:return source.copy()
    endpoint=phi+float(beta)*delta;changed=(endpoint>=0)!=(phi>=0)
    mask=np.asarray(roi,dtype=bool)&changed
    result=source.copy();z=np.clip(endpoint/TEMPERATURE,-60,60);q=1/(1+np.exp(-z))
    result[mask]=q[mask];assert np.array_equal(result[~np.asarray(roi,dtype=bool)],source[~np.asarray(roi,dtype=bool)])
    return result.astype(np.float32,copy=False)

class LocalProposer(nn.Module):
    def __init__(self,mode):
        super().__init__();assert mode in ('flow','direct');self.mode=mode;self.field=PixelVelocity(base=8)
    def velocity(self,state,time,probability,rgb):return CAP*torch.tanh(self.field(state,time,probability,rgb)/CAP)
    def delta(self,phi,probability,rgb,roi):
        if self.mode=='direct':return roi*self.velocity(phi,torch.zeros(len(phi),device=phi.device),probability,rgb)
        result=torch.zeros_like(phi)
        for k in range(2):result=result+.5*roi*self.velocity(phi+result,torch.full((len(phi),),k/2,device=phi.device),probability,rgb)
        return result

def loss(model,batch,time):
    phi,p,rgb,roi,g,target,sk,thin=batch;endpoint=phi+model.delta(phi,p,rgb,roi);valid=torch.ones_like(g)
    probability=roi*torch.sigmoid(endpoint/TEMPERATURE)+(1-roi)*p
    pixel=F.binary_cross_entropy_with_logits(endpoint/TEMPERATURE,g,reduction='none')
    supports=[roi*g,roi*(1-g)];terms=[mean_support(pixel,s) for s in supports if bool(s.sum()>0)]
    bce=torch.stack(terms).mean() if terms else pixel.sum()*0;dl=dice_loss(probability,g,valid)
    # Reconstruct endpoint logits only where the field may act; outside is frozen O3.
    logits=roi*endpoint/TEMPERATURE+(1-roi)*torch.logit(p.clamp(1e-4,1-1e-4))
    cl,tl,sl=structural_losses(logits,g,valid,sk,thin)
    regression=mean_support(F.smooth_l1_loss(endpoint,target,reduction='none'),roi)
    fm=endpoint.sum()*0
    if model.mode=='flow':
        target_delta=(target-phi)*roi;state=phi+time[:,None,None,None]*target_delta
        sq=(model.velocity(state,time,p,rgb)-target_delta).square()
        fm_terms=[mean_support(sq,s) for s in (roi*thin,roi*g*(1-thin),roi*(1-g)) if bool(s.sum()>0)]
        if fm_terms:fm=torch.stack(fm_terms).mean()
    total=.25*bce+.5*dl+.5*(cl+.5*tl+.25*sl)+.25*regression+.5*fm
    return total,dict(bce=bce,dice=dl,cldice=cl,thin=tl,skeleton=sl,sdf=regression,fm=fm)

@torch.inference_mode()
def predict_delta(model,source,rgb,geometry,device):
    from flow_matching_refiner.native_patch_model import positive_hann
    phi=torch.from_numpy(geometry['phi']);p=torch.from_numpy(source);im=torch.from_numpy(rgb);roi=torch.from_numpy(geometry['roi'].astype(np.float32))
    accum=torch.zeros_like(phi);coverage=torch.zeros_like(phi);window=positive_hann(TILE);coords=[]
    for y in range(0,512-TILE+1,STRIDE):
        for x in range(0,512-TILE+1,STRIDE):
            if bool(roi[:,y:y+TILE,x:x+TILE].any()):coords.append((y,x))
    was=model.training;model.eval()
    try:
        for start in range(0,len(coords),16):
            loc=coords[start:start+16];b=[torch.stack([a[:,y:y+TILE,x:x+TILE] for y,x in loc]).to(device) for a in (phi,p,im,roi)]
            d=model.delta(*b).cpu()
            for j,(y,x) in enumerate(loc):accum[:,y:y+TILE,x:x+TILE]+=d[j]*window;coverage[:,y:y+TILE,x:x+TILE]+=window
    finally:model.train(was)
    assert bool((coverage[roi.bool()]>0).all())
    result=torch.where(coverage>0,accum/coverage.clamp_min(1e-12),torch.zeros_like(accum))*roi
    assert bool(torch.isfinite(result).all()) and float(result.abs().max())<=CAP+1e-5
    return result.numpy()

def local_features(current,candidate,source,rgb,geometry,region):
    """Only image and prediction information; no GT or outcome-based mask inputs."""
    y,x=region['y'],region['x'];s=np.s_[:,y:y+REGION,x:x+REGION]
    a=current[s][0];b=candidate[s][0];p=source[s][0];r=geometry['roi'][s][0];v=geometry['vesselness'][s][0]
    ap=a>=.5;bp=b>=.5;add=(~ap)&bp;remove=ap&(~bp);changed=add|remove
    mean=lambda z,m:float(z[m].mean()) if bool(m.any()) else 0.
    green=rgb[1,y:y+REGION,x:x+REGION];std=float(green.std())+1e-6
    def morphology(mask):
        sk=skeletonize(mask);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
        return [float(sk.mean()),float((sk&(degree==1)).sum())/32,float(ndi.label(mask,np.ones((3,3)))[1])/32]
    features=[y/512,x/512,float(r.mean()),float(changed.mean()),float(add.mean()),float(remove.mean()),float(ap.mean()),float(bp.mean()),float(current.mean()),float(a.mean()),float(a.std()),float(b.mean()),float(b.std()),float(np.abs(b-a).mean()),mean(p,add),mean(p,remove),mean(v,add),mean(v,remove),mean(v,r),mean(green,add),mean(green,remove),mean(green,ap),std,(mean(green,ap)-mean(green,add))/std,*morphology(ap),*morphology(bp)]
    result=np.asarray(features,dtype=np.float32);assert result.shape==(30,) and np.isfinite(result).all();return result

class QualityCritic(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(30,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,6))
    def forward(self,x):
        q=self.net(x);return q[:,:3],q[:,3:].clamp(-4,3)

def region_candidate(current,candidate,region):
    p=current.copy();y,x=region['y'],region['x'];p[:,y:y+REGION,x:x+REGION]=candidate[:,y:y+REGION,x:x+REGION];return p

@torch.inference_mode()
def accept_corrections(source,delta,rgb,geometry,critic,scalers,beta_max,kappa,device='cpu'):
    if beta_max==0:return source.copy(),[]
    candidates={b:candidate_probability(source,geometry['phi'],delta,geometry['roi'],b) for b in BETAS if b<=beta_max}
    current=source.copy();accepted=[];critic.eval()
    for region in geometry['regions']:
        offers=[]
        for beta,candidate in candidates.items():
            p=region_candidate(current,candidate,region)
            if np.array_equal(p>=.5,current>=.5):continue
            feature=local_features(current,p,source,rgb,geometry,region);x=(feature-scalers['xmean'])/scalers['xstd']
            mu,logstd=critic(torch.from_numpy(x)[None].to(device));mean=mu[0].cpu().numpy()*scalers['ystd']+scalers['ymean'];sd=np.exp(logstd[0].cpu().numpy())*scalers['ystd'];lower=mean-float(kappa)*sd
            if bool((lower>=0).all()):offers.append((float((lower/scalers['ystd']).sum()),beta,p,mean,lower))
        if offers:
            _,beta,current,mean,lower=max(offers,key=lambda z:(z[0],-z[1]));accepted.append(dict(**region,beta=beta,predicted_gain=mean.tolist(),lower_gain=lower.tolist()))
    outside=~geometry['roi'].astype(bool);assert np.array_equal(current[outside],source[outside]);return current,accepted

def self_test():
    import inspect
    torch.set_num_threads(2);torch.manual_seed(29)
    a=LocalProposer('flow');b=LocalProposer('direct');assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    phi=torch.full((2,1,32,32),-.25);p=torch.full_like(phi,.2);rgb=torch.rand(2,3,32,32);roi=torch.zeros_like(phi);roi[:,:,8:24,8:24]=1;g=torch.zeros_like(phi);g[:,:,8:24,15:17]=1;target=(2*g-1)*.25
    batch=(phi,p,rgb,roi,g,target,g.clone(),g.clone())
    for m in (a,b):
        v,parts=loss(m,batch,torch.tensor([0.,.5]));assert torch.isfinite(v);v.backward();assert m.field.head.weight.grad.abs().sum()>0
        with torch.no_grad():m.field.head.bias.fill_(.1)
        d=m.delta(phi,p,rgb,roi);assert bool((d[roi==0]==0).all()) and float(d.detach().abs().max())<=CAP
    source=np.full((1,512,512),.1,np.float32);source[:,100:110,200:202]=.8;image=np.random.default_rng(1).random((3,512,512),dtype=np.float32);geo=image_geometry(source,image)
    assert np.array_equal(candidate_probability(source,geo['phi'],np.zeros_like(source),geo['roi'],1),source)
    assert np.array_equal(candidate_probability(source,geo['phi'],np.ones_like(source),geo['roi'],0),source)
    assert all('gt' not in x and 'truth' not in x for x in inspect.signature(accept_corrections).parameters)
    if geo['regions']:assert local_features(source,source,source,image,geo,geo['regions'][0]).shape==(30,)
    critic=QualityCritic()
    with torch.no_grad():
        for p in critic.parameters():p.zero_()
        critic.net[-1].bias[:3].fill_(-1.)
    scalers=dict(xmean=np.zeros(30,np.float32),xstd=np.ones(30,np.float32),ymean=np.zeros(3,np.float32),ystd=np.ones(3,np.float32))
    result,accepted=accept_corrections(source,np.ones_like(source),image,geo,critic,scalers,1.,0.)
    assert not accepted and np.array_equal(result,source),'A negative-score critic accepted an edit'
    print('Synthetic checks passed: matched capacity, nonzero gradients, exact identity, zero correction outside ROI, GT-free acceptance API.',flush=True)

if __name__=='__main__':self_test()
