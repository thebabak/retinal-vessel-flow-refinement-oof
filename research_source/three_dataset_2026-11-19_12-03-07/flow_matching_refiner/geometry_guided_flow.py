"""Geometry-guided native Flow; all guide inputs are RGB and frozen O3 only."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from flow_matching_refiner.pixel_model import PixelVelocity,Block,smooth_time_embedding
from flow_matching_refiner.native_rollout_pilot import CorrectionField,CAP,objective as previous_objective,_balanced
from flow_matching_refiner.topology_oof_flow import mean_support,aligned_batch
from flow_matching_refiner.esfm_model import source_logits

class GuidedVelocity(PixelVelocity):
    def __init__(self):super().__init__(base=8);self.encoders[0]=Block(9,8)
    def forward(self,state,time,p,rgb,geometry):
        embedding=self.time(smooth_time_embedding(time));x=torch.cat((state,p,2*rgb-1,geometry),1);skips=[]
        for j,block in enumerate(self.encoders):
            x=block(x,embedding);skips.append(x)
            if j<3:x=F.avg_pool2d(x,2)
        for block,skip in zip(self.decoders,reversed(skips[:-1])):x=block(torch.cat((F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1),embedding)
        return self.head(x)

class GeometryFlow(CorrectionField):
    def __init__(self,mode='flow'):
        nn.Module.__init__(self);assert mode in ('flow','direct');self.mode=mode;self.field=GuidedVelocity();self.guide=nn.Sequential(nn.Conv2d(4,16,5,padding=2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,16,3,padding=2,dilation=2),nn.GroupNorm(4,16),nn.SiLU(),nn.Conv2d(16,4,1))
    def guide_fields(self,p,rgb):
        raw=self.guide(torch.cat((p,2*rgb-1),1));direction=F.normalize(raw[:,:2],dim=1,eps=1e-3);width=raw[:,2:3].sigmoid();support=raw[:,3:4].sigmoid();return raw,torch.cat((direction,width,support),1)
    def velocity(self,state,time,p,rgb):
        _,guide=self.guide_fields(p,rgb);v=CAP*torch.tanh(self.field(state,time,p,rgb,guide)/CAP)
        # Background additions need image/geometry support; removals stay possible.
        support=.25+.75*guide[:,3:4];return torch.minimum(v,torch.zeros_like(v))+F.relu(v)*support

def training_geometry(gt):
    """Full native TRAINING mask targets. Low-coherence junction directions omitted."""
    g=np.asarray(gt,dtype=bool);soft=ndi.gaussian_filter(g.astype(np.float32),1.);gy,gx=np.gradient(soft);xx=ndi.gaussian_filter(gx*gx,3.);yy=ndi.gaussian_filter(gy*gy,3.);xy=ndi.gaussian_filter(gx*gy,3.);den=np.sqrt((xx-yy)**2+4*xy**2);cos=(yy-xx)/(den+1e-8);sin=-2*xy/(den+1e-8);coherence=den/(xx+yy+1e-8)
    sk=skeletonize(g);valid=sk&(coherence>.4);radius=ndi.distance_transform_edt(np.pad(g,1))[1:-1,1:-1];width=np.log1p(np.minimum(radius,12))/np.log(13.)
    return torch.from_numpy(np.stack((cos,sin,width,valid),0).astype(np.float32))

def geometry_crop(target,meta):
    y,x=meta['top'],meta['left'];k=meta['rotation90'];t=torch.rot90(target[:,y:y+256,x:x+256],k,(-2,-1)).clone();t[:2]*=(-1)**k
    if meta['flip_horizontal']:t=t.flip(-1);t[1]*=-1
    if meta['flip_vertical']:t=t.flip(-2);t[1]*=-1
    return t.contiguous()

def objective(model,p,rgb,g,fov,sk,thin,alignment,time,cfg,targets):
    base,parts=previous_objective(model,p,rgb,g,fov,sk,thin,alignment,time,cfg);raw,features=model.guide_fields(p,rgb)
    guide_bce=_balanced(F.binary_cross_entropy_with_logits(raw[:,3:4],g,reduction='none'),[fov*g,fov*(1-g)])
    orientation=mean_support(1-(features[:,:2]*targets[:,:2]).sum(1,keepdim=True),targets[:,3:4]*fov)
    radius=mean_support(F.smooth_l1_loss(features[:,2:3],targets[:,2:3],reduction='none'),fov*g)
    delta=model.delta(p,rgb,steps=4);endpoint=source_logits(p)+delta;band=F.max_pool2d((p>=.5).float(),7,1,3)*(1-g)*fov;boundary=mean_support(F.softplus(endpoint),band)
    ap,ag,av=[torch.stack([r[k] for r in alignment]).to(p.device) for k in ('source','gt','fov')];az=source_logits(ap)+aligned_batch(delta,alignment);ab=F.max_pool2d((ap>=.5).float(),3,1,1)*(1-ag)*av;boundary=.5*(boundary+mean_support(F.softplus(az),ab))
    value=base+.25*guide_bce+.05*orientation+.1*radius+.5*boundary
    return value,dict(**parts,base_loss=base,guide_bce=guide_bce,guide_direction=orientation,guide_radius=radius,boundary_FP=boundary)

def eligible(candidate,base):
    return candidate['dice']>=base['dice']-1e-12 and candidate['precision']>=base['precision']-1e-12 and candidate['cldice']>base['cldice']+1e-12 and candidate['thin_skeleton_recall']>base['thin_skeleton_recall']+1e-12

def self_test():
    torch.set_num_threads(2);torch.manual_seed(1821);m=GeometryFlow();p=torch.rand(2,1,32,32)*.6+.2;rgb=torch.rand(2,3,32,32);g=torch.zeros_like(p);g[:,:,4:28,15]=1;v=torch.ones_like(g);targets=torch.zeros(2,4,32,32);targets[:,0]=1;targets[:,2]=.3;targets[:,3]=g[:,0];records=[dict(iy=torch.arange(32),ix=torch.arange(32),rotation90=0,flip_horizontal=False,flip_vertical=False,gt=g[j],fov=v[j],skeleton=g[j],thin=g[j],source=p[j]) for j in range(2)]
    assert torch.equal(m.delta(p,rgb),torch.zeros_like(p));loss,parts=objective(m,p,rgb,g,v,g,g,records,torch.tensor([.2,.7]),dict(preservation_weight=1.),targets);assert torch.isfinite(loss);loss.backward();assert m.field.head.weight.grad.abs().sum()>0 and m.guide[0].weight.grad.abs().sum()>0
    q=torch.zeros(4,256,256);q[0]=1;q[1]=0;q[2]=.2;q[3]=1;rot=geometry_crop(q,dict(top=0,left=0,rotation90=1,flip_horizontal=False,flip_vertical=False));assert (rot[0]==-1).all() and (rot[1]==0).all()
    direct=GeometryFlow('direct');assert sum(p.numel() for p in direct.parameters())==sum(p.numel() for p in m.parameters());print('Synthetic checks passed: initial identity, real guide/Flow gradients, vector augmentation and matched direct parameter budget.',flush=True)
if __name__=='__main__':self_test()
