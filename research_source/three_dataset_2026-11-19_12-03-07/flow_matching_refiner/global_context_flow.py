"""Frozen whole-image ImageNet context plus native patch Flow/direct correction."""
import torch
from torch import nn
from torch.nn import functional as F
from flow_matching_refiner.pixel_model import PixelVelocity,smooth_time_embedding
from flow_matching_refiner.native_rollout_pilot import CorrectionField,CAP
from flow_matching_refiner.native_patch_model import positive_hann,tile_starts

class ContextVelocity(PixelVelocity):
    def __init__(self):
        super().__init__(base=8);self.context_spatial=nn.Sequential(nn.Conv2d(256,64,1,bias=False),nn.GroupNorm(4,64));self.context_global=nn.Linear(512,128)
    def forward(self,state,time,p,rgb,context,global_vector):
        embedding=self.time(smooth_time_embedding(time));x=torch.cat((state,p,2*rgb-1),1);skips=[]
        for j,block in enumerate(self.encoders):
            x=block(x,embedding);skips.append(x)
            if j<3:x=F.avg_pool2d(x,2)
        assert context.shape==(len(x),256,*x.shape[-2:]) and global_vector.shape==(len(x),512)
        x=x+.1*torch.tanh(self.context_spatial(context));gain,bias=self.context_global(global_vector).chunk(2,1);x=x*(1+.1*torch.tanh(gain[:,:,None,None]))+.1*torch.tanh(bias[:,:,None,None])
        for block,skip in zip(self.decoders,reversed(skips[:-1])):x=block(torch.cat((F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1),embedding)
        return self.head(x)

class GlobalContextCorrection(CorrectionField):
    def __init__(self,mode='flow'):
        nn.Module.__init__(self);assert mode in ('flow','direct');self.mode=mode;self.field=ContextVelocity();self._context=None
    def bind(self,spatial,global_vector):
        assert not spatial.requires_grad and not global_vector.requires_grad and spatial.ndim==4 and global_vector.ndim==2;self._context=(spatial,global_vector)
    def unbind(self):self._context=None
    def velocity(self,state,time,p,rgb):
        if self._context is None:raise RuntimeError('Explicit same-image context is required; no local-only fallback')
        c,g=self._context;assert len(c)==len(p) and c.device==p.device and g.device==p.device
        return CAP*torch.tanh(self.field(state,time,p,rgb,c,g)/CAP)

def sample_context(feature,vector,shape,top,left,k=0,fh=False,fv=False):
    """Sample canonicalized global features at native encoder cell centers.

    Features were encoded from the matching augmented WHOLE image, then their
    grid was inverted to canonical coordinates. Crop first, then reapply patch
    augmentation so RGB, source and context coordinates remain aligned.
    """
    assert feature.shape==(256,14,14) and vector.shape==(512,);h,w=shape
    ys=top+3.5+8*torch.arange(32,dtype=torch.float32);xs=left+3.5+8*torch.arange(32,dtype=torch.float32);yy,xx=torch.meshgrid(ys,xs,indexing='ij');grid=torch.stack((2*(xx+.5)/w-1,2*(yy+.5)/h-1),-1)[None]
    c=F.grid_sample(feature[None],grid,mode='bilinear',padding_mode='border',align_corners=False)[0];c=torch.rot90(c,k,(-2,-1))
    if fh:c=c.flip(-1)
    if fv:c=c.flip(-2)
    return c.contiguous(),vector

@torch.inference_mode()
def predict_delta(model,source,rgb,feature,global_vector,device,steps=4,batch=8):
    assert source.device.type==rgb.device.type=='cpu';h,w=source.shape[-2:];coords=[(y,x) for y in tile_starts(h,256,128) for x in tile_starts(w,256,128)];window=positive_hann(256);acc=torch.zeros_like(source);coverage=torch.zeros_like(source);was=model.training;model.eval()
    try:
        for start in range(0,len(coords),batch):
            loc=coords[start:start+batch];p=torch.stack([source[:,y:y+256,x:x+256] for y,x in loc]).to(device);image=torch.stack([rgb[:,y:y+256,x:x+256] for y,x in loc]).to(device);contexts=[sample_context(feature,global_vector,(h,w),y,x) for y,x in loc];c=torch.stack([v[0] for v in contexts]).to(device);g=torch.stack([v[1] for v in contexts]).to(device);model.bind(c,g);d=model.delta(p,image,steps=steps).cpu();model.unbind()
            for j,(y,x) in enumerate(loc):acc[:,y:y+256,x:x+256]+=d[j]*window;coverage[:,y:y+256,x:x+256]+=window
    finally:model.unbind();model.train(was)
    assert coverage.min()>0;delta=acc/coverage;assert torch.isfinite(delta).all() and delta.abs().max()<=CAP+1e-5
    return delta,dict(tile=256,stride=128,steps=steps,tiles=len(coords),velocity_calls=len(coords)*(1 if model.mode=='direct' else steps),context_from_same_whole_RGB=True,GT_free=True)

def self_test():
    import inspect
    torch.set_num_threads(2);torch.manual_seed(1801);a=GlobalContextCorrection('flow');b=GlobalContextCorrection('direct');b.load_state_dict(a.state_dict());assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    p=torch.rand(2,1,32,32);rgb=torch.rand(2,3,32,32);c=torch.rand(2,256,4,4);g=torch.rand(2,512)
    try:a.delta(p,rgb)
    except RuntimeError:pass
    else:raise AssertionError('Missing context was accepted')
    a.bind(c,g);assert torch.equal(a.delta(p,rgb),torch.zeros_like(p));opt=torch.optim.SGD(a.parameters(),lr=.01)
    for _ in range(2):opt.zero_grad();loss=(a.delta(p,rgb)-.1).square().mean();loss.backward();opt.step()
    assert a.field.context_global.weight.grad.abs().sum()>0 and a.field.context_spatial[0].weight.grad.abs().sum()>0
    first=a.delta(p,rgb).detach();a.bind(c+1,g+1);assert not torch.equal(first,a.delta(p,rgb));a.unbind()
    feature=torch.ones(256,14,14);ctx,vec=sample_context(feature,torch.ones(512),(960,999),128,256,1,True,False);assert ctx.shape==(256,32,32) and torch.allclose(ctx,torch.ones_like(ctx))
    assert not any('gt' in x or 'truth' in x or 'fov' in x for x in inspect.signature(predict_delta).parameters)
    print('Synthetic tests passed: context required, exact initial identity, matched counts, context gradients/causal sensitivity, native sampling and label-free inference.',flush=True)
if __name__=='__main__':self_test()
