"""Frozen pretrained whole-image context cache; no fitting, no GT input."""
import json,gc,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from PIL import Image
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.geometry_flow_data import OLD,OT
from flow_matching_refiner.run_refinement_v3 import stamp,write_json

WEIGHTS=Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
AUGS=[(k,fh,fv) for k in range(4) for fh in (False,True) for fv in (False,True)]
def rgb_sha(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
def encoder(device):
    assert WEIGHTS.is_file() and sha256_file(WEIGHTS).startswith('f37072fd');m=resnet18(weights=None);m.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True));return m.to(device).eval().requires_grad_(False)
def augment(x,k,fh,fv):
    x=torch.rot90(x,k,(-2,-1))
    if fh:x=x.flip(-1)
    if fv:x=x.flip(-2)
    return x
def invert(x,k,fh,fv):
    if fv:x=x.flip(-2)
    if fh:x=x.flip(-1)
    return torch.rot90(x,-k,(-2,-1))

@torch.inference_mode()
def encode_rgb(model,rgb,augs,device):
    mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None];sp=[];vectors=[]
    # Whole-field square resize; no ImageNet center crop that could remove retina.
    for start in range(0,len(augs),8):
        aa=augs[start:start+8];x=torch.stack([F.interpolate(augment(rgb,*a)[None],size=(224,224),mode='bilinear',align_corners=False)[0] for a in aa]).to(device);x=(x-mean)/std;x=model.maxpool(model.relu(model.bn1(model.conv1(x))));x=model.layer1(x);x=model.layer2(x);s=model.layer3(x);v=model.layer4(s).mean((-2,-1))
        for j,a in enumerate(aa):sp.append(invert(s[j].cpu(),*a).contiguous());vectors.append(v[j].cpu())
    return torch.stack(sp).numpy().copy(),torch.stack(vectors).numpy().copy()

def build_context(cases,out,tag,status,all_augmentations=True):
    out.mkdir(parents=True,exist_ok=False);m=encoder('cuda');augs=AUGS if all_augmentations else [(0,False,False)];cache={};records=[]
    for n,c in cases.items():
        status('frozen_RGB_context_extraction',image=n,augmentations=len(augs));sp,g=encode_rgb(m,c['rgb'],augs,'cuda');path=out/f'{n}_context_{tag}.npz';np.savez_compressed(path,spatial=sp,global_vector=g);assert sp.shape==(len(augs),256,14,14) and g.shape==(len(augs),512) and np.isfinite(sp).all() and np.isfinite(g).all();cache[n]=dict(spatial=torch.from_numpy(sp),global_vector=torch.from_numpy(g),augs=augs,rgb_sha256=rgb_sha(c['rgb']),path=str(path),sha256=sha256_file(path));records.append(dict(image=n,path=str(path),sha256=cache[n]['sha256'],RGB_array_sha256=cache[n]['rgb_sha256'],shape_native=list(c['rgb'].shape[-2:])))
    manifest=dict(created_at=stamp(),pretrained='torchvision ResNet18 ImageNet1K V1',checkpoint=str(WEIGHTS),checkpoint_sha256=sha256_file(WEIGHTS),frozen=True,preprocess='Full RGB bilinear224square, ImageNet mean/std; no center crop',spatial='layer3 256x14x14 canonicalized after encoding matching full-image augmentation',global_vector='layer4 512 mean pooled',augmentations=augs,GT_input=False,records=records);write_json(out/f'context_manifest_{tag}.json',manifest);del m;gc.collect();torch.cuda.empty_cache();return cache,manifest

def load_test_cases(seed):
    """Called only after the new global system lock; uses retained frozen sources."""
    mf=OLD/f'test_sources/seed_{seed}/source_manifest_{OT}.json';manifest=json.loads(mf.read_text());assert manifest['parts']==['test'];cases={}
    for r in manifest['records']:
        assert r['split']=='test' and sha256_file(r['path'])==r['sha256'];n=r['image'];pr=r['provenance'];assert n not in pr['fit_images'] and sha256_file(pr['O3_checkpoint'])==pr['O3_sha256']
        with np.load(r['path'],allow_pickle=False) as z:p=torch.from_numpy(z['source_native'].copy());p512=torch.from_numpy(z['source512'].copy());coarse=torch.from_numpy(z['coarse512'].copy())
        with Image.open(ROOT/'data'/f'{n}.jpg') as im:rgb=torch.from_numpy((np.asarray(im.convert('RGB'))/255.).astype(np.float32)).permute(2,0,1).contiguous()
        with Image.open(ROOT/'data'/f'{n}_1stHO.png') as im:g=torch.from_numpy((np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>127).copy())[None].float()
        cases[n]=dict(name=n,source=p,source512=p512,coarse512=coarse,rgb=rgb,g512=g,record=r)
    assert len(cases)==8;return cases
