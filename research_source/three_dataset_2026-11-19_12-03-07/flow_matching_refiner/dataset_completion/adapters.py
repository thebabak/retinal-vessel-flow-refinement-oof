"""Checked, snapshotted dataset adapters; historical implementations stay untouched."""
from __future__ import annotations
import ast,types,sys,json,math,gc,copy
from pathlib import Path
import numpy as np
import torch
from dataclasses import replace
from .common import ROOT,load,sha256_file,write_json,savecsv,csvrows

def module(ctx,name,changes=()):
    path=ROOT/'flow_matching_refiner'/f'{name}.py';ctx.protect(path)
    source=path.read_text(encoding='utf-8-sig');adaptations=[]
    for before,after in changes:
        count=source.count(before)
        assert count>0,(name,'adapter source contract changed',before)
        source=source.replace(before,after);adaptations.append(dict(before=before,after=after,count=count))
    dest=ctx.out/'adapted_source';dest.mkdir(exist_ok=True)
    p=dest/f'{name}_{ctx.tag}.py'
    if p.exists():assert p.read_text(encoding='utf-8')==source
    else:p.write_text(source,encoding='utf-8')
    write_json(dest/f'{name}_adaptations_{ctx.tag}.json',dict(original=str(path),original_sha256=sha256_file(path),adapted=str(p),adapted_sha256=sha256_file(p),changes=adaptations))
    key='flow_matching_refiner.dataset_completion._'+name
    m=types.ModuleType(key);m.__file__=str(path);m.__package__='flow_matching_refiner';sys.modules[key]=m
    exec(compile(ast.parse(source),str(p),'exec'),m.__dict__)
    return m

def variable_budget():
    return [('seen==64 and updates==4',"seen==len(ref.manifest['train'])*4 and updates==(seen+15)//16")]

def native_ref(ctx):
    from flow_matching_refiner.native_patch_data import load_native_dataset
    ref,audit=load_native_dataset(ctx.dataset,ctx.out/'input_audit','cuda')
    ctx.protected.update(audit['source_sha256_before'])
    return ref

def supervision(ref):
    from flow_matching_refiner.topology_oof_flow import prepare_supervision
    if ref.dataset!='HRF':return prepare_supervision(ref)
    # OpenCV's original INTER_NEAREST samples floor(i*input/output), unlike PIL.
    from skimage.morphology import skeletonize
    from flow_matching_refiner.structure_metrics_all import thin_skeleton_mask
    h=min(math.floor(256*512/ref.data[n]['n_g'].shape[-2])-1 for n in ref.manifest['train'])
    w=min(math.floor(256*512/ref.data[n]['n_g'].shape[-1])-1 for n in ref.manifest['train'])
    assert h>6 and w>6;cache={}
    for n in ref.manifest['train']:
        d=ref.data[n];ng=d['n_g'].numpy()[0]>=.5;nf=d['n_fov'].numpy()[0]>.5
        g=d['g'].numpy()[0]>=.5;f=d['fov'].numpy()[0]>.5
        ys=np.floor(np.arange(512)*ng.shape[0]/512).astype(np.int64);xs=np.floor(np.arange(512)*ng.shape[1]/512).astype(np.int64)
        assert np.array_equal(ng[ys[:,None],xs[None,:]],g),n
        cache[n]=dict(native_skeleton=torch.from_numpy(skeletonize(ng&nf).copy())[None],gt=d['g'].float(),fov=d['fov'].float(),skeleton=torch.from_numpy(skeletonize(g&f).copy())[None].float(),thin=torch.from_numpy(thin_skeleton_mask(g,1.,f).copy())[None].float(),ys=ys,xs=xs,shape=(h,w))
    return cache

def completed_fit(path,tag,run,epochs=None):
    marker=Path(path)/f'selection_locked_{tag}.json'
    if marker.exists():
        d=load(marker)
        for k,v in d.items():
            if k.endswith('_path') and Path(str(v)).is_file() and k[:-5]+'_sha256' in d:assert sha256_file(v)==d[k[:-5]+'_sha256']
        if epochs is not None:
            logs=list(Path(path).glob(f'training*_{tag}.csv'))
            assert len(logs)==1 and len(csvrows(logs[0]))==epochs,(path,epochs)
        return d
    assert not Path(path).exists(),f'Partial fit preserved; recovery needed: {path}'
    d=run()
    if epochs is not None:
        logs=list(Path(path).glob(f'training*_{tag}.csv'))
        assert len(logs)==1 and len(csvrows(logs[0]))==epochs,(path,epochs)
    return d

def lock(ctx,name,decisions,**extra):
    p=ctx.out/f'{name}_{ctx.tag}.json';value=dict(decisions=decisions,selection_uses_test=False,**extra)
    if p.exists():assert load(p)==value
    else:write_json(p,value)
    return p

def with_coarse(ref,maps):
    from torch.nn import functional as F
    data=dict(ref.data)
    for n,c in maps.items():
        c=torch.as_tensor(c).clone().float();d=dict(ref.data[n],coarse=c)
        d['n_coarse']=F.interpolate(c[None],size=d['n_rgb'].shape[-2:],mode='bilinear',align_corners=False)[0].contiguous();data[n]=d
    return replace(ref,data=data,namespace=dict(ref.namespace,data=data))

def cleanup():gc.collect();torch.cuda.empty_cache()
