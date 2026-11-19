"""Read-only reuse of audited CHASE OOF sources for a new validation-only pilot."""
import json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from flow_matching_refiner.data import ROOT,sha256_file
OLD=ROOT/'outputs/native_rollout_CHASE_pilot_2026-11-17_23-14-19';OT='2026-11-17_23-14-19'
def load_cases(seed=42):
    protocol=ROOT/'paired_reproduction/all_datasets_five_full_runs_2026-10-08_15-38-24/CHASE/train_seed_42/protocol.json';split=json.loads(protocol.read_text())['split'];manifest=OLD/f'sources/seed_{seed}/source_manifest_{OT}.json';m=json.loads(manifest.read_text());assert m['all_training_sources_whole_chain_OOF'];protected={str(protocol):sha256_file(protocol),str(manifest):sha256_file(manifest)};cases={};data={}
    for r in m['records']:
        n=r['image'];assert r['split'] in ('train','validation') and n in split[r['split']];assert sha256_file(r['path'])==r['sha256'];protected[r['path']]=r['sha256'];pr=r['provenance'];assert sha256_file(pr['O3_checkpoint'])==pr['O3_sha256'];protected[pr['O3_checkpoint']]=pr['O3_sha256']
        if r['split']=='train':
            assert pr['whole_chain_OOF'] and n not in pr['fit_images'];fp=json.loads(Path(pr['fold_protocol']).read_text());held=set(pr['held_images']);assert all(not held&set(x['baseline_fit_images']) and x['image'] not in x['baseline_fit_images'] for x in fp['training_input_lineage']);assert sha256_file(pr['baseline_checkpoint'])==pr['baseline_sha256'];protected[pr['baseline_checkpoint']]=pr['baseline_sha256']
        rgbpath=ROOT/'data'/f'{n}.jpg';gtpath=ROOT/'data'/f'{n}_1stHO.png';protected[str(rgbpath)]=sha256_file(rgbpath);protected[str(gtpath)]=sha256_file(gtpath)
        with Image.open(rgbpath) as im:rgb=torch.from_numpy((np.asarray(im.convert('RGB'))/255.).astype(np.float32)).permute(2,0,1).contiguous()
        with Image.open(gtpath) as im:g=torch.from_numpy((np.asarray(im.convert('L'))>127).copy())[None].float();g512=torch.from_numpy((np.asarray(im.convert('L').resize((512,512),Image.Resampling.NEAREST))>127).copy())[None].float()
        with np.load(r['path'],allow_pickle=False) as z:p=torch.from_numpy(z['source_native'].copy());p512=torch.from_numpy(z['source512'].copy());coarse=torch.from_numpy(z['coarse512'].copy())
        assert p.shape==g.shape==(1,960,999) and p512.shape==g512.shape==(1,512,512)
        f=torch.ones_like(g);f512=torch.ones_like(g512,dtype=torch.bool);cases[n]=dict(name=n,source=p,source512=p512,coarse512=coarse,rgb=rgb,g=g,fov=f,record=r);data[n]=dict(n_g=g,n_fov=f,n_rgb=rgb,g=g512,fov=f512,coarse=coarse)
    assert len(cases)==20 and not set(split['test'])&set(cases)
    return SimpleNamespace(dataset='CHASE',device=torch.device('cuda'),manifest=split,data=data),cases,protected
