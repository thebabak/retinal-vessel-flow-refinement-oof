"""Regenerate native OOF O3 inputs through a held-subject baseline as well.

Earlier O3 cache files are not used as inputs: their coarse512 field is tied to
the retained reference baseline. This cache records every producer checkpoint.
"""
from __future__ import annotations
import gc,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import write_json,stamp
from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,predict_tiled,project_residual_512,apply_residual
from flow_matching_refiner.run_native_patch_flow_v2 import anchor_path

CROSS=ROOT/'outputs/strict_o3_crossfit_2026-11-17_03-16-57';CT='2026-11-17_03-16-57'

def build_cases(ref,seed,out,tag,status,parts=('train','validation'),test_unlocked=False):
    assert ref.dataset=='CHASE' and (test_unlocked or 'test' not in parts)
    out.mkdir(parents=True,exist_ok=False);allchoices=json.loads((CROSS/f'all40_O3_choices_locked_before_OOF_{CT}.json').read_text());decisions=[d for d in allchoices['decisions'] if d['dataset']=='CHASE' and d['refiner_seed']==seed];assert len(decisions)==4
    cases={};records=[];protected={};training=set(ref.manifest['train'])
    def infer(names,coarse_maps,cp,provenance,part):
        assert sha256_file(cp)==provenance['O3_sha256'];protected[str(cp)]=sha256_file(cp);net=ErrorSelectiveModel(False).to(ref.device);net.load_state_dict(torch.load(cp,map_location=ref.device,weights_only=True)['state']);net.eval().requires_grad_(False)
        for n in names:
            status('building_native_OOF_sources' if part=='train' else 'building_'+part+'_sources',seed=seed,image=n);d=ref.data[n];coarse=coarse_maps[n].cpu().float();nc=F.interpolate(coarse[None],size=d['n_rgb'].shape[-2:],mode='bilinear',align_corners=False)[0].contiguous()
            _,anchor,meta=predict_tiled(net,nc,d['n_rgb'],ref.device,batch=8,return_residual=True);pn=apply_residual(nc,anchor);p512=apply_residual(coarse,project_residual_512(anchor,'CHASE'))
            path=out/f'{n}_native_source_{tag}.npz';np.savez_compressed(path,source_native=pn.numpy(),source512=p512.numpy(),coarse512=coarse.numpy(),anchor_residual_native=anchor.numpy())
            record=dict(image=n,split=part,seed=seed,path=str(path),sha256=sha256_file(path),provenance=provenance,**meta);records.append(record);cases[n]=dict(name=n,source=pn,source512=p512,coarse512=coarse,rgb=d['n_rgb'],g=d['n_g'],fov=d['n_fov'].float(),record=record)
        del net;gc.collect();torch.cuda.empty_cache()
    if 'train' in parts:
        for dec in decisions:
            protocol=json.loads(Path(dec['fold_protocol']).read_text());held=set(protocol['held_images']);fit=set(protocol['fit_images']);assert not held&fit and held<=training and fit|held==training
            for dep in protocol['training_input_lineage']:assert dep['image'] not in dep['baseline_fit_images'] and not held&set(dep['baseline_fit_images'])
            bcp=Path(protocol['outer_baseline_checkpoint']);assert sha256_file(bcp)==protocol['outer_baseline_sha256'];protected[str(bcp)]=sha256_file(bcp);obj=torch.load(bcp,map_location=ref.device,weights_only=False)
            assert set(obj['protocol']['fit_images'])==fit and set(obj['protocol']['held_out_images'])==held
            baseline=ref.LNet_RA(in_channels=1).to(ref.device);baseline.load_state_dict(obj['state']);baseline.eval().requires_grad_(False)
            with torch.inference_mode():maps={n:baseline(ref.data[n]['x'][None].to(ref.device)).cpu()[0] for n in protocol['held_images']}
            del baseline,obj;gc.collect();torch.cuda.empty_cache()
            cp=Path(dec['raw_path']);pr=dict(whole_chain_OOF=True,held_images=protocol['held_images'],fit_images=protocol['fit_images'],baseline_checkpoint=str(bcp),baseline_sha256=sha256_file(bcp),O3_checkpoint=str(cp),O3_sha256=dec['raw_sha256'],fold_protocol=dec['fold_protocol'],fold_protocol_sha256=sha256_file(dec['fold_protocol']),source_validation_excludes_held_subjects=True,training_input_lineage=protocol['training_input_lineage'])
            infer(protocol['held_images'],maps,cp,pr,'train')
    for part in ('validation','test'):
        if part not in parts:continue
        names=ref.manifest[part];cp=anchor_path('CHASE',seed);pr=dict(whole_chain_OOF=False,split=part,fit_images=ref.manifest['train'],held_images=names,O3_checkpoint=str(cp),O3_sha256=sha256_file(cp),baseline_checkpoint=str(ROOT/'paired_reproduction/all_datasets_five_full_runs_2026-10-08_15-38-24/CHASE/train_seed_42/baseline.pt'),baseline_fit_images=ref.manifest['train'])
        infer(names,{n:ref.data[n]['coarse'] for n in names},cp,pr,part)
    assert set(cases)=={n for part in parts for n in ref.manifest[part]};write_json(out/f'source_manifest_{tag}.json',dict(locked_at=stamp(),seed=seed,parts=list(parts),records=records,protected=protected,old_OOF_caches_used=False,all_training_sources_whole_chain_OOF=True));return cases,records,protected
