"""Whole-chain OOF local Flow/direct proposers and learned acceptance, FOV-aware."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .common import ROOT,load,write_json,sha256_file
from .adapters import module,native_ref
from .sources import read_strict,load_case_records,load_case_record,oof_cases

def local_acceptor(ctx):
    ref=native_ref(ctx);fr,_,m=read_strict(ctx,ref)
    core=module(ctx,'local_acceptance_core',[
        ('phi,p,rgb,roi,g,target,sk,thin=batch;endpoint=phi+model.delta(phi,p,rgb,roi);valid=torch.ones_like(g)',
         'phi,p,rgb,roi,g,target,sk,thin,valid=batch;roi=roi*valid;endpoint=phi+model.delta(phi,p,rgb,roi)')])
    changes=[
      ("sdf(g[0]>=.5)[None]","sdf((g[0]>=.5)&(f[0]>.5))[None]"),
      ("skeletonize(g[0]>=.5)[None]","skeletonize((g[0]>=.5)&(f[0]>.5))[None]"),
      ("c['target'],c['sk'],c['thin']]","c['target'],c['sk'],c['thin'],c['fov'].astype(np.float32)]"),
      ('for j in range(8)]','for j in range(9)]'),
      ('assert len(held)==4 and','assert len(held)==len(sources_folds[k-1]) and'),
      ('len(set(held_images))==16',"len(set(held_images))==len(ref.manifest['train'])"),
      ('n_images=16,n_subjects=8','n_images=len(ref.manifest["train"]),n_subjects=None'),
      ('all_direct_indirect_and_selection_producers_exclude_held_subjects=True','all_direct_indirect_and_selection_producers_exclude_held_images=True'),
      ('held-subject OOF','held-image OOF'),
      ('four reused validation images',f'{len(ref.manifest["validation"])} reused validation images'),
      ("print('CHASE',mode.upper()", "print(ref.dataset,mode.upper()"),
      ("'local examples from16images/8subjects; examples are dependent.'", "f'local examples from{len(ref.manifest[\"train\"])}training images; examples are dependent.'"),
      ('OOF_producer_fit_images=12','OOF_producer_fit_images="variable: excludes one complete group"'),
      ("assert ref.dataset=='CHASE' and","assert ref.dataset in ('DRIVE','HRF') and"),
      ('assert len(train)==12 and',"assert len(train)==len(ref.manifest['train'])-len(held) and"),
      ("f'CHASE_{AT}/native_patch_direct/seed_{seed}/{n}_predictions_{AT}.npz'","f'{ref.dataset}_{AT}/native_patch_direct/seed_{seed}/{n}_predictions_{AT}.npz'"),
      ('assert len(records)==80 and',"assert len(records)==len(ref.manifest['test'])*10 and"),
      ('prediction_files=80','prediction_files=len(records)')]
    r=module(ctx,'local_acceptor_workflow',changes);r.loss=core.loss;r.sources_folds=m['folds']
    def make_cases(unused,sources,cache):
        pairs={};outer={}
        for entry in m['pair_case_records']:
            key='_'.join(str(j+1) for j in entry['groups'])
            for n,c in load_case_records(ref,entry['records']).items():
                pp=c['record']['provenance'];protocol=dict(fit_images=pp['fit_images'],validation_images=pp['validation_images'],coarse_training_fit_dependencies=pp['coarse_training_fit_dependencies'])
                q=r.case(fr,n,c['source512'].numpy(),dict(role='pair',protocol=protocol,record=c['record']),cache);q['coarse']=c['coarse512'].numpy();pairs[key,n]=q
        for entry in m['outer_case_records']:
            if entry['seed']!=42:continue
            key=str(entry['group']+1)
            for n,c in load_case_records(ref,entry['records']).items():
                pp=c['record']['provenance'];protocol=dict(fit_images=pp['fit_images'],validation_images=pp['validation_images'],coarse_training_fit_dependencies=pp['coarse_training_fit_dependencies'])
                q=r.case(fr,n,c['source512'].numpy(),dict(role='outer',protocol=protocol,record=c['record']),cache);q['coarse']=c['coarse512'].numpy();outer[key,n]=q
        return pairs,outer
    def full(unused,seed,out,tag,cache):
        _,cases,_=oof_cases(ctx,ref,seed);training=[];validation=[]
        for n,c in cases.items():
            q=r.case(fr,n,c['source512'].numpy(),dict(role='strict_nested_train' if n in ref.manifest['train'] else 'full_O3_validation',record=c['record']),cache);q['coarse']=c['coarse512'].numpy()
            (training if n in ref.manifest['train'] else validation).append(q)
        return training,validation
    r.make_source_cases=make_cases;r.full_source_cases=full
    marker=ctx.root/'strict_sources'/ctx.dataset/f'completed_{ctx.tag}.json'
    sources=dict(folds=m['folds'],global_lock=str(marker),global_lock_sha256=sha256_file(marker))
    result=r.run_study(fr,sources,ctx.out,ctx.tag,ctx.status)
    result.pop('new_triple_baselines',None);result.pop('new_pair_O3_fits',None)
    result['upstream_source_fits_in_this_phase']=0
    return ctx.done(dict(**result,FOV_supervision=True,nested_source_phase='strict_sources',source_fits_counted_separately=True))
