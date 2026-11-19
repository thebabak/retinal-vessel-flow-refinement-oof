"""Strict nested baseline/O3 cross-fitting for CHASE and DRIVE only.

Six unordered pairs of excluded folds need only six inner baselines/dataset.
For O3 outer fold k, each fitting image in fold j uses a baseline excluding k+j.
Thus held-fold labels cannot leak indirectly through O3 fitting coarse maps.
"""
from __future__ import annotations
import argparse,csv,gc,itertools,json,random,shutil,traceback
from dataclasses import replace
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
DS=('CHASE','DRIVE');SEEDS=(42,123,2026,3000,3001);OT='2026-10-09_23-30-40'
OOF=ROOT/f'revise/oof_comparison_{OT}'
V2=ROOT/'outputs/native_patch_flow_v2_2026-11-17_02-17-42';VT='2026-11-17_02-17-42'

def require_lineage(folds,outer,fit_ids,val_ids,inner_fit_by_image):
    held=set(folds[outer]);fit=set(fit_ids);val=set(val_ids)
    assert not held&(fit|val)
    assert fit==set().union(*(set(f) for k,f in enumerate(folds) if k!=outer))
    for image in fit:
        deps=set(inner_fit_by_image[image])
        assert image not in deps and not held&deps,'Indirect held-fold label leakage'

def self_test():
    f=[['a','b'],['c','d'],['e','f'],['g','h']]
    fit=[x for fold in f[1:] for x in fold];deps={x:[v for k,fold in enumerate(f) if k!=0 and x not in fold for v in fold] for x in fit}
    require_lineage(f,0,fit,['v'],deps)
    try:require_lineage(f,0,fit,['v'],{**deps,'c':['a','e','f']})
    except AssertionError:pass
    else:raise AssertionError('Indirect leakage was accepted')
    print('Synthetic nested fold leakage guards passed; no real data accessed.')

def train_inner_baseline(ref,fit,excluded,out,tag,status):
    import numpy as np
    import torch
    from torchvision.transforms.functional import rotate
    from torchvision.transforms import InterpolationMode
    out.mkdir(parents=True,exist_ok=False)
    assert len(fit)==8 and not set(fit)&set(excluded) and not set(excluded)&set(ref.manifest['validation'])
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);rng=random.Random(42)
    model=ref.LNet_RA(in_channels=1).to(ref.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    best=-1.;stale=0;best_epoch=None;checkpoint=out/f'baseline_{tag}.pt';log=[]
    protocol=dict(dataset=ref.dataset,fit_images=fit,excluded_images=excluded,validation_images=ref.manifest['validation'],test_excluded=ref.manifest['test'],seed=42,max_epochs=1000,patience=20,loss='1-global softDice entire512grid, matching retained baseline recipe',learning_rate=.001,weight_decay=.001)
    write_json(out/f'protocol_{tag}.json',protocol)
    for epoch in range(1,1001):
        status('nested_baseline_training',dataset=ref.dataset,epoch=epoch,excluded_folds=out.name)
        model.train();names=fit.copy();rng.shuffle(names);losses=[]
        for start in range(0,len(names),4):
            samples=[]
            for name in names[start:start+4]:
                d=ref.data[name];x=d['x'].clone();g=d['g'].clone();angle=rng.choice([0,15,30,45,90,100,120])
                x=rotate(x,angle,interpolation=InterpolationMode.BILINEAR);g=rotate(g,angle,interpolation=InterpolationMode.NEAREST)
                for dim in (-1,-2):
                    if rng.random()<.5:x=x.flip(dim);g=g.flip(dim)
                samples.append((x,g))
            x=torch.stack([v[0] for v in samples]).to(ref.device);g=torch.stack([v[1] for v in samples]).to(ref.device)
            p=model(x);loss=1-(2*(p*g).sum()+1e-6)/(p.sum()+g.sum()+1e-6)
            assert bool(torch.isfinite(loss));opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.item()))
        model.eval();vs=[]
        with torch.no_grad():
            names=ref.manifest['validation'];x=torch.stack([ref.data[n]['x'] for n in names]).to(ref.device);p=model(x).cpu()
            for j,n in enumerate(names):
                d=ref.data[n];score=ref.score(p[j],d['g']) if ref.dataset=='CHASE' else ref.score(p[j],d['g'],d['fov']>.5)
                vs.append(score['dice'])
        val=float(np.mean(vs));log.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_dice=val,optimizer_updates=2))
        if val>best:
            best=val;best_epoch=epoch;stale=0;torch.save(dict(state=model.state_dict(),epoch=epoch,validation_dice=val,protocol=protocol),checkpoint)
        else:stale+=1
        if epoch%10==0:print(ref.dataset,'INNER',out.name,epoch,'val',val,'best',best,'stale',stale,flush=True)
        if stale>=20:break
    with (out/f'training_{tag}.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
    model.load_state_dict(torch.load(checkpoint,map_location=ref.device,weights_only=True)['state']);model.eval().requires_grad_(False)
    maps={}
    with torch.inference_mode():
        for n in excluded:
            a=model(ref.data[n]['x'][None].to(ref.device)).cpu()[0].numpy().copy()
            assert a.dtype==np.float32 and a.shape==(1,512,512) and np.isfinite(a).all();maps[n]=a
    archive=out/f'excluded_fold_coarse_maps_{tag}.npz';np.savez_compressed(archive,**maps)
    with np.load(archive,allow_pickle=False) as z:assert set(z.files)==set(excluded) and all(np.array_equal(z[n],maps[n]) for n in excluded)
    decision=dict(**protocol,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),selected_epoch=best_epoch,selected_validation_dice=best,epochs_run=len(log),maps_path=str(archive),maps_sha256=sha256_file(archive),locked_before_prediction=True)
    write_json(out/f'completed_{tag}.json',decision)
    del model,opt;gc.collect();torch.cuda.empty_cache();return decision,maps

def fold_reference(ref,folds,outer,inner_maps,inner_decisions,out,tag):
    import numpy as np
    import torch
    from torch.nn import functional as F
    held=folds[outer];fit=[n for n in ref.manifest['train'] if n not in held]
    outer_cp=OOF/ref.dataset/'seed_42'/f'fold_{outer+1}'/'baseline.pt'
    cp=torch.load(outer_cp,map_location=ref.device,weights_only=False)
    assert cp['protocol']['fit_images']==fit and cp['protocol']['held_out_images']==held
    model=ref.LNet_RA(in_channels=1).to(ref.device);model.load_state_dict(cp['state']);model.eval().requires_grad_(False)
    val_maps={}
    with torch.inference_mode():
        for n in ref.manifest['validation']:val_maps[n]=model(ref.data[n]['x'][None].to(ref.device)).cpu()[0].numpy().copy()
    del model,cp;torch.cuda.empty_cache()
    data={};deps={};lineage=[]
    for n in fit:
        inner=next(k for k,fold in enumerate(folds) if n in fold);pair=tuple(sorted((outer,inner)))
        deps[n]=inner_decisions[pair]['fit_images'];coarse=inner_maps[pair][n]
        data[n]=dict(ref.data[n],coarse=torch.from_numpy(coarse.copy()))
        lineage.append(dict(image=n,source='nested baseline',excluded_folds=[v+1 for v in pair],baseline_fit_images=deps[n],baseline_checkpoint_sha256=inner_decisions[pair]['checkpoint_sha256']))
    for n in ref.manifest['validation']:data[n]=dict(ref.data[n],coarse=torch.from_numpy(val_maps[n].copy()))
    for n in held:data[n]=dict(ref.data[n])
    for n,d in data.items():d['n_coarse']=F.interpolate(d['coarse'][None],size=d['n_rgb'].shape[-2:],mode='bilinear',align_corners=False)[0].contiguous()
    require_lineage(folds,outer,fit,ref.manifest['validation'],deps)
    protocol=dict(dataset=ref.dataset,outer_fold=outer+1,fit_images=fit,held_images=held,validation_images=ref.manifest['validation'],test_images_excluded=ref.manifest['test'],training_input_lineage=lineage,outer_baseline_checkpoint=str(outer_cp),outer_baseline_sha256=sha256_file(outer_cp),strict_nested_exclusions_verified=True)
    out.mkdir(parents=True,exist_ok=True);write_json(out/f'fold_protocol_{tag}.json',protocol)
    return replace(ref,data=data,manifest=dict(train=fit,validation=ref.manifest['validation'].copy(),test=held.copy()),namespace=dict(ref.namespace,data=data)),protocol

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--self-test',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.self_test:self_test();return
    if not args.execute:print('Inert; --execute required for12nestedbaselines +40foldO3fits. No HRF.');return
    out=(args.output or ROOT/'outputs'/f'strict_o3_crossfit_{stamp()}').resolve();out.mkdir(parents=True,exist_ok=False);tag=out.name[-19:]
    state=dict(output=str(out),tag=tag,datasets=list(DS),seeds=list(SEEDS),planned_nested_baselines=12,planned_O3_fold_fits=40,completed_nested_baselines=0,completed_O3_fold_fits=0,HRF_started=False)
    def status(stage,**extra):state.update(stage=stage,updated_at=stamp(),**extra);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_strict_o3_crossfit.json',state)
    status('protecting_previous_artifacts')
    protected=json.loads((V2/f'protected_before_including_loaded_sources_{VT}.json').read_text());assert all(sha256_file(p)==h for p,h in protected.items())
    for p in V2.rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    write_json(out/f'protected_before_{tag}.json',protected)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):
        target=out/'source'/f'{p.stem}_{tag}.py';target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
    write_json(out/f'plan_{tag}.json',dict(**state,baseline='Six unordered excluded-fold pairs per dataset, seed42,8fitimages,4externalvalidation, max1000epochs patience20. Each model reused for both directions of a pair.',O3='Fourouterfolds x5seeds x2datasets;12fitimages,4externalvalidation; exactnativeO3recipe120epochs. Innerbaselineinputs exclude both outerheld and input-own folds.',selection='Only external validation for all checkpoints. No heldfold scoring used to choose O3models. All40O3choices globally locked before OOFanchor generation.',cache='Native stitched O3residuals on heldimages, producing16OOFtraininganchor maps per seed and dataset. No test refiner outputs here.',old_results_unchanged=True))
    try:
        import numpy as np
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        from flow_matching_refiner.native_patch_model import training_center_pools,predict_tiled,ErrorSelectiveModel
        from flow_matching_refiner.run_native_patch import train_one
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.is_available();write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),TF32=False))
        refs={};foldrefs={};all_decisions=[];inner_all=[]
        for ds in DS:
            job=out/f'{ds}_{tag}';status('loading_audited_data',dataset=ds);ref,audit=load_native_dataset(ds,job/'input_audit','cuda');refs[ds]=ref
            for p,h in audit['source_sha256_before'].items():
                if p in protected:assert protected[p]==h
                protected[p]=h
            protocol=json.loads((OOF/ds/'seed_42'/f'oof_protocol_{OT}.json').read_text());assert protocol['split']==ref.manifest;folds=protocol['folds']
            assert len(folds)==4 and all(len(f)==4 for f in folds)
            if ds=='CHASE':
                ss=[{ref.data[n]['subject'] for n in f} for f in folds]
                assert all(not ss[i]&ss[j] for i,j in itertools.combinations(range(4),2))
            imaps={};idec={}
            for pair in itertools.combinations(range(4),2):
                excluded=folds[pair[0]]+folds[pair[1]];fit=[n for n in ref.manifest['train'] if n not in excluded]
                d,m=train_inner_baseline(ref,fit,excluded,job/'nested_baselines'/f'exclude_{pair[0]+1}_{pair[1]+1}',tag,status);idec[pair]=d;imaps[pair]=m;inner_all.append(dict(excluded_folds=[i+1 for i in pair],**d));status('nested_baseline_complete',completed_nested_baselines=len(inner_all))
            for outer in range(4):
                fjob=job/f'outer_fold_{outer+1}';fr,fp=fold_reference(ref,folds,outer,imaps,idec,fjob,tag);foldrefs[ds,outer]=fr
                pools={n:training_center_pools(fr.data[n]['n_g'],fr.data[n]['n_fov']) for n in fr.manifest['train']}
                for seed in SEEDS:
                    status('O3_fold_training',dataset=ds,outer_fold=outer+1,seed=seed)
                    dec=train_one(fr,fjob,tag,seed,pools,status);all_decisions.append(dict(dataset=ds,outer_fold=outer+1,fit_images=fp['fit_images'],held_images=fp['held_images'],validation_images=fp['validation_images'],fold_protocol=str(fjob/f'fold_protocol_{tag}.json'),raw_path=str(fjob/'native_patch_direct'/f'seed_{seed}'/f'best_raw_model_{tag}.pt'),**dec));status('O3_fold_complete',completed_O3_fold_fits=len(all_decisions))
                del pools
            write_json(job/f'inner_baseline_inventory_{tag}.json',[v for v in inner_all if v['dataset']==ds]);del imaps;gc.collect();torch.cuda.empty_cache()
        assert len(inner_all)==12 and len(all_decisions)==40
        lock=out/f'all40_O3_choices_locked_before_OOF_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=all_decisions,inner_baselines=inner_all));lh=sha256_file(lock)
        entries=[]
        for dec in all_decisions:
            ds=dec['dataset'];fr=foldrefs[ds,dec['outer_fold']-1];model=ErrorSelectiveModel(False).to(fr.device);assert sha256_file(dec['raw_path'])==dec['raw_sha256'];model.load_state_dict(torch.load(dec['raw_path'],map_location=fr.device,weights_only=True)['state']);model.eval().requires_grad_(False)
            dest=out/f'{ds}_{tag}/OOF_anchor_cache/seed_{dec["refiner_seed"]}';dest.mkdir(parents=True,exist_ok=True)
            for name in dec['held_images']:
                status('generating_strict_OOF_anchors',dataset=ds,outer_fold=dec['outer_fold'],seed=dec['refiner_seed'],image=name)
                d=fr.data[name];_,a,meta=predict_tiled(model,d['n_coarse'],d['n_rgb'],fr.device,return_residual=True)
                path=dest/f'{name}_OOF_anchor_{tag}.npz';np.savez_compressed(path,anchor_residual_native=a.numpy(),coarse512=d['coarse'].numpy())
                with np.load(path,allow_pickle=False) as z:assert np.array_equal(z['anchor_residual_native'],a.numpy()) and np.array_equal(z['coarse512'],refs[ds].data[name]['coarse'].numpy())
                entries.append(dict(dataset=ds,seed=dec['refiner_seed'],image=name,outer_fold=dec['outer_fold'],path=str(path),sha256=sha256_file(path),anchor_model_sha256=dec['raw_sha256'],anchor_model_path=dec['raw_path'],fit_images=dec['fit_images'],validation_images=dec['validation_images'],held_images=dec['held_images'],fold_protocol=dec['fold_protocol'],strict_nested_exclusion=True,**meta))
            del model;gc.collect();torch.cuda.empty_cache()
        assert len(entries)==160 and len({(e['dataset'],e['seed'],e['image']) for e in entries})==160 and sha256_file(lock)==lh
        write_json(out/f'OOF_anchor_manifest_{tag}.json',dict(records=entries,global_lock_path=str(lock),global_lock_sha256=lh))
        status('checking_previous_sources');after={p:sha256_file(p) for p in protected};assert after==protected
        write_json(out/f'protected_before_including_loaded_sources_{tag}.json',protected);write_json(out/f'protected_after_{tag}.json',after)
        write_json(out/f'completed_strict_crossfit_{tag}.json',dict(completed_at=stamp(),nested_baseline_fits=12,O3_fold_fits=40,OOF_anchor_maps=160,training_images_per_seed_dataset=16,strict_indirect_leakage_exclusions_verified=True,old_files_unchanged=True,HRF_started=False,lock_sha256=lh))
        status('strict_O3_crossfit_complete');print('STRICT_O3_CROSSFIT_COMPLETE',str(out),flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
