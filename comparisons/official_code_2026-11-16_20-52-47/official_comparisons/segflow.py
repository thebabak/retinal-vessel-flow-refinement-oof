"""Controlled-budget retinal adaptation of pinned official FMS2 SegFlow code.

The official architecture, independent conditional path, velocity MSE,
self-conditioning and Euler implementation are loaded unchanged. The fitting
budget/optimizer schedule, data protocol, validation-only selection, foreground
polarity and final comparison-grid projection are explicitly adapted. This is
not a reproduction of the original FMS2 paper or its full synthesis pipeline.
"""
from __future__ import annotations
import argparse, ast, copy, csv, hashlib, importlib, json, math, random
from pathlib import Path
from types import ModuleType, SimpleNamespace
import shutil, subprocess, sys, time, traceback, typing, warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from . import portable_api as portable

WORKSPACE = None
PINNED_COMMIT = 'ab18a8f33c7aa27425d7265c3dc9a69bf1d8c5a4'
SEEDS = [42,123,2026,3000,3001]
DATASETS = ['CHASE','DRIVE','HRF']
PILOT_SEED = 91011
LRS = [1e-5,1e-4,3e-4]
OLD = {}

def stamp(): return datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    temporary.replace(path)

def official_components(source):
    """Load unchanged official files; extract pure definitions to omit unused deps."""
    import torch
    source=Path(source).resolve()
    commit=portable.verify_snapshot(source,PINNED_COMMIT)['commit']
    if commit!=PINNED_COMMIT:raise ValueError('Official source commit differs from the locked version')
    package='fms2_pinned_official'
    for suffix,relative in [('', 'torchcfm'),('.models','torchcfm/models'),('.models.unet','torchcfm/models/unet')]:
        name=package+suffix
        if name not in sys.modules:
            module=ModuleType(name);module.__path__=[str(source/relative)]
            sys.modules[name]=module
        elif list(sys.modules[name].__path__) != [str(source/relative)]:
            raise RuntimeError('An official-source namespace points at different bytes')
    network=importlib.import_module(package+'.models.unet.unet_residal').UNetModel
    ns=dict(torch=torch,Union=typing.Union,math=math,warnings=warnings,
            FLAGS=SimpleNamespace(self_cond=True))
    extracted={}
    for relative,names in [
        ('torchcfm/conditional_flow_matching.py',{'pad_t_like_x','ConditionalFlowMatcher'}),
        ('FMS/segflow/train.py',{'generate_samples_with_neural_ode'}),
        ('FMS/synflow/utils.py',{'ema'}),
    ]:
        path=source/relative;tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
        if {node.name for node in nodes}!=names:raise RuntimeError('Official definition contract changed')
        subset=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(subset)
        exec(compile(subset,str(path),'exec'),ns)
        extracted[relative]=dict(file_sha256=sha(path),definitions=sorted(names),
            definition_AST_sha256=hashlib.sha256(ast.dump(subset,include_attributes=False).encode()).hexdigest())
    return SimpleNamespace(Network=network,Matcher=ns['ConditionalFlowMatcher'],
        generate=ns['generate_samples_with_neural_ode'],ema=ns['ema'],
        source=source,commit=commit,extracted=extracted)

def make_model(official,image_size=256):
    return official.Network(image_size=image_size,in_channels=6,model_channels=32,
        out_channels=3,num_res_blocks=1,attention_resolutions=(32,16,8),dropout=0.,
        channel_mult=(1,2,4),use_checkpoint=False,use_fp16=False,num_heads=4,
        num_head_channels=32,num_heads_upsample=-1,use_scale_shift_norm=True,
        resblock_updown=True,use_new_attention_order=False)

def image_inputs(rgb,size=256):
    """GT-free deployment preprocessing of retained RGB fields."""
    import torch
    from PIL import Image
    import numpy as np
    images=[]
    for image in rgb:
        # The 512 tensors are quantized retained RGB fields; round-trip to their bytes.
        a=(image.detach().cpu().permute(1,2,0).numpy()*255).round().clip(0,255).astype('uint8')
        a=np.asarray(Image.fromarray(a).resize((size,size),Image.Resampling.BICUBIC)).copy()
        images.append(torch.from_numpy(a).permute(2,0,1).float()/127.5-1)
    return torch.stack(images)

def target_inputs(gt,size=256):
    """Training-only target formatting in the official foreground convention."""
    import torch
    from PIL import Image
    import numpy as np
    targets=[]
    for truth in gt:
        g=(truth.detach().cpu()[0].numpy()>=.5).astype('uint8')
        # Official to_binary_mask uses state.mean(channel)<0 as foreground.
        mask=np.asarray(Image.fromarray(255*(1-g)).resize((size,size),Image.Resampling.NEAREST)).copy()
        targets.append(torch.from_numpy(mask).float()[None].expand(3,-1,-1).clone()/127.5-1)
    return torch.stack(targets)

def transport_inputs(rgb,gt,size=256):
    """Training-only convenience wrapper; deployment uses image_inputs alone."""
    return image_inputs(rgb,size),target_inputs(gt,size)

def probability_from_state(state,comparison_size=512):
    import torch.nn.functional as F
    p=((1-state.mean(1,keepdim=True))/2).clamp(0,1)
    if p.shape[-2:]!=(comparison_size,comparison_size):
        p=F.interpolate(p,size=(comparison_size,comparison_size),mode='bilinear',align_corners=False)
    return p.clamp(0,1)

def predict(official,model,image256,device):
    import torch
    with torch.inference_mode():
        state=official.generate(model,image256.to(device),num_steps=10,integration_method='euler')
        return probability_from_state(state).cpu(),state.cpu()

def load_data(dataset,out,device):
    """Portable manifest loader; private test I/O remains held by controller."""
    return portable.load_data(dataset)

def _score(ref,p,name):
    d=ref.data[name]
    return ref.score(p,d['g'])['dice'] if ref.dataset=='CHASE' else ref.score(p,d['g'],d['fov']>.5)['dice']

def val_score(official,model,ref,prepared,device):
    import numpy as np
    model.eval();values=[]
    for name in ref.manifest['validation']:
        p,_=predict(official,model,prepared[name][0][None],device)
        values.append(_score(ref,p[0],name))
    return float(np.mean(values))

def prepared_images(ref):
    # Test data is converted only during the explicitly separated test phase.
    names=ref.manifest['train']+ref.manifest['validation']
    result={}
    for name in names:
        x,y=transport_inputs(ref.data[name]['rgb'][None],ref.data[name]['g'][None])
        result[name]=(x[0],y[0])
    return result

def train_fit(official,ref,prepared,dest,tag,seed,lr,epochs,device,microbatch=1):
    import torch
    import torch.nn.functional as F
    import numpy as np
    done=dest/f'selection_locked_{tag}.json'
    if done.exists():
        selection=json.loads(done.read_text());assert sha(dest/selection['checkpoint'])==selection['checkpoint_sha256']
        return selection
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=make_model(official).to(device)
    ema_model=copy.deepcopy(model).requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    updates_per_epoch=math.ceil(len(ref.manifest['train'])/4)
    total_updates=epochs*updates_per_epoch;warmup_updates=max(1,round(.05*total_updates))
    warm=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1e-8,total_iters=warmup_updates)
    cosine=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=total_updates-warmup_updates,eta_min=lr*.1)
    schedule=torch.optim.lr_scheduler.SequentialLR(optimizer,[warm,cosine],milestones=[warmup_updates])
    matcher=official.Matcher(sigma=0.)
    shuffle=random.Random(seed+27101);self_condition=random.Random(seed+27102)
    torch.cuda.manual_seed_all(seed+27103)
    best=dict(epoch=None,validation_dice=-1.,checkpoint=None)
    global_step=0;start_epoch=1
    resume=dest/f'last_training_state_{tag}.pt'
    if resume.exists():
        checkpoint=torch.load(resume,map_location=device,weights_only=False)
        assert checkpoint['config']==dict(seed=seed,lr=lr,epochs=epochs)
        model.load_state_dict(checkpoint['model']);ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer']);schedule.load_state_dict(checkpoint['schedule'])
        start_epoch=checkpoint['epoch']+1;global_step=checkpoint['global_step'];best=checkpoint['best']
        shuffle.setstate(checkpoint['shuffle']);self_condition.setstate(checkpoint['self_condition'])
        torch.set_rng_state(checkpoint['torch_cpu_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['torch_cuda_rng']])
    log=dest/f'training_validation_{tag}.csv'
    fields=['epoch','updates','mean_velocity_mse','learning_rate','ema_validation_dice','best_epoch','best_validation_dice']
    if not log.exists():
        with log.open('x',newline='',encoding='utf-8') as h:csv.DictWriter(h,fieldnames=fields).writeheader()
    for epoch in range(start_epoch,epochs+1):
        model.train();order=list(ref.manifest['train']);shuffle.shuffle(order);loss_values=[]
        for start in range(0,len(order),4):
            names=order[start:start+4]
            x=torch.stack([prepared[n][0] for n in names]).to(device)
            y=torch.stack([prepared[n][1] for n in names]).to(device)
            t=torch.full((len(names),),(global_step%11)/10,device=device,dtype=x.dtype)
            t,xt,ut=matcher.sample_location_and_conditional_flow(x,y,t=t)
            do_self_condition=self_condition.random()<.5
            optimizer.zero_grad(set_to_none=True);batch_loss=0.
            for offset in range(0,len(names),microbatch):
                mb=xt[offset:offset+microbatch];tm=t[offset:offset+microbatch];target=ut[offset:offset+microbatch]
                sc=torch.zeros_like(mb)
                if do_self_condition:
                    with torch.no_grad():
                        v0=model(x=torch.cat([mb,sc],1),timesteps=tm*1000.)
                        sc=(mb+(1-tm[:,None,None,None])*v0).detach()
                velocity=model(x=torch.cat([mb,sc],1),timesteps=tm*1000.)
                loss=F.mse_loss(velocity,target,reduction='none').mean()*len(mb)/len(names)
                loss.backward();batch_loss+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();schedule.step();official.ema(model,ema_model,.99)
            global_step+=1;loss_values.append(batch_loss)
        val=None
        if epoch%10==0 or epoch==epochs:
            val=val_score(official,ema_model,ref,prepared,device)
            candidate=dest/f'ema_candidate_epoch{epoch:03d}_{tag}.pt'
            if not candidate.exists():torch.save(dict(state=ema_model.state_dict(),epoch=epoch,validation_dice=val),candidate)
            if val>best['validation_dice']+1e-12:
                best=dict(epoch=epoch,validation_dice=val,checkpoint=candidate.name)
            print(f'{ref.dataset} seed={seed} lr={lr:g} epoch={epoch} val={val:.7f} best={best}',flush=True)
        with log.open('a',newline='',encoding='utf-8') as h:
            csv.DictWriter(h,fieldnames=fields).writerow(dict(epoch=epoch,updates=global_step,mean_velocity_mse=float(np.mean(loss_values)),
                learning_rate=optimizer.param_groups[0]['lr'],ema_validation_dice=val,best_epoch=best['epoch'],best_validation_dice=best['validation_dice']))
        # Only a working restart alias is overwritten; epoch candidates remain immutable.
        temporary=resume.with_suffix('.tmp')
        torch.save(dict(config=dict(seed=seed,lr=lr,epochs=epochs),model=model.state_dict(),ema_model=ema_model.state_dict(),
            optimizer=optimizer.state_dict(),schedule=schedule.state_dict(),epoch=epoch,global_step=global_step,best=best,
            shuffle=shuffle.getstate(),self_condition=self_condition.getstate(),torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all()),temporary)
        temporary.replace(resume)
    if best['checkpoint'] is None:raise RuntimeError('No learned validation checkpoint selected')
    selection=dict(dataset=ref.dataset,seed=seed,learning_rate=lr,**best,locked_at=stamp(),
        checkpoint_sha256=sha(dest/best['checkpoint']),training_parameter_count=sum(p.numel() for p in model.parameters()),
        epochs=epochs,optimizer_updates=global_step,logical_batch_size=4,microbatch_size=microbatch,
        official_source_commit=official.commit,selected_weights='EMA decay0.99')
    write_json(done,selection)
    del model,ema_model,optimizer,schedule;torch.cuda.empty_cache()
    return selection

def evaluate_fit(official,ref,prepared,dest,tag,selection,device,cutoff):
    import torch,numpy as np
    from .structure_metrics_all import metrics
    checkpoint=dest/selection['checkpoint'];assert sha(checkpoint)==selection['checkpoint_sha256']
    model=make_model(official).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state']);model.eval().requires_grad_(False)
    rows=[]
    for name in ref.manifest['test']:
        d=ref.data[name]
        x=image_inputs(d['rgb'][None])
        p,state=predict(official,model,x,device)
        with np.load(portable.baseline_archive(ref.dataset,name),allow_pickle=False) as original:
            baseline=original['coarse'].copy()
            assert np.array_equal(original['ground_truth'],d['g'].numpy())
            assert np.array_equal(original['fov'].astype(bool),d['fov'].numpy().astype(bool))
        path=dest/f'{name}_segflow_predictions_{tag}.npz'
        if not path.exists():
            np.savez_compressed(path,baseline=baseline,raw=p.numpy()[0],state_256=state.numpy()[0],
                ground_truth=d['g'].numpy(),fov=d['fov'].numpy().astype(bool))
        with np.load(path,allow_pickle=False) as saved:
            assert np.array_equal(saved['raw'],p.numpy()[0]) and np.array_equal(saved['state_256'],state.numpy()[0])
            assert np.array_equal(saved['baseline'],baseline) and np.array_equal(saved['ground_truth'],d['g'].numpy())
            assert np.array_equal(probability_from_state(torch.from_numpy(saved['state_256'])[None]).numpy()[0],saved['raw'])
            values=metrics(saved['raw'][0],saved['ground_truth'][0],saved['baseline'][0],cutoff,fov=saved['fov'][0])
        rows.append(dict(arm='official_FMS2_SegFlow_controlled_budget_adaptation',refiner_seed=selection['seed'],image=name,
            subject=int(d['subject']) if ref.dataset=='CHASE' else ref.manifest['test'].index(name),output='raw',
            selected_epoch=selection['epoch'],velocity_calls=10,gate_calls=0,**values))
    with (dest/f'test_records_{tag}.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(dest/f'completed_test_{tag}.json',dict(records=len(rows),predictions_verified_against_saved_state=True))
    del model;torch.cuda.empty_cache()
    return rows

def _record_gating_amendment(out,tag,old_plan,new_plan,allow):
    if old_plan['wrapper_sha256']!=new_plan['wrapper_sha256']:
        raise RuntimeError('Portable packaging is for new runs; it cannot silently adopt an executed-original run')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source',type=Path,default=Path(__file__).parent/'vendor'/'fms2')
    p.add_argument('--workspace-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True,help='Workspace-relative private prepared-input manifest')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--epochs',type=int,default=120);p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--memory-fraction',type=float,default=.35)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--finalize-tests',action='store_true',help='Evaluate only after all intended dataset final selections exist')
    p.add_argument('--resume-gating-amendment',action='store_true')
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:print('No experiment executed; --execute is required.');return
    if args.epochs<10 or args.microbatch not in [1,2,4] or not 0<args.memory_fraction<=1 or len(set(args.datasets))!=len(args.datasets):
        raise ValueError('Invalid training budget, dataset gate or GPU memory fraction')
    global WORKSPACE
    portable.configure(args.workspace_root,args.output_root,args.input_manifest)
    WORKSPACE=portable.configuration().workspace_root
    selected=[d for d in DATASETS if d in args.datasets]
    out=portable.checked_output(args.output);tag=out.name[-19:];datetime.strptime(tag,'%Y-%m-%d_%H-%M-%S')
    if out.is_relative_to(args.official_source.resolve()):raise ValueError('Outputs cannot modify official source')
    if out.exists() and any(out.iterdir()) and not (out/f'plan_{tag}.json').is_file():raise FileExistsError('Unrelated nonempty output tree')
    if (out/f'completed_training_evaluation_{tag}.json').is_file():print('Completed study exists; no files changed.');return
    out.mkdir(parents=True,exist_ok=True)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for final fits')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction);torch.set_num_threads(4);device='cuda'
    official=official_components(args.official_source)
    sources={str(x.relative_to(official.source)).replace('\\','/'):sha(x) for x in official.source.rglob('*.py')}
    plan=dict(version=tag,portable_export=True,portable_input_manifest_sha256=portable.configuration().manifest_sha256,method='Official FMS2 SegFlow controlled-budget retinal adaptation',official_repository='https://github.com/BabakAsadi94/FMS2',official_commit=official.commit,license='MIT including upstream Tong CFM notices',source_sha256=sources,definitions=official.extracted,
        datasets=DATASETS,training_seeds=SEEDS,pilot_seed=PILOT_SEED,learning_rate_candidates=LRS,epochs=args.epochs,logical_batch_size=4,microbatch_size=args.microbatch,GPU_memory_fraction_limit=args.memory_fraction,
        optimizer='Adam;weight_decay0',warmup_fraction=.05,EMA_decay=.99,time_sampling='Official Euler eleven-point cycle0,0.1,...,1',
        official_reference_schedule_NOT_executed=dict(updates=48002,learning_rate=1e-5,warmup_updates=10000,EMA_decay=.9999),
        architecture='Unchanged official UNetModel;6input/32base/3output,mults1,2,4,selfcond0.5',segmentation_only=True,synthesis_module_trained=False,coarse_maps_used_as_model_input=False,image_input_size=256,comparison_grid=512,
        foreground_mapping='Negative state is vessel, official g<0 convention',objective='Exact official CFM sigma0 and full-image velocity MSE',
        inference='Exact10Euler selfconditioned; probabilityclip((1-channelmeanstate)/2);bilinear512align_cornersFalse',
        selection_policy='Dataset LR locks before its five finals; all intended datasets final choices lock before tests; --finalize-tests explicitly required',
        default_test_execution=False,source_resolution_and_budget_are_adapted=True,original_paper_reproduction=False,wrapper_sha256=sha(__file__),state_created_at=stamp())
    plan_path=out/f'plan_{tag}.json'
    if plan_path.exists():
        old=json.loads(plan_path.read_text());assert old['epochs']==args.epochs and old['official_commit']==official.commit
        assert old['portable_input_manifest_sha256']==portable.configuration().manifest_sha256
        _record_gating_amendment(out,tag,old,plan,args.resume_gating_amendment)
    else:write_json(plan_path,plan)
    license_copy=out/f'FMS2_LICENSE_{tag}.txt'
    if license_copy.exists():assert sha(license_copy)==sha(official.source/'LICENSE')
    else:shutil.copy2(official.source/'LICENSE',license_copy)
    configs={};pilots={}
    for dataset in selected:
        ref=load_data(dataset,out,device);prepared=prepared_images(ref);choices=[]
        for lr in LRS:
            choices.append(train_fit(official,ref,prepared,out/dataset/'pilots'/f'lr_{lr:g}',tag,PILOT_SEED,lr,args.epochs,device,args.microbatch))
            write_json(out/f'status_{tag}.json',dict(stage='pilots',dataset=dataset,learning_rate=lr,updated_at=stamp(),tests_deferred=True))
            if (out/'PAUSE_REQUESTED').exists():write_json(out/f'status_{tag}.json',dict(stage='paused_after_completed_fit',dataset=dataset,updated_at=stamp(),tests_deferred=True));return
        chosen=sorted(choices,key=lambda c:c['learning_rate'])[0]
        for c in sorted(choices,key=lambda c:c['learning_rate']):
            if c['validation_dice']>chosen['validation_dice']+1e-12:chosen=c
        configs[dataset]=dict(learning_rate=chosen['learning_rate'],pilot_validation_dice=chosen['validation_dice']);pilots[dataset]=choices
        config_lock=out/dataset/f'configuration_before_dataset_final_fitting_{tag}.json'
        value=dict(locked_at=stamp(),dataset=dataset,configuration=configs[dataset],pilot_fits=choices)
        if config_lock.exists():assert json.loads(config_lock.read_text())['configuration']==value['configuration']
        else:write_json(config_lock,value)
        del ref,prepared
    for dataset in selected:
        ref=load_data(dataset,out,device);prepared=prepared_images(ref)
        for seed in SEEDS:
            train_fit(official,ref,prepared,out/dataset/'final'/f'seed_{seed}',tag,seed,configs[dataset]['learning_rate'],args.epochs,device,args.microbatch)
            write_json(out/f'status_{tag}.json',dict(stage='final_fitting',dataset=dataset,seed=seed,updated_at=stamp(),tests_deferred=True))
            if (out/'PAUSE_REQUESTED').exists():write_json(out/f'status_{tag}.json',dict(stage='paused_after_completed_fit',dataset=dataset,updated_at=stamp(),tests_deferred=True));return
        del ref,prepared
    decisions={};missing=[];config_hashes={}
    for dataset in DATASETS:
        decisions[dataset]=[];config=out/dataset/f'configuration_before_dataset_final_fitting_{tag}.json'
        if config.exists():config_hashes[dataset]=sha(config)
        else:missing.append(dataset+':configuration')
        for seed in SEEDS:
            decision=out/dataset/'final'/f'seed_{seed}'/f'selection_locked_{tag}.json'
            if decision.exists():decisions[dataset].append(json.loads(decision.read_text()))
            else:missing.append(dataset+':seed_'+str(seed))
    if missing or not args.finalize_tests:
        status=dict(stage='selected_dataset_fitting_completed_TESTS_HELD',datasets_fitted=selected,updated_at=stamp(),missing_intended_selections=missing,test_execution_requested=args.finalize_tests,tests_executed=False)
        write_json(out/f'status_{tag}.json',status);write_json(out/f'partial_stage_completed_{"_".join(selected)}_{tag}.json',status)
        print('FITTING_COMPLETE_TESTS_HELD',json.dumps(status),flush=True);return
    lock=out/f'all_selections_before_test_{tag}.json'
    if not lock.exists():write_json(lock,dict(locked_at=stamp(),decisions=decisions,dataset_configuration_locks_sha256=config_hashes))
    portable.authorize_test_inputs(out,tag,DATASETS,SEEDS,args.finalize_tests)
    all_rows=[]
    for dataset in DATASETS:
        ref=load_data(dataset,out,device)
        cutoff=portable.thin_cutoff(dataset)
        for selection in decisions[dataset]:
            dest=out/dataset/'final'/f'seed_{selection["seed"]}'
            if (dest/f'completed_test_{tag}.json').exists():
                with (dest/f'test_records_{tag}.csv').open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
            else:rows=evaluate_fit(official,ref,None,dest,tag,selection,device,cutoff)
            all_rows.extend(dict(dataset=dataset,**row) for row in rows)
        del ref
    with (out/f'all_test_records_{tag}.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    assert all(sha(official.source/name)==value for name,value in sources.items())
    write_json(out/f'completed_training_evaluation_{tag}.json',dict(completed_at=stamp(),pilot_fits=9,final_fits=15,test_image_seed_records=len(all_rows),all_official_source_bytes_unchanged=True,original_paper_reproduction=False,method=plan['method']))
    print('OFFICIAL_SEGFlow_ADAPTATION_COMPLETE',out,flush=True)

if __name__=='__main__':main()
