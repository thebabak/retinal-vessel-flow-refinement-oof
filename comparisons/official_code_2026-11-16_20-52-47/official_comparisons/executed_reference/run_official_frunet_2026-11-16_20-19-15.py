"""Pinned official FR-UNet network under a new controlled retinal protocol.

No pretrained weights or buggy upstream DTI helper is used. Original model
definitions and initialization are loaded unchanged. The existing green512
input,120-epoch budget, logicalbatch4/microbatch1 and validation selection are
explicit adaptations, not a reproduction of the original paper's scores.
"""
from __future__ import annotations
import argparse,ast,csv,hashlib,importlib.util,json,math,random,shutil,subprocess
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
COMMON_SOURCE=HERE/'run_official_segflow_2026-11-16_19-59-19.py'
_spec=importlib.util.spec_from_file_location('frunet_retained_data_helpers',COMMON_SOURCE)
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
WORKSPACE=common.WORKSPACE
stamp,sha,write_json=common.stamp,common.sha,common.write_json
SEEDS=[42,123,2026,3000,3001];DATASETS=['CHASE','DRIVE','HRF'];LRS=[1e-5,1e-4,3e-4];PILOT_SEED=91011
PINNED_COMMIT='1856bc206055fdfccde184d0ebf9ac8c3207387d'

def official_components(source):
    import torch
    source=Path(source).resolve()
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=PINNED_COMMIT:raise ValueError('Official FR-UNet checkout differs from pinned commit')
    init_path=source/'models/utils.py'
    tree=ast.parse(init_path.read_text(encoding='utf-8-sig'))
    nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='InitWeights_He']
    if len(nodes)!=1:raise RuntimeError('Official initialization class changed')
    ns=dict(torch=torch,nn=torch.nn,trunc_normal_=torch.nn.init.trunc_normal_)
    extract=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(extract)
    exec(compile(extract,str(init_path),'exec'),ns)
    model_path=source/'models/fr_unet.py';modeltree=ast.parse(model_path.read_text(encoding='utf-8-sig'))
    removed=[n for n in modeltree.body if isinstance(n,ast.ImportFrom) and n.level==1 and n.module=='utils']
    if len(removed)!=1:raise RuntimeError('Unexpected official model import contract')
    modeltree.body=[n for n in modeltree.body if n not in removed];ast.fix_missing_locations(modeltree)
    exec(compile(modeltree,str(model_path),'exec'),ns)
    # FR_UNet has no Linear submodule; the unused timm trunc_normal dependency is
    # never called. All actually executed convolution/normalization initialization
    # is the exact extracted official InitWeights_He definition.
    return SimpleNamespace(Network=ns['FR_UNet'],source=source,commit=commit,
        source_sha256={str(p.relative_to(source)).replace('\\','/'):sha(p) for p in source.rglob('*.py')},
        initialization_AST_sha256=hashlib.sha256(ast.dump(extract,include_attributes=False).encode()).hexdigest(),
        model_AST_sha256=hashlib.sha256(ast.dump(modeltree,include_attributes=False).encode()).hexdigest())

def make_model(official):
    import torch
    model=official.Network(num_classes=1,num_channels=1,feature_scale=2,dropout=.2,fuse=True,out_ave=True)
    assert not any(isinstance(m,torch.nn.Linear) for m in model.modules())
    return model

def image_inputs(green):
    """GT-free green/CLAHE reference input; no image or label resizing."""
    if green.ndim!=4 or green.shape[1]!=1 or green.shape[-2:]!=(512,512):
        raise ValueError('Expected retained Bx1x512x512 green input')
    return green.float()

def _score(ref,p,name):
    d=ref.data[name]
    return ref.score(p,d['g'])['dice'] if ref.dataset=='CHASE' else ref.score(p,d['g'],d['fov']>.5)['dice']

def predict(model,green,device):
    import torch
    with torch.inference_mode():return torch.sigmoid(model(image_inputs(green).to(device))).cpu()

def validation_score(model,ref,device):
    import numpy as np
    model.eval();values=[]
    for name in ref.manifest['validation']:
        p=predict(model,ref.data[name]['x'][None],device)
        values.append(_score(ref,p[0],name))
    return float(np.mean(values))

def train_fit(official,ref,dest,tag,seed,lr,epochs,device,microbatch=1):
    import torch,numpy as np
    from torch.nn import functional as F
    done=dest/f'selection_locked_{tag}.json'
    if done.exists():
        decision=json.loads(done.read_text());assert sha(dest/decision['checkpoint'])==decision['checkpoint_sha256'];return decision
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=make_model(official).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    shuffle=random.Random(seed+28101);augment=random.Random(seed+28102)
    best=dict(epoch=None,validation_dice=-1.,checkpoint=None);start_epoch=1;updates=0
    resume=dest/f'last_training_state_{tag}.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=device,weights_only=False)
        assert saved['config']==dict(seed=seed,lr=lr,epochs=epochs)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['schedule'])
        shuffle.setstate(saved['shuffle']);augment.setstate(saved['augment']);start_epoch=saved['epoch']+1;updates=saved['updates'];best=saved['best']
        torch.set_rng_state(saved['torch_cpu_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['torch_cuda_rng']])
    log=dest/f'training_validation_{tag}.csv'
    fields=['epoch','optimizer_updates','mean_BCE','learning_rate','validation_dice','best_epoch','best_validation_dice']
    if not log.exists():
        with log.open('x',newline='',encoding='utf-8') as h:csv.DictWriter(h,fieldnames=fields).writeheader()
    for epoch in range(start_epoch,epochs+1):
        model.train();order=list(ref.manifest['train']);shuffle.shuffle(order);losses=[]
        for start in range(0,len(order),4):
            names=order[start:start+4];x=[];g=[]
            for name in names:
                xx=ref.data[name]['x'];gg=ref.data[name]['g']
                # Preserve official aligned H/V flips and right-angle rotations.
                if augment.random()<.5:xx=xx.flip(-1);gg=gg.flip(-1)
                if augment.random()<.5:xx=xx.flip(-2);gg=gg.flip(-2)
                k=augment.randrange(4);x.append(xx.rot90(k,[-2,-1]));g.append(gg.rot90(k,[-2,-1]))
            x=image_inputs(torch.stack(x)).to(device);g=torch.stack(g).to(device)
            optimizer.zero_grad(set_to_none=True);batch_loss=0.
            for offset in range(0,len(names),microbatch):
                logits=model(x[offset:offset+microbatch]);target=g[offset:offset+microbatch]
                # Official config is BCEWithLogits(pos_weight=1), unmasked full
                # image; only device placement differs from upstream .cuda().
                loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=torch.ones(1,device=device))*len(logits)/len(names)
                loss.backward();batch_loss+=float(loss.detach())
            optimizer.step();updates+=1;losses.append(batch_loss)
        schedule.step();val=None
        if epoch%10==0 or epoch==epochs:
            val=validation_score(model,ref,device)
            candidate=dest/f'candidate_epoch{epoch:03d}_{tag}.pt'
            if not candidate.exists():torch.save(dict(state=model.state_dict(),epoch=epoch,validation_dice=val),candidate)
            if val>best['validation_dice']+1e-12:best=dict(epoch=epoch,validation_dice=val,checkpoint=candidate.name)
            print(f'{ref.dataset} FR_UNet seed={seed} lr={lr:g} epoch={epoch} val={val:.7f} best={best}',flush=True)
        with log.open('a',newline='',encoding='utf-8') as h:
            csv.DictWriter(h,fieldnames=fields).writerow(dict(epoch=epoch,optimizer_updates=updates,mean_BCE=float(np.mean(losses)),learning_rate=optimizer.param_groups[0]['lr'],validation_dice=val,best_epoch=best['epoch'],best_validation_dice=best['validation_dice']))
        temporary=resume.with_suffix('.tmp')
        torch.save(dict(config=dict(seed=seed,lr=lr,epochs=epochs),model=model.state_dict(),optimizer=optimizer.state_dict(),schedule=schedule.state_dict(),epoch=epoch,updates=updates,best=best,shuffle=shuffle.getstate(),augment=augment.getstate(),torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all()),temporary)
        temporary.replace(resume)
    if best['checkpoint'] is None:raise RuntimeError('No learned validation checkpoint')
    decision=dict(dataset=ref.dataset,seed=seed,learning_rate=lr,**best,locked_at=stamp(),checkpoint_sha256=sha(dest/best['checkpoint']),training_parameter_count=sum(p.numel() for p in model.parameters()),epochs=epochs,optimizer_updates=updates,logical_batch_size=4,microbatch_size=microbatch,selected_weights='raw model',official_source_commit=official.commit)
    write_json(done,decision);del model,optimizer,schedule;torch.cuda.empty_cache();return decision

def evaluate_fit(official,ref,dest,tag,selection,device,cutoff):
    import torch,numpy as np
    from flow_matching_refiner.structure_metrics_all import metrics
    checkpoint=dest/selection['checkpoint'];assert sha(checkpoint)==selection['checkpoint_sha256']
    model=make_model(official).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state']);model.eval().requires_grad_(False)
    rows=[];reference_root=WORKSPACE/'paired_reproduction/all_datasets_five_full_runs_2026-10-08_15-38-24'/ref.dataset/'train_seed_42'
    for name in ref.manifest['test']:
        d=ref.data[name];p=predict(model,d['x'][None],device)
        with np.load(reference_root/f'{name}_reference.npz',allow_pickle=False) as src:
            baseline=src['coarse'].copy();assert np.array_equal(src['ground_truth'],d['g'].numpy());assert np.array_equal(src['fov'].astype(bool),d['fov'].numpy().astype(bool))
        target=dest/f'{name}_FR_UNet_predictions_{tag}.npz'
        if not target.exists():np.savez_compressed(target,baseline=baseline,raw=p.numpy()[0],ground_truth=d['g'].numpy(),fov=d['fov'].numpy().astype(bool))
        with np.load(target,allow_pickle=False) as saved:
            assert np.array_equal(saved['raw'],p.numpy()[0]);assert np.array_equal(saved['baseline'],baseline);assert np.array_equal(saved['ground_truth'],d['g'].numpy())
            values=metrics(saved['raw'][0],saved['ground_truth'][0],saved['baseline'][0],cutoff,fov=saved['fov'][0])
        rows.append(dict(arm='official_FR_UNet_network_controlled_protocol_adaptation',refiner_seed=selection['seed'],image=name,subject=int(d['subject']) if ref.dataset=='CHASE' else ref.manifest['test'].index(name),output='raw',selected_epoch=selection['epoch'],velocity_calls=1,gate_calls=0,**values))
    with (dest/f'test_records_{tag}.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(dest/f'completed_test_{tag}.json',dict(records=len(rows),all_saved_predictions_reopened_and_verified=True))
    del model;torch.cuda.empty_cache();return rows

def _record_gating_amendment(out,tag,old_plan,new_plan,allow):
    previous=HERE/'run_official_frunet_2026-11-16_20-11-08.py'
    if old_plan['wrapper_sha256']==new_plan['wrapper_sha256']:return
    if not allow or sha(previous)!=old_plan['wrapper_sha256']:raise RuntimeError('Controller amendment requires preserved original wrapper and explicit flag')
    old={n.name:n for n in ast.parse(previous.read_text()).body if isinstance(n,ast.FunctionDef) and n.name!='main'}
    new={n.name:n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    for name,node in old.items():assert ast.dump(node,include_attributes=False)==ast.dump(new[name],include_attributes=False),name
    target=out/'dataset_gating_amendment_2026-11-16_20-19-15.json'
    if not target.exists():write_json(target,dict(created_at=stamp(),previous_wrapper_sha256=sha(previous),new_wrapper_sha256=sha(__file__),unchanged_function_AST=list(old),reason='Hold DRIVE and every test while source/protocol decision is pending',selection_policy='Per-dataset configuration before its finals; all intended final choices before test',default_test_execution=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-source',type=Path,default=HERE/'FR-UNet');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=120);p.add_argument('--microbatch',type=int,default=1);p.add_argument('--memory-fraction',type=float,default=.2)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS);p.add_argument('--finalize-tests',action='store_true')
    p.add_argument('--resume-gating-amendment',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print('No experiment executed; --execute required.');return
    if a.epochs<10 or a.microbatch!=1 or not 0<a.memory_fraction<=1 or len(set(a.datasets))!=len(a.datasets):raise ValueError('Invalid training controls')
    selected=[d for d in DATASETS if d in a.datasets]
    out=a.output.resolve();tag=out.name[-19:];common.datetime.strptime(tag,'%Y-%m-%d_%H-%M-%S')
    if out.is_relative_to(a.official_source.resolve()):raise ValueError('Outputs cannot modify official source')
    plan_path=out/f'plan_{tag}.json'
    if out.exists() and any(out.iterdir()) and not plan_path.exists():raise FileExistsError('Existing unrelated output folder')
    if (out/f'completed_training_evaluation_{tag}.json').exists():print('Already completed; no files changed.');return
    out.mkdir(parents=True,exist_ok=True)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for final fits')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction);torch.set_num_threads(4);device='cuda'
    official=official_components(a.official_source)
    plan=dict(version=tag,created_at=stamp(),method='Official FR-UNet network controlled-protocol adaptation',official_repository='https://github.com/lseventeen/FR-UNet',official_commit=official.commit,source_sha256=official.source_sha256,model_AST_sha256=official.model_AST_sha256,initialization_AST_sha256=official.initialization_AST_sha256,wrapper_sha256=sha(__file__),shared_loader_helper_sha256=sha(COMMON_SOURCE),license='MIT',pretrained_weights_used=False,original_paper_reproduction=False,
        datasets=DATASETS,training_seeds=SEEDS,pilot_seed=PILOT_SEED,learning_rate_candidates=LRS,epochs=a.epochs,optimizer='Adam;weight_decay1e-5;cosine120epochs',logical_batch_size=4,microbatch_size=1,GPU_memory_fraction_limit=a.memory_fraction,
        objective='Official config full-image BCEWithLogits(posweight1),noDice',architecture='Unchanged official FR_UNet1channel,feature_scale2,dropout.2,fuseTrue,out_aveTrue',input='Retained512green/CLAHE,GT-freeinference',comparison_grid=512,evaluation_threshold=.5,
        training_augmentation='Official alignedH/Vflips andrandomright-angle rotations',validation_augmentation=False,
        selection='ThreeLRpilots per dataset,nine total; datasetconfiguration locks before its5freshfinalfits; all intended15finalchoices globally lock beforeANYtest; --finalize-tests explicitlyrequired',default_test_execution=False,
        adaptations=['Original40epoch48patch/batch512 changedto120epochswhole512/logicalbatch4','Retainedgreenpreprocessingandfixedpartitions','Microbatch1gradientaccumulation:BatchNorm remainsmicrobatch1,NOTequivalentBNbatch4','FP32insteadupstreamAMP','Validation-only3LRpilotsandcheckpointselection,nopretrainedweights'],DTI_executed=False,DTI_reason='PinnedslowhelperBooleanall/aliases/unboundedneighbors;fasthelperignoreshighthreshold;notexecutedorsilentlyrepaired')
    if plan_path.exists():
        previous=json.loads(plan_path.read_text());assert previous['epochs']==a.epochs and previous['official_commit']==official.commit
        _record_gating_amendment(out,tag,previous,plan,a.resume_gating_amendment)
    else:write_json(plan_path,plan)
    license_copy=out/f'FR_UNet_LICENSE_{tag}.txt'
    if license_copy.exists():assert sha(license_copy)==sha(official.source/'LICENSE')
    else:shutil.copy2(official.source/'LICENSE',license_copy)
    configurations={}
    for dataset in selected:
        ref=common.load_data(dataset,out,device);choices=[]
        for lr in LRS:
            choices.append(train_fit(official,ref,out/dataset/'pilots'/f'lr_{lr:g}',tag,PILOT_SEED,lr,a.epochs,device,a.microbatch))
            write_json(out/f'status_{tag}.json',dict(stage='pilots',dataset=dataset,learning_rate=lr,updated_at=stamp(),tests_deferred=True))
            if (out/'PAUSE_REQUESTED').exists():write_json(out/f'status_{tag}.json',dict(stage='paused_after_completed_fit',dataset=dataset,updated_at=stamp(),tests_deferred=True));return
        chosen=sorted(choices,key=lambda c:c['learning_rate'])[0]
        for c in sorted(choices,key=lambda c:c['learning_rate']):
            if c['validation_dice']>chosen['validation_dice']+1e-12:chosen=c
        configurations[dataset]=dict(learning_rate=chosen['learning_rate'],pilot_validation_dice=chosen['validation_dice'])
        target=out/dataset/f'configuration_before_dataset_final_fitting_{tag}.json'
        value=dict(locked_at=stamp(),dataset=dataset,configuration=configurations[dataset],pilot_fits=choices)
        if target.exists():assert json.loads(target.read_text())['configuration']==value['configuration']
        else:write_json(target,value)
        del ref
    for dataset in selected:
        ref=common.load_data(dataset,out,device)
        for seed in SEEDS:
            train_fit(official,ref,out/dataset/'final'/f'seed_{seed}',tag,seed,configurations[dataset]['learning_rate'],a.epochs,device,a.microbatch)
            write_json(out/f'status_{tag}.json',dict(stage='final_fitting',dataset=dataset,seed=seed,updated_at=stamp(),tests_deferred=True))
            if (out/'PAUSE_REQUESTED').exists():write_json(out/f'status_{tag}.json',dict(stage='paused_after_completed_fit',dataset=dataset,updated_at=stamp(),tests_deferred=True));return
        del ref
    decisions={};missing=[];config_hashes={}
    for dataset in DATASETS:
        decisions[dataset]=[];config=out/dataset/f'configuration_before_dataset_final_fitting_{tag}.json'
        if config.exists():config_hashes[dataset]=sha(config)
        else:missing.append(dataset+':configuration')
        for seed in SEEDS:
            decision=out/dataset/'final'/f'seed_{seed}'/f'selection_locked_{tag}.json'
            if decision.exists():decisions[dataset].append(json.loads(decision.read_text()))
            else:missing.append(dataset+':seed_'+str(seed))
    if missing or not a.finalize_tests:
        status=dict(stage='selected_dataset_fitting_completed_TESTS_HELD',datasets_fitted=selected,updated_at=stamp(),missing_intended_selections=missing,test_execution_requested=a.finalize_tests,tests_executed=False)
        write_json(out/f'status_{tag}.json',status);write_json(out/f'partial_stage_completed_{"_".join(selected)}_{tag}.json',status)
        print('FITTING_COMPLETE_TESTS_HELD',json.dumps(status),flush=True);return
    lock=out/f'all_selections_before_test_{tag}.json'
    if not lock.exists():write_json(lock,dict(locked_at=stamp(),decisions=decisions,dataset_configuration_locks_sha256=config_hashes))
    all_rows=[]
    for dataset in DATASETS:
        ref=common.load_data(dataset,out,device);old=common.OLD[dataset];cutoff=float(json.loads((old/f'thin_definition_{old.name[-19:]}.json').read_text())['cutoff'])
        for choice in decisions[dataset]:
            dest=out/dataset/'final'/f'seed_{choice["seed"]}'
            if (dest/f'completed_test_{tag}.json').exists():
                with (dest/f'test_records_{tag}.csv').open(newline='',encoding='utf-8') as h:rows=list(csv.DictReader(h))
            else:rows=evaluate_fit(official,ref,dest,tag,choice,device,cutoff)
            all_rows.extend(dict(dataset=dataset,**row) for row in rows)
        del ref
    with (out/f'all_test_records_{tag}.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    assert all(sha(official.source/p)==value for p,value in official.source_sha256.items())
    write_json(out/f'completed_training_evaluation_{tag}.json',dict(completed_at=stamp(),pilot_fits=9,final_fits=15,test_image_seed_records=len(all_rows),official_source_bytes_unchanged=True,original_paper_reproduction=False,method=plan['method']))
    print('OFFICIAL_FR_UNET_ADAPTATION_COMPLETE',out,flush=True)

if __name__=='__main__':main()
