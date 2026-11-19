"""Training, nested OOF acceptance labels and locked test execution. CHASE only."""
from __future__ import annotations
import csv,gc,hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from flow_matching_refiner.local_acceptance_core import (LocalProposer,QualityCritic,sdf,image_geometry,loss,predict_delta,candidate_probability,local_features,region_candidate,accept_corrections,BETAS,KAPPAS,TILE)
from flow_matching_refiner.structure_metrics_all import metrics,thin_skeleton_mask
from flow_matching_refiner.local_acceptor_sources import CROSS,CT,baseline_predictions,with_coarse,predict_anchor
SEEDS=(42,123,2026,3000,3001);MODES=('flow','direct');KEYS=('dice','cldice','thin_skeleton_recall')

def save_csv(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def case(ref,name,p,provenance,target_cache):
    from skimage.morphology import skeletonize
    d=ref.data[name];p=np.asarray(p,dtype=np.float32);assert p.shape==(1,512,512) and np.isfinite(p).all()
    rgb=d['rgb'].numpy();geometry=image_geometry(p,rgb)
    if name not in target_cache:
        g=d['g'].numpy();f=d['fov'].numpy();target_cache[name]=dict(g=g,fov=f,target=sdf(g[0]>=.5)[None],sk=skeletonize(g[0]>=.5)[None].astype(np.float32),thin=thin_skeleton_mask(g[0]>=.5,1.,f[0])[None].astype(np.float32))
    return dict(name=name,source=p,rgb=rgb,geometry=geometry,coarse=d['coarse'].numpy(),provenance=provenance,**target_cache[name])

def score(c,p):return metrics(p[0],c['g'][0],c['coarse'][0],1.,c['fov'][0])

def mean_scores(cases,probabilities):return {k:float(np.mean([score(c,p)[k] for c,p in zip(cases,probabilities)])) for k in KEYS}

def subset(cases,region_only=False):
    return [dict(image=c['name'],O3_sha256=array_sha(c['source']),rgb_sha256=array_sha(c['rgb']),roi_sha256=array_sha(c['geometry']['roi']),regions=c['geometry']['regions'],provenance=c['provenance']) for c in cases]

def make_source_cases(ref,sources,cache):
    lookup={(r['role'],r['key'],r['image']):r for r in sources['records']};pairs={};outer={}
    for key,r in lookup.items():
        assert sha256_file(r['path'])==r['sha256'] and sha256_file(r['source_checkpoint'])==r['source_sha256']
        with np.load(r['path'],allow_pickle=False) as z:p=z['O3'].copy();coarse=z['coarse'].copy()
        c=case(ref,r['image'],p,dict(role=r['role'],cache_path=r['path'],cache_sha256=r['sha256'],producer_checkpoint=r['source_checkpoint'],producer_sha256=r['source_sha256'],protocol=r['protocol']),cache)
        c['coarse']=coarse
        (pairs if r['role']=='pair' else outer)[(r['key'],r['image'])]=c
    return pairs,outer

def full_source_cases(ref,seed,out,tag,cache):
    from flow_matching_refiner.native_patch_model import project_residual_512,apply_residual,ErrorSelectiveModel
    from flow_matching_refiner.run_native_patch_flow_v2 import anchor_path
    manifest=json.loads((CROSS/f'OOF_anchor_manifest_{CT}.json').read_text());rows=[r for r in manifest['records'] if r['dataset']=='CHASE' and r['seed']==seed];assert len(rows)==16
    training=[]
    for r in rows:
        assert sha256_file(r['path'])==r['sha256'] and r['image'] not in r['fit_images']+r['validation_images']
        with np.load(r['path'],allow_pickle=False) as z:
            assert np.array_equal(z['coarse512'],ref.data[r['image']]['coarse'].numpy());p=apply_residual(torch.from_numpy(z['coarse512'].copy()),project_residual_512(torch.from_numpy(z['anchor_residual_native'].copy()),'CHASE')).numpy()
        training.append(case(ref,r['image'],p,dict(role='strict_fold_O3',record=r),cache))
    cp=anchor_path('CHASE',seed);net=ErrorSelectiveModel(False).to(ref.device);net.load_state_dict(torch.load(cp,map_location=ref.device,weights_only=True)['state']);net.eval().requires_grad_(False);validation=[]
    for n in ref.manifest['validation']:
        p,_=predict_anchor(ref,net,ref.data[n]);validation.append(case(ref,n,p.numpy(),dict(role='full_O3_validation',checkpoint=str(cp),sha256=sha256_file(cp)),cache))
    del net;gc.collect();torch.cuda.empty_cache();return training,validation

def train_proposer(ref,mode,seed,training,validation,out,tag,status,epochs,aux_fold=None):
    out.mkdir(parents=True,exist_ok=False);torch.manual_seed(seed+31415);torch.cuda.manual_seed_all(seed+31415);model=LocalProposer(mode).to(ref.device);initial=state_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    rng=random.Random(seed+141001);trng=torch.Generator(device=ref.device).manual_seed(seed+20261117);best=None;checkpoint=out/f'best_proposer_{tag}.pt';validation_rows=[];part_names=('bce','dice','cldice','thin','skeleton','sdf','fm')
    write_json(out/f'input_lineage_{tag}.json',dict(training=subset(training),validation=subset(validation),auxiliary_held_fold=aux_fold,comparison_grid=512,training_region_selection_uses_GT=False))
    with (out/f'training_{tag}.csv').open('x',newline='') as tf,(out/f'patch_draws_{tag}.csv').open('x',newline='') as pf:
        tw=csv.writer(tf);pw=csv.writer(pf);tw.writerow(['epoch','loss',*part_names,'lr','patches','optimizer_updates']);pw.writerow(['epoch','patch_index','image','top','left','rotation90','flip_horizontal','flip_vertical','source_sha256'])
        for epoch in range(1,epochs+1):
            status('auxiliary_proposer_training' if aux_fold else 'final_proposer_training',mode=mode,seed=seed,epoch=epoch,epochs=epochs,held_fold=aux_fold);model.train();names=list(range(len(training)))*4;rng.shuffle(names);total=np.zeros(1+len(part_names));seen=0;updates=0;lr=opt.param_groups[0]['lr']
            for start in range(0,len(names),16):
                batch=[]
                for index,j in enumerate(names[start:start+16],start=start):
                    c=training[j];regions=c['geometry']['regions']
                    if regions and rng.random()<.8:
                        r=rng.choice(regions);top=min(512-TILE,max(0,r['y']+16-64+rng.randint(-16,16)));left=min(512-TILE,max(0,r['x']+16-64+rng.randint(-16,16)))
                    else:top=rng.randrange(512-TILE+1);left=rng.randrange(512-TILE+1)
                    arrays=[c['geometry']['phi'],c['source'],c['rgb'],c['geometry']['roi'].astype(np.float32),c['g'],c['target'],c['sk'],c['thin']]
                    k=rng.randrange(4);fh=rng.random()<.5;fv=rng.random()<.5;fields=[torch.rot90(torch.from_numpy(a[:,top:top+TILE,left:left+TILE].copy()),k,(-2,-1)) for a in arrays]
                    if fh:fields=[t.flip(-1) for t in fields]
                    if fv:fields=[t.flip(-2) for t in fields]
                    batch.append(fields);pw.writerow([epoch,index,c['name'],top,left,k,fh,fv,array_sha(c['source'])])
                tensors=[torch.stack([b[j] for b in batch]).to(ref.device) for j in range(8)];time=torch.rand(len(batch),device=ref.device,generator=trng)
                opt.zero_grad(set_to_none=True);value,parts=loss(model,tensors,time);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                total+=np.asarray([value.item(),*(parts[k].item() for k in part_names)])*len(batch);seen+=len(batch);updates+=1
            schedule.step();assert seen==4*len(training);tw.writerow([epoch,*(total/seen).tolist(),lr,seen,updates]);tf.flush();pf.flush()
            if epoch%10==0:
                status('proposer_validation',mode=mode,seed=seed,epoch=epoch,held_fold=aux_fold);pred=[]
                for c in validation:
                    delta=predict_delta(model,c['source'],c['rgb'],c['geometry'],ref.device);p=candidate_probability(c['source'],c['geometry']['phi'],delta,c['geometry']['roi'],1.);m=score(c,p);pred.append(m);validation_rows.append(dict(epoch=epoch,image=c['name'],**{k:m[k] for k in KEYS}))
                means={k:float(np.mean([m[k] for m in pred])) for k in KEYS};rank=means['dice']+.1*means['cldice']+.02*means['thin_skeleton_recall']
                if best is None or rank>best['rank']+1e-12:
                    best=dict(epoch=epoch,rank=rank,metrics=means);torch.save(dict(state=model.state_dict(),mode=mode,selection=best),checkpoint)
                print('CHASE',mode.upper(),'AUX'+str(aux_fold) if aux_fold else 'FINAL',seed,epoch,'val',means,'selected_epoch',best['epoch'],flush=True)
    final=out/f'last_proposer_{tag}.pt';torch.save(dict(state=model.state_dict(),mode=mode,epoch=epochs),final);save_csv(out/f'validation_{tag}.csv',validation_rows)
    decision=dict(mode=mode,seed=seed,auxiliary_fold=aux_fold,epochs=epochs,fit_images=[c['name'] for c in training],validation_images=[c['name'] for c in validation],checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),last_checkpoint=str(final),last_sha256=sha256_file(final),initial_state_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),selection=best,locked_at=stamp(),source_grid=512,patches_per_image=4,GT_free_regions=True)
    write_json(out/f'proposer_locked_{tag}.json',decision);del model,opt,schedule;gc.collect();torch.cuda.empty_cache();return decision

def load_proposer(decision,device):
    assert sha256_file(decision['checkpoint'])==decision['checkpoint_sha256'];m=LocalProposer(decision['mode']).to(device);m.load_state_dict(torch.load(decision['checkpoint'],map_location=device,weights_only=True)['state']);return m.eval()

def generate_bank(ref,mode,auxiliary,outer,out,tag,status):
    """The entire producer chain excludes each labeled image's subject fold."""
    xs=[];ys=[];records=[];rng=random.Random(20261117);out.mkdir(parents=True,exist_ok=False);expected=0;held_images=[]
    for dec in auxiliary:
        if dec['mode']!=mode:continue
        k=dec['auxiliary_fold'];model=load_proposer(dec,ref.device)
        held=[c for (key,n),c in outer.items() if key==str(k) and n not in ref.manifest['validation']]
        assert len(held)==4 and not set(dec['fit_images'])&{c['name'] for c in held}
        for c in held:
            expected+=len(c['geometry']['regions'])*len(BETAS)*2;held_images.append(c['name'])
            assert c['name'] not in c['provenance']['protocol']['fit_images'];status('generating_OOF_acceptance_labels',mode=mode,held_fold=k,image=c['name'])
            delta=predict_delta(model,c['source'],c['rgb'],c['geometry'],ref.device)
            path=out/f'{c["name"]}_OOF_proposals_{tag}.npz';np.savez_compressed(path,source=c['source'],delta=delta,roi=c['geometry']['roi'])
            for beta in BETAS:
                candidate=candidate_probability(c['source'],c['geometry']['phi'],delta,c['geometry']['roi'],beta)
                for walk in range(2):
                    current=c['source'].copy();before=score(c,current)
                    regions=c['geometry']['regions'].copy();rng.shuffle(regions)
                    for step,region in enumerate(regions):
                        proposed=region_candidate(current,candidate,region);feature=local_features(current,proposed,c['source'],c['rgb'],c['geometry'],region);after=score(c,proposed)
                        gain=np.asarray([after[k]-before[k] for k in KEYS],dtype=np.float64);xs.append(feature);ys.append(gain)
                        advance=rng.random()<.5
                        records.append(dict(row=len(xs)-1,image=c['name'],held_fold=k,beta=beta,walk=walk,step=step,y=region['y'],x=region['x'],dice_gain=gain[0],cldice_gain=gain[1],thin_gain=gain[2],state_advance_random=advance,GT_used_to_advance=False,producer_checkpoint=dec['checkpoint'],producer_sha256=dec['checkpoint_sha256'],O3_input_source_sha256=array_sha(c['source']),proposal_path=str(path),proposal_sha256=sha256_file(path)))
                        if advance:current=proposed;before=after
        del model;gc.collect();torch.cuda.empty_cache()
    x=np.asarray(xs,np.float32);y=np.asarray(ys,np.float64);assert len(set(held_images))==16 and len(x)==expected and x.shape[1]==30 and y.shape==(len(x),3)
    bank=out/f'OOF_bank_{tag}.npz';np.savez_compressed(bank,x=x,y=y);save_csv(out/f'OOF_bank_labels_{tag}.csv',records)
    meta=dict(mode=mode,path=str(bank),sha256=sha256_file(bank),n_examples=len(x),n_images=16,n_subjects=8,dependent_local_examples=True,shared_proposer_seed=91011,producer_fits=4,OOF_producer_fit_images=12,all_direct_indirect_and_selection_producers_exclude_held_subjects=True,random_state_advancement_GT_free=True)
    write_json(out/f'bank_manifest_{tag}.json',meta);print('OOF ACCEPTANCE BANK',mode,len(x),'local examples from16images/8subjects; examples are dependent.',flush=True);return meta

def train_critic(bank,seed,out,tag,status):
    assert sha256_file(bank['path'])==bank['sha256'];out.mkdir(exist_ok=False)
    with np.load(bank['path'],allow_pickle=False) as z:x=z['x'].copy();y=z['y'].astype(np.float32)
    scalers=dict(xmean=x.mean(0),xstd=x.std(0).clip(1e-6),ymean=y.mean(0),ystd=y.std(0).clip(1e-6))
    xx=torch.from_numpy((x-scalers['xmean'])/scalers['xstd']);yy=torch.from_numpy((y-scalers['ymean'])/scalers['ystd'])
    torch.manual_seed(seed+91231);model=QualityCritic();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=torch.Generator().manual_seed(seed+7891);rows=[]
    for epoch in range(1,101):
        status('acceptor_training',mode=bank['mode'],seed=seed,epoch=epoch,epochs=100);model.train();order=torch.randperm(len(xx),generator=rng);s=0.;updates=0
        for start in range(0,len(order),128):
            indices=order[start:start+128];mu,logstd=model(xx[indices]);value=(.5*((yy[indices]-mu)*torch.exp(-logstd)).square()+logstd).mean()
            assert torch.isfinite(value);opt.zero_grad(set_to_none=True);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();s+=value.item()*len(indices);updates+=1
        rows.append(dict(epoch=epoch,loss=s/len(xx),samples=len(xx),optimizer_updates=updates))
        if epoch%25==0:print('ACCEPTOR',bank['mode'],seed,epoch,'training_NLL',s/len(xx),flush=True)
    cp=out/f'acceptor_{tag}.pt';torch.save(dict(state=model.state_dict(),scalers={k:torch.from_numpy(v.copy()) for k,v in scalers.items()},bank=bank,seed=seed,epochs=100),cp);save_csv(out/f'acceptor_training_{tag}.csv',rows)
    decision=dict(mode=bank['mode'],seed=seed,epochs=100,checkpoint=str(cp),checkpoint_sha256=sha256_file(cp),parameter_count=sum(p.numel() for p in model.parameters()),training_bank=bank,locked_at=stamp(),checkpoint_selection='Preset100epochs; system calibration uses validation only')
    write_json(out/f'acceptor_locked_{tag}.json',decision);return model.eval(),scalers,decision

def load_critic(decision):
    assert sha256_file(decision['checkpoint'])==decision['checkpoint_sha256'];state=torch.load(decision['checkpoint'],map_location='cpu',weights_only=True);m=QualityCritic();m.load_state_dict(state['state']);return m.eval(),{k:v.numpy().copy() for k,v in state['scalers'].items()}

def calibrate(ref,proposer,critic,scalers,cases,out,tag,status,critic_record):
    model=load_proposer(proposer,ref.device);deltas=[predict_delta(model,c['source'],c['rgb'],c['geometry'],ref.device) for c in cases];base=mean_scores(cases,[c['source'] for c in cases]);rows=[]
    best=dict(beta_max=0.,kappa=0.,metrics=base,structural_gain=0.,identity=True)
    for b in BETAS:
        for kappa in KAPPAS:
            status('system_validation_calibration',mode=proposer['mode'],seed=proposer['seed'],beta_max=b,kappa=kappa);outputs=[]
            for c,d in zip(cases,deltas):
                p,accepted=accept_corrections(c['source'],d,c['rgb'],c['geometry'],critic,scalers,b,kappa);m=score(c,p);outputs.append(p);rows.append(dict(image=c['name'],beta_max=b,kappa=kappa,accepted_regions=len(accepted),**{k:m[k] for k in KEYS}))
            mm=mean_scores(cases,outputs);gain=.5*(mm['cldice']-base['cldice']+mm['thin_skeleton_recall']-base['thin_skeleton_recall']);feasible=all(mm[k]>=base[k]-1e-12 for k in KEYS)
            if feasible and gain>best['structural_gain']+1e-12:best=dict(beta_max=b,kappa=kappa,metrics=mm,structural_gain=gain,identity=False)
    save_csv(out/f'system_validation_{tag}.csv',rows);decision=dict(mode=proposer['mode'],seed=proposer['seed'],proposer=proposer,critic=critic_record,critic_checkpoint=str(out/f'acceptor_{tag}.pt'),critic_sha256=sha256_file(out/f'acceptor_{tag}.pt'),O3_validation=base,selection=best,locked_at=stamp(),selection_uses_test=False)
    write_json(out/f'system_locked_{tag}.json',decision);print('VALIDATION SYSTEM CHOICE',proposer['mode'],proposer['seed'],best,flush=True);del model;gc.collect();torch.cuda.empty_cache();return decision

def run_study(ref,sources,out,tag,status):
    assert ref.dataset=='CHASE' and sha256_file(sources['global_lock'])==sources['global_lock_sha256']
    a=LocalProposer('flow');b=LocalProposer('direct');field_count=sum(p.numel() for p in a.parameters());assert field_count==sum(p.numel() for p in b.parameters());critic_count=sum(p.numel() for p in QualityCritic().parameters())
    write_json(out/f'method_protocol_{tag}.json',dict(proposal_grid=512,patch_size=128,stride=64,SDF_radius=4.,SDF_temperature=.25,delta_cap=2.,local_region_size=32,max_regions=24,GT_free_geometric_ROIs=True,RGB='audited retained512RGB; not a new native-resolution claim',unchanged_probabilities_preserved_except_class_flips=True,
        flow_parameters=field_count,direct_parameters=field_count,acceptor_parameters=critic_count,Flow_total_trainable=field_count+critic_count,Direct_total_trainable=field_count+critic_count,Flow_velocity_calls_per_tile=2,Direct_calls_per_tile=1,
        shared_auxiliary_seed=91011,auxiliary_epochs=80,final_epochs=120,acceptor_epochs=100,pilot_hyperparameter_training=False,fixed_field_training='AdamW3e-4,wd1e-4,cosine,4patches/image/epoch,batch16; coefficients same except Flow FM0.5; same initialization and crop/augmentation streams per matched seed',
        proposer_checkpoint='Max validation Dice +0.1clDice +0.02thin recall; trained epochs10..end only. Final system retains exact O3 identity as fallback.',acceptance_training='Gaussian heteroscedastic regression of three global metric changes for local offers; all fitting targets from whole-pipeline held-subject OOF predictions;2GT-free random state walks per beta.',
        calibration='beta maximum0.25/0.5/1 and conservatism kappa0/0.5/1 on validation only; all three mean validation metrics >=O3 required, then max mean structural gain; identity candidate always eligible. Local predicted lower gains must all >=0.',
        limitations='Shared one-seed auxiliary OOF bank, four reused validation images, previously examined test cohort and training-size shift. Parameter-matched; inference calls/compute differ. No test guarantee.'))
    del a,b;targets={};paircases,outer=make_source_cases(ref,sources,targets);folds=sources['folds'];auxiliary=[]
    for k in range(4):
        held=set(folds[k]);train=[]
        for j in range(4):
            if j==k:continue
            key='_'.join(str(v+1) for v in sorted((k,j)))
            for n in folds[j]:
                c=paircases[key,n];assert not held&set(c['provenance']['protocol']['fit_images']);assert n not in c['provenance']['protocol']['fit_images'];train.append(c)
        validation=[outer[str(k+1),n] for n in ref.manifest['validation']]
        assert len(train)==12 and all(not held&set(c['provenance']['protocol']['fit_images']) for c in validation)
        for mode in MODES:
            d=train_proposer(ref,mode,91011,train,validation,out/'auxiliary_proposers'/f'fold_{k+1}'/mode,tag,status,80,k+1);auxiliary.append(d);status('auxiliary_proposer_complete',completed_auxiliary_proposers=len(auxiliary))
        x,y=auxiliary[-2:];assert x['initial_state_sha256']==y['initial_state_sha256']
        px=Path(x['checkpoint']).parent/f'patch_draws_{tag}.csv';py=Path(y['checkpoint']).parent/f'patch_draws_{tag}.csv';assert sha256_file(px)==sha256_file(py),'Control crops differ'
    auxlock=out/f'all8_auxiliary_choices_locked_before_bank_{tag}.json';write_json(auxlock,dict(locked_at=stamp(),decisions=auxiliary));auxhash=sha256_file(auxlock)
    banks={mode:generate_bank(ref,mode,auxiliary,outer,out/'acceptance_banks'/mode,tag,status) for mode in MODES};assert sha256_file(auxlock)==auxhash
    decisions=[];critic_decisions=[];proposers=[]
    for seed in SEEDS:
        train,validation=full_source_cases(ref,seed,out,tag,targets);seed_proposers=[]
        for mode in MODES:
            dest=out/'final'/mode/f'seed_{seed}';d=train_proposer(ref,mode,seed,train,validation,dest/'proposer',tag,status,120);seed_proposers.append(d);proposers.append(d);status('final_proposer_complete',completed_final_proposers=len(proposers))
            critic,scalers,cd=train_critic(banks[mode],seed,dest/'acceptor',tag,status);critic_decisions.append(cd);status('acceptor_complete',completed_acceptors=len(critic_decisions))
            decision=calibrate(ref,d,critic,scalers,validation,dest/'acceptor',tag,status,cd);decision['critic']=cd;decisions.append(decision);status('final_system_locked',completed_final_systems=len(decisions));del critic;gc.collect()
        x,y=seed_proposers;assert x['parameter_count']==y['parameter_count']==field_count and x['initial_state_sha256']==y['initial_state_sha256']
        assert sha256_file(Path(x['checkpoint']).parent/f'patch_draws_{tag}.csv')==sha256_file(Path(y['checkpoint']).parent/f'patch_draws_{tag}.csv')
        del train,validation;gc.collect();torch.cuda.empty_cache()
    assert len(decisions)==len(proposers)==len(critic_decisions)==10
    lock=out/f'all10_system_choices_locked_before_test_{tag}.json';write_json(lock,dict(locked_at=stamp(),decisions=decisions,auxiliary_lock_sha256=auxhash,method_protocol_sha256=sha256_file(out/f'method_protocol_{tag}.json')));lh=sha256_file(lock);records=[]
    from flow_matching_refiner.run_native_patch_flow_v2 import O3,AT
    for decision in decisions:
        mode=decision['mode'];seed=decision['seed'];model=load_proposer(decision['proposer'],ref.device);critic,scalers=load_critic(decision['critic']);selection=decision['selection'];dest=out/'final'/mode/f'seed_{seed}'/'test';dest.mkdir(exist_ok=False)
        for n in ref.manifest['test']:
            status('locked_test_inference',mode=mode,seed=seed,image=n);d=ref.data[n];oldpath=O3/f'CHASE_{AT}/native_patch_direct/seed_{seed}/{n}_predictions_{AT}.npz'
            with np.load(oldpath,allow_pickle=False) as z:
                assert all(np.array_equal(z[k],d[v].numpy()) for k,v in (('baseline','coarse'),('ground_truth','g'),('fov','fov')));source=z['raw'].copy()
            rgb=d['rgb'].numpy();geometry=image_geometry(source,rgb);delta=predict_delta(model,source,rgb,geometry,ref.device);raw=candidate_probability(source,geometry['phi'],delta,geometry['roi'],1.)
            accepted,regions=accept_corrections(source,delta,rgb,geometry,critic,scalers,selection['beta_max'],selection['kappa']);outside=~geometry['roi'].astype(bool);assert np.array_equal(raw[outside],source[outside]) and np.array_equal(accepted[outside],source[outside])
            if selection['identity']:assert np.array_equal(accepted,source)
            path=dest/f'{n}_{mode}_predictions_{tag}.npz';np.savez_compressed(path,baseline=d['coarse'].numpy(),O3=source,raw=raw,accepted=accepted,ground_truth=d['g'].numpy(),fov=d['fov'].numpy(),roi=geometry['roi'],phi=geometry['phi'],delta=delta)
            detail=dest/f'{n}_{mode}_accepted_regions_{tag}.json';write_json(detail,dict(regions=regions,proposed_regions=geometry['regions'],GT_free_inference=True))
            records.append(dict(mode=mode,seed=seed,image=n,path=str(path),sha256=sha256_file(path),region_path=str(detail),region_sha256=sha256_file(detail),accepted_regions=len(regions),selection=selection,old_O3_path=str(oldpath),old_O3_sha256=sha256_file(oldpath),proposer_checkpoint=decision['proposer']['checkpoint'],proposer_sha256=decision['proposer']['checkpoint_sha256'],acceptor_checkpoint=decision['critic']['checkpoint'],acceptor_sha256=decision['critic']['checkpoint_sha256']))
        del model,critic;gc.collect();torch.cuda.empty_cache()
    assert len(records)==80 and sha256_file(lock)==lh
    write_json(out/f'prediction_manifest_{tag}.json',dict(records=records,global_lock_path=str(lock),global_lock_sha256=lh))
    return dict(new_triple_baselines=4,new_pair_O3_fits=6,auxiliary_proposer_fits=8,final_proposer_fits=10,final_acceptor_fits=10,prediction_files=80,source_training_seed=42,auxiliary_training_seed=91011,final_seeds=list(SEEDS),shared_conditional_OOF_bank=True,identity_selections={mode:sum(d['mode']==mode and d['selection']['identity'] for d in decisions) for mode in MODES},flow_parameters=field_count+critic_count,direct_parameters=field_count+critic_count,global_lock_sha256=lh,all_test_inference_GT_free=True)
