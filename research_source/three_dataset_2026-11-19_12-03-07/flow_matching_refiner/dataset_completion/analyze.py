"""Independent replay and new, explicitly exploratory combined comparison tables."""
from __future__ import annotations
import argparse,csv,json,re,math,itertools,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats
from .common import ROOT,SEEDS,load,write_json,stamp,sha256_file,savecsv
from flow_matching_refiner.structure_metrics_all import metrics
KEYS=('dice','precision','cldice','thin_skeleton_recall')
SECONDARY=('iou','recall','specificity','accuracy','topology_precision','topology_sensitivity','new_error_rate','fixed_error_rate','betti0_absolute_error')
PHASE_NAMES={'native_flow':'F1 native FM','anchored':'F2 anchored FM','fully_oof':'F3 strict OOF','topology':'F4 topology','retained':'F5 retained','local_acceptor':'F6 local acceptor','rollout':'F7 native rollout','context':'F8 global context','fresh_cv':'F9 fresh CV','pixel':'AE-free mask flow','encoded':'Encoded preliminary flow','segflow':'SegFlow'}

def seed_from(path):
    for p in reversed(path.parts):
        m=re.fullmatch(r'(?:refiner_|training_)?seed_(\d+)',p)
        if m:return int(m[1])
    raise ValueError('No training-seed binding: '+str(path))

def cohort_from(path,phase):
    if phase=='calibration':return 'fresh_nested_CV_calibration'
    if phase!='fresh_cv':return 'retained_fixed_split'
    return 'fresh_nested_CV'

def candidates(folder,tag):
    for p in sorted(folder.rglob(f'*_{tag}.npz')):
        if not ('prediction' in p.name or 'probability' in p.name):continue
        if any(w in p.name for w in ('native_predictions','native_residual','_native_')):continue
        yield p

def image_from(path,phase):
    name=path.name
    name=re.sub(r'_(?:flow|direct)_predictions_.*$','',name)
    name=re.sub(r'_(?:predictions?|probability)_.*$','',name)
    return name

def read_folder(folder,phase,dataset,tag,bindings):
    rows=[];seen=set()
    for path in candidates(folder,tag):
        if phase=='encoded' and 'test' not in path.parts:continue
        with np.load(path,allow_pickle=False) as z:
            if not {'ground_truth','fov'}<=set(z.files):continue
            g=z['ground_truth'];f=z['fov'];base=z['baseline'] if 'baseline' in z else z['coarse']
            assert g.shape==f.shape==base.shape==(1,512,512)
            assert np.isfinite(base).all() and np.isfinite(g).all()
            seed=seed_from(path);name=image_from(path,phase);scope=cohort_from(path,phase)
            mode=next((v for v in reversed(path.parts) if v in ('flow','direct')),'') if phase in ('local_acceptor','rollout','context','fresh_cv','calibration') else ''
            arm=next((path.parts[i-1] for i,v in enumerate(path.parts) if re.fullmatch(r'seed_\d+',v)),phase) if phase in ('controls','initial_controls','initial_objectives','O3') else PHASE_NAMES.get(phase,phase)
            field_candidates=[k for k in ('raw','selected','blend','refined','accepted') if k in z and z[k].shape==(1,512,512)]
            if 'probabilities' in z:
                assert phase=='encoded';p=z['probabilities'];assert p.shape==(1,5,1,512,512)
                field_candidates=['noise_mean_metric']
            anchor=z['O3'] if 'O3' in z else base
            am=metrics(anchor[0],g[0],base[0],1.,f[0]);bm=metrics(base[0],g[0],base[0],1.,f[0])
            fold=next((int(m.group(1)) for v in path.parts if (m:=re.fullmatch(r'outer_fold_(\d+)',v))),0)
            for kind in field_candidates:
                reported_kind='selected' if kind=='refined' else kind
                if kind=='noise_mean_metric':
                    mm=[metrics(p[0,j,0],g[0],base[0],1.,f[0]) for j in range(5)];result={k:float(np.mean([r[k] for r in mm])) for k in mm[0]};assert len(set(z['sampling_seeds'].tolist()))==5
                else:
                    a=z[kind];assert a.dtype==np.float32 and np.isfinite(a).all() and a.min()>=0 and a.max()<=1;result=metrics(a[0],g[0],base[0],1.,f[0])
                key=(scope,dataset,phase,arm,mode,reported_kind,seed,name,fold);assert key not in seen,key;seen.add(key)
                signature=hashlib.sha256(base.astype(np.float32).tobytes()+(g>=.5).tobytes()+(f>=.5).tobytes()).hexdigest()
                rows.append(dict(cohort=scope,dataset=dataset,phase=phase,method=arm,mode=mode,output=reported_kind,seed=seed,image=name,outer_fold=fold,path=str(path),prediction_sha256=sha256_file(path),input_signature=signature,anchor=('O3_calibrated' if phase=='calibration' else 'O3') if 'O3' in z else 'baseline',**result,**{'anchor_'+k:am[k] for k in KEYS},**{'baseline_'+k:bm[k] for k in KEYS}))
        bindings[str(path)]=sha256_file(path)
    return rows

def old_folders():
    out=[];mapping={'native_patch_flow':'native_flow','native_patch_flow_v2':'anchored','fully_oof_flow':'fully_oof','topology_oof_flow':'topology','retained_oof_flow':'retained','local_flow_acceptor':'local_acceptor','native_rollout':'rollout','global_context':'context','precision_subject_cv':'fresh_cv'}
    for name,phase in mapping.items():
        state=ROOT/'tmp'/f'current_{name}_delivery.json';d=load(state);folder=Path(d.get('output',d.get('experiment','')));tag=folder.name[-19:]
        if phase in ('native_flow','anchored','fully_oof','topology','retained'):
            for ds in ('CHASE','DRIVE'):out.append((folder/f'{ds}_{tag}',phase,ds,tag))
        else:out.append((folder,phase,'CHASE',tag))
    for ds in ('CHASE','DRIVE','HRF'):
        t='2026-11-16_16-53-08';out.append((ROOT/f'outputs/native_patch_study_{t}/{ds}_{t}','O3',ds,t))
        t='2026-11-15_23-28-26';out.append((ROOT/f'outputs/refinement_v3_{t}/{ds}_{t}','initial_objectives',ds,t))
        t='2026-11-15_01-59-17' if ds=='CHASE' else '2026-11-15_02-35-09'
        folder=ROOT/f'outputs/esfm_oof_chase_{t}' if ds=='CHASE' else ROOT/f'outputs/esfm_all_datasets_{t}/{ds}_{t}'
        out.append((folder,'initial_controls',ds,t))
    t='2026-11-14_23-46-03';out.append((ROOT/f'outputs/flow_matching_{t}/CHASE','encoded','CHASE',t))
    t='2026-11-15_01-12-53';out.append((ROOT/f'outputs/flow_matching_v2_{t}','pixel','CHASE',t))
    t='2026-11-17_00-29-51'
    for ds in ('CHASE','DRIVE'):out.append((ROOT/f'outputs/segflow_inference_CHASE_DRIVE_{t}/{ds}','segflow',ds,t))
    return out

def saved_controls(bindings):
    manifest=ROOT/'outputs/private_test_comparisons_2026-11-17_00-38-33/normalized_saved_comparison_manifest_2026-11-17_00-39-15.json';m=load(manifest);bindings[str(manifest)]=sha256_file(manifest);rows=[]
    for r in m['controls']:
        assert sha256_file(r['path'])==r['sha256'] and sha256_file(r['input_path'])==r['input_sha256']
        with np.load(r['path'],allow_pickle=False) as z:p=z['raw'].copy()
        with np.load(r['input_path'],allow_pickle=False) as z:base=z['baseline'].copy();g=z['ground_truth'].copy();f=z['fov'].copy()
        mm=metrics(p[0],g[0],base[0],1.,f[0]);bm=metrics(base[0],g[0],base[0],1.,f[0]);signature=hashlib.sha256(base.astype(np.float32).tobytes()+(g>=.5).tobytes()+(f>=.5).tobytes()).hexdigest()
        rows.append(dict(cohort='retained_fixed_split',dataset=r['dataset'],phase='controls',method=r['arm'],mode='',output='raw',seed=r['seed'],image=r['image'],outer_fold=0,path=r['path'],prediction_sha256=r['sha256'],input_signature=signature,anchor='baseline',**mm,**{'anchor_'+k:bm[k] for k in KEYS},**{'baseline_'+k:bm[k] for k in KEYS}))
        bindings[r['path']]=r['sha256'];bindings[r['input_path']]=r['input_sha256']
    return rows

def holm(values):
    n=len(values);order=np.argsort(values);out=np.empty(n);prev=0.
    for rank,i in enumerate(order):prev=max(prev,(n-rank)*values[i]);out[i]=min(1.,prev)
    return out.tolist()

def unit_name(ds,image):
    if ds=='CHASE':
        m=re.search(r'Image_(\d+)',image);assert m,image;return m.group(1)
    return image

def exact_signed_rank(v):
    q=np.asarray(v,dtype=float);q=q[q!=0]
    if not len(q):return 1.
    ranks=stats.rankdata(np.abs(q),method='average');observed=abs(float(np.sign(q)@ranks));values=[abs(float(np.array(signs)@ranks)) for signs in itertools.product((-1.,1.),repeat=len(q))]
    return sum(x>=observed-1e-12 for x in values)/len(values)

def sample_sd(values):
    q=np.asarray(values,dtype=float)
    if len(q)<2:return None
    # np.std can return a rounding residual for an exactly constant vector.
    if np.all(q==q[0]):return 0.
    return float(np.std(q,ddof=1))

def paired_result(rr,k,reference):
    seeds=sorted({r['seed'] for r in rr});images=sorted({r['image'] for r in rr});n=len(seeds)
    assert n in (1,5) and len(rr)==len(images)*n
    a=np.array([[next(r[k]-r[reference+k] for r in rr if r['seed']==s and r['image']==im) for im in images] for s in seeds]);v=a.mean(1);mean=float(v.mean());sd=sample_sd(v)
    if n==1:return dict(mean_difference=mean,SD_difference=None,CI_low=None,CI_high=None,t=None,p=None,Wilcoxon_p=None,seed_count=1)
    if np.all(v==0):t=0.;p=1.;wp=1.;ci=(0.,0.)
    elif sd==0:t=None;p=None;wp=exact_signed_rank(v);ci=(None,None)
    else:
        test=stats.ttest_1samp(v,0);t=float(test.statistic) if np.isfinite(test.statistic) else None;p=float(test.pvalue);se=sd/math.sqrt(n);h=stats.t.ppf(.975,n-1)*se;ci=(mean-h,mean+h)
        wp=exact_signed_rank(v)
    # Cross seed/cohort resampling, stratified by outer fold when present.
    units=sorted({unit_name(rr[0]['dataset'],im) for im in images});byunit={u:[i for i,im in enumerate(images) if unit_name(rr[0]['dataset'],im)==u] for u in units}
    strata=defaultdict(list)
    for u in units:
        f={int(r['outer_fold']) for r in rr if unit_name(r['dataset'],r['image'])==u};assert len(f)==1
        pathology=u.split('_')[-1] if rr[0]['dataset']=='HRF' else ''
        strata[next(iter(f)),pathology].append(u)
    rng=np.random.default_rng(20261118);boot=[]
    assert len({len(v) for v in byunit.values()})==1
    uu=np.stack([a[:,byunit[u]].mean(1) for u in units],axis=1);position={u:i for i,u in enumerate(units)}
    # Bounded chunks retain equal weighting of resampled cohort images.
    for start in range(0,50000,500):
        count=min(500,50000-start);si=rng.integers(0,n,(count,n));selected=[]
        for ss in strata.values():
            picks=rng.integers(0,len(ss),(count,len(ss)));selected.append(np.array([position[u] for u in ss])[picks])
        ui=np.concatenate(selected,axis=1);boot.extend(uu[si[:,:,None],ui[:,None,:]].mean((1,2)).tolist())
    b=np.quantile(boot,[.025,.975])
    return dict(mean_difference=mean,SD_difference=sd,CI_low=float(ci[0]) if ci[0] is not None else None,CI_high=float(ci[1]) if ci[1] is not None else None,t=t,p=p,Wilcoxon_p=wp,seed_count=n,crossed_bootstrap_low=float(b[0]),crossed_bootstrap_high=float(b[1]),cohort_units=len(units),cohort_unit='subject' if rr[0]['dataset']=='CHASE' else 'image')

def summarize(rows,out,tag):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in ('cohort','dataset','phase','method','mode','output'))].append(r)
    means=[];tests=[]
    for key,rr in groups.items():
        meta=dict(zip(('cohort','dataset','phase','method','mode','output'),key));seeds=sorted({r['seed'] for r in rr})
        for k in KEYS+SECONDARY:
            vals=[float(np.mean([r[k] for r in rr if r['seed']==s])) for s in seeds]
            means.append(dict(**meta,metric=k,mean=float(np.mean(vals)),SD=sample_sd(vals),training_seed_count=len(seeds),images=len({r['image'] for r in rr}),seed_means=json.dumps(dict(zip(seeds,vals)))))
            if k not in KEYS:continue
            for reference in (('anchor_','baseline_') if rr[0]['anchor'].startswith('O3') else ('baseline_',)):
                tests.append(dict(**meta,metric=k,reference='matched_'+rr[0]['anchor'] if reference=='anchor_' else 'matched_baseline',**paired_result(rr,k,reference)))
        if meta['mode']=='flow':
            direct_key=(*key[:4],'direct',key[5]);direct=groups.get(direct_key)
            if direct:
                matched=[]
                for r in rr:
                    dd=[d for d in direct if (d['seed'],d['image'],d['outer_fold'])==(r['seed'],r['image'],r['outer_fold'])];assert len(dd)==1
                    assert r['input_signature']==dd[0]['input_signature']
                    matched.append(dict(r,**{'direct_'+k:dd[0][k] for k in KEYS}))
                for k in KEYS:tests.append(dict(**meta,metric=k,reference='matched_direct',**paired_result(matched,k,'direct_')))
        if meta['phase']=='controls':
            cp='native_patch_direct' if meta['method'].startswith(('patch_','native_')) else 'esfm_endpoint_anchor' if meta['method']=='o2_no_rgb' else 'esfm' if meta['method'].startswith('o2_') else 'direct'
            cp_phase='O3' if cp=='native_patch_direct' else 'initial_objectives' if cp=='esfm_endpoint_anchor' else 'initial_controls'
            ck=(key[0],key[1],cp_phase,cp,'',key[5]);control=groups.get(ck);assert control,ck;matched=[]
            for r in rr:
                dd=[d for d in control if (d['seed'],d['image'])==(r['seed'],r['image'])];assert len(dd)==1
                assert r['input_signature']==dd[0]['input_signature'];matched.append(dict(r,**{'control_'+k:dd[0][k] for k in KEYS}))
            for k in KEYS:tests.append(dict(**meta,metric=k,reference='matched_'+cp,**paired_result(matched,k,'control_')))
        if meta['phase']=='native_flow':
            control=groups[key[0],key[1],'O3','native_patch_direct','','raw'];matched=[]
            for r in rr:
                dd=next(d for d in control if (d['seed'],d['image'])==(r['seed'],r['image']));assert r['input_signature']==dd['input_signature'];matched.append(dict(r,**{'control_'+k:dd[k] for k in KEYS}))
            for k in KEYS:tests.append(dict(**meta,metric=k,reference='matched_O3',**paired_result(matched,k,'control_')))
        previous={'anchored':('native_flow','raw'),'fully_oof':('anchored','selected'),'topology':('fully_oof','selected'),'retained':('topology','selected'),'context':('rollout','selected')}
        if meta['phase'] in previous and meta['output']=='selected' and meta['cohort']=='retained_fixed_split':
            phase,output=previous[meta['phase']];previous_mode=meta['mode'] if meta['phase']=='context' else ''
            control=groups[key[0],key[1],phase,PHASE_NAMES[phase],previous_mode,output];matched=[]
            for r in rr:
                dd=[d for d in control if (d['seed'],d['image'],d['outer_fold'])==(r['seed'],r['image'],r['outer_fold'])];assert len(dd)==1
                assert r['input_signature']==dd[0]['input_signature'];matched.append(dict(r,**{'predecessor_'+k:dd[0][k] for k in KEYS}))
            for k in KEYS:tests.append(dict(**meta,metric=k,reference='matched_predecessor_'+phase+'_'+output,contrast_scope='Exploratory paired recipe comparison; multiple recipe factors can differ.',**paired_result(matched,k,'predecessor_')))
    # Shared upstream estimators are described once, without inventing five
    # independently trained baselines/O3 models in the fresh outer CV.
    for ds in ('CHASE','DRIVE','HRF'):
        for cohort in ('retained_fixed_split','fresh_nested_CV','fresh_nested_CV_calibration'):
            choices=[r for r in rows if r['dataset']==ds and r['cohort']==cohort]
            if not choices:continue
            ss=sorted({r['seed'] for r in choices});chosen=[r for r in choices if r['seed']==ss[0]]
            unique={}
            for r in chosen:
                im=r['image']
                if im in unique:assert unique[im]['input_signature']==r['input_signature']
                else:unique[im]=r
            for prefix,method,mode in [('baseline_','Frozen baseline','baseline')]+([('anchor_','O3 shared upstream','O3')] if cohort.startswith('fresh_nested_CV') else []):
                # For fresh CV use an O3-bearing context row, not an arbitrary
                # historical fixed-stage array lacking an O3 key.
                uu=list(unique.values())
                for k in KEYS:means.append(dict(cohort=cohort,dataset=ds,phase=('calibration' if cohort.endswith('_calibration') else 'fresh_cv') if cohort.startswith('fresh_nested_CV') else 'baseline',method=method,mode=mode,output='anchor',metric=k,mean=float(np.mean([r[prefix+k] for r in uu])),SD=None,training_seed_count=1,images=len(uu),seed_means=json.dumps({'42':float(np.mean([r[prefix+k] for r in uu]))})))
    # Same five-seed factorial contrasts as the original CHASE controls.
    for ds in ('CHASE','DRIVE','HRF'):
        for output in ('raw','blend'):
            for label,terms in {
                'O1_interaction':[(1,'initial_objectives','direct_thin_anchor'),(-1,'controls','o1_thin_only'),(-1,'controls','o1_anchor_only'),(1,'initial_controls','direct')],
                'O2_interaction':[(1,'initial_objectives','esfm_endpoint_anchor'),(-1,'controls','o2_endpoint_only'),(-1,'controls','o2_anchor_only'),(1,'initial_controls','esfm')],
                'native_information_interaction':[(1,'controls','native_degraded_both'),(-1,'controls','native_degraded_rgb'),(-1,'controls','native_degraded_gt'),(1,'O3','native_patch_direct')]
            }.items():
                gg=[groups.get(('retained_fixed_split',ds,phase,method,'',output)) for sign,phase,method in terms]
                if not all(gg):
                    assert output=='blend' and ds in ('CHASE','DRIVE'),(ds,output,label)
                    continue # Earlier CHASE/DRIVE scope exported raw outputs only.
                rr=[]
                for first in gg[0]:
                    match=[next(r for r in a if (r['image'],r['seed'])==(first['image'],first['seed'])) for a in gg]
                    assert len({r['input_signature'] for r in match})==1
                    rr.append(dict(first,**{k:sum(term[0]*r[k] for term,r in zip(terms,match)) for k in KEYS},**{'zero_'+k:0. for k in KEYS}))
                meta=dict(cohort='retained_fixed_split',dataset=ds,phase='factorial_controls',method=label,mode='',output=output)
                for k in KEYS:tests.append(dict(**meta,metric=k,reference='zero_interaction',**paired_result(rr,k,'zero_')))
    families=defaultdict(list)
    for i,t in enumerate(tests):
        if t['seed_count']==5:families[(t['cohort'],t['phase'],t['method'],t['mode'],t['output'],t['reference'])].append(i)
    for family,indices in families.items():
        for field,dest in [('p','Holm_p'),('Wilcoxon_p','Holm_Wilcoxon_p')]:
            valid=[i for i in indices if tests[i][field] is not None];adjusted=holm([tests[i][field] for i in valid])
            for i,v in zip(valid,adjusted):tests[i][dest]=v;tests[i][dest+'_family_size']=len(valid);tests[i]['Holm_family']=json.dumps(family)
    # Rectangular schema even for the one-training-run preliminary experiment.
    columns=list(dict.fromkeys(k for t in tests for k in t))
    savecsv(out/f'mean_SD_{tag}.csv',means);savecsv(out/f'paired_tests_{tag}.csv',[{k:t.get(k) for k in columns} for t in tests])
    return means,tests

def run(root):
    root=Path(root).resolve();tag=root.name[-19:];plan=load(root/f'queue_{tag}.json');assert plan['sealed']
    for j in plan['jobs']:
        if j['phase']=='analysis':continue
        assert (root/j['phase']/j['dataset']/f'completed_{tag}.json').is_file(),j
    out=root/'analysis';out.mkdir(exist_ok=False);rows=[];bindings={}
    for j in plan['jobs']:
        if j['phase'] in ('strict_sources','fresh_sources','geometry','frunet','gap','analysis','encoded_diagnosis','bias_probe','precision_reselection'):continue
        rows.extend(read_folder(root/j['phase']/j['dataset'],j['phase'],j['dataset'],tag,bindings))
    cache=root/'preverified_legacy';marker=cache/f'completed_{tag}.json'
    if marker.exists():
        m=load(marker);p=cache/f'verified_existing_rows_{tag}.json';b=cache/f'prediction_bindings_{tag}.json';assert sha256_file(p)==m['rows_sha256'] and sha256_file(b)==m['bindings_sha256'];old_bindings=load(b);assert all(sha256_file(p)==h for p,h in old_bindings.items());bindings.update(old_bindings);rows.extend(load(p))
    else:
        for folder,phase,ds,t in old_folders():
            assert folder.is_dir(),folder;rows.extend(read_folder(folder,phase,ds,t,bindings))
        rows.extend(saved_controls(bindings))
    # The initial control named "direct" is a method, not the paired-mode
    # directory used by later Flow/direct studies. Normalize cached metadata
    # only; no cached file or numerical metric is changed.
    for r in rows:
        if r['phase'] not in ('local_acceptor','rollout','context','fresh_cv','calibration'):r['mode']=''
    assert rows
    # Saved FP32 probabilities, not reported rounded tables, are the only inputs.
    keys=list(dict.fromkeys(k for r in rows for k in r));savecsv(out/f'all_verified_per_image_{tag}.csv',[{k:r.get(k) for k in keys} for r in rows]);means,tests=summarize(rows,out,tag)
    assert all(sha256_file(p)==h for p,h in bindings.items());write_json(out/f'prediction_bindings_{tag}.json',bindings)
    write_json(out/f'completed_analysis_{tag}.json',dict(completed_at=stamp(),prediction_files=len(bindings),per_image_rows=len(rows),summary_rows=len(means),paired_rows=len(tests),all_input_bytes_unchanged=True,seed_CIs='t df4 conditional correction-stage uncertainty; not independent whole-pipeline or patient generalization CIs',bootstrap='50000 crossed seed and subject/image draws; stratified by outerfold; exploratory',historical_table_numbers_not_used=True))
    return out

def collect_existing(root):
    root=Path(root).resolve();tag=root.name[-19:];out=root/'preverified_legacy';out.mkdir(exist_ok=False);rows=[];bindings={}
    for folder,phase,ds,t in old_folders():
        assert folder.is_dir(),folder;new=read_folder(folder,phase,ds,t,bindings);assert new,(phase,ds,folder);rows.extend(new);print('EXISTING FP32 REPLAY',phase,ds,len(new),flush=True)
        write_json(out/f'status_{tag}.json',dict(stage='existing_prediction_replay',phase=phase,dataset=ds,rows=len(rows),updated_at=stamp()))
    rows.extend(saved_controls(bindings));p=out/f'verified_existing_rows_{tag}.json';b=out/f'prediction_bindings_{tag}.json';write_json(p,rows);write_json(b,bindings)
    assert all(sha256_file(p)==h for p,h in bindings.items());write_json(out/f'completed_{tag}.json',dict(completed_at=stamp(),rows=len(rows),prediction_and_input_files=len(bindings),rows_sha256=sha256_file(p),bindings_sha256=sha256_file(b),old_bytes_unchanged=True,no_training=True,new_predictions_computed=False))
    print('EXISTING REPLAY COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--collect-existing',action='store_true');p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.execute:print(collect_existing(a.root) if a.collect_existing else run(a.root),flush=True)
