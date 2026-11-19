"""Separate hard-metric implementation and checkpoint/log inventory validation."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from .common import ROOT,SEEDS,load,write_json,stamp,sha256_file,csvrows,savecsv
KEYS=('dice','precision','cldice','thin_skeleton_recall')

def bound_paths(value,parent,verified):
    if isinstance(value,list):
        for x in value:bound_paths(x,parent,verified)
    elif isinstance(value,dict):
        for key,path in value.items():
            if isinstance(path,(dict,list)):bound_paths(path,parent,verified);continue
            if not isinstance(path,str):continue
            digest=None
            if key=='path':digest=value.get('sha256')
            elif key.endswith('_path'):digest=value.get(key[:-5]+'_sha256')
            elif key=='checkpoint':digest=value.get('checkpoint_sha256')
            elif key=='last_checkpoint':digest=value.get('last_sha256')
            if digest:
                p=Path(path);p=p if p.is_absolute() else parent/p
                assert p.is_file() and sha256_file(p)==digest,(key,str(p));verified.add(str(p.resolve()))

def truth(g,f):
    valid=np.asarray(f)>=.5;gt=(np.asarray(g)>=.5)&valid;sk=skeletonize(gt);radius=distance_transform_edt(np.pad(gt,1))[1:-1,1:-1]
    return dict(valid=valid,gt=gt,skeleton=sk,thin=sk&(radius<=1.))

def hard(p,t):
    pred=(np.asarray(p)>=np.float32(.5))&t['valid'];g=t['gt'];tp=int((pred&g).sum());fp=int((pred&~g).sum());fn=int((~pred&g).sum());empty=not pred.any() and not g.any();sp=skeletonize(pred);pn=int(sp.sum());gn=int(t['skeleton'].sum());a=float((sp&g).sum()/pn) if pn else float(empty);b=float((t['skeleton']&pred).sum()/gn) if gn else float(empty);cl=2*a*b/(a+b) if a+b else float(empty);thin=int(t['thin'].sum())
    return dict(dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,precision=tp/(tp+fp) if tp+fp else float(empty),cldice=cl,thin_skeleton_recall=float((t['thin']&pred).sum()/thin) if thin else 1.)

def coverage(root,rows):
    """Check that every planned test scope appears in the combined replay."""
    tag=root.name[-19:];report=[]
    test_phases={'controls','encoded','pixel','native_flow','anchored','fully_oof','topology','retained','local_acceptor','rollout','context','fresh_cv','calibration','segflow'}
    for job in load(root/f'queue_{tag}.json')['jobs']:
        phase=job['phase'];ds=job['dataset']
        if phase not in test_phases:continue
        completion=load(root/phase/ds/f'completed_{tag}.json')
        rr=[r for r in rows if r['phase']==phase and r['dataset']==ds]
        if phase=='calibration' and not completion['pilot_gate_passed']:
            assert not rr,(phase,ds,'Test rows exist despite failed calibration gate')
            report.append(dict(**job,test_scope='Not triggered by validation gate',rows=0));continue
        assert rr,(phase,ds,'Completed test scope missing from combined replay')
        expected_seeds={42} if phase=='encoded' else set(SEEDS)
        expected_images=({'DRIVE':40,'HRF':45}[ds] if phase in ('fresh_cv','calibration') else {'DRIVE':20,'HRF':12}[ds])
        assert {int(r['seed']) for r in rr}==expected_seeds,(phase,ds,'Seed coverage')
        assert len({r['image'] for r in rr})==expected_images,(phase,ds,'Image coverage')
        outputs=defaultdict(list)
        for r in rr:outputs[r['method'],r['mode'],r['output']].append(r)
        for key,group in outputs.items():
            assert len(group)==len(expected_seeds)*expected_images,(phase,ds,key,'Incomplete output group')
        files=len({r['path'] for r in rr})
        for key in ('test_prediction_files','prediction_files'):
            if key in completion:assert files==completion[key],(phase,ds,key,files,completion[key])
        if phase=='controls':
            assert len({r['method'] for r in rr})==16 and len(rr)==completion['test_records']
        report.append(dict(**job,rows=len(rr),prediction_files=files,output_groups=len(outputs),training_seed_labels=sorted(expected_seeds),held_out_images=expected_images))
    return report

def run(root,legacy_only=False):
    root=Path(root).resolve();tag=root.name[-19:];out=root/('preverified_legacy' if legacy_only else 'analysis')
    if legacy_only:
        m=load(out/f'completed_{tag}.json');p=out/f'verified_existing_rows_{tag}.json';assert sha256_file(p)==m['rows_sha256'];rows=load(p)
    else:
        assert (out/f'completed_analysis_{tag}.json').exists();rows=csvrows(out/f'all_verified_per_image_{tag}.csv')
    coverage_report=None if legacy_only else coverage(root,rows)
    if coverage_report is not None:write_json(out/f'test_scope_coverage_{tag}.json',coverage_report)
    grouped=defaultdict(list)
    for r in rows:grouped[r['path']].append(r)
    norm=load(ROOT/'outputs/private_test_comparisons_2026-11-17_00-38-33/normalized_saved_comparison_manifest_2026-11-17_00-39-15.json');control={r['path']:r['input_path'] for r in norm['controls']};gtcache={};checked=0;identities=[]
    for path,rr in grouped.items():
        assert sha256_file(path)==rr[0]['prediction_sha256']
        with np.load(path,allow_pickle=False) as z:
            if 'ground_truth' in z:
                g=z['ground_truth'][0];f=z['fov'][0];base=z['baseline'][0] if 'baseline' in z else z['coarse'][0]
            else:
                with np.load(control[path],allow_pickle=False) as source:g=source['ground_truth'][0].copy();f=source['fov'][0].copy();base=source['baseline'][0].copy()
            anchor=z['O3'][0] if 'O3' in z else base;valid=f>=.5
            key=hashlib.sha256((g>=.5).tobytes()+(f>=.5).tobytes()).hexdigest()
            if key not in gtcache:gtcache[key]=truth(g,f)
            for r in rr:
                kind=r['output']
                if kind=='noise_mean_metric':
                    mm=[hard(p[0],gtcache[key]) for p in z['probabilities'][0]];result={k:float(np.mean([m[k] for m in mm])) for k in KEYS}
                else:
                    array_key='refined' if kind=='selected' and 'selected' not in z else kind
                    probability=z[array_key][0];result=hard(probability,gtcache[key]);changed=int(np.count_nonzero(((probability>=.5)!=(anchor>=.5))&valid))
                    identity_probability=z['probability'][0] if r['phase']=='calibration' else probability
                    identity_source=z['O3_probability'][0] if r['phase']=='calibration' else anchor
                    identities.append(dict(**{k:r[k] for k in ('cohort','dataset','phase','method','mode','output','seed','image','outer_fold','anchor')},changed_FOV_pixels=changed,valid_pixels=int(valid.sum()),exact_FOV_mask_identity=changed==0,exact_FOV_probability_identity=bool(np.array_equal(identity_probability[valid],identity_source[valid])),path=path,prediction_sha256=r['prediction_sha256']))
                assert all(result[k]==float(r[k]) for k in KEYS),(path,kind,result,{k:r[k] for k in KEYS});checked+=1
        if checked%500<4:print('INDEPENDENT HARD METRICS',checked,'/',len(rows),flush=True)
    if legacy_only:
        report=dict(verified_at=stamp(),metric_rows_verified=checked,scalar_metrics_verified=checked*4,rows_sha256=m['rows_sha256'],independent_hard_formulas=True,GT_FOV_fields=len(gtcache),training_performed=False)
        write_json(out/f'independent_hard_verification_{tag}.json',report);print(json.dumps(report,indent=2),flush=True);return report
    # Verify new training log lengths and locked local checkpoint hashes.
    decisions=0;logs=0;verified_paths=set()
    for phase in load(root/f'queue_{tag}.json')['jobs']:
        folder=root/phase['phase']/phase['dataset']
        for p in folder.rglob(f'*locked_{tag}.json'):
            d=load(p)
            if not isinstance(d,dict):continue
            bound_paths(d,p.parent,verified_paths)
            if phase['phase']=='controls' and 'arm' in d:
                for stem,key in [('best_raw_model','raw_sha256'),('selected_blend_model','blend_sha256'),('final_model','final_sha256')]:
                    cp=p.parent/f'{stem}_{tag}.pt';assert sha256_file(cp)==d[key];verified_paths.add(str(cp.resolve()))
            decisions+=1
        if phase['phase']=='encoded':
            models=list(folder.rglob(f'flow_model_{tag}.pt'));assert len(models)==1
            cp=models[0];completion=load(cp.parent/f'completed_{tag}.json');assert sha256_file(cp)==completion['checkpoint_sha256'];verified_paths.add(str(cp.resolve()))
        for p in folder.rglob(f'*training*_{tag}.csv'):
            rr=csvrows(p)
            if not rr or 'epoch' not in rr[0]:continue
            assert [int(r['epoch']) for r in rr]==list(range(1,len(rr)+1)),p;logs+=1
    expected=load(root/f'queue_{tag}.json')['planned_new_training_fits'];assert logs==expected,(logs,expected)
    savecsv(out/f'verified_identity_masks_{tag}.csv',identities)
    protected=next(root.glob('prior_manuscripts_sources_protected_*.json'));assert all(sha256_file(p)==h for p,h in load(protected).items())
    report=dict(verified_at=stamp(),independently_implemented_metrics=list(KEYS),metric_rows_verified=checked,scalar_metrics_verified=checked*4,unique_GT_FOV_fields=len(gtcache),locked_decision_files=decisions,distinct_bound_checkpoint_and_input_files_verified=len(verified_paths),consecutive_training_logs=logs,test_scopes_covered=len(coverage_report),all_planned_test_images_seeds_and_output_groups_covered=True,all_previous_bound_manuscripts_and_sources_unchanged=True,implementation='Independent NumPy confusion counts and hard skeleton/EDT formulas; same pinned scipy/skimage primitives, no use of the original metric function.')
    write_json(out/f'independent_verification_{tag}.json',report);print(json.dumps(report,indent=2),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--legacy-only',action='store_true');p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.execute:run(a.root,a.legacy_only)
