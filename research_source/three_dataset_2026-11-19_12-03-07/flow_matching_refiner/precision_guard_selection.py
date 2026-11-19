"""Validation-only re-selection from the unchanged context-Flow checkpoints."""
import csv,json
from pathlib import Path
import numpy as np
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json

KEYS=('dice','precision','cldice','thin_skeleton_recall')
def eligible(mm,base):
    return mm['dice']>=base['dice']-1e-12 and mm['precision']>=base['precision']-1e-12 and mm['cldice']>base['cldice']+1e-12 and mm['thin_skeleton_recall']>base['thin_skeleton_recall']+1e-12
def gain(mm,base):return .5*(mm['cldice']-base['cldice']+mm['thin_skeleton_recall']-base['thin_skeleton_recall'])
def save_csv(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run_existing(out,tag):
    old=ROOT/'outputs/global_context_CHASE_pilot_2026-11-18_01-42-41';ot=old.name[-19:];rows=[];decisions=[];protected={}
    for seed in (42,123,2026,3000,3001):
        for mode in ('flow','direct'):
            folder=old/'final'/mode/f'seed_{seed}';dp=folder/f'selection_locked_{ot}.json';vp=folder/f'validation_all_{ot}.csv';protected[str(dp)]=sha256_file(dp);protected[str(vp)]=sha256_file(vp);d=json.loads(dp.read_text());base=d['baseline_validation'];best=None
            with vp.open(encoding='utf-8-sig',newline='') as f:vv=list(csv.DictReader(f))
            for ep in (20,40,60,80):
                for steps in ((2,4,8) if mode=='flow' else (1,)):
                    cp=folder/f'field_epoch{ep}_{ot}.pt';protected[str(cp)]=sha256_file(cp)
                    for beta in (.125,.25,.5,1.):
                        v=[r for r in vv if int(r['epoch'])==ep and int(r['steps'])==steps and float(r['beta'])==beta];assert len(v)==4
                        mm={k:float(np.mean([float(r[k]) for r in v])) for k in KEYS};q=dict(mode=mode,seed=seed,epoch=ep,steps=steps,beta=beta,**mm,qualifies=eligible(mm,base),structural_gain=gain(mm,base));rows.append(q)
                        if q['qualifies'] and (best is None or q['structural_gain']>best['structural_gain']+1e-12):best=dict(**q,checkpoint=str(cp),checkpoint_sha256=protected[str(cp)])
            chosen=best if best else dict(mode=mode,seed=seed,epoch=0,steps=0,beta=0.,**{k:base[k] for k in KEYS},qualifies=False,structural_gain=0.,checkpoint=None,checkpoint_sha256=None)
            decisions.append(dict(mode=mode,seed=seed,selection=chosen,baseline_validation=base,qualifying_candidates=sum(r['qualifies'] for r in rows if r['mode']==mode and r['seed']==seed)))
    assert len(rows)==320 and len(decisions)==10;save_csv(out/f'existing_validation_precision_guard_candidates_{tag}.csv',rows);result=dict(completed_at=stamp(),rule='Validation mean Dice and precision >=O3 with tolerance1e-12, clDice and thin recall strictly higher; maximize mean structural gain; exact identity if no candidate.',decisions=decisions,training_performed=False,test_accessed=False,protected_inputs=protected);write_json(out/f'existing_validation_precision_guard_selections_{tag}.json',result);assert all(sha256_file(p)==h for p,h in protected.items());print('EXISTING CHECKPOINT RESELECTION',[(d['mode'],d['seed'],d['qualifying_candidates']) for d in decisions],flush=True);return result

def self_test():
    b=dict(dice=.8,precision=.8,cldice=.8,thin_skeleton_recall=.6);assert not eligible(b,b);assert eligible(dict(b,cldice=.81,thin_skeleton_recall=.61),b);assert not eligible(dict(b,precision=.79999,cldice=.81,thin_skeleton_recall=.61),b);assert not eligible(dict(b,dice=.79999,cldice=.81,thin_skeleton_recall=.61),b)
if __name__=='__main__':self_test();m=json.loads((ROOT/'tmp/current_precision_subject_cv_launch.json').read_text());run_existing(Path(m['output']),m['tag'])
