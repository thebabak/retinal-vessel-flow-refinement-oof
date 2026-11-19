"""One explicitly dispatched GPU phase. Old artifact directories are read-only."""
from __future__ import annotations
import argparse,sys,gc,traceback
from pathlib import Path
from .common import ROOT,SEEDS,Context,setup,load,write_json,stamp,sha256_file,csvrows,savecsv

def controls(ctx):
    # Import the exact registered implementation without invoking its old main.
    package=ROOT/'outputs/reviewer_completion_2026-11-16_19-37-22/controlled_ablations_2026-11-16_19-41-32/code'
    sys.path.insert(0,str(package))
    from reviewer_ablations import runner as r
    from reviewer_ablations.specs import ARMS,EPOCHS
    ctx.protect(package/'reviewer_ablations/runner.py');ctx.protect(package/'reviewer_ablations/operations.py');ctx.protect(package/'reviewer_ablations/specs.py')
    ctx.status('loading_audited_native_inputs',fits_completed=0,fits_expected=80)
    ref,audit,thin=r.prepare(ctx.dataset,ctx.out,ctx.tag,'training')
    ctx.protected.update(audit['source_sha256_before']);choices=[]
    for spec in ARMS:
        views,pools,vrgb=r.create_views(ref,spec)
        for seed in SEEDS:
            job=ctx.out/spec.name/f'seed_{seed}';marker=job/f'fit_verified_{ctx.tag}.json'
            if marker.exists():d=load(marker);r.verify_decision(ctx.out,ctx.tag,d)
            else:
                assert not job.exists(),f'Incomplete fit requires explicit recovery: {job}'
                d=r.train_one(ref,ctx.out,ctx.tag,spec,seed,views,pools,vrgb,ctx.status)
                r.verify_decision(ctx.out,ctx.tag,d)
                assert len(csvrows(job/f'training_{ctx.tag}.csv'))==EPOCHS==120
                write_json(marker,d)
            choices.append(d);ctx.status('training_progress',fits_completed=len(choices),fits_expected=80)
        del views,pools,vrgb;gc.collect()
    assert len(choices)==80
    lock=ctx.out/f'all80_new_choices_locked_before_test_{ctx.tag}.json'
    if lock.exists():assert load(lock)['decisions']==choices
    else:write_json(lock,dict(locked_at=stamp(),decisions=choices,scope='HRF-only completion after previously inspected CHASE/DRIVE results. Not a retrospective global240beforealltests claim.'))
    lh=sha256_file(lock);rows=[]
    for d in choices:
        dest=ctx.out/d['arm']/f'seed_{d["refiner_seed"]}'
        if (dest/f'evaluation_completed_{ctx.tag}.json').exists():rows.extend(csvrows(dest/f'test_records_{ctx.tag}.csv'))
        else:rows.extend(r.evaluate_one(ref,ctx.out,ctx.tag,d,thin['cutoff'],ctx.status))
    assert sha256_file(lock)==lh
    savecsv(ctx.out/f'all_per_image_records_{ctx.tag}.csv',rows)
    return ctx.done(dict(final_fits=80,epochs_per_fit=120,test_records=len(rows),selection_lock=str(lock),selection_lock_sha256=lh,thin_cutoff=thin['cutoff']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',required=True);p.add_argument('--dataset',choices=['DRIVE','HRF'],required=True);a=p.parse_args()
    if not a.execute:print('Inert without --execute.');return
    setup();ctx=Context(a.root,a.phase,a.dataset)
    marker=ctx.out/f'completed_{ctx.tag}.json'
    if marker.exists():print('ALREADY COMPLETE',marker,flush=True);return
    try:
        from .basic import encoded,pixel,native_flow,anchored
        from .sources import strict_sources
        from .oof_families import run_family
        from .corrections import fixed_correction,fresh_sources,fresh_cv
        from .local import local_acceptor
        from .geometry import geometry
        from .diagnostics import calibration,gap,encoded_diagnosis,bias_probe,precision_reselection
        from .external import frunet,segflow
        functions={'controls':controls,'encoded':encoded,'encoded_diagnosis':encoded_diagnosis,'pixel':pixel,'native_flow':native_flow,'anchored':anchored,'strict_sources':strict_sources,'fully_oof':run_family,'topology':run_family,'retained':run_family,'rollout':fixed_correction,'context':fixed_correction,'fresh_sources':fresh_sources,'fresh_cv':fresh_cv,'local_acceptor':local_acceptor,'geometry':geometry,'calibration':calibration,'gap':gap,'frunet':frunet,'segflow':segflow,'bias_probe':bias_probe,'precision_reselection':precision_reselection}
        if a.phase not in functions:raise ValueError('Phase dispatcher is not implemented: '+a.phase)
        functions[a.phase](ctx)
    except BaseException:
        write_json(ctx.out/f'failed_{ctx.tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));ctx.status('failed',status='failed');raise
if __name__=='__main__':main()
