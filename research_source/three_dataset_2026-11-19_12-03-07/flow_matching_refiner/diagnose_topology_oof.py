"""Training/validation-only structural diagnosis of the preceding OOF Flow."""
import csv,json,gc
from pathlib import Path
import numpy as np
import torch
from skimage.morphology import skeletonize
from flow_matching_refiner.run_refinement_v3 import write_json
from flow_matching_refiner.data import sha256_file
from flow_matching_refiner.run_native_patch_flow_v2 import model_for,rows
from flow_matching_refiner.native_patch_flow_v2 import predict_tiled
from flow_matching_refiner.native_patch_model import apply_residual,project_residual_512
from flow_matching_refiner.structure_metrics_all import metrics,thin_skeleton_mask

OLD=Path(__file__).resolve().parents[1]/'outputs/fully_oof_flow_2026-11-17_03-45-32';OT='2026-11-17_03-45-32'
KEYS=('dice','precision','recall','cldice','topology_precision','topology_sensitivity','thin_skeleton_recall','thin_skeleton_hit','thin_skeleton_missed')
def savecsv(path,rr):
    with Path(path).open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

def diagnose(ref,out,tag,routes,status):
    out.mkdir(parents=True,exist_ok=False);decisions=json.loads((OLD/f'all10_choices_locked_before_test_{OT}.json').read_text())['decisions']
    per=[];candidate=[];transitions=[];native=[];history=[];oof=[]
    def score(p,d):return metrics(p.numpy()[0],d['g'].numpy()[0],d['coarse'].numpy()[0],1.,d['fov'].numpy()[0])
    for dec in [q for q in decisions if q['dataset']==ref.dataset]:
        seed=dec['seed'];status('diagnosing_old_validation',dataset=ref.dataset,seed=seed)
        folder=Path(dec['selected_path']).parent;t=rows(folder/f'training_{OT}.csv');v=rows(folder/f'validation_{OT}.csv')
        history.append(dict(dataset=ref.dataset,seed=seed,first_soft_cldice_loss=float(t[0]['cldice']),final_soft_cldice_loss=float(t[-1]['cldice']),first_FM=float(t[0]['fm']),final_FM=float(t[-1]['fm']),selected_by='Dice only',selected_epoch=dec['selected']['epoch'],beta=dec['selected']['beta']))
        model=model_for(ref.dataset,seed,seed,dec['config']['cap'],ref.device);states={}
        for kind in ('raw','selected'):
            cp=Path(dec[f'{kind}_path']);assert sha256_file(cp)==dec[f'{kind}_sha256']
            if kind=='selected' and dec['selected']['beta']==0:continue
            epoch=dec[kind]['epoch']
            if epoch not in states:states[epoch]=torch.load(cp,map_location=ref.device,weights_only=True)['state']
        selected_outputs={};base_outputs={}
        for epoch,state in states.items():
            model.field.load_state_dict(state);aggregate={b:[] for b in (0.,.25,.5,1.)}
            for n in ref.manifest['validation']:
                d=ref.data[n];a,f,_=predict_tiled(model,d['n_coarse'],d['n_rgb'],ref.device)
                a512=project_residual_512(a,ref.dataset);f512=project_residual_512(f,ref.dataset);base=apply_residual(d['coarse'],a512);base_outputs[n]=base
                for beta in aggregate:
                    p=base if beta==0 else apply_residual(d['coarse'],a512+beta*f512);m=score(p,d);aggregate[beta].append(m)
                    per.append(dict(dataset=ref.dataset,seed=seed,checkpoint_epoch=epoch,beta=beta,image=n,**{k:m[k] for k in KEYS}))
                    if epoch==dec['selected']['epoch'] and beta==dec['selected']['beta']:selected_outputs[n]=p
                # A validation resolution probe, never a replacement primary score.
                ng=d['n_g'].numpy()[0];nf=d['n_fov'].numpy()[0]
                for kind,res,b512 in [('O3',a,base),('old_raw_beta1',a+f,apply_residual(d['coarse'],a512+f512))]:
                    npred=apply_residual(d['n_coarse'],res).numpy()[0];nm=metrics(npred,ng,d['n_coarse'].numpy()[0],1.,nf);em=score(b512,d)
                    native.append(dict(dataset=ref.dataset,seed=seed,epoch=epoch,image=n,arm=kind,native_Dice=nm['dice'],native_clDice=nm['cldice'],evaluation512_Dice=em['dice'],evaluation512_clDice=em['cldice'],thin_native_NOT_primary=True))
            for beta,mm in aggregate.items():candidate.append(dict(dataset=ref.dataset,seed=seed,checkpoint_epoch=epoch,beta=beta,**{k:float(np.mean([z[k] for z in mm])) for k in KEYS}))
        if dec['selected']['beta']==0:selected_outputs=base_outputs
        assert set(selected_outputs)==set(ref.manifest['validation'])
        for n,p in selected_outputs.items():
            d=ref.data[n];g=d['g'].numpy()[0]>=.5;valid=d['fov'].numpy()[0]>.5;sk=skeletonize(g&valid);thin=thin_skeleton_mask(g,1.,valid)
            b=base_outputs[n].numpy()[0]>=.5;q=p.numpy()[0]>=.5
            transitions.append(dict(dataset=ref.dataset,seed=seed,image=n,thin_recovered=int((thin&~b&q).sum()),thin_lost=int((thin&b&~q).sum()),skeleton_recovered=int((sk&~b&q).sum()),skeleton_lost=int((sk&b&~q).sum()),selected_beta=dec['selected']['beta']))
        del model,states;gc.collect();torch.cuda.empty_cache()
        for n in ref.manifest['train']:
            e=routes[ref.dataset,seed,n];assert sha256_file(e['path'])==e['sha256']
            with np.load(e['path'],allow_pickle=False) as z:
                delta=project_residual_512(torch.from_numpy(z['anchor_residual_native']),ref.dataset);d=ref.data[n]
                assert np.array_equal(z['coarse512'],d['coarse'].numpy());p=apply_residual(d['coarse'],delta);m=score(p,d)
                oof.append(dict(dataset=ref.dataset,seed=seed,image=n,source='strict_OOF_train_anchor',**{k:m[k] for k in KEYS}))
    savecsv(out/f'old_validation_per_image_{tag}.csv',per);savecsv(out/f'old_validation_candidates_{tag}.csv',candidate)
    savecsv(out/f'old_validation_thin_transitions_{tag}.csv',transitions);savecsv(out/f'validation_resolution_probe_{tag}.csv',native);savecsv(out/f'training_OOF_anchor_quality_{tag}.csv',oof)
    summary=dict(dataset=ref.dataset,validation_only=True,new_weights_fitted=False,history=history,
       thin_recovered=sum(r['thin_recovered'] for r in transitions),thin_lost=sum(r['thin_lost'] for r in transitions),skeleton_recovered=sum(r['skeleton_recovered'] for r in transitions),skeleton_lost=sum(r['skeleton_lost'] for r in transitions),
       training_OOF_means={k:float(np.mean([r[k] for r in oof])) for k in KEYS},
       observations=['Prior checkpoint/beta selection uses only Dice; its loss has no explicit thin support and weight0.1 native soft-clDice.',
          'Prior FM target is exactly zero on correct GT centerline pixels; probability preservation also discourages their change.',
          'Native soft-skeleton training and hard 512 evaluation have distinct grids/skeletons; resolution probe is diagnostic only.',
          'OOF source fits12images with nested baselines fit8; deployment anchor fits16. Training-size shift remains.'],
       causal_scope='Observed code/probes motivate hypotheses. The new combined variant is not an isolated causal ablation.')
    write_json(out/f'diagnosis_{tag}.json',summary);return summary
