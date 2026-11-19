"""New DRIVE/HRF connection-repair screen on validation images only.

Replays five existing Option-3 training seeds; never confuses them with new
Flow trainings. A dataset must pass the locked feasibility gate before any
connection-Flow pilot may be fitted. Test RGB/GT are not loaded by this runner.
"""
from __future__ import annotations
import argparse,csv,gc,json,shutil,traceback
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.torch_version import TorchVersion
from scipy import ndimage as ndi
from skimage.morphology import disk
from flow_matching_refiner.data import ROOT,REFERENCE_ROOT,sha256_file
from flow_matching_refiner.native_patch_data import _source_inventory,_read_native_arrays
from flow_matching_refiner.native_patch_model import ErrorSelectiveModel,predict_tiled,project_residual_512,apply_residual
from flow_matching_refiner.gap_repair_diagnosis import CONFIG,diagnose
from flow_matching_refiner.calibration_metrics import KEYS,qualifies
from flow_matching_refiner.structure_metrics_all import metrics
from flow_matching_refiner.run_refinement_v3 import stamp,write_json
from paired_reproduction.official_model import LNet_RA
O3=ROOT/'outputs/native_patch_study_2026-11-16_16-53-08';OT='2026-11-16_16-53-08'
SEEDS=(42,123,2026,3000,3001)

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def protect(p,protected):h=sha256_file(p);protected[str(p)]=h;return h
def savecsv(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def diagnose_fov(prob,rgb,g,fov):
    p=np.where(fov,prob,0).astype(np.float32);g=np.asarray(g,dtype=bool)&fov
    result,arrays=diagnose(p,rgb,g);tube=np.zeros_like(g);accepted=[]
    for q in result['proposals']:
        xy=np.asarray(q['path_yx'])
        if not fov[xy[:,0],xy[:,1]].all():continue
        path=np.zeros_like(g);path[xy[:,0],xy[:,1]]=True
        tube|=ndi.binary_dilation(path,structure=disk(CONFIG['proposal_tube_radius']))&fov
        accepted.append(dict(q,candidate_id=len(accepted)))
    b=arrays['binary'];oracle=b|(tube&g);raw=b|tube
    arrays.update(proposal_tube=tube,oracle_mask=oracle,unfiltered_mask=raw,fov=fov)
    result['proposals']=accepted;result['metrics']={}
    for name,mask in [('O3',b),('oracle_GT_filtered_proposals',oracle),('unfiltered_proposal_union',raw)]:
        m=metrics(mask,g,p,1.,fov);result['metrics'][name]={k:m[k] for k in KEYS}
    s=result['stats'];s.update(candidate_count=len(accepted),proposal_covered_missing_skeleton_points=int((tube&arrays['gt_skeleton']&~b).sum()),proposal_covered_missing_thin_points=int((tube&arrays['missing_thin']).sum()),proposed_new_GT_pixels=int((tube&g&~b).sum()),proposed_new_false_positive_pixels=int((tube&~g&~b).sum()),oracle_added_pixels=int((oracle&~b).sum()))
    return result,arrays

def figure(path,rgb,arrays,ds,seed,name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13.5,3.7));axes[0].imshow(rgb);axes[0].set_title('RGB / missed thin skeleton')
    overlay=np.zeros((*arrays['gt'].shape,4),np.float32);overlay[arrays['missing_thin']]=[1,.1,.1,.9];overlay[arrays['bounded_gaps']]=[0,1,1,1];axes[0].imshow(overlay)
    axes[1].imshow(arrays['binary'],cmap='gray',vmin=0,vmax=1);axes[1].set_title('Option 3 / threshold 0.5')
    axes[2].imshow(rgb);overlay=np.zeros_like(overlay);overlay[arrays['proposal_tube']]=[1,.8,0,.7];axes[2].imshow(overlay);axes[2].set_title('GT-free candidate tubes')
    axes[3].imshow(arrays['oracle_mask'],cmap='gray',vmin=0,vmax=1);axes[3].set_title('GT-filtered oracle (not a model)')
    for ax in axes:ax.axis('off')
    fig.suptitle(f'{ds} validation | O3 seed {seed} | {name} | red: missed thin; cyan: GT-bounded gap',fontsize=10);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def prepare(ds,protected,status):
    protocol,inventory,_=_source_inventory(ds);sp=protocol['split'];base=REFERENCE_ROOT/ds/'train_seed_42';protect(base/'protocol.json',protected);protect(base/'file_provenance.json',protected)
    cp=base/'baseline.pt';ch=protect(cp,protected)
    with torch.serialization.safe_globals([TorchVersion]):obj=torch.load(cp,map_location='cuda',weights_only=True)
    assert obj['protocol']['split']==sp
    net=LNet_RA(in_channels=1).cuda().eval().requires_grad_(False);net.load_state_dict(obj['state']);cases={}
    for n in sp['validation']:
        status('validation_input_and_baseline',dataset=ds,image=n)
        record=inventory[n]
        for k in ('image_path','gt_path','fov_path'):
            if record.get(k):protect(record[k],protected)
        arrays,info=_read_native_arrays(ds,record,'validation');rgb=arrays['rgb'];green=cv2.createCLAHE(clipLimit=5.,tileGridSize=(32,32)).apply(rgb[:,:,1]);green=np.uint8(np.clip((green/255.)**(1/1.2)*255,0,255));green=cv2.resize(green,(512,512),interpolation=cv2.INTER_LINEAR)
        x=torch.tensor(green/255.,dtype=torch.float32)[None,None].cuda()
        with torch.inference_mode():coarse=net(x).cpu()[0].contiguous()
        native_rgb=torch.from_numpy((rgb/255.).astype(np.float32)).permute(2,0,1).contiguous()
        native_coarse=F.interpolate(coarse[None],size=rgb.shape[:2],mode='bilinear',align_corners=False)[0].contiguous()
        cases[n]=dict(rgb=native_rgb,rgb512=cv2.resize(rgb,(512,512),interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.,coarse=coarse,native_coarse=native_coarse,g=arrays['small_g'],fov=arrays['small_fov'],record=record,source_info=info,baseline=str(cp),baseline_sha256=ch)
    del net,obj;gc.collect();torch.cuda.empty_cache();return cases,sp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if not args.execute:print('Inert without --execute. Validation only.');return
    out=args.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and out.is_dir() and not (out/f'plan_{tag}.json').exists();state=dict(output=str(out),tag=tag,training_performed=False,test_accessed=False)
    def status(stage,**kw):state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_gap_external_validation.json',state)
    try:
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();protected={}
        for p in (ROOT/'flow_matching_refiner').glob('*.py'):protect(p,protected)
        protect(ROOT/'paired_reproduction/official_model.py',protected)
        for name in ('template261006.tex','last.tex','tmp/current_gap_diagnosis_delivery.json'):
            p=ROOT/name
            if p.is_file():protect(p,protected)
        plan=dict(locked_at=stamp(),datasets=['DRIVE','HRF'],source_O3_study=str(O3),O3_seeds=SEEDS,training_performed=False,proposal_config=CONFIG,FOV='Mask probability and GT before skeletons; discard candidate paths leaving FOV and clip tube dilation to FOV. All four metrics use the existing FOV-aware scorer.',scope='New validation inference and diagnostic calculations from five retained O3 training seeds, not five new Flow fits. No test RGB/GT loaded. Existing full-training baseline supplies validation coarse maps.',gate=dict(bounded_gap_share_of_missed_thin_at_least=.20,candidate_coverage_of_missed_thin_at_least=.10,oracle_four_metric_gain_required=True,minimum_qualifying_O3_seeds=3,total_O3_seeds=5),gate_adaptation='Unlike CHASE three-fold diagnosis, DRIVE/HRF reuse one fixed4/6-image validation cohort under5retainedO3trainingseeds. A majority3/5source-realization gate assesses engineering stability, not independent-cohort replication. Fixed before new diagnosis; no pooling with CHASE.',pilot_if_passes='Only a passing dataset proceeds to new connection/path Flow plus matched direct control with strict OOF training inputs; validation-only choice. Diagnostic oracle never counts as Flow efficacy.',data='Existing retained annotations/conversions unchanged. No new DRIVE archive/source provenance investigation; test-label provenance caveat retained. HRF source-pixel audit reused.',old_papers_and_results_unchanged=True)
        write_json(out/f'plan_{tag}.json',plan);write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True))
        source=out/'source';source.mkdir()
        for n in ('run_gap_external_validation','gap_repair_diagnosis','native_patch_data','native_patch_model'):
            shutil.copy2(ROOT/'flow_matching_refiner'/f'{n}.py',source/f'{n}_{tag}.py')
        allrows=[];seedrows=[];records=[];datasets=[]
        for ds in ('DRIVE','HRF'):
            cases,split=prepare(ds,protected,status);job=out/ds;job.mkdir();(job/'figures').mkdir();(job/'arrays').mkdir();ds_seeds=[]
            for seed in SEEDS:
                fold=O3/f'{ds}_{OT}/native_patch_direct'/f'seed_{seed}';selection=load(fold/f'selection_locked_{OT}.json');protect(fold/f'selection_locked_{OT}.json',protected);cp=fold/f'best_raw_model_{OT}.pt';ch=protect(cp,protected);assert ch==selection['raw_sha256'];obj=torch.load(cp,map_location='cuda',weights_only=True);net=ErrorSelectiveModel(False).cuda().eval().requires_grad_(False);net.load_state_dict(obj['state']);rr=[]
                for n,c in cases.items():
                    status('validation_O3_inference',dataset=ds,seed=seed,image=n)
                    _,delta,meta=predict_tiled(net,c['native_coarse'],c['rgb'],'cuda',batch=selection['infer_batch'],return_residual=True)
                    d512=project_residual_512(delta,ds);p=apply_residual(c['coarse'],d512).numpy()[0].copy();result,arrays=diagnose_fov(p,c['rgb512'],c['g'],c['fov']);path=job/'arrays'/f'seed{seed}_{n}_validation_gap_{tag}.npz';np.savez_compressed(path,probability=p,coarse=c['coarse'].numpy()[0],delta512=d512.numpy()[0],**arrays)
                    rec=dict(dataset=ds,seed=seed,image=n,result=result,arrays=str(path),arrays_sha256=sha256_file(path),O3_checkpoint=str(cp),O3_sha256=ch,source=c['record'],inference=meta);records.append(rec);rr.append(rec);row=dict(dataset=ds,seed=seed,image=n,**result['stats'])
                    for method,mm in result['metrics'].items():row.update({method+'_'+k:v for k,v in mm.items()})
                    allrows.append(row);write_json(job/f'seed{seed}_{n}_records_{tag}.json',rec)
                    if seed==42:figure(job/'figures'/f'{n}_seed{seed}_validation_gap_{tag}.png',c['rgb512'],arrays,ds,seed,n)
                    del delta,arrays;gc.collect()
                sums={k:sum(r['result']['stats'][k] for r in rr) for k in rr[0]['result']['stats']};missing=sums['missing_thin_points'];gap=sums['bounded_gap_thin_points']/missing if missing else 0.;coverage=sums['proposal_covered_missing_thin_points']/missing if missing else 0.;means={m:{k:float(np.mean([r['result']['metrics'][m][k] for r in rr])) for k in KEYS} for m in rr[0]['result']['metrics']}
                assert abs(means['O3']['dice']-selection['raw']['validation_dice'])<1e-12,'Retained O3 validation inference changed'
                oracle_ok=qualifies(means['oracle_GT_filtered_proposals'],means['O3']);passed=gap>=.20 and coverage>=.10 and oracle_ok
                sr=dict(dataset=ds,seed=seed,validation_images=len(cases),bounded_gap_fraction=gap,candidate_coverage=coverage,oracle_goal=oracle_ok,feasibility_passed=passed,counts=sums,mean_metrics=means,O3_validation_Dice_replayed_exact=True);ds_seeds.append(sr);flat={k:v for k,v in sr.items() if k not in ('counts','mean_metrics')};flat.update(sums)
                for method,mm in means.items():flat.update({method+'_'+k:v for k,v in mm.items()})
                seedrows.append(flat);print('EXTERNAL GAP',ds,seed,'bounded',gap,'coverage',coverage,'passed',passed,flush=True);del net,obj;gc.collect();torch.cuda.empty_cache()
            passed_count=sum(s['feasibility_passed'] for s in ds_seeds);summary=dict(dataset=ds,source_seeds=ds_seeds,qualifying_seeds=passed_count,pilot_authorized=passed_count>=3,training_performed=False,test_accessed=False);datasets.append(summary);write_json(job/f'dataset_gate_locked_{tag}.json',dict(locked_at=stamp(),**summary));del cases;gc.collect()
        savecsv(out/f'validation_per_image_{tag}.csv',allrows);savecsv(out/f'validation_per_source_seed_{tag}.csv',seedrows);write_json(out/f'prediction_and_diagnostic_manifest_{tag}.json',records);assert all(sha256_file(p)==h for p,h in protected.items());write_json(out/f'protected_inputs_unchanged_{tag}.json',protected)
        result=dict(completed_at=stamp(),datasets=datasets,validation_image_seed_records=len(records),new_O3_training=False,new_Flow_training=False,training_performed=False,test_accessed=False,old_files_unchanged=True,protected_inputs=len(protected),conditional_pilot_pending=[d['dataset'] for d in datasets if d['pilot_authorized']]);write_json(out/f'completed_external_diagnosis_{tag}.json',result);status('diagnosis_complete',conditional_pilot_pending=result['conditional_pilot_pending']);print('EXTERNAL GAP SCREEN COMPLETE',result['conditional_pilot_pending'],flush=True)
    except BaseException:write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
