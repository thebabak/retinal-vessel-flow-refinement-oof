"""Isolated CHASE-only experiment, with live VS Code output and staged source locks."""
from __future__ import annotations
import argparse,json,shutil,time,traceback
from pathlib import Path
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if not a.execute:print('Inert. Explicit --execute required. CHASE only.');return
    out=a.output.resolve();tag=out.name[-19:];assert out.parent==ROOT/'outputs' and (out/f'live_setup_{tag}.json').is_file()
    assert not (out/f'plan_{tag}.json').exists(),'Never overwrite a previous experiment'
    state=dict(output=str(out),tag=tag,dataset='CHASE',seeds=[42,123,2026,3000,3001],HRF_started=False,DRIVE_started=False)
    def status(stage,**kw):
        state.update(stage=stage,updated_at=stamp(),**kw);write_json(out/f'status_{tag}.json',state);write_json(ROOT/'tmp/current_local_flow_acceptor.json',state)
    previous=ROOT/'outputs/retained_oof_flow_2026-11-17_19-47-46';pt='2026-11-17_19-47-46'
    protected=json.loads((previous/f'protected_before_including_loaded_sources_{pt}.json').read_text());assert all(sha256_file(p)==h for p,h in protected.items())
    for p in (ROOT/'outputs/local_flow_acceptor_CHASE_2026-11-17_22-03-44').rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in previous.rglob('*'):
        if p.is_file():protected[str(p)]=sha256_file(p)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):protected[str(p)]=sha256_file(p)
    write_json(out/f'protected_before_{tag}.json',protected)
    for p in (ROOT/'flow_matching_refiner').glob('*.py'):
        d=out/'source'/f'{p.stem}_{tag}.py';d.parent.mkdir(exist_ok=True);shutil.copy2(p,d)
    write_json(out/f'plan_{tag}.json',dict(**state,source_stage='4 triple-excluded coarse baselines (4 fit images) +6 pair-excluded O3 models (8fit), common source seed42, minimum200optimizer updates and validation softDice tie-break for4image baselines; reuse existing subject-grouped outer O3 models. All direct/indirect held labels excluded, including validation-map producers.',
        proposal_stage='Local signed-distance Flow (2 bounded Euler updates) and one-step direct endpoint refiner with exactly equal trainable architecture/parameter counts. Sparse GT-free proposal regions on unchanged512evaluation grid. Preset common recipe, no test tuning.',
        acceptance_stage='8 auxiliary proposer fits:4held subject-folds x2arms, seed91011. Each fit uses pair-excluded O3 training inputs; its held and validation O3/coarse producers also exclude held subjects. Resulting OOF correction examples train5acceptor seeds/arm. Shared OOF bank is conditional, not5new full pipelines.',
        final_stage='5matched seeds per arm:10new final proposers plus10acceptors. Fixed120epoch proposers/100epoch acceptors. Checkpoints and candidate strength/acceptance conservatism selected only by external validation. Ten full decisions lock before test scoring.',
        evaluation='Same frozen O3, RGB/GT/FOV/threshold>=.5 and thin-radius1. Primary:Flowaccepted-O3,Flowaccepted-Directaccepted,Directaccepted-O3, eachDice/clDice/thin:9-test Holm family; seed and4subject sensitivity separate. Raw arms descriptive. Identity fallback explicitly counted.',
        limitations='Previously inspected test cohort, reused4validation images, shared auxiliary bank/source models and source/deployment training-size shift. Parameter-matched, not equal inference calls. No claim of guaranteed test no-drop or Flow-specific benefit before matched control.',old_manuscript_results_unchanged=True))
    try:
        import torch
        from flow_matching_refiner.native_patch_data import load_native_dataset
        from flow_matching_refiner.local_acceptor_sources_v2 import build_sources
        torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available()
        status('loading_CHASE_only');print('SOURCE STAGE: CHASE only; building leakage-free inputs for the acceptance model.',flush=True)
        write_json(out/f'environment_{tag}.json',dict(torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),FP32=True,TF32=False))
        ref,audit=load_native_dataset('CHASE',out/'input_audit','cuda')
        for p,h in audit['source_sha256_before'].items():
            if p in protected:assert protected[p]==h
            protected[p]=h
        sources=build_sources(ref,out,tag,status)
        ready=out/f'workflow_ready_{tag}.json';status('verifying_local_model_implementation')
        while not ready.exists():
            print('Nested source stage complete. Local-model implementation/guards are being verified; waiting before further fitting.',flush=True);time.sleep(30)
        manifest=json.loads(ready.read_text());assert all(sha256_file(p)==h for p,h in manifest['source_sha256'].items())
        for p,h in manifest['source_sha256'].items():
            protected[p]=h;d=out/'source'/f'{Path(p).stem}_{tag}.py'
            if not d.exists():shutil.copy2(p,d)
        from flow_matching_refiner.local_acceptor_workflow import run_study
        result=run_study(ref,sources,out,tag,status)
        status('checking_previous_artifacts');assert all(sha256_file(p)==h for p,h in protected.items())
        write_json(out/f'protected_before_including_loaded_sources_{tag}.json',protected);write_json(out/f'completed_experiment_{tag}.json',dict(completed_at=stamp(),**result,old_files_unchanged=True,HRF_started=False,DRIVE_started=False));status('experiment_complete');print('CHASE LOCAL FLOW ACCEPTOR EXPERIMENT COMPLETE',str(out),flush=True)
    except BaseException:
        write_json(out/f'failed_{tag}.json',dict(failed_at=stamp(),traceback=traceback.format_exc()));status('failed');raise

if __name__=='__main__':main()
