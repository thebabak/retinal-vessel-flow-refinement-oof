"""Dependent task pipeline: finish fits, replay, verify, draft, compile and render."""
from __future__ import annotations
import argparse,subprocess,time,sys,os,traceback,shutil
from pathlib import Path
from .common import ROOT,load,write_json,stamp,sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if not a.execute:return
    root=a.root.resolve();assert root.parent==ROOT/'outputs';tag=root.name[-19:]
    state=dict(root=str(root),tag=tag,pid=os.getpid(),stage='waiting_for_authorized_training_queue',status='running')
    def status(stage,**extra):
        state.update(stage=stage,updated_at=stamp(),**extra);write_json(root/f'postprocessing_status_{tag}.json',state)
    def run(stage,args):
        status(stage)
        with (root/f'{stage}_{tag}.log').open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','flow_matching_refiner.dataset_completion.'+args[0],*args[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            status(stage,child_pid=proc.pid)
            while proc.poll() is None:time.sleep(10)
            assert proc.returncode==0,(stage,proc.returncode)
    try:
        status(state['stage'])
        while not (root/f'queue_completed_{tag}.json').exists():
            if (root/f'queue_failed_{tag}.json').exists():status('training_queue_failure_requires_repair',status='failed');return
            time.sleep(10)
        if not (root/'analysis'/f'completed_analysis_{tag}.json').exists():run('new_all_dataset_analysis',['analyze','--execute','--root',str(root)])
        if not (root/'analysis'/f'independent_verification_{tag}.json').exists():run('independent_final_verification',['verify','--execute','--root',str(root)])
        run('assemble_new_MDPI_source',['delivery','--execute','--root',str(root)])
        build=load(ROOT/'tmp/current_all_dataset_MDPI_build.json');assert build['run']==str(root);state_path=Path(build['output'])/f'BUILD_STATE_{build["created_at"]}.json'
        run('compile_and_render_all_pages',['finalize','--prepare','--state',str(state_path)])
        status('compiled_draft_requires_visual_review',status='awaiting_visual_review',build_state=str(state_path),visual_review_completed=False,ZIP_packaging_after_visual_review=True)
    except BaseException:
        status('failed',status='failed',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
