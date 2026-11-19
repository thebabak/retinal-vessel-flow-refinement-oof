"""Serial GPU queue. Failed phases halt; scientific completion is never inferred."""
from __future__ import annotations
import argparse,subprocess,sys,time,os,json
from pathlib import Path
from .common import ROOT,load,write_json,stamp

def progress(root,jobs,tag):
    completed=sum((root/j['phase']/j['dataset']/f'completed_{tag}.json').exists() for j in jobs);active=root/f'active_job_{tag}.json';state=load(active) if active.exists() else {}
    lines=[f'Updated (Asia/Tehran): {stamp()}',f'Completed phases: {completed} / {len(jobs)}',f'Active: {state.get("phase", "")} / {state.get("dataset", "")}',f'Stage: {state.get("stage", "")}',f'Training seed: {state.get("seed", "")}; epoch: {state.get("epoch", "")}',f'Completed fits in active phase: {state.get("fits_completed", "")}', 'Previous manuscript/results are preserved. Settings are selected on validation only.', 'Running/planned work is not a completed result. Final tables and MDPI files follow verified completion.']
    (root/'PROGRESS_CURRENT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if not a.execute:return
    root=a.root.resolve();tag=root.name[-19:];assert root.parent==ROOT/'outputs'
    plan=root/f'queue_{tag}.json'
    while True:
        jobs=load(plan)['jobs'];pending=[j for j in jobs if not (root/j['phase']/j['dataset']/f'completed_{tag}.json').exists()]
        progress(root,jobs,tag)
        if not pending:
            if load(plan).get('sealed'):
                write_json(root/f'queue_completed_{tag}.json',dict(completed_at=stamp(),jobs=jobs,all_phase_markers_verified=True));return
            time.sleep(5);continue
        j=pending[0];out=root/j['phase']/j['dataset'];out.mkdir(parents=True,exist_ok=True)
        current=out/f'status_{tag}.json'
        if current.exists():
            status=load(current);pid=status.get('pid')
            # Adopt the already-authorized first worker; do not launch a duplicate.
            if status.get('status')!='failed' and pid:
                alive=subprocess.run(['powershell','-NoProfile','-Command',f'if (Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue) {{exit 0}} else {{exit 1}}'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW).returncode==0
                if alive:time.sleep(10);continue
        failed=out/f'failed_{tag}.json'
        if failed.exists():
            write_json(root/f'queue_failed_{tag}.json',dict(job=j,failed_at=stamp(),failure=str(failed)));return
        logfile=root/f'{j["phase"]}_{j["dataset"]}_{tag}.log'
        with logfile.open('ab',buffering=0) as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','flow_matching_refiner.dataset_completion.worker','--execute','--root',str(root),'--phase',j['phase'],'--dataset',j['dataset']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            write_json(ROOT/'tmp/current_all_dataset_completion.json',dict(root=str(root),tag=tag,controller_pid=os.getpid(),active_pid=proc.pid,active_phase=j['phase'],active_dataset=j['dataset'],log=str(logfile),jobs=len(jobs)))
            while proc.poll() is None:progress(root,jobs,tag);time.sleep(10)
            result=proc.returncode
        if result or not (out/f'completed_{tag}.json').exists():
            write_json(root/f'queue_failed_{tag}.json',dict(job=j,exit_code=result,failed_at=stamp(),log=str(logfile)));return
if __name__=='__main__':main()
