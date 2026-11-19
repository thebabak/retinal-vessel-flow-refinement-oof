"""Shared provenance, checkpoint and resumable phase utilities."""
from __future__ import annotations
import json,hashlib,sys,csv,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from flow_matching_refiner.data import ROOT,sha256_file
from flow_matching_refiner.run_refinement_v3 import stamp,write_json as _original_write_json
SEEDS=(42,123,2026,3000,3001)

def load(path):
    # Antivirus/indexing and an atomic replace can briefly lock live JSON on Windows.
    for attempt in range(12):
        try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.05*2**attempt,.5))

def write_json(path,value):
    for attempt in range(12):
        try:return _original_write_json(Path(path),value)
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.05*2**attempt,.5))
def savecsv(path,rows):
    path=Path(path)
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def setup():
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
def manifest(paths):return {str(Path(p).resolve()):sha256_file(p) for p in paths}
def ensure_unchanged(protected):assert all(sha256_file(p)==h for p,h in protected.items())
def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

class Context:
    def __init__(self,root,phase,dataset):
        self.root=Path(root).resolve();assert self.root.parent==ROOT/'outputs'
        self.tag=self.root.name[-19:];self.phase=phase;self.dataset=dataset
        self.out=self.root/phase/dataset;self.out.mkdir(parents=True,exist_ok=True)
        self.state=dict(phase=phase,dataset=dataset,root=str(self.root),output=str(self.out),tag=self.tag,pid=os.getpid(),status='starting')
        self.secondary=os.environ.get('FLOW_COMPLETION_SECONDARY','')=='1'
        self.status_channel=os.environ.get('FLOW_COMPLETION_SECONDARY_CHANNEL',dataset)
        assert self.status_channel and all(c.isascii() and (c.isalnum() or c=='_') for c in self.status_channel)
        if self.secondary:
            self.state['secondary_independent_dataset_worker']=True
            self.state['secondary_status_channel']=self.status_channel
        self.protected={}
        source=self.out/'executed_source';source.mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob('*.py'):
            target=source/f'{path.stem}_{self.tag}.py'
            if not target.exists():shutil.copy2(path,target)
    def status(self,stage,**extra):
        self.state.update(stage=stage,updated_at=stamp(),**extra)
        write_json(self.out/f'status_{self.tag}.json',self.state)
        if self.secondary:
            write_json(self.root/f'secondary_job_{self.status_channel}_{self.tag}.json',self.state)
            write_json(ROOT/f'tmp/current_all_dataset_completion_secondary_{self.status_channel}.json',self.state)
        else:
            write_json(self.root/f'active_job_{self.tag}.json',self.state)
            write_json(ROOT/'tmp/current_all_dataset_completion_job.json',self.state)
    def protect(self,path):
        path=Path(path);h=sha256_file(path);self.protected[str(path)]=h;return h
    def done(self,extra):
        ensure_unchanged(self.protected)
        record=dict(completed_at=stamp(),phase=self.phase,dataset=self.dataset,old_sources_unchanged=True,**extra)
        write_json(self.out/f'protected_inputs_{self.tag}.json',self.protected)
        write_json(self.out/f'completed_{self.tag}.json',record);self.status('complete',status='complete',**extra)
        print('PHASE COMPLETE',self.phase,self.dataset,json.dumps(extra),flush=True)
        return record
