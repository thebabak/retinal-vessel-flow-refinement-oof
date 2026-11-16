"""Local manifest/path plumbing. Import performs no I/O and imports no torch.

Prepared inputs are user-supplied private files; this package redistributes no
dataset, ground truth, upstream weight, or correspondence. The packaging layer
has not itself been used to produce the executed experiment's results.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path,PureWindowsPath
from types import SimpleNamespace
import hashlib,json,math

DATASETS=('CHASE','DRIVE','HRF')
SEEDS=(42,123,2026,3000,3001)
SIZES={'CHASE':(16,4,8),'DRIVE':(16,4,20),'HRF':(27,6,12)}
_CONFIG=None

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def local_path(root,relative):
    """Manifest paths must be portable relatives confined to the workspace."""
    if not isinstance(relative,str) or not relative or '\\' in relative or Path(relative).is_absolute() or PureWindowsPath(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError('Manifest paths must be nonempty relative paths')
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise ValueError('Manifest path escapes workspace')
    return path

@dataclass
class Configuration:
    workspace_root:Path
    output_root:Path
    manifest_path:Path
    input_root:Path
    manifest:dict
    manifest_sha256:str
    test_access_allowed:bool=False

def configure(workspace_root,output_root,input_manifest):
    global _CONFIG
    workspace=Path(workspace_root).resolve(strict=True)
    if not workspace.is_dir():raise ValueError('Workspace root must be a directory')
    output=Path(output_root).resolve();manifest_path=local_path(workspace,str(input_manifest))
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1 or set(manifest.get('datasets',{}))!=set(DATASETS):raise ValueError('Manifest must declare schema1 and all three intended datasets')
    fixed=json.loads((Path(__file__).with_name('FIXED_SPLITS.json')).read_text(encoding='utf-8'))
    source=local_path(workspace,manifest['input_root'])
    if source==workspace:raise ValueError('Private input_root must be a dedicated subdirectory')
    package=Path(__file__).resolve().parent
    if (output.is_relative_to(source) or source.is_relative_to(output) or output.is_relative_to(package) or package.is_relative_to(output)):
        raise ValueError('Output root overlaps private inputs or package source')
    for dataset in DATASETS:
        spec=manifest['datasets'][dataset];split=spec['split']
        if set(split)!={'train','validation','test'}:raise ValueError('Unexpected split keys')
        if split!=fixed[dataset]:raise ValueError('Partition IDs/order differ from the executed fixed protocol')
        if tuple(len(split[key]) for key in ('train','validation','test'))!=SIZES[dataset]:raise ValueError('Partition sizes differ from the executed fixed protocol')
        names=[name for key in ('train','validation','test') for name in split[key]]
        if any(not isinstance(n,str) or not n or '/' in n or '\\' in n or n in {'.','..'} for n in names) or len(set(names))!=len(names):raise ValueError('Image IDs must be unique and path-free across partitions')
        if set(spec['samples'])!=set(names):raise ValueError('Sample inventory differs from partition')
        if set(spec['baseline_references'])!=set(split['test']):raise ValueError('Baseline references must cover exactly the held-out images')
        try:cutoff=float(spec['thin_cutoff'])
        except (TypeError,ValueError):raise ValueError('Supply the fixed training-derived thin cutoff') from None
        if not math.isfinite(cutoff) or cutoff<0:raise ValueError('Missing fixed training-derived thin cutoff')
        if not spec.get('ground_truth_provenance') or spec['ground_truth_provenance'].startswith('REPLACE'):raise ValueError('Explicit GT provenance/limitations required')
        sample_paths=[]
        for record in list(spec['samples'].values())+list(spec['baseline_references'].values()):
            path=local_path(workspace,record['path'])
            if not path.is_relative_to(source):raise ValueError('Prepared artifact is outside dedicated private input root')
            if len(record.get('sha256',''))!=64 or any(c not in '0123456789abcdef' for c in record['sha256']):raise ValueError('Every prepared artifact requires its SHA256')
        for name,record in spec['samples'].items():
            sample_paths.append(local_path(workspace,record['path']))
            if dataset=='CHASE' and (not isinstance(record.get('subject'),int) or record['subject']!=int(name.split('_')[1][:2])):raise ValueError('CHASE subject IDs must match the fixed filename identities')
        if len(set(sample_paths))!=len(sample_paths):raise ValueError('Input files may not alias distinct image IDs')
        if dataset=='CHASE':
            groups=[{spec['samples'][name]['subject'] for name in split[key]} for key in ('train','validation','test')]
            if any(groups[a]&groups[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('CHASE subject partitions overlap')
    _CONFIG=Configuration(workspace,output,manifest_path,source,manifest,sha(manifest_path));return _CONFIG

def configuration():
    if _CONFIG is None:raise RuntimeError('Configure --workspace-root,--output-root,--input-manifest before accessing local files')
    return _CONFIG

def configure_backend_policy():
    """Make the executed private loader's inherited policy explicit here.

    This runs only inside an explicitly executed CLI, before any official model
    is constructed or fit. Imports stay inert and training/inference math is
    unchanged. The four assignments match all three producer loader prefixes.
    No stronger global deterministic-algorithms policy is introduced.
    """
    import torch
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return dict(cudnn_deterministic=bool(torch.backends.cudnn.deterministic),cudnn_benchmark=bool(torch.backends.cudnn.benchmark),cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),policy_source='Explicit copy of the four assignments inherited by executed wrappers from retained producer loader prefixes',strict_global_deterministic_algorithms_policy_modified=False)

def checked_output(path):
    c=configuration();value=Path(path)
    target=value.resolve() if value.is_absolute() else (c.output_root/value).resolve()
    if target==c.output_root or not target.is_relative_to(c.output_root):raise ValueError('Run output must be strictly inside --output-root')
    if target.is_relative_to(c.input_root) or c.input_root.is_relative_to(target) or c.manifest_path.is_relative_to(target):raise ValueError('Run output overlaps source inputs')
    return target

def verify_snapshot(source,expected_commit):
    source=Path(source).resolve();manifest=json.loads((source/'SOURCE.json').read_text(encoding='utf-8'))
    if manifest['commit']!=expected_commit:raise RuntimeError('Official snapshot commit mismatch')
    for relative,expected in manifest['files'].items():
        if sha(local_path(source,relative))!=expected:raise RuntimeError('Official snapshot bytes changed: '+relative)
    return manifest

def _verified_record(record):
    c=configuration();path=local_path(c.workspace_root,record['path'])
    if sha(path)!=record['sha256']:raise RuntimeError('Prepared artifact bytes differ from declared source')
    return path

def _score(p,g,fov=None):
    # Same >=.5 pixel bookkeeping as the executed CHASE/DRIVE/HRF loaders.
    if fov is not None:p=p[fov];g=g[fov]
    p=p>=.5;g=g>=.5
    tp=(p&g).sum().item();fp=(p&~g).sum().item();fn=(~p&g).sum().item();tn=(~p&~g).sum().item()
    return dict(dice=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),accuracy=(tp+tn)/(tp+fp+fn+tn),tp=tp,fp=fp,fn=fn,tn=tn)

class LazySamples(Mapping):
    def __init__(self,dataset):self.dataset=dataset;self.cache={}
    def __iter__(self):return iter(configuration().manifest['datasets'][self.dataset]['samples'])
    def __len__(self):return len(configuration().manifest['datasets'][self.dataset]['samples'])
    def __getitem__(self,name):
        c=configuration();spec=c.manifest['datasets'][self.dataset]
        if name in spec['split']['test'] and not c.test_access_allowed:raise RuntimeError('Test input/GT access held until all15final choices are globally locked')
        import numpy as np,torch
        if name not in self.cache:
            record=spec['samples'][name];path=_verified_record(record)
            with np.load(path,allow_pickle=False) as stored:
                values={key:stored[key].copy() for key in ('x','rgb','g','fov')}
            for key,shape in [('x',(1,512,512)),('rgb',(3,512,512)),('g',(1,512,512)),('fov',(1,512,512))]:
                a=values[key]
                if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid prepared shape/value: '+key)
                if key in ('x','rgb') and (a.dtype!=np.float32 or np.any((a<0)|(a>1))):raise ValueError('Input must be retained FP32[0,1]')
                if key in ('g','fov') and not np.all((a==0)|(a==1)):raise ValueError('GT/FOV must be binary')
            if not values['fov'].any():raise ValueError('FOV cannot be empty')
            if self.dataset=='CHASE' and not values['fov'].all():raise ValueError('Executed CHASE region is the whole512grid')
            self.cache[name]={key:torch.from_numpy(a.astype(bool) if key=='fov' else a.astype(np.float32,copy=False)) for key,a in values.items()}
            if self.dataset=='CHASE':self.cache[name]['subject']=record['subject']
        return self.cache[name]

def load_data(dataset):
    c=configuration();spec=c.manifest['datasets'][dataset]
    return SimpleNamespace(dataset=dataset,manifest=spec['split'],data=LazySamples(dataset),score=_score)

def baseline_archive(dataset,name):
    c=configuration()
    if not c.test_access_allowed:raise RuntimeError('Paired test reference access is held')
    return _verified_record(c.manifest['datasets'][dataset]['baseline_references'][name])

def thin_cutoff(dataset):return float(configuration().manifest['datasets'][dataset]['thin_cutoff'])

def authorize_test_inputs(output,tag,datasets,seeds,finalize_requested):
    """The public adapter adds lazy test I/O behind the same global choice gate."""
    c=configuration()
    if not finalize_requested or tuple(datasets)!=DATASETS or tuple(seeds)!=SEEDS:raise RuntimeError('All intended datasets/seeds and explicit finalization required')
    root=checked_output(output);lock=json.loads((root/f'all_selections_before_test_{tag}.json').read_text())
    for dataset in DATASETS:
        choices=lock['decisions'][dataset]
        if sorted(x['seed'] for x in choices)!=sorted(SEEDS):raise RuntimeError('Final choice inventory incomplete')
        config=root/dataset/f'configuration_before_dataset_final_fitting_{tag}.json'
        if sha(config)!=lock['dataset_configuration_locks_sha256'][dataset]:raise RuntimeError('Dataset configuration differs from global lock')
        for choice in choices:
            folder=root/dataset/'final'/f'seed_{choice["seed"]}'
            if json.loads((folder/f'selection_locked_{tag}.json').read_text())!=choice:raise RuntimeError('Final selection differs from global lock')
            checkpoint=local_path(folder,choice['checkpoint'])
            if sha(checkpoint)!=choice['checkpoint_sha256']:raise RuntimeError('Final checkpoint differs from lock')
    if sha(c.manifest_path)!=c.manifest_sha256:raise RuntimeError('Input manifest changed since fitting began')
    c.test_access_allowed=True
