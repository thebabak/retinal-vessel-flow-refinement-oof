"""Public numerical table reanalysis only; default inert.

Consumes three published scalar CSVs and a relative SHA-bound input manifest.
No private attestation, training, model, prediction, image, ZIP, source-data
verification or network is required or performed.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath,PureWindowsPath
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np
import scipy

HERE=Path(__file__).resolve().parent
BINDINGS=HERE/'code_math_bindings_2026-11-16_21-47-36.json'
DATASETS=('CHASE','DRIVE','HRF')
SEEDS=(42,123,2026,3000,3001)
N_IMAGES=dict(CHASE=8,DRIVE=20,HRF=12)

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:return json.load(f)

def relative_file(base,value):
    require(isinstance(value,str) and value and '\\' not in value and '\0' not in value,'Expected a relative POSIX input path')
    p=PurePosixPath(value)
    require(not p.is_absolute() and not PureWindowsPath(value).drive and p.as_posix()==value and
            all(s not in ('.','..') and ':' not in s for s in p.parts),'Unsafe/absolute input path')
    base=Path(base).resolve();candidate=base/p
    for parent in [candidate,*candidate.parents]:
        if parent==base:break
        require(not parent.is_symlink() and not getattr(parent,'is_junction',lambda:False)(),'Linked input path is unsupported')
    require(candidate.resolve().is_relative_to(base),'Input path escapes manifest directory')
    require(candidate.is_file(),'Published input file is missing: '+value)
    return candidate

def load_core():
    bindings=read_json(BINDINGS)
    path=relative_file(HERE,bindings['core_name'])
    require(sha(path)==bindings['core_sha256'],'Public paired-math code changed')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for item in bindings['extracted_functions']:
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==item['function'])
        actual=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        require(actual==item['AST_sha256'],'A paired-math function differs from its preserved definition')
    spec=importlib.util.spec_from_file_location('public_scalar_math',path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    return core,bindings

def validate_manifest(manifest,base,bindings):
    require(manifest.get('schema_version')==1 and manifest.get('numerical_table_reanalysis_only') is True,'Wrong numerical manifest schema')
    require(manifest.get('ready') is True,'Input template is deliberately incomplete; published CSV bindings are not ready')
    plan_path=relative_file(base,manifest['plan']['path'])
    require(sha(plan_path)==manifest['plan']['sha256'],'Public mathematical plan SHA256 differs')
    plan=read_json(plan_path)
    mathematical={k:plan[k] for k in ('datasets','seeds','ablation_contrasts','external_contrasts','families')}
    signature=hashlib.sha256(json.dumps(mathematical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(signature==bindings['registered_math_sha256'],'Registered 57/15 mathematical families changed')
    require(tuple(plan['datasets'])==DATASETS and tuple(plan['seeds'])==SEEDS,'Dataset/seed scope changed')
    require([d['name'] for d in manifest['datasets']]==list(DATASETS),'All three datasets in the registered order are required')
    metrics=manifest['descriptive_metrics']
    require(metrics and 'dice' in metrics and len(metrics)==len(set(metrics)),'Invalid descriptive metric list')
    sources=[]
    for dataset in manifest['datasets']:
        d=dataset['name'];images=dataset['image_ids']
        require(dataset['expected_images']==N_IMAGES[d] and len(images)==len(set(images))==N_IMAGES[d] and
                images==sorted(images),'Full sorted image inventory is required: '+d)
        csv_spec=dataset['csv']
        require(isinstance(csv_spec.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',csv_spec['sha256']),
                'Missing published CSV SHA256: '+d)
        require(type(csv_spec.get('size_bytes')) is int and csv_spec['size_bytes']>0,'Missing CSV size: '+d)
        path=relative_file(base,csv_spec['path']);require(path.suffix.lower()=='.csv','Only scalar CSV inputs are supported')
        require(path.stat().st_size==csv_spec['size_bytes'] and sha(path)==csv_spec['sha256'],'CSV size/SHA256 differs: '+d)
        sources.append(dict(dataset=d,path=path,relative_path=csv_spec['path'],sha256=csv_spec['sha256'],size_bytes=csv_spec['size_bytes']))
    require(len({s['path'] for s in sources})==3,'Each dataset requires a distinct published CSV')
    return plan,sources,plan_path

def load_scalar_records(manifest,plan,sources):
    needed={'Baseline'}|{arm for key in ('ablation_contrasts','external_contrasts') for c in plan[key] for arm in c['coefficients']}
    metrics=manifest['descriptive_metrics'];records={};tables={}
    for dataset,source in zip(manifest['datasets'],sources):
        d=dataset['name'];images=dataset['image_ids'];row_count=0
        with source['path'].open(encoding='utf-8-sig',newline='') as stream:
            reader=csv.DictReader(stream)
            require({'arm','refiner_seed','image','subject','output',*metrics}<=set(reader.fieldnames or []),
                    'Scalar CSV columns are incomplete: '+d)
            for row in reader:
                row_count+=1;arm=row['arm'];kind=row['output'];seed=int(row['refiner_seed']);image=row['image']
                require(arm in needed and kind in ('raw','blend') and image in images,'Unregistered arm/output/image: '+d)
                require((seed==-1 and kind=='raw') if arm=='Baseline' else seed in SEEDS,'Invalid matched seed/output')
                key=d,arm,kind,seed,image;require(key not in records,'Duplicate scalar image/seed row')
                values={m:None if row[m] in ('',None) else float(row[m]) for m in metrics}
                require(all(v is None or np.isfinite(v) for v in values.values()),'Nonfinite scalar metric')
                require(values['dice'] is not None and 0<=values['dice']<=1,'Raw Dice is missing or invalid')
                subject=row['subject'] if d=='CHASE' else image
                records[key]=dict(dataset=d,arm=arm,output=kind,seed=seed,image=image,subject=subject,metrics=values)
        for arm in needed:
            kinds={k[2] for k in records if k[0]==d and k[1]==arm}
            require('raw' in kinds,'Required raw arm is missing: '+d+' '+arm)
            for kind in kinds:
                seeds=(-1,) if arm=='Baseline' else SEEDS
                expected={(d,arm,kind,s,i) for s in seeds for i in images}
                require({k for k in records if k[:3]==(d,arm,kind)}==expected,
                        'Full matched arm/seed/image coverage is required; families cannot shrink')
        subjects={i:records[d,'Baseline','raw',-1,i]['subject'] for i in images}
        groups=defaultdict(list)
        for image,subject in subjects.items():groups[subject].append(image)
        if d=='CHASE':
            require(len(groups)==4 and all(len(g)==2 for g in groups.values()),'CHASE requires four two-eye subject units')
            require(all({i[-1] for i in pair}=={'L','R'} for pair in groups.values()),'CHASE eye pairing is invalid')
        for key,r in records.items():
            if key[0]==d:require(r['subject']==subjects[r['image']],'Subject labels differ within matched CSV')
        tables[d]=dict(images=images,subjects=subjects,groups=dict(groups))
        source['rows']=row_count
    return records,tables

def write_json(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def write_csv(path,rows):
    require(bool(rows),'Empty reanalysis table')
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def execute(manifest_path,output):
    initial_manifest_sha=sha(manifest_path)
    core,bindings=load_core();manifest=read_json(manifest_path);base=manifest_path.resolve().parent
    require(not output.exists(),'Use a NEW output directory; existing results are preserved')
    require(re.search(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}',output.name),'Output directory needs a Tehran timestamp')
    plan,sources,plan_path=validate_manifest(manifest,base,bindings)
    initial_plan_sha=manifest['plan']['sha256']
    records,tables=load_scalar_records(manifest,plan,sources)
    result=core.analyze(records,tables,plan,manifest['descriptive_metrics'])
    require(all(sha(s['path'])==s['sha256'] for s in sources),'CSV bytes changed during numerical reanalysis')
    require(sha(manifest_path)==initial_manifest_sha and sha(plan_path)==initial_plan_sha,'Manifest/plan changed during reanalysis')
    tag=datetime.now(ZoneInfo('Asia/Tehran')).strftime('%Y-%m-%d_%H-%M-%S')
    output.mkdir(parents=True,exist_ok=False)
    public_sources=[{k:v for k,v in s.items() if k!='path'} for s in sources]
    report=dict(created_at=datetime.now(ZoneInfo('Asia/Tehran')).isoformat(),
        numerical_table_reanalysis_only=True,fresh_training_performed=False,predictions_evaluated=False,
        dataset_source_or_execution_provenance_verified=False,private_authorization_records_required=False,
        input_manifest_sha256=initial_manifest_sha,plan_sha256=initial_plan_sha,
        code_math_sha256=bindings['core_sha256'],sources=public_sources,
        software=dict(Python=sys.version.split()[0],NumPy=np.__version__,SciPy=scipy.__version__),
        interpretation='Post-hoc exploratory conditional statistics from supplied scalar CSVs; not new measured efficacy or independent verification of seed roles, patients, predictions or source labels.',
        score_averaging='Average five individual metric scores per image, not probability maps; no ensemble is evaluated.',
        baseline='One fixed baseline score per image; its training-seed SD is unavailable.',
        units='Five matched training-seed labels; CHASE four two-eye subject labels; DRIVE/HRF twenty/twelve image units with patient independence unverified.',
        multiplicity='Complete separate 57-test ablation and 15-test external families, separately for t/exact W and for seed/patient-image sensitivity.',
        **result)
    write_json(output/f'numerical_reanalysis_{tag}.json',report)
    write_json(output/f'input_manifest_snapshot_{tag}.json',manifest)
    write_json(output/f'public_mathematical_plan_{tag}.json',plan)
    write_csv(output/f'paired_tests_{tag}.csv',[core.flat_test(r) for r in result['tests']])
    for key,stem in [('per_image_contrasts','per_image_Dice_contrasts'),('per_image_metrics','per_image_metric_scores'),
                     ('seed_metrics','training_seed_metric_means'),('mean_SD','descriptive_mean_SD')]:
        write_csv(output/f'{stem}_{tag}.csv',result[key])
    print(json.dumps(dict(status='NUMERICAL_TABLE_REANALYSIS_COMPLETE',output=str(output),tests=len(result['tests']),
                         model_or_source_verification=False),indent=2))
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='INERT_NO_CSV_READ',numerical_table_reanalysis_only=True,
            note='Explicit --execute, complete relative SHA manifest and a new timestamped output are required. No private gates are altered.'),indent=2));return 0
    require(args.output is not None,'A new timestamped output is required')
    execute(args.manifest,args.output);return 0

if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,KeyError,FileNotFoundError,csv.Error) as error:print('BLOCKED: '+str(error));sys.exit(2)
