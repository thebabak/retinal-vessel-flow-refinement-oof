"""Build a new MDPI draft only after every authorized phase and replay completes."""
from __future__ import annotations
import argparse,csv,json,re,shutil,subprocess,zipfile,hashlib,os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from .common import ROOT,load,write_json,stamp,sha256_file,csvrows,savecsv
KEYS=('dice','precision','cldice','thin_skeleton_recall')
DATASETS=('CHASE','DRIVE','HRF')

def esc(value):
    return ''.join({'&':r'\&','%':r'\%','_':r'\_','#':r'\#'}.get(c,c) for c in str(value))

def fmt(x):
    if x in (None,'','None'):return r'---'
    v=float(x)
    if 0<abs(v)<1e-5:
        a,b=f'{v:.3e}'.split('e');return r'\ensuremath{'+a+r'\times10^{'+str(int(b))+'}}'
    return f'{v:.6f}'

def pvalue(x):
    if x in (None,'','None'):return r'---'
    return r'$<10^{-6}$' if float(x)<1e-6 else f'{float(x):.6f}'

def prose(text):
    macros={}
    def protect(m):
        key='PROSEMACRO'+chr(65+len(macros)//26)+chr(65+len(macros)%26);macros[key]=m.group(0);return key
    text=re.sub(r'\\(?:cite|ref|label|eqref)\{[^}]+\}',protect,text)
    text=re.sub(r'([A-Za-z])(?=\d)',r'\1 ',text)
    text=re.sub(r'(\d)(?=[A-Za-z])',r'\1 ',text)
    for symbol in ('F','O'):text=re.sub(r'\b'+symbol+r' ([1-9])\b',symbol+r'\1',text)
    text=text.replace('ResNet 18','ResNet18')
    for before,after in [('FP 32','FP32'),('SHA 256','SHA256'),('CHASE-DB 1','CHASE-DB1')]:text=text.replace(before,after)
    for key,value in macros.items():text=text.replace(key,value)
    return text

def cell(r):return '$'+fmt(r['mean'])+(r'\pm '+fmt(r['SD']) if r['SD'] not in (None,'','None') else '')+'$'

def order_bibliography(source):
    """Number the unchanged manual entries by their first manuscript citation."""
    begin=source.index(r'\begin{thebibliography}')
    end=source.index(r'\end{thebibliography}',begin)
    block=source[begin:end]
    matches=list(re.finditer(r'\\bibitem\{([^}]+)\}',block))
    entries={m.group(1):block[m.start():(matches[i+1].start() if i+1<len(matches) else len(block))] for i,m in enumerate(matches)}
    assert len(entries)==len(matches)
    cited=[]
    for m in re.finditer(r'\\cite(?:t|p)?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',source[:begin]):
        for key in m.group(1).split(','):
            key=key.strip();assert key in entries,key
            if key not in cited:cited.append(key)
    order=cited+[key for key in entries if key not in cited]
    revised=block[:matches[0].start()]+''.join(entries[key] for key in order)
    reordered=list(re.finditer(r'\\bibitem\{([^}]+)\}',revised))
    unchanged=[revised[m.start():(reordered[i+1].start() if i+1<len(reordered) else len(revised))] for i,m in enumerate(reordered)]
    assert sorted(entries.values())==sorted(unchanged)
    return source[:begin]+revised+source[end:]

def longtable(caption,label,headers,rows,widths,rowstretch=1.12):
    spec='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+'cm}' for w in widths)+'@{}';head=' & '.join(headers)+r'\\'
    capwidth=r'\dimexpr '+f'{sum(widths):.3f}cm+{6*(len(widths)-1)}pt'+r'\relax'
    wide_caption=r'\makeatletter\long\def\caption@LT@make#1{\LT@mcol\LT@cols{@{}l@{}}{\parbox[t]{\LTcapwidth}{#1}}}\makeatother'
    return '\n'+r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{'+str(rowstretch)+r'}\setlength{\LTleft}{-\extralength}\setlength{\LTright}{0pt plus1fill}\setlength{\LTcapwidth}{'+capwidth+r'}\captionsetup[longtable]{margin=0pt,width=\LTcapwidth,singlelinecheck=false,justification=raggedright}'+wide_caption+ '\n'+r'\begin{longtable}{'+spec+'}\n'+r'\caption{'+prose(caption)+r'}\label{'+label+r'}\\'+'\n'+r'\toprule'+'\n'+head+'\n'+r'\midrule\endfirsthead'+'\n'+r'\toprule'+'\n'+head+'\n'+r'\midrule\endhead'+'\n'+r'\bottomrule\endfoot'+'\n'+'\n'.join(' & '.join(map(str,r))+r'\\' for r in rows)+'\n'+r'\end{longtable}\endgroup'+'\n'

def figure(filename,label,caption):
    return '\n'+r'\begin{figure}[htbp]\begin{adjustwidth}{-\extralength}{0cm}\centering'+'\n'+r'\includegraphics[width=\fulllength]{figures/'+filename+'}\n'+r'\end{adjustwidth}\caption{'+prose(caption)+r'}\label{'+label+r'}\end{figure}'+'\n'

def diagnostic_tables(root,tag):
    """Explicitly expose the completed validation screens, without test claims."""
    import numpy as np
    used=[];text=r'\subsection{Validation Diagnostics and Rejected Candidates}'+'\n'
    def read(p):
        p=Path(p);used.append(p);return load(p)
    encoded=[];solver=[];geometry=[];guard=[];calib=[];gap=[];bias=[]
    for ds in DATASETS:
        if ds=='CHASE':
            p=ROOT/'outputs/flow_matching_v2_2026-11-15_01-12-53/diagnostics/v1_validation_diagnostic_2026-11-15_01-14-22/diagnosis_2026-11-15_01-14-22.json'
        else:
            pp=list((root/'encoded_diagnosis'/ds).glob('v1_validation_diagnostic_*/diagnosis_*.json'));assert len(pp)==1; p=pp[0]
        d=read(p)
        for mode,value in d['validation_mean_dice'].items():encoded.append([ds,esc(mode),fmt(value)])
        if ds=='HRF':p=root/'anchored'/ds/f'validation_diagnosis_{tag}.json'
        else:
            a=load(ROOT/'tmp/current_native_patch_flow_v2_delivery.json');folder=Path(a.get('output',a.get('experiment','')));p=folder/f'{ds}_{folder.name[-19:]}'/f'validation_diagnosis_{folder.name[-19:]}.json'
        d=read(p)
        for r in d['solver_probe']:solver.append([ds,r['steps'],fmt(r['validation_mean_dice'])])
        p=ROOT/'outputs/geometry_flow_CHASE_pilot_2026-11-18_01-11-21/diagnosis/diagnosis_2026-11-18_01-11-21.json' if ds=='CHASE' else root/'geometry'/ds/'diagnosis'/f'diagnosis_{tag}.json'
        d=read(p);geometry.append([ds,d['total']['added_FP'],d['total']['boundary_adjacent_added_FP'],fmt(d['boundary_adjacent_fraction'] if d['total']['added_FP'] else None)])
        p=ROOT/'outputs/precision_guarded_subject_CV_CHASE_2026-11-18_02-14-28/existing_validation_precision_guard_selections_2026-11-18_02-14-28.json' if ds=='CHASE' else root/'precision_reselection'/ds/f'precision_guard_choices_{tag}.json'
        d=read(p)
        for mode in ('flow','direct'):
            dd=[x for x in d['decisions'] if x['mode']==mode];assert len(dd)==5
            guard.append([ds,mode,sum(x['selection']['beta']>0 for x in dd),5-sum(x['selection']['beta']>0 for x in dd)])
        if ds=='CHASE':
            folder=ROOT/'outputs/validation_calibration_CHASE_2026-11-18_11-01-15';t='2026-11-18_11-01-15';g=read(folder/f'pilot_gate_locked_{t}.json');decisions=g['decisions'];bases=[read(folder/f'fold{f}_Option3_calibration_locked_{t}.json') for f in (1,2,3)];passed=g['passed']
        else:
            d=read(root/'calibration'/ds/f'all_validation_calibration_choices_{tag}.json');decisions=d['decisions'];bases=d['baseline_calibrations'];passed=d['pilot_gate_passed']
        systems=[('O3 threshold0.5',[b['fixed_threshold_metrics'] for b in bases],0),('Calibrated O3',[b['calibrated_selection']['metrics'] for b in bases],0)]
        for mode in ('flow','direct'):
            dd=[x for x in decisions if x['mode']==mode and x['seed']==42];assert len(dd)==3
            systems.append((mode+' calibration pilot',[x['selection']['metrics'] for x in dd],sum(x['selection']['beta']>0 for x in dd)))
        for label,mm,nonzero in systems:calib.append([ds,esc(label),nonzero,*(fmt(np.mean([m[k] for m in mm])) for k in KEYS),'pass' if passed else 'fail'])
        p=ROOT/'outputs/connection_flow_diagnosis_CHASE_2026-11-18_12-24-44/diagnosis_results_2026-11-18_12-24-44.json' if ds=='CHASE' else root/'gap'/ds/f'diagnosis_results_{tag}.json'
        d=read(p)
        for f in d['folds']:gap.append([ds,f['outer_fold'],fmt(f.get('bounded_gap_fraction',f.get('bounded_gap_fraction_of_missed_thin'))),fmt(f.get('candidate_coverage',f.get('candidate_fraction_of_missed_thin'))),'yes' if f.get('geometric_gate',f.get('geometric_feasibility_passed')) else 'no'])
        p=ROOT/'outputs/validation_calibration_probe_2026-11-17_19-39-48/validation_calibration_candidates_2026-11-17_19-39-48.csv' if ds=='CHASE' else root/'bias_probe'/ds/f'validation_calibration_candidates_{tag}.csv';used.append(p);rr=csvrows(p);assert len(rr)==24
        def yes(v):return str(v).lower() in ('true','1')
        bias.append([ds,len(rr),sum(yes(r['pareto']) for r in rr),sum(yes(r['full_retention']) for r in rr)])
    text+=longtable('Encoded-representation diagnostics: validation Dice only, one trained model. AE(GT) is a representation probe, not an attainable-performance upper bound. Euler100 and zero-noise outputs are diagnostic replays, not replacement test models.','tab:encoded-diagnostics',['Data','Probe','Validation Dice'],encoded,[1.5,9.0,3.5])
    text+=longtable('Native F1 solver probe, validation only, seed42. The training and deployed solver recipe is unchanged by this diagnostic.','tab:solver-diagnostics',['Data','Euler updates','Validation Dice'],solver,[2.0,4.0,4.0])
    text+=longtable('Validation added-false-positive diagnosis of the preceding seed42 rollout. Boundary adjacency means distance at most1.5evaluation pixels from O3. A fraction is unavailable when no FP is added. Counts are restricted to the retained FOV; they do not establish anatomical false bridges. The geometry pilot extension is run regardless of whether its original CHASE adjacency heuristic holds.','tab:boundary-diagnosis',['Data','Added FP','Boundary-adjacent FP','Fraction'],geometry,[1.5,3.0,4.5,3.0])
    text+=longtable('Retrospective four-metric precision guard on the fixed-cohort F8 validation grids. Each mode has five stored training seeds. Identity is an explicit rejection outcome, not a learned gain.','tab:precision-reselection',['Data','Arm','Nonzero choices','Identity choices'],guard,[1.5,2.5,4.0,4.0])
    text+=longtable('Threshold-calibration pilot: equal-fold validation means for seed42, not test scores. NZ counts the three fold-specific nonzero Flow/direct corrections. The same aggregate four-metric Flow gate determines expansion to the other stored seeds; calibrated O3 gains alone are not Flow gains.','tab:calibration-validation',['Data','System','NZ folds','Dice','Precision','clDice','Thin recall','Flow gate'],calib,[1.0,2.8,.8,2.1,2.1,2.1,2.1,.9])
    text+=longtable('Fresh-fold connection-screen geometry, validation only. Bounded-gap share and candidate coverage are fractions of missed thin-skeleton points. A geometric pass needs at least0.20and0.10respectively; the additional oracle four-metric rule is reported in the source evidence. No connection-Flow model is fitted or scored as a consequence of an oracle gain.','tab:connection-screen',['Data','Outer fold','Bounded-gap share','Candidate coverage','Geometric pass'],gap,[1.4,1.7,3.6,3.6,3.0])
    text+=longtable('The24-point validation strength/bias probe: counts of eligible settings under the original tolerance and prior-structure retention rules. No new test outcome or updated weight is inferred.','tab:bias-probe',['Data','Candidates','Pareto-eligible','Full-retention candidates'],bias,[1.5,2.0,4.0,5.0])
    return text,used

def scientific_plots(means,rows,tests,figdir,tag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    groups=defaultdict(dict)
    for r in means:groups[tuple(r[k] for k in ('cohort','dataset','phase','method','mode','output'))][r['metric']]=r
    labels={'dice':'Dice','precision':'Precision','cldice':'clDice','thin_skeleton_recall':'Thin recall'}
    initial_labels={'Frozen baseline':'B','direct':'D','endpoint_iterative':'I','full_fm':'FM','selective_no_gate':'S','esfm':'ES','direct_thin_anchor':'O1','esfm_endpoint_anchor':'O2','native_patch_direct':'O3'}
    phase_labels={'native_flow':'F1','anchored':'F2','fully_oof':'F3','topology':'F4','retained':'F5','local_acceptor':'F6','rollout':'F7','context':'F8'}
    files=[]
    for phase_group,phases in [('initial',{'baseline','initial_controls','initial_objectives','O3'}),('followup',{'native_flow','anchored','fully_oof','topology','retained','local_acceptor','rollout','context'})]:
        fig,axes=plt.subplots(3,4,figsize=(7.2,8.0))
        for i,ds in enumerate(DATASETS):
            items=[(key,mm) for key,mm in groups.items() if key[0]=='retained_fixed_split' and key[1]==ds and key[2] in phases and key[5] in (('raw','anchor') if phase_group=='initial' else ('selected','accepted','blend'))]
            phase_order={p:j for j,p in enumerate(('baseline','initial_controls','initial_objectives','O3','native_flow','anchored','fully_oof','topology','retained','local_acceptor','rollout','context'))}
            items.sort(key=lambda a:(phase_order[a[0][2]],a[0][3],a[0][4],a[0][5]))
            for j,k in enumerate(KEYS):
                ax=axes[i,j];names=[initial_labels[key[3]] if phase_group=='initial' else phase_labels[key[2]]+('-F' if key[4]=='flow' else '-D' if key[4]=='direct' else '') for key,_ in items];v=[float(mm[k]['mean']) for _,mm in items];sd=[float(mm[k]['SD'] or 0.) for _,mm in items];ax.errorbar(range(len(v)),v,yerr=sd,fmt='o',capsize=2,ms=3,color='#146c94');ax.set_xticks(range(len(v)),names,rotation=90,ha='center',fontsize=7);ax.set_title(ds+' / '+labels[k],fontsize=8);ax.tick_params(axis='y',labelsize=7);ax.ticklabel_format(axis='y',useOffset=False,style='plain');ax.grid(axis='y',alpha=.25)
        fig.suptitle('Fixed-split '+phase_group+' comparisons: mean +/- training-seed SD',fontsize=9);fig.tight_layout(rect=(0,0,1,.97));name=f'{phase_group}_three_dataset_scores_{tag}.pdf';fig.savefig(figdir/name);plt.close(fig);files.append(name)
    # Prespecified display cases: first image ID in each original test split,
    # seed42, selected context Flow/direct, with no score-based case selection.
    fig,axes=plt.subplots(3,6,figsize=(7.2,5.4))
    from PIL import Image
    import cv2
    for i,ds in enumerate(DATASETS):
        rr=[r for r in rows if r['dataset']==ds and r['phase']=='context' and int(r['seed'])==42 and r['output']=='selected'];name=sorted({r['image'] for r in rr})[0];pair={m:next(r for r in rr if r['image']==name and r['mode']==m) for m in ('flow','direct')};arrays={}
        for m,r in pair.items():
            with np.load(r['path'],allow_pickle=False) as z:
                arrays[m]=z['selected' if 'selected' in z else 'refined'][0].copy();arrays['GT']=z['ground_truth'][0].copy();arrays['baseline']=z['coarse' if 'coarse' in z else 'baseline'][0].copy();arrays['O3']=z['O3'][0].copy()
        if ds=='CHASE':ip=ROOT/'data'/f'{name}.jpg'
        else:ip=Path(load(ROOT/f'paired_reproduction/all_datasets_five_full_runs_2026-10-08_15-38-24/{ds}/train_seed_42/file_provenance.json')[name]['image_path'])
        with Image.open(ip) as im:rgb=cv2.resize(np.asarray(im.convert('RGB')),(512,512),interpolation=cv2.INTER_LINEAR)
        for j,label in enumerate(('RGB','GT','baseline','O3','flow','direct')):
            ax=axes[i,j];ax.imshow(rgb if label=='RGB' else arrays[label]>=.5,cmap=None if label=='RGB' else 'gray',vmin=None if label=='RGB' else 0,vmax=None if label=='RGB' else 1);ax.set_title(ds+' '+name+'\n'+label,fontsize=7);ax.axis('off')
    fig.tight_layout();name=f'prespecified_real_prediction_panels_{tag}.pdf';fig.savefig(figdir/name);plt.close(fig);files.append(name)
    # Keep the new outer-CV cohort separate from the fixed-split panels.
    fig,axes=plt.subplots(3,4,figsize=(7.2,7.2))
    candidates=[('flow','raw'),('flow','selected'),('direct','raw'),('direct','selected')]
    for i,ds in enumerate(DATASETS):
        for j,metric in enumerate(KEYS):
            ax=axes[i,j]
            for x,(mode,output) in enumerate(candidates):
                match=[r for r in tests if r['cohort']=='fresh_nested_CV' and r['dataset']==ds and r['phase']=='fresh_cv' and r['mode']==mode and r['output']==output and r['metric']==metric and r['reference']=='matched_O3']
                if not match:
                    assert ds=='CHASE' and output=='raw',(ds,metric,mode,output)
                    ax.text(x,.5,'NA',transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=7,color='#777777')
                    continue
                assert len(match)==1,(ds,metric,mode,output)
                r=match[0];value=float(r['mean_difference']);low=r['CI_low'];high=r['CI_high'];error=None
                if low not in (None,'','None') and high not in (None,'','None'):
                    error=np.asarray([[max(0.,value-float(low))],[max(0.,float(high)-value)]])
                ax.errorbar(x,value,yerr=error,fmt='o',capsize=2,ms=3,color='#146c94' if mode=='flow' else '#c05712')
            ax.axhline(0.,color='#666666',lw=.6,ls='--')
            ax.set_xticks(range(4),['Flow raw','Flow selected','Direct raw','Direct selected'],rotation=90,fontsize=7)
            ax.set_title(ds+' / '+labels[metric],fontsize=8)
            ax.tick_params(axis='y',labelsize=7);ax.ticklabel_format(axis='y',useOffset=False,style='plain');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fresh outer CV: paired changes from matched O3',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.97));name=f'fresh_CV_paired_changes_{tag}.pdf';fig.savefig(figdir/name);plt.close(fig);files.append(name)
    cal_ds=[ds for ds in DATASETS if any(r['cohort']=='fresh_nested_CV_calibration' and r['dataset']==ds for r in tests)]
    if cal_ds:
        fig,axes=plt.subplots(len(cal_ds),4,figsize=(7.2,2.6*len(cal_ds)),squeeze=False)
        contrasts=[('flow','matched_O3_calibrated','Flow - O3 cal.','#146c94'),('direct','matched_O3_calibrated','Direct - O3 cal.','#c05712'),('flow','matched_direct','Flow - Direct','#555555')]
        for i,ds in enumerate(cal_ds):
            for j,metric in enumerate(KEYS):
                ax=axes[i,j]
                for x,(mode,reference,label,color) in enumerate(contrasts):
                    matches=[r for r in tests if r['cohort']=='fresh_nested_CV_calibration' and r['dataset']==ds and r['mode']==mode and r['output']=='selected' and r['metric']==metric and r['reference']==reference]
                    assert len(matches)==1,(ds,metric,mode,reference)
                    r=matches[0];value=float(r['mean_difference']);error=None
                    if r['CI_low'] not in (None,'','None') and r['CI_high'] not in (None,'','None'):error=np.asarray([[max(0.,value-float(r['CI_low']))],[max(0.,float(r['CI_high'])-value)]])
                    ax.errorbar(x,value,yerr=error,fmt='o',capsize=2,ms=3,color=color)
                ax.axhline(0.,color='#666666',lw=.6,ls='--');ax.set_xticks(range(3),[c[2] for c in contrasts],rotation=90,fontsize=7)
                ax.set_title(ds+' / '+labels[metric],fontsize=8);ax.tick_params(axis='y',labelsize=7);ax.ticklabel_format(axis='y',useOffset=False,style='plain');ax.grid(axis='y',alpha=.2)
        fig.suptitle('Calibrated fresh CV: paired changes',fontsize=9);fig.tight_layout(rect=(0,0,1,.97));name=f'calibrated_CV_paired_changes_{tag}.pdf';fig.savefig(figdir/name);plt.close(fig);files.append(name)
    return files

def build(root):
    root=Path(root).resolve();run_tag=root.name[-19:];analysis=root/'analysis';assert (analysis/f'completed_analysis_{run_tag}.json').exists();plan=load(root/f'queue_{run_tag}.json');assert plan['sealed']
    verification=load(analysis/f'independent_verification_{run_tag}.json')
    assert verification['consecutive_training_logs']==plan['planned_new_training_fits'] and verification['all_planned_test_images_seeds_and_output_groups_covered']
    completion=[]
    for j in plan['jobs']:completion.append(load(root/j['phase']/j['dataset']/f'completed_{run_tag}.json'))
    summary_alias=root/'current_statistical_summary.json';summary=load(summary_alias) if summary_alias.exists() else None
    mean_path=Path(summary['mean_SD_path']) if summary else analysis/f'mean_SD_{run_tag}.csv'
    test_path=Path(summary['paired_tests_path']) if summary else analysis/f'paired_tests_{run_tag}.csv'
    if summary:
        assert sha256_file(mean_path)==summary['mean_SD_sha256'] and sha256_file(test_path)==summary['paired_tests_sha256']
        assert sha256_file(analysis/f'all_verified_per_image_{run_tag}.csv')==summary['source_rows_sha256']
    means=csvrows(mean_path);tests=csvrows(test_path);rows=csvrows(analysis/f'all_verified_per_image_{run_tag}.csv')
    tag=stamp();out=ROOT/'outputs'/f'all_dataset_flow_MDPI_{tag}';project=out/'Overleaf';project.mkdir(parents=True,exist_ok=False);figdir=project/'figures';figdir.mkdir()
    old=load(ROOT/'tmp/current_comprehensive_MDPI_delivery.json');base_path=Path(old['project'])/'main.tex';base=base_path.read_text(encoding='utf-8-sig');shutil.copytree(Path(old['project'])/'Definitions',project/'Definitions')
    preamble=base[:base.index(r'\begin{document}')]
    preamble+='\n'+r'\usepackage{needspace}'+'\n'
    local_date=datetime.strptime(tag[:10],'%Y-%m-%d')
    fixed_date=local_date.strftime('%B')+' '+str(local_date.day)+', '+str(local_date.year)
    preamble+='\n% Freeze the local draft date so the extracted Overleaf ZIP reproduces the PDF.\n'+r'\renewcommand{\today}{'+fixed_date+'}\n'
    title='A Three-Dataset Evaluation of Flow Matching for Retinal Vessel Refinement with Native Patches and Nested Out-of-Fold Inputs'
    preamble=re.sub(r'\\Title\{[^\n]+\}',lambda m:r'\Title{'+title+'}',preamble)
    # Every result in the abstract is pulled from the new bound CSV.
    groups=defaultdict(dict)
    for r in means:groups[tuple(r[k] for k in ('cohort','dataset','phase','method','mode','output'))][r['metric']]=r
    o3=[]
    for ds in DATASETS:
        mm=groups['retained_fixed_split',ds,'O3','native_patch_direct','','raw'];o3.append(fmt(mm['dice']['mean']))
    cv_dice=[];cv_change=[];cv_cl=[];cv_thin=[]
    for ds in DATASETS:
        mm=groups['fresh_nested_CV',ds,'fresh_cv','F9 fresh CV','flow','selected'];cv_dice.append(fmt(mm['dice']['mean']))
        for metric,dest in [('dice',cv_change),('cldice',cv_cl),('thin_skeleton_recall',cv_thin)]:
            match=[r for r in tests if r['cohort']=='fresh_nested_CV' and r['dataset']==ds and r['phase']=='fresh_cv' and r['mode']=='flow' and r['output']=='selected' and r['metric']==metric and r['reference']=='matched_O3'];assert len(match)==1;dest.append(fmt(match[0]['mean_difference']))
    def calibrated_test(ds,metric,reference,mode='flow'):
        matches=[r for r in tests if r['cohort']=='fresh_nested_CV_calibration' and r['dataset']==ds and r['phase']=='calibration' and r['mode']==mode and r['output']=='selected' and r['metric']==metric and r['reference']==reference]
        assert len(matches)==1,(ds,metric,reference,mode)
        return matches[0]
    calibrated_datasets=[ds for ds in DATASETS if any(r['cohort']=='fresh_nested_CV_calibration' and r['dataset']==ds for r in tests)]
    assert calibrated_datasets==['DRIVE','HRF']
    cal_dice=[fmt(calibrated_test(ds,'dice','matched_O3_calibrated')['mean_difference']) for ds in calibrated_datasets]
    cal_cl=[fmt(calibrated_test(ds,'cldice','matched_O3_calibrated')['mean_difference']) for ds in calibrated_datasets]
    cal_thin=[fmt(calibrated_test(ds,'thin_skeleton_recall','matched_O3_calibrated')['mean_difference']) for ds in calibrated_datasets]
    cal_vs_direct=[fmt(calibrated_test(ds,'dice','matched_direct')['mean_difference']) for ds in calibrated_datasets]
    identities=csvrows(analysis/f'verified_identity_masks_{run_tag}.csv')
    selected_cv=[r for r in identities if r['phase']=='fresh_cv' and r['output']=='selected' and r['mode'] in ('flow','direct')]
    assert len(selected_cv)==2*5*(28+40+45) and all(r['exact_FOV_probability_identity']=='True' for r in selected_cv)
    abstract=('Flow Matching offers an iterative formulation for retinal vessel refinement, but its benefit must be separated from native image information and output selection. '
              'We report a three-dataset extension covering native-patch correction, nested out-of-fold (OOF) source maps, structural and preservation objectives, local acceptance and frozen whole-image context. '
              'Final comparisons use five matched correction-training seeds; shared upstream fits and single-fit pilots retain separate uncertainty units. '
              'Fresh three-fold evaluations cover28 subject-grouped CHASE-DB1 images, 40 DRIVE images and45 pathology-stratified HRF images. '
              'At the fixed0.5threshold, validation-selected Flow and capacity-matched direct controls preserve the native-patch direct refiner (O3) probabilities within the valid field of view, with mean Dice '+', '.join(cv_dice)+', respectively. '
              'A validation-only calibration extension passes its prespecified gate on DRIVE and HRF. Relative to separately calibrated O3, Flow changes mean Dice by '+ ' and '.join(cal_dice)+', clDice by '+' and '.join(cal_cl)+', and thin-skeleton recall by '+' and '.join(cal_thin)+', respectively. '
              'Flow-minus-Direct mean Dice is '+' and '.join(cal_vs_direct)+'. '
              'Saved FP32 arrays support paired seed tests, Holm correction and exploratory cohort/seed intervals. Calibration exposes useful corrections, especially on HRF, while matched direct controls attain slightly higher mean Dice. '
              'Repeated validation use, shared upstream estimators and annotation-provenance limits keep these findings conditional; a Flow-specific or prospective clinical advantage is not established.')
    preamble=re.sub(r'\\abstract\{[^\n]+\}',lambda m:r'\abstract{'+prose(abstract)+'}',preamble)
    intro=base[base.index(r'\section{Introduction}'):base.index(r'\section{Materials and Methods}')]
    intro=intro.replace('fresh subject CV','dataset-appropriate fresh CV')
    intro=intro.replace('The additional controls and sequential Flow follow-ups below qualify that ranking and do not establish a joint four-metric Flow advantage.','The additional controls and sequential Flow follow-ups below test how that ranking changes under explicit preservation criteria and matched controls.')
    intro=intro.replace('subsequent phases test Flow corrections, stricter source construction, matched controls and validation-only connection screens.','subsequent phases test Flow corrections, stricter source construction, matched controls and validation-only connection screens, now extended to all three datasets.')
    methods=base[base.index(r'\section{Materials and Methods}'):base.index(r'\subsection{Outcomes and Conditional Uncertainty}')]
    methods=methods.replace(r'\subsection{Fitting, Sequential Selection and Experimental Inventory}',r'\subsection{Initial-Phase Fitting and Selection}')
    methods=methods.replace('All final recipes use five independent refiner training seeds:','The initial O1/O2/O3 comparisons use five independent refiner training seeds:')
    methods=methods.replace('their saved predictions and\nstatistics are reused unchanged.','their saved predictions remain unchanged; the combined analysis below independently replays their FP32 arrays.')
    workflow=base[base.index(r'\begin{figure}[htbp]',base.index(r'\subsection{Computational Provenance')):base.index(r'\section{Results}')]
    workflow=workflow.replace(r'\begin{figure}[htbp]',r'\begin{figure}[H]',1)
    extra=r'''
\subsection{Completed Sequential Experiments}\label{sec:sequential-methods}
The initial controls and O1/O2/O3 recipes are separate from the subsequent
F1--F9 development sequence. Applying Flow Matching to retinal vessels is not
claimed as a first use. Each phase has its own input lineage and validation
history. The later missing-dataset extension uses new dataset-specific
validation records, not a retrospective global lock over previously seen tests.
Previously inspected cohorts and repeated validation use make these exploratory
comparisons. No earlier prediction, checkpoint or manuscript is replaced.
The earlier encoded-flow pilot and AE-free mask-grid prototype use in-sample
coarse maps; they precede the OOF recipe series and are labelled separately.
The encoded pilot is one trained model with five initial-noise realizations.
The mask-grid prototype instead has five independently initialized refiner fits.

F1 trains an AE-free native-patch Flow for120epochs with8Euler updates.
F2 retains a frozen full-training O3 anchor and learns a zero-initialized bounded
Flow field; four40-epoch pilots compare caps1/2 and FM coefficients0.01/0.1,
followed by five120-epoch fields. Validation selects checkpoint and strength
from0,0.25,0.5,1; strength0 is exactly the retained O3 prediction. Raw strength1
is separate. Solver/path diagnostics use validation only. F3 instead obtains
every training O3 input from a whole-chain nested exclusion: an O3 model
excludes the input's group, and each of its fitting coarse producers excludes
both that held group and the fitting input's own group. The deployed O3 remains
the retained full-training model. Four groups are used for the new nested
HRF source chain, distinct from its original three-fold coarse-map recipe.

F4 adds native and exactly aligned512-grid structural supervision. PIL nearest
indices are used for CHASE/DRIVE; HRF uses the original OpenCV nearest indices.
Four40-epoch cap/structure pilots precede five120-epoch fits. Its original
selection permits a Dice decrease of0.0005 and a precision decrease of0.002;
this rule is not renamed Dice preservation. F5 evaluates four signed-field
pilots and four warm-start retention pilots before five120-epoch fits. Its
selection disallows Dice decrease, retains a0.002precision tolerance and ranks
retention of prior structural gain. Depending on the validation-selected family,
the final seeds are signed-field fits or warm fine-tunes of the five matched
F4 fields. A warm epoch-zero choice reuses the retained field rather than
establishing a new learning improvement. A24-point validation-only strength/bias
probe is reported separately and is not a test-selected replacement.

F6 operates on the512grid with GT-free geometrical regions, matched Flow/direct
local proposers and a learned acceptance critic. Four auxiliary proposer fits
per arm use pair-excluded O3 inputs; their held-group proposals supply critic
targets. Pair-excluded O3 fitting inputs require triple-excluded baselines.
These baselines retain the200optimizer-update minimum and validation-softDice
tie break. Eight auxiliary80-epoch proposers, ten final120-epoch proposers and
ten100-epoch critics are used per dataset. The shared auxiliary seed is91011.
Validation calibrates strength and critic conservatism; identity remains
eligible. The critic predicts changes in Dice, clDice and thin recall. Its
three-metric acceptance rule does not guarantee precision or test improvement.

F7 uses native256patches, on-policy rollout supervision and matched direct
fields; two80-epoch preservation pilots precede five80-epoch pairs. F8 adds
frozen ImageNet ResNet18 whole-image features to both arms. Its encoder is not
fitted on retinal labels. Context is extracted from the full RGB field resized
to224square with ImageNet normalization, without a centre crop. Both arms use
the same initialized trainable capacity, crop/augmentation draws and image
budget. Flow uses more field evaluations; parameter matching is not compute
matching. F7/F8 search2/4/8Flow steps versus one direct call, so their candidate
counts are not generally matched. Their pilots use seed42, which also appears
among the five final seeds; pilot fits are not additional independent repeats.
The F7 direct arm inherits the preservation
coefficient chosen by the Flow pilot. For F7--F9, the raw row is the
validation-ranked learned candidate across checkpoints, strengths and solver
settings without requiring the preservation gate; it need not use strength1.
The selected row requires the phase-specific gate or returns identity.
F7/F8 retain their original three-metric validation eligibility
rule. The requested missing-dataset five-seed comparisons are completed even
when their pilot fails; unqualified deployed outputs remain exact identity.
Geometry-guided Flow retains the one80-epoch seed42 pilot scope used on CHASE;
it is not presented as a five-seed result. FR-UNet likewise retains three
validation-learning-rate pilots per dataset. Its official one-channel network
uses the retained512green fields, full-image BCE, Adam with weight decay
$10^{-5}$, a120-epoch cosine schedule and learning-rate candidates
$10^{-5},10^{-4},3\times10^{-4}$. The logical batch is4 with microbatch1;
BatchNorm statistics use microbatch1. Dual-threshold iteration is not executed.
The official SegFlow adaptation uses its existing five EMA checkpoints,
RGB inputs resized to256, zero initial self-conditioning and ten Euler updates
starting from the RGB state, followed by the fixed512probability projection.
Its missing HRF test outputs are generated without
retraining. These are local protocol adaptations, not reproductions of the
external papers' reported benchmark scores.

\subsection{Nested Source Exclusion}
Let $\mathcal{T}$ be a fitting partition and $G_k$ one of its four inner
groups. The baseline operator $B$ includes the unchanged image preprocessing.
An O3 source model excluding $G_k$ is fitted using nested coarse inputs:
\begin{equation}
b_u^{-(k,j)}=B_{-(k,j)}(I_u),\quad u\in G_j,\ j\ne k,
\qquad
p_i^{\mathrm{OOF}}=\mathrm{O3}_{-k}\bigl(I_i,B_{-k}(I_i)\bigr),\quad i\in G_k.
\label{eq:nested-source}
\end{equation}
Subscripts list excluded groups, rather than random seeds. Thus the labels of
$i$ are absent from the O3 source fit, its checkpoint-selection inputs and
every coarse producer used to fit that O3 model. External validation images
remain outside $\mathcal{T}$. For F6's pair-excluded critic source
$\mathrm{O3}_{-(k,j)}$, a fitting image in $G_l$ instead receives a coarse map
from $B_{-(k,j,l)}$. The stored lineage records enumerate these actual sets;
merely using a final held-image prediction is insufficient to establish
indirect exclusion. Each source stores aligned native and comparison-grid
fields, with the residual projected onto the original coarse map. In fresh
outer CV, the outer-test images are additionally excluded from all the above
fits and validation decisions.

\subsection{Executable Rollout and Context Correction}
For F7--F9, let $p^n$ denote a frozen native O3 field and $c$ the optional
same-image frozen RGB context. The correction state starts at zero, and its
implemented Euler updates are
\begin{equation}
r_0=0,\qquad
r_{j+1}=r_j+\frac{1}{S}v_\theta(r_j,j/S,p^n,I,c),\qquad
v_\theta=6\tanh\!\left(f_\theta/6\right).
\label{eq:correction-rollout}
\end{equation}
Training unrolls four steps. Validation searches $S\in\{2,4,8\}$ for Flow;
the matched direct control uses the same initialized field architecture in a
single call at state and time zero. These phases use deterministic inference
conditional on the fitted weights and locked choices, with no initial-noise
sampling. Positive-Hann fusion of native patch corrections is followed by the
dataset's recorded nearest projection $\mathcal{P}_{512}$. Writing the final
comparison-grid residual as $q=\beta\mathcal{P}_{512}(r)$, inference preserves
the original O3 probability exactly wherever the correction is zero:
\begin{equation}
p^*(x)=
\begin{cases}
p^{512}(x),&q(x)=0,\\
\sigma\!\left(\operatorname{logit}_{\epsilon}(p^{512}(x))+q(x)\right),&q(x)\ne0,
\end{cases}
\qquad \epsilon=10^{-4}.
\label{eq:correction-output}
\end{equation}
Here $\operatorname{logit}_{\epsilon}$ clips its probability argument to
$[\epsilon,1-\epsilon]$. This zero-residual branch prevents an FP32
logit/sigmoid round trip from changing a threshold tie.

Each fitting image supplies four augmented native256patches per epoch.
Requested crop-centre strata are missed-thin40\%, near-false-positive25\%,
correct-thin20\% and random-valid15\%; an empty stratum falls back to valid
random pixels. Only training GT defines these sampling supports. Matched arms
share crop draws, rotations by multiples of $90^\circ$ and horizontal/vertical
flips. Optimization uses AdamW with
learning rate $3\times10^{-4}$, weight decay $10^{-4}$, cosine decay without
warm-up, gradient clipping at1 and effective batch8 with microbatches4.
Inference uses256tiles, stride128and batch8, with strictly positive Hann
weights. The full-image context and every inference input remain GT-free.

For training only, the expert residual is the clipped correction toward
margin $1.5$, restricted to erroneous O3 pixels:
\begin{equation}
r^*=\mathbf{1}_{\{(p^n\geq0.5)\ne g\}}
\operatorname{clip}\!\left(1.5(2g-1)-\operatorname{logit}_{\epsilon}(p^n),-6,6\right).
\label{eq:rollout-expert}
\end{equation}
Conditional field regression samples the straight state $t r^*$ and target
$r^*$. The additional on-policy imitation term uses a detached model state
after two of four Euler updates and a clipped remaining-velocity target
$\operatorname{clip}((r^*-r_{1/2})/0.5,-6,6)$. This imitation term is
distinguished from conditional Flow Matching itself. Both controls share the
endpoint, preservation and structural terms; the two field-regression terms
apply only to Flow. With $n$ and $e$ denoting native and aligned evaluation
grids, respectively, the implemented objective is
\begin{equation}
\begin{aligned}
\mathcal{L}={}&0.05\mathcal{L}_{\mathrm{FM}}+0.05\mathcal{L}_{\mathrm{on}}
+0.125(B_n+B_e)+0.25(D_n+D_e)\\
&+0.5\mathcal{L}_{\mathrm{structure}}+\lambda_P\mathcal{L}_{\mathrm{preserve}}
+2(R_n+R_e),\\
\mathcal{L}_{\mathrm{structure}}={}&\tfrac12\bigl[(C_n+0.5T_n+0.25K_n)
+(C_e+0.5T_e+0.25K_e)\bigr].
\end{aligned}
\label{eq:rollout-objective}
\end{equation}
$B$ is the class-balanced BCE term, $D$ soft-Dice loss, $C$ soft-clDice loss,
$T$ positive-skeleton softplus loss and $K$ skeleton/near-background BCE.
$R$ penalizes a soft-Dice loss exceeding the source's loss. Preservation
combines a sign-margin hinge with a $0.25$ probability-change penalty on
correctly classified source pixels, balanced between foreground and background
and averaged across the two grids. F7 tests $\lambda_P=1$ and $4$ using its
two validation pilots; F8/F9 use $1$. Native thin training support has EDT
radius at most two native pixels; aligned comparison-grid support uses the
fixed one-pixel cutoff. Structural training uses ten soft-skeleton iterations
and excludes a three-pixel crop boundary. These training surrogates differ
from the full-image hard evaluation skeletons. GT supplies training targets
and losses only; it is absent from correction inference.

\subsection{Fresh Outer Cross-Validation and Precision Guard}
F9 uses three outer folds. CHASE retains subject grouping over28images;
DRIVE uses all40retained images with native-size annotations and image-level
folds; HRF uses45images stratified by pathology. DRIVE/HRF patient linkage is
unverified, so their folds are not called patient-disjoint. DRIVE reserves four
inner-validation images per fold; HRF reserves two per pathology. Each outer
training set has four inner OOF groups. Each outer fold fits one full,
four single-exclusion and six pair-exclusion baselines, then one full and
four excluded-group O3 models. Across three folds this gives33baseline and
15O3fits per dataset. Five correction seeds then share each fold's upstream
models: these are30correction fits, not five complete pipeline repetitions.
Outer-test images are excluded from every fitting and checkpoint-selection
dependency. No retinal checkpoint from the earlier fixed split initializes this
fresh chain; only the frozen ImageNet context is reused.

The F9 candidate must preserve validation Dice and precision (tolerance
$10^{-12}$) and strictly improve both clDice and thin recall. It maximizes mean
structural gain; otherwise the selected output is exactly O3. All30correction
choices are locked before fresh-pipeline outer-test inference. The fixed-cohort F8
checkpoints also receive a separate retrospective validation precision-guard
re-selection, without retraining. Threshold calibration mirrors the original
single-seed three-fold diagnostic: epochs20/40/60/80, four strengths,81thresholds
from0.10to0.90, and matched Flow/direct evaluation. Average precision excludes
pixels outside FOV. AP-max configurations are optimistic validation diagnostics,
not evidence of test gain. The original aggregate four-metric pilot gate
determines whether the remaining four stored correction seeds are calibrated
and evaluated. All30choices are then locked before the new calibrated outer
evaluation; calibration-only O3 gains are never credited to Flow.
Connection-repair diagnostics compare GT-free
candidate tubes, an unfiltered union and an explicitly GT-filtered oracle;
oracle gains never count as a trained Flow result.

\subsection{Outcomes, Statistical Units and Verification}
Fixed-threshold primary scores use the unchanged512-grid GT and probability threshold
$p\geq0.5$, with the original valid FOV. Thin support is the GT skeleton with
padded Euclidean distance-transform radius at most one evaluation-grid pixel.
It is an operational thin-vessel definition, not a calibrated anatomical width.
The separate calibration comparison uses its locked fold-specific thresholds.
Soft training skeletons and hard evaluation skeletons are different quantities.
IoU, recall, specificity, accuracy, skeleton precision/sensitivity, correction
and damage rates and component-count discrepancy are supplied as secondary
descriptive seed summaries; primary test families retain the four stated outcomes.
FP32 probabilities, GT and FOV are reopened before independent metric replay;
raw, selected and identity outputs are kept distinct. Initial-noise repetitions
of one encoded model are averaged as metrics and are not independent trainings.
Native-resolution validation probes do not replace primary512-grid scores.
The word raw retains each phase's recorded rule: fixed-final encoded/mask
prototypes; validation-chosen unblended initial/F1 checkpoints; strength1
F2--F5 fields; and the F6 strength1 proposer before critic acceptance. Newly
exported F7--F9 raw rows are the validation-ranked diagnostic candidate,
including its recorded checkpoint, strength and solver. They are not assumed
to be fixed-final or strength1 outputs. Their earlier CHASE raw counterparts
were not exported and are not invented.

Means and sample SDs describe the five matched training-seed means. Paired
two-sided t intervals use four degrees of freedom. Exact two-sided signed-rank
tests are separate; with five nonzero concordant pairs their minimum attainable
p value is0.0625. An all-zero identity contrast has interval[0,0] and p=1;
it does not establish equivalence. Newly combined Holm families are declared by
cohort, phase, arm/output and contrast across the available dataset/metric cells;
the full-precision tables include family membership and size. Direct/Flow
comparisons use matched seeds, source images and exact input hashes.
Recipe-specific controls use their matched retained control, and factorial
interactions are kept separate. Exploratory adjacent-recipe contrasts compare
F2 with F1 raw, F3 with F2 selected, F4 with F3 selected, F5 with F4 selected,
and each F8 arm with its corresponding F7 arm. These are paired system
comparisons; changes in source, capacity, objective or selection prevent a
single-factor causal interpretation. Historical single-dataset p values are not
silently relabelled as corrected over newly added datasets.

Exploratory crossed bootstrap intervals use50,000seed/cohort resamples with
seed20261118. CHASE resamples subjects with both eyes; DRIVE/HRF resample
images. Resampling preserves outer-fold strata and HRF pathology strata.
These intervals are conditional on the retained cohort, source fits and
selection history; they are not prospective clinical confidence bounds.
The exploratory inference is not rescued by treating dependent patches,
pixels, sampling seeds or critic offers as additional independent subjects.
All loss and checkpoint settings are fixed or validation-selected, never chosen
by test performance. Negative results and exact fallback remain in the tables.

\subsection{Computational Provenance and AI Assistance}
The extension retains FP32 execution, disabled TF32, deterministic cuDNN
settings and disabled cuDNN benchmarking. These recorded settings are not a
claim that every training operation is bitwise reproducible across software
versions or devices. Independent DRIVE work overlaps the primary HRF worker;
frozen-model validation diagnostics also overlap fitting when memory permits.
Each process has separate random generators, selection records and output
paths. No overlapping-job wall time is reported as an isolated latency
benchmark. Source versions, initialization and crop checks, checkpoint hashes
and saved predictions accompany the execution records. The large number of
source, pilot and final fitting calls is not an inferential sample size.
OpenAI Codex assisted implementation, workflow orchestration, verification and
drafting. Scientific plots use Matplotlib and real saved arrays; the schematic
uses TikZ. Exact AI-version and author-disclosure details remain subject to
the author confirmations stated in the acknowledgments.
'''
    extra=extra.replace('FR-UNet likewise',r'FR-UNet~\cite{Liu2022FRUNet} likewise')
    files=scientific_plots(means,rows,tests,figdir,tag)
    result=r'\section{Results}\label{sec:results}'+'\n'
    result+='The test comparisons below were independently recomputed from saved probabilities. Separate validation-only tables identify their recorded sources. Original fixed cohorts and fresh outer-CV cohorts are reported separately. A raw output is a learned candidate; a selected identity is the frozen anchor. Warm epoch-zero field reuse and calibration-only changes are distinguished from newly learned corrections.\n'
    result+=r'\subsection{Initial Recipes and Completed Controls}'+'\n'
    prelim=[]
    for key,mm in sorted(groups.items()):
        if key[2] in ('encoded','pixel','segflow'):prelim.append([key[1],esc(key[3]+' / '+key[5]),mm['dice']['training_seed_count'],*(cell(mm[k]) for k in KEYS)])
    result+=longtable('Preliminary representation studies and the official SegFlow adaptation. The encoded prototype has one training fit and five noise realizations, so no training-seed SD is invented. The AE-free mask prototype and SegFlow each have five conditional training-seed fits.','tab:preliminary-all',['Data','Method / output','Fits','Dice','Precision','clDice','Thin recall'],prelim,[1.1,3.1,.7,2.5,2.5,2.5,2.5])
    for ds in DATASETS:
        rr=[]
        for key,mm in sorted(groups.items()):
            if key[0]=='retained_fixed_split' and key[1]==ds and key[2] in ('baseline','initial_controls','initial_objectives','O3') and key[5] in ('raw','anchor'):rr.append([esc(key[3]),mm['dice']['training_seed_count'],*(cell(mm[k]) for k in KEYS)])
        result+=longtable(ds+r': initial raw comparisons. Mean $\pm$ sample SD of conditional training-seed means; all fields use the same fixed test cohort.','tab:initial-'+ds,['Method','Seeds','Dice','Precision','clDice','Thin recall'],rr,[4.1,.7,2.5,2.5,2.5,2.5])
    result+=figure(files[0],'fig:initial-comparison','Initial raw recipes on all three retained fixed cohorts. B: frozen baseline; D: Direct; I: endpoint-iterative; FM: full-target Flow; S: selective without gate; ES: ESFM; O1/O2/O3: the three initial recipes. Error bars show sample SD across five refiner-training seed means, conditional on the retained upstream models. The one frozen baseline has no training-seed SD. These bars are not population confidence intervals.')
    result+=r'\subsection{Sequential Flow and Matched Direct Results}'+'\n'
    for ds in DATASETS:
        rr=[]
        sequence={p:j for j,p in enumerate(('native_flow','anchored','fully_oof','topology','retained','local_acceptor','rollout','context'))}
        for key,mm in sorted(groups.items(),key=lambda q:(sequence.get(q[0][2],99),q[0])):
            if key[0]=='retained_fixed_split' and key[1]==ds and key[2] in ('native_flow','anchored','fully_oof','topology','retained','local_acceptor','rollout','context'):
                rr.append([esc((key[3]+' '+key[4]+' / '+key[5]).strip()),mm['dice']['training_seed_count'],*(cell(mm[k]) for k in KEYS)])
        result+=longtable(ds+r': sequential follow-ups. Raw and validation-selected outputs are separate. Source and selection changes across phases prevent an additive causal interpretation.','tab:followup-'+ds,['Method / output','Seeds','Dice','Precision','clDice','Thin recall'],rr,[4.1,.7,2.5,2.5,2.5,2.5])
    result+=figure(files[1],'fig:followup-comparison','Validation-selected fixed-split outputs. F1--F8 follow the method definitions; suffix F denotes Flow and D denotes its matched direct control. Parameter matching is phase-specific; field-evaluation and candidate-grid counts differ. Selection includes exact anchor fallback, and repeated validation use is exploratory. These phases are not an additive learning curve.')
    result+=figure(files[2],'fig:real-panels','Real saved seed42 predictions for the first image identifier in each original test split. The cases are selected by identifier before comparing scores. Columns show the original RGB image, GT, baseline, O3 and the validation-selected whole-image-context Flow and direct outputs. Identical panels are retained when the system selects identity; no generated prediction or manually repaired vessel is substituted.')
    result+=r'\subsection{Fresh Outer-CV Comparison}'+'\n'
    cv=[]
    for key,mm in sorted(groups.items()):
        if key[0]=='fresh_nested_CV':cv.append([key[1],esc(key[4]+' / '+key[5]),mm['dice']['images'],*(cell(mm[k]) for k in KEYS)])
    result+=longtable('Fresh outer-CV scores. CHASE uses28images, DRIVE40and HRF45. CHASE retains selected CV outputs only; raw scores are reported where archived on DRIVE/HRF. SD describes correction-stage seed variation conditional on shared fresh upstream models; it is not SD over five complete pipelines.','tab:fresh-cv',['Dataset','Method / output','Images','Dice','Precision','clDice','Thin recall'],cv,[1.3,2.4,1.0,2.5,2.5,2.5,2.5])
    result+=figure(files[3],'fig:fresh-cv-changes',r'Fresh outer-CV changes relative to the matched O3 sources. Points are means over five correction seeds; bars show pointwise seed-level paired $t$ 95\% confidence intervals, where defined. NA marks the unarchived CHASE raw CV outputs; no score is imputed. Both arms share the freshly trained upstream models within each fold. Raw denotes the validation-ranked learned candidate without the preservation gate; selected outputs include exact identity. The intervals are conditional on the retained cohort and shared upstream fits, and do not quantify full-pipeline or prospective clinical uncertainty.')
    calibrated=[]
    for key,mm in sorted(groups.items()):
        if key[0]=='fresh_nested_CV_calibration':calibrated.append([key[1],esc(key[4]+' / '+key[5]),mm['dice']['images'],*(cell(mm[k]) for k in KEYS)])
    if calibrated:
        result+=longtable('Conditional calibration extension on datasets whose single-seed validation pilot passes. The calibrated O3 baseline is separate from the original0.5baseline. No unexecuted CHASE calibration-test result is inferred.','tab:calibrated-cv',['Data','Output','Images','Dice','Precision','clDice','Thin recall'],calibrated,[1.3,2.4,1.0,2.5,2.5,2.5,2.5])
        cal_plot=next(f for f in files if f.startswith('calibrated_CV_paired_changes_'))
        result+=figure(cal_plot,'fig:calibrated-cv-changes',r'Conditional calibrated outer-CV contrasts. Points are five-seed mean differences and bars are pointwise paired $t$ 95\% intervals. Each fold calibrates O3 separately on validation data. The gray contrast compares Flow directly with its matched direct control. These intervals describe correction-seed variation conditional on the shared source fits and retained cohort. The plotted comparisons do not establish a general Flow-specific advantage.')
    result+=r'\subsection{Paired Contrasts and Diagnostic Scope}'+'\n'
    result+='The full paired tables include both seed-based and crossed-bootstrap intervals, unadjusted tests, Holm families and identity cases. Five-seed exact signed-rank tests have coarse resolution; a nonsignificant result is not an equivalence claim. The encoded-flow and geometry pilots remain single-training diagnostics. FR-UNet has three validation learning-rate pilots per dataset; no five-seed FR-UNet test study is inferred from them. The newly completed SegFlow HRF comparison uses five previously fitted checkpoints. Threshold, precision-guard and connection/oracle diagnostics are validation-only and are not pooled with test efficacy.\n'
    pilotrows=[]
    for ds in DATASETS:
        p=ROOT/'outputs/geometry_flow_CHASE_pilot_2026-11-18_01-11-21/selection_locked_2026-11-18_01-11-21.json' if ds=='CHASE' else root/'geometry'/ds/f'selection_locked_{run_tag}.json';d=load(p)
        for label,mm in [('O3 validation',d['baseline_validation']),('Selected geometry pilot',d['selection']['metrics']),('Raw validation-ranked candidate',d['best_raw']['metrics'])]:pilotrows.append([ds,label,*(fmt(mm[k]) for k in KEYS)])
    result+=longtable('Single-fit geometry pilots: validation means only. The selected and raw validation-ranked candidates are distinct; no SD, five-seed confirmation or new test score is implied.','tab:geometry-pilots',['Data','Candidate','Dice','Precision','clDice','Thin recall'],pilotrows,[1.1,4.0,2.5,2.5,2.5,2.5])
    retention=[]
    for r in tests:
        if r['phase']=='retained' and r['output']=='selected' and r['reference']=='matched_predecessor_topology_selected':
            label=dict(dice='Dice',precision='Precision',cldice='clDice',thin_skeleton_recall='Thin recall')[r['metric']]
            retention.append([r['dataset'],label,fmt(r['mean_difference']),r'$['+fmt(r['CI_low'])+','+fmt(r['CI_high'])+']$',pvalue(r.get('Holm_p')),pvalue(r.get('Wilcoxon_p'))])
    assert len(retention)==12
    result+=longtable('F5 selected minus preceding F4 selected on matched test images and training seeds. Validation retention does not imply test retention. CI is the seed-level t interval; exact signed-rank p is unadjusted. Multiple recipe factors differ, so this is a system comparison.','tab:retention-vs-prior',['Data','Outcome','Mean change',r'95\% CI','Holm t p','Exact W p'],retention,[1.2,3.0,2.0,3.6,2.2,2.0])
    from .analyze import old_folders
    old_paths={(phase,ds):(folder,t) for folder,phase,ds,t in old_folders()};reuse=[]
    for ds in DATASETS:
        folder,t=(root/'retained'/ds,run_tag) if ds=='HRF' else old_paths['retained',ds]
        pp=sorted((folder/'final').glob(f'seed_*/selection_locked_{t}.json'));assert len(pp)==5
        dd=[load(p) for p in pp];families={d.get('family',d['config'].get('family')) for d in dd};assert len(families)==1
        reuse.append([ds,esc(next(iter(families))),sum(d['config'].get('family')=='warm' for d in dd),sum(d['config'].get('family')=='warm' and d['selected']['epoch']==0 and d['selected']['beta']>0 for d in dd),sum(d['selected']['beta']==0 for d in dd)])
    result+=longtable('F5 initialization and reuse accounting across five final seeds. An epoch-zero warm choice uses an already retained F4 field; it is not credited as a newly learned F5 improvement. Zero strength instead returns O3.','tab:warm-reuse',['Data','Selected family','Warm starts','Nonzero epoch-zero reuse','Zero-strength choices'],reuse,[1.5,3.0,2.5,4.2,3.0])
    frrows=[];ft='2026-11-16_20-28-20';frroot=ROOT/f'outputs/reviewer_completion_2026-11-16_19-37-22/official_frunet_retinal_adaptation_{ft}'
    for ds in DATASETS:
        paths=list((root/'frunet'/ds/'pilots').glob(f'*/selection_locked_{run_tag}.json')) if ds=='DRIVE' else list((frroot/ds/'pilots').glob(f'*/selection_locked_{ft}.json'));assert len(paths)==3,(ds,paths)
        for p in sorted(paths):d=load(p);frrows.append([ds,str(d['learning_rate']),str(d['epoch']),fmt(d['validation_dice'])])
    result+=longtable('FR-UNet validation-learning-rate pilots. Each row is one seed91011fit; these are tuning records, not three independent training seeds or a test benchmark.','tab:frunet-pilots',['Data','Learning rate','Selected epoch','Validation Dice'],frrows,[2.0,3.0,3.0,3.0])
    diagnostic_text,diagnostic_sources=diagnostic_tables(root,run_tag);result+=diagnostic_text
    selected_tests=[r for r in tests if r['phase'] in ('local_acceptor','rollout','context','fresh_cv','calibration') and r['output'] in ('selected','accepted') and r['reference'] in ('matched_O3','matched_O3_calibrated','matched_direct')]
    pt=[]
    for r in selected_tests:
        ci=r'$['+fmt(r['CI_low'])+','+fmt(r['CI_high'])+']$'
        metric_label=dict(dice='Dice',precision='Precision',cldice='clDice',thin_skeleton_recall='Thin recall')[r['metric']]
        reference_label={'matched_O3':'O3','matched_O3_calibrated':'O3 cal.','matched_direct':'Direct'}[r['reference']]
        pt.append([r['dataset'],esc(r['method']+' '+r['mode']+'/'+r['output']),reference_label,metric_label,fmt(r['mean_difference']),ci,pvalue(r.get('Holm_p')),pvalue(r.get('Wilcoxon_p'))])
    discussion=r'''
\section{Discussion}
These comparisons separate a transport formulation from access to image detail,
source construction, context, structural supervision and output acceptance.
Native patches can supply information discarded by a512-grid input, while
windowed residual fusion preserves the original comparison grid. A local field
can nevertheless trade thin-skeleton recovery against foreground precision or
whole-mask overlap. A jointly constrained validation choice can instead return
the exact anchor. This is useful operational conservatism, but not evidence that
the field learned a beneficial transformation. Parameter matching controls one
capacity measure; Flow's extra numerical evaluations and auxiliary source/critic
fits remain distinct computational work.

The fixed-threshold F7--F9 correction also has a shared feasibility limit:
its bounded logit residual can leave a sufficiently confident false negative
below the decision threshold even at maximum allowed strength. Euler updates
over a unit interval and positive-Hann averaging retain the velocity cap.
This limitation is shared by the matched direct parameterization. Its
prevalence and causal contribution to the observed outcomes remain unmeasured.

The completed missing-dataset extension addresses the earlier coverage gap; it
does not make the sequential search confirmatory. Cohorts and validation sets
were repeatedly encountered, source-training sizes differ from deployed models,
and choices are conditional on the recorded partitions. The fresh CV prevents
direct and indirect held-image fitting/selection leakage within its execution,
but is not a previously unseen external cohort. DRIVE and HRF remain image-level
evaluations without verified patient linkage. DRIVE's new40-image CV must not be
confused with the original20-image official fixed test split. Reuse of the
retained local test-label version does not independently establish acquisition
history. HRF's recorded source-pixel audit establishes a stated foreground
conversion, not recovery of the historical converter implementation.

Thin recall uses an evaluation-grid skeleton and radius cutoff. It does not
measure every anatomical small vessel, and clDice is not a clinical topology
guarantee. Local false connections require more than matching a global component
count. GT-filtered connection oracles show an opportunity conditional on labels;
their gains cannot be assigned to a learned model. Single-seed pilots, five
initial-noise draws, five correction-stage fits and five whole pipelines are
different experimental units. Their uncertainty must remain separate.

\section{Conclusions}
The completed three-dataset study provides phase-specific comparisons of Direct
and Flow Matching under recorded OOF inputs, native-patch inference and explicit
preservation rules. The tables retain negative outcomes, raw corrections and
exact identity fallback. Any favorable mean is conditional on the stated cohort,
source chain and validation history; it does not establish a general Flow
advantage, prospective clinical benefit or equivalence of nonsignificant systems.

\section*{Reproducibility and Availability}
The earlier source-code repository is
\url{https://github.com/thebabak/retinal-vessel-flow-refinement-oof}.
Its immutable initial artifact release is
\url{https://github.com/thebabak/retinal-vessel-flow-refinement-oof/releases/tag/artifacts-2026-11-16_18-53-15}.
That release covers the earlier initial reference/control and
three-recipe artifacts; it is not a release of every newly completed
sequential experiment. The new extension's checkpoints, source maps and FP32
outputs are preserved separately. Its accompanying manuscript package provides
the new full-precision scalar tables, verification records, phase completion
registry, source manifests and executed-code snapshots. Large model/output
arrays are not embedded in the Overleaf ZIP. No new GitHub publication is
performed by this manuscript build.

Raw DRIVE dataset files are not included in the code/evidence release because
redistribution permission was not granted; retrieval follows the original
providers cited here. Existing public DRIVE prediction-only containers omit
GT/FOV, and privacy derivatives of HRF annotations omit personal labeler-login
fields. Such derivatives do not reproduce every original-byte checksum.
Dataset acquisition history, exact local pixel correspondence and model-output
verification are separate claims. The retained local DRIVE test-label source
is not independently established. No reviewer correspondence is included in
the public repository or this new manuscript package.
'''
    direct_dice_excess=[fmt(-float(calibrated_test(ds,'dice','matched_direct')['mean_difference'])) for ds in calibrated_datasets]
    direct_contrasts=[calibrated_test(ds,k,'matched_direct') for ds in calibrated_datasets for k in KEYS]
    assert all(float(r['Holm_p'])>=.05 for r in direct_contrasts)
    findings=(r'At the fixed0.5threshold in fresh outer CV, both selected correction arms preserve the O3 probabilities throughout the valid FOV for every held-out image and seed on all three datasets (Table~\ref{tab:identity-audit}). '
              'Their zero correction-stage SD reflects identity selection with shared upstream estimates; it is not evidence of negligible full-pipeline or population uncertainty. '
              'The selection rule therefore provides no additional fixed-threshold gain over O3 in this evaluation.\n\n'
              r'The prespecified calibration extension changes that outcome on DRIVE and HRF (Table~\ref{tab:calibrated-cv}). Relative to separately calibrated O3, Flow mean Dice changes are '+' and '.join(cal_dice)+', clDice changes are '+' and '.join(cal_cl)+', and thin-recall changes are '+' and '.join(cal_thin)+', respectively. '
              'The DRIVE clDice interval and both precision intervals include zero, so these observations do not establish a joint four-metric benefit. '
              'Matched Direct attains higher mean Dice by '+' and '.join(direct_dice_excess)+', respectively; none of the eight Holm-corrected Flow-versus-Direct endpoint contrasts is significant. '
              'The exact five-seed signed-rank tests retain their coarse probability resolution. These results support calibrated correction under this protocol, while providing no evidence that the Flow formulation itself improves on the matched direct alternative.\n\n')
    discussion=discussion.replace(r'\section{Discussion}'+'\n',r'\section{Discussion}'+'\n'+findings,1)
    discussion=discussion.replace('The completed three-dataset study provides phase-specific comparisons of Direct\nand Flow Matching under recorded OOF inputs, native-patch inference and explicit\npreservation rules. The tables retain negative outcomes, raw corrections and\nexact identity fallback.',
        'Fixed-threshold fresh-CV selection preserves O3 outputs within the valid FOV on all three datasets. Validation-only calibration exposes mean Dice and thin-recall gains on DRIVE and HRF and a clearer clDice gain on HRF; matched direct correction attains slightly higher mean Dice. The completed study therefore supports careful calibration and matched controls rather than a Flow-specific advantage. Negative outcomes, raw candidates and exact identity fallback remain explicit.')
    discussion=prose(discussion)
    back=base[base.index(r'\authorcontributions{'):base.index(r'\appendixtitles{yes}')]
    back=back.replace('DRIVE source images, annotations and FOV are not redistributed.','Raw DRIVE dataset files are not included in the evidence/code release.')
    appendix=r'\appendixtitles{yes}\appendixstart\appendix'+'\n'+r'\section{Completed Controlled Ablations}'+'\n'
    for ds in DATASETS:
        rr=[]
        for key,mm in sorted(groups.items()):
            if key[1]==ds and key[2]=='controls':rr.append([esc(key[3]+' / '+key[5]),*(cell(mm[k]) for k in KEYS)])
        appendix+=longtable(ds+':16controlled recipes. Earlier CHASE/DRIVE exports contain raw outputs only; their missing blends are not invented. Newly completed HRF raw and validation-blend outputs are distinct.','tab:controls-'+ds,['Recipe / output','Dice','Precision','clDice','Thin recall'],rr,[5.0,2.5,2.5,2.5,2.5])
    appendix+=r'\section{Paired Selected-Output Comparisons}'+'\n'+longtable('Two-sided paired comparisons of matched training-seed means. O3 cal. denotes the matched calibrated O3 reference; Direct denotes the phase-matched direct control. CI is the seed-level t interval; the full CSV separately reports crossed-bootstrap bounds. Holm families and all full-precision values accompany the manuscript.','tab:paired-all',['Data','System','Ref.','Metric','Mean change',r'95\% CI','Holm t p','Exact W p'],pt,[1.1,3.2,1.0,2.0,1.5,2.9,1.2,1.2])
    identity=csvrows(analysis/f'verified_identity_masks_{run_tag}.csv');identity_groups=defaultdict(list)
    for r in identity:
        if r['phase'] in ('native_flow','anchored','fully_oof','topology','retained','local_acceptor','rollout','context','fresh_cv','calibration') and r['output'] in ('selected','blend','accepted'):
            identity_groups[tuple(r[k] for k in ('cohort','dataset','phase','method','mode','output','anchor'))].append(r)
    identical=[]
    for key,rr in sorted(identity_groups.items()):
        seeds=sorted({r['seed'] for r in rr});counts={}
        for k in ('exact_FOV_mask_identity','exact_FOV_probability_identity'):counts[k]=sum(all(r[k]=='True' for r in rr if r['seed']==s) for s in seeds)
        identical.append([key[1],esc(key[3]+' '+key[4]+' / '+key[5]),esc(key[6]),len(rr),str(counts['exact_FOV_mask_identity'])+' / '+str(len(seeds)),str(counts['exact_FOV_probability_identity'])+' / '+str(len(seeds))])
    appendix+=r'\section{Unchanged-Output Accounting}'+'\n'+longtable('Array-based identity audit in the valid FOV. Same masks/probabilities counts seed-level systems for which every held-out image is identical to the named input reference; each denominator is the number of training seeds. Fresh-CV seeds are summarized over all outer-test images. Calibration compares masks after their locked thresholds and original probabilities before thresholding. Image-seed output counts are bookkeeping, not additional independent samples. Mask identity alone does not imply probability identity or equivalence.','tab:identity-audit',['Data','Phase / arm / output','Reference','Image-seed outputs','Same masks','Same probabilities'],identical,[1.0,4.2,2.0,1.6,2.7,2.7])
    appendix+=r'\section{Source Correspondence and Execution Registry}\label{app:source-audit}'+'\n'
    appendix+=r'Native and 512-grid RGB, GT and FOV versions are bound by SHA256 manifests. All 45 retained HRF native targets match the official TIFF foreground defined by luminance greater than 254, with matching OpenCV nearest-neighbour 512-grid conversions. Masks 11\_h, 12\_h and 13\_h contain intermediate gray levels, so a different threshold defines different targets. This establishes the stated pixel correspondence; the historical annotation converter and original acquisition history have not been independently recovered. No source target is replaced in this extension. The recorded DRIVE audit marks native GT available at 565 by 584 for all 40 retained images; no resized label is promoted to native supervision. Official-source access and data acquisition history remain distinct from exact local pixel correspondence.'+'\n'
    registry=[]
    for c in completion:registry.append([esc(c['phase']),c['dataset'],esc(', '.join(f'{k}={v}' for k,v in c.items() if k in ('final_fits','training_fits','pilot_fits','fresh_baseline_fits','fresh_O3_fits','outer_O3_fits','pair_O3_fits','nested_baseline_fits','auxiliary_proposer_fits','final_proposer_fits','final_acceptor_fits','test_prediction_files','test_records','validation_images','systems')))])
    appendix+=longtable('Completed new extension scopes. Initial CHASE and already completed DRIVE/HRF records are retained separately; this table counts the newly dispatched phases only.','tab:execution-registry',['Phase','Dataset','Verified completed scope'],registry,[3.0,1.4,11.0],rowstretch=1.04)
    biblio=base[base.index(r'\begin{thebibliography}'):base.index(r'\end{thebibliography}')+len(r'\end{thebibliography}')]
    fr_reference=r'''\bibitem{Liu2022FRUNet}
Liu, W.; Yang, H.; Tian, T.; Cao, Z.; Pan, X.; Xu, W.; Jin, Y.; Gao, F.
Full-Resolution Network and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation.
\textit{IEEE J. Biomed. Health Inform.} \textbf{2022}, \textit{26}, 4623--4634.
\url{https://doi.org/10.1109/JBHI.2022.3188710}.
'''
    biblio=biblio.replace(r'\end{thebibliography}',fr_reference+'\n'+r'\end{thebibliography}')
    source=preamble+r'\begin{document}'+'\n'+intro+methods+prose(extra)+workflow+result+discussion+back+appendix+'\n'+r'\begin{adjustwidth}{-\extralength}{0cm}'+'\n'+biblio+'\n'+r'\PublishersNote{}\end{adjustwidth}\end{document}'+'\n'
    result_guides={
        'Initial Recipes and Completed Controls':r'Table~\ref{tab:preliminary-all} reports the preliminary representation studies. Tables~\ref{tab:initial-CHASE}--\ref{tab:initial-HRF} and Figure~\ref{fig:initial-comparison} compare the initial recipes on the fixed cohorts. The controlled recipe results are retained separately in Tables~\ref{tab:controls-CHASE}--\ref{tab:controls-HRF}.',
        'Sequential Flow and Matched Direct Results':r'Tables~\ref{tab:followup-CHASE}--\ref{tab:followup-HRF} and Figure~\ref{fig:followup-comparison} summarize the sequential comparisons. Figure~\ref{fig:real-panels} displays identifier-selected examples from the saved outputs.',
        'Fresh Outer-CV Comparison':r'Table~\ref{tab:fresh-cv} and Figure~\ref{fig:fresh-cv-changes} report the fresh held-out comparisons. Table~\ref{tab:calibrated-cv} and Figure~\ref{fig:calibrated-cv-changes} retain the separate, conditional calibration extension.',
        'Paired Contrasts and Diagnostic Scope':r'Table~\ref{tab:paired-all} contains the selected-output paired contrasts. Table~\ref{tab:geometry-pilots} reports the single-fit geometry pilots; Tables~\ref{tab:retention-vs-prior} and~\ref{tab:warm-reuse} separate F5 retention from warm-field reuse. Table~\ref{tab:frunet-pilots} lists the FR-UNet validation fits.',
        'Validation Diagnostics and Rejected Candidates':r'Tables~\ref{tab:encoded-diagnostics} and~\ref{tab:solver-diagnostics} retain the representation and solver probes. Tables~\ref{tab:boundary-diagnosis} and~\ref{tab:precision-reselection} report the boundary and precision screens. Calibration, connection and strength/bias diagnostics appear in Tables~\ref{tab:calibration-validation}--\ref{tab:bias-probe}.',
    }
    for heading,paragraph in result_guides.items():
        marker=r'\subsection{'+heading+'}'
        assert source.count(marker)==1
        source=source.replace(marker,marker+'\n'+paragraph+'\n\n',1)
    marker=r'\section{Source Correspondence and Execution Registry}\label{app:source-audit}'
    source=source.replace(marker,marker+'\n'+r'Table~\ref{tab:execution-registry} records the completed extension scopes.'+'\n\n',1)
    source=source.replace(r'\begin{thebibliography}',r'\reftitle{References}'+'\n'+r'\begin{thebibliography}',1)
    source=source.replace(r'\authorcontributions{','\n\n'+r'\authorcontributions{',1)
    source=source.replace('frozen ImageNet ResNet18 whole-image features','frozen ImageNet ResNet18~'+r'\cite{He2016}'+' whole-image features',1)
    source=source.replace(r'\section{Discussion}',r'\FloatBarrier'+'\n'+r'\section{Discussion}',1)
    source=source.replace(r'\section{Paired Selected-Output Comparisons}',r'\clearpage'+'\n'+r'\section{Paired Selected-Output Comparisons}',1)
    source=source.replace(r'\subsection{',r'\Needspace{8\baselineskip}'+'\n'+r'\subsection{')
    source=order_bibliography(source)
    # MDPI numbering and symbols use the same established class/font stack.
    (project/'main.tex').write_text(source,encoding='utf-8');tex=out/f'retinal_flow_all_datasets_{tag}.tex';tex.write_text(source,encoding='utf-8')
    evidence=out/'evidence';evidence.mkdir();
    for p in analysis.glob('*'):
        if not p.is_file():continue
        if summary and p.name in (f'mean_SD_{run_tag}.csv',f'paired_tests_{run_tag}.csv'):
            historical=evidence/'superseded_statistics';historical.mkdir(exist_ok=True);shutil.copy2(p,historical/p.name)
        else:shutil.copy2(p,evidence/p.name)
    if summary:
        note=evidence/'superseded_statistics'/f'README_{tag}.txt'
        note.write_text('Historical, superseded statistical products. Exact-constant seed vectors produced floating-point SD residuals in 12 summary cells and 2 invalid t/CI entries. The immutable correction manifest documents the replacement. Current products are ../'+mean_path.name+' and ../'+test_path.name+'. All measured means, per-image metrics, predictions and prior files were preserved.\n',encoding='utf-8')
    diag=evidence/'validation_diagnostics';diag.mkdir();diagnostic_manifest=[]
    for i,p in enumerate(diagnostic_sources):
        dest=diag/(f'{i:03d}_'+p.name);shutil.copy2(p,dest);assert sha256_file(dest)==sha256_file(p);diagnostic_manifest.append(dict(source=str(p),copied_to=str(dest),sha256=sha256_file(p)))
    write_json(evidence/f'validation_diagnostic_sources_{tag}.json',diagnostic_manifest)
    implementation=evidence/'dataset_completion_code';implementation.mkdir()
    for p in (ROOT/'flow_matching_refiner/dataset_completion').glob('*.py'):shutil.copy2(p,implementation/(p.stem+'_'+tag+'.py'))
    executed=evidence/'executed_sources';executed.mkdir();execution_source_manifest=[]
    for j in plan['jobs']:
        for source_kind in ('executed_source','adapted_source'):
            folder=root/j['phase']/j['dataset']/source_kind
            if not folder.exists():continue
            target=executed/j['phase']/j['dataset']/source_kind;target.mkdir(parents=True,exist_ok=True)
            for p in folder.iterdir():
                if not p.is_file():continue
                q=target/p.name;shutil.copy2(p,q);assert sha256_file(q)==sha256_file(p)
                execution_source_manifest.append(dict(phase=j['phase'],dataset=j['dataset'],kind=source_kind,original=str(p),member=str(q.relative_to(evidence)),sha256=sha256_file(p)))
    write_json(evidence/f'executed_source_inventory_{tag}.json',execution_source_manifest)
    for p in root.glob('*.json'):shutil.copy2(p,evidence/p.name)
    write_json(out/f'completed_scope_registry_{tag}.json',completion)
    summary_note=('Current statistical products: evidence/'+mean_path.name+' and evidence/'+test_path.name+'. Earlier products are preserved only in evidence/superseded_statistics/; the correction manifest documents constant-vector SD roundoff.\n') if summary else ''
    instructions=out/f'OVERLEAF_INSTRUCTIONS_{tag}.txt';instructions.write_text('Upload the contents of Overleaf/ as one project. Main document: main.tex. Compiler: pdfLaTeX. Definitions/ and figures/ are required. This is a new research draft; the published paper and previous results are unchanged. Study-specific contributor, funding, ethics/consent, conflict and exact AI-version declarations still require author confirmation. No reviewer-response letter is included and nothing is published to GitHub by this build.\n'+summary_note,encoding='utf-8')
    state=dict(created_at=tag,run=str(root),run_tag=run_tag,output=str(out),project=str(project),tex=str(tex),title=title,status='source_built_requires_compile_and_visual_review',previous_manuscript=str(base_path),previous_manuscript_sha256=sha256_file(base_path),analysis=str(analysis),statistical_mean_source=str(mean_path),statistical_test_source=str(test_path),statistical_summary_manifest=summary,figures=files)
    write_json(out/f'BUILD_STATE_{tag}.json',state);write_json(ROOT/'tmp/current_all_dataset_MDPI_build.json',state);return state

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.execute:print(json.dumps(build(a.root),indent=2),flush=True)
