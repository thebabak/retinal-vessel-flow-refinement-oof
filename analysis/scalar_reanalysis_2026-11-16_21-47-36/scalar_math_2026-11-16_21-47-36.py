"""Identical paired mathematics extracted from the preserved private runner.
CSV numerical reanalysis only; no training, model/prediction or source verification.
"""

from __future__ import annotations
from typing import Any
import hashlib
from pathlib import Path
import numpy as np
from scipy import stats


DATASETS=('CHASE','DRIVE','HRF')
SEEDS=(42,123,2026,3000,3001)


def require(ok,message):
    if not ok:raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def holm(values: list[float | None]) -> list[float | None]:
    """Holm's step-down correction; undefined tests conservatively occupy p=1."""
    p = np.asarray([1.0 if v is None else v for v in values], dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid p-value')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, float(p[index]) * (len(p) - rank))
        adjusted[index] = min(1.0, running)
    return [None if original is None else float(adjusted[i]) for i, original in enumerate(values)]

def exact_signed_rank(differences: np.ndarray) -> dict[str, Any]:
    """Exact two-sided random-sign signed-rank distribution, including ties.

    Drop exact zero differences (wilcox convention), average tied absolute
    ranks and double ranks to make integer DP support. No normal approximation.
    """
    x = np.asarray(differences, dtype=np.float64)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError('Differences must be a finite vector')
    nonzero = x[x != 0]
    if len(nonzero) == 0:
        return dict(statistic=0.0, pvalue=1.0, nonzero_n=0, zero_n=len(x),
                    exact=True, zero_method='wilcox', tied_absolute_differences=False)
    ranks = stats.rankdata(np.abs(nonzero), method='average')
    doubled = np.rint(2 * ranks).astype(int)
    assert np.array_equal(2 * ranks, doubled)
    positive = int(doubled[nonzero > 0].sum())
    negative = int(doubled[nonzero < 0].sum())
    observed = min(positive, negative)
    counts = [1]
    for weight in doubled:
        extended = counts + [0] * int(weight)
        for index, count in enumerate(counts):
            extended[index + int(weight)] += count
        counts = extended
    denominator = 2 ** len(nonzero)
    assert sum(counts) == denominator
    pvalue = min(1.0, 2 * sum(counts[:observed + 1]) / denominator)
    return dict(statistic=observed / 2, pvalue=pvalue,
                nonzero_n=len(nonzero), zero_n=len(x) - len(nonzero), exact=True,
                zero_method='wilcox',
                tied_absolute_differences=len(np.unique(np.abs(nonzero))) != len(nonzero))

def describe_differences(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError('At least two finite paired differences are required')
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / np.sqrt(n)
    if sd == 0:
        t_statistic = pvalue = None
        ci = [mean, mean]
        t_status = 'unavailable_zero_sample_variance'
    else:
        t_statistic = mean / se
        pvalue = float(2 * stats.t.sf(abs(t_statistic), df=n - 1))
        margin = float(stats.t.ppf(0.975, df=n - 1) * se)
        ci = [mean - margin, mean + margin]
        t_status = 'computed_conditional_on_unit_assumptions'
        reference = stats.ttest_1samp(x, 0)
        assert np.isclose(t_statistic, reference.statistic, rtol=1e-12, atol=1e-14)
        assert np.isclose(pvalue, reference.pvalue, rtol=1e-12, atol=1e-16)
    return dict(n=n, mean_difference=mean, sample_SD_difference=sd,
                t95ci=ci, paired_t_statistic=t_statistic, paired_t_df=n - 1,
                paired_t_pvalue=pvalue, paired_t_status=t_status,
                exact_Wilcoxon=exact_signed_rank(x),
                positive_differences=int((x > 0).sum()),
                negative_differences=int((x < 0).sum()), zero_differences=int((x == 0).sum()),
                paired_differences=x.tolist())

def mean_individual_scores(records,dataset,arm,kind,image,metric):
    seeds=(-1,) if arm=='Baseline' else SEEDS
    values=[records[dataset,arm,kind,s,image]['metrics'][metric] for s in seeds]
    return None if any(v is None for v in values) else float(np.mean(values))

def contrast_vectors(records, table, dataset, coefficients):
    """A contrast uses metrics scored separately, never probability averaging."""
    seed_values=[]
    image_values={}
    for image in table['images']:
        block=[]
        for seed in SEEDS:
            block.append(sum(coef*records[dataset,arm,'raw',-1 if arm=='Baseline' else seed,image]['metrics']['dice']
                             for arm,coef in coefficients.items()))
        image_values[image]=float(np.mean(block))
    for seed in SEEDS:
        seed_values.append(float(np.mean([
            sum(coef*records[dataset,arm,'raw',-1 if arm=='Baseline' else seed,image]['metrics']['dice']
                for arm,coef in coefficients.items()) for image in table['images']])))
    unit_names=sorted(table['groups'])
    units=[float(np.mean([image_values[i] for i in table['groups'][unit]])) for unit in unit_names]
    return dict(seed=np.asarray(seed_values),units=np.asarray(units),unit_names=unit_names,images=image_values)

def analyze(records,tables,plan,metrics):
    tests=[];deltas=[]
    for family in ('ablation','external'):
        for contrast in plan[family+'_contrasts']:
            for dataset in DATASETS:
                vectors=contrast_vectors(records,tables[dataset],dataset,contrast['coefficients'])
                for level,key,names in [('refiner_seed','seed',list(SEEDS)),
                                        ('patient_or_image_sensitivity','units',vectors['unit_names'])]:
                    row=dict(family=family,analysis_level=level,dataset=dataset,contrast=contrast['id'],
                        coefficients=contrast['coefficients'],unit_names=names,
                        unit='refiner training seed' if key=='seed' else ('two-eye subject' if dataset=='CHASE' else 'image (patient independence unverified)'),
                        **describe_differences(vectors[key]))
                    tests.append(row)
                deltas.extend(dict(family=family,dataset=dataset,contrast=contrast['id'],image=i,
                    subject_or_image_unit=tables[dataset]['subjects'][i],
                    mean_of_five_individual_Dice_contrasts=v) for i,v in vectors['images'].items())
    for family,size in [('ablation',57),('external',15)]:
        for level in ('refiner_seed','patient_or_image_sensitivity'):
            rows=[r for r in tests if r['family']==family and r['analysis_level']==level]
            require(len(rows)==size,'Registered family is incomplete')
            t=holm([r['paired_t_pvalue'] for r in rows]);w=holm([r['exact_Wilcoxon']['pvalue'] for r in rows])
            for r,tp,wp in zip(rows,t,w):
                r.update(family_size=size,paired_t_Holm_pvalue=tp,exact_Wilcoxon_Holm_pvalue=wp)
    image_metrics=[];seed_metrics=[];summaries=[]
    arms=sorted({k[1] for k in records})
    for dataset,table in tables.items():
        for arm in arms:
            kinds=sorted({k[2] for k in records if k[0]==dataset and k[1]==arm})
            for kind in kinds:
                seeds=(-1,) if arm=='Baseline' else SEEDS
                for metric in metrics:
                    for image in table['images']:
                        values=[records[dataset,arm,kind,s,image]['metrics'][metric] for s in seeds]
                        valid=all(v is not None for v in values)
                        image_metrics.append(dict(dataset=dataset,arm=arm,output=kind,metric=metric,image=image,
                            subject_or_image_unit=table['subjects'][image],fitted_models=1 if arm=='Baseline' else 5,
                            mean_individual_score=float(np.mean(values)) if valid else None,
                            sample_SD_individual_scores=float(np.std(values,ddof=1)) if valid and len(values)>1 else None,
                            status='available' if valid else 'unavailable_in_source_CSV'))
                    means=[]
                    for seed in seeds:
                        values=[records[dataset,arm,kind,seed,i]['metrics'][metric] for i in table['images']]
                        mean=None if any(v is None for v in values) else float(np.mean(values))
                        means.append(mean)
                        seed_metrics.append(dict(dataset=dataset,arm=arm,output=kind,metric=metric,refiner_seed=seed,
                            image_count=len(values),mean_individual_image_score=mean))
                    valid=all(v is not None for v in means)
                    summaries.append(dict(dataset=dataset,arm=arm,output=kind,metric=metric,
                        role='primary_raw_Dice' if kind=='raw' and metric=='dice' else 'descriptive_only',
                        fitted_models=1 if arm=='Baseline' else 5,
                        mean_of_training_seed_means=float(np.mean(means)) if valid else None,
                        sample_SD_training_seed_means=float(np.std(means,ddof=1)) if valid and len(means)>1 else None,
                        status='available' if valid else 'unavailable_in_source_CSV'))
    return dict(tests=tests,per_image_contrasts=deltas,per_image_metrics=image_metrics,
                seed_metrics=seed_metrics,mean_SD=summaries)

def flat_test(r):
    return dict(family=r['family'],analysis_level=r['analysis_level'],dataset=r['dataset'],contrast=r['contrast'],
        n=r['n'],unit=r['unit'],mean_difference=r['mean_difference'],sample_SD_difference=r['sample_SD_difference'],
        t95ci_low=r['t95ci'][0],t95ci_high=r['t95ci'][1],paired_t_statistic=r['paired_t_statistic'],
        paired_t_df=r['paired_t_df'],paired_t_pvalue=r['paired_t_pvalue'],paired_t_Holm_pvalue=r['paired_t_Holm_pvalue'],
        exact_Wilcoxon_statistic=r['exact_Wilcoxon']['statistic'],exact_Wilcoxon_pvalue=r['exact_Wilcoxon']['pvalue'],
        exact_Wilcoxon_Holm_pvalue=r['exact_Wilcoxon_Holm_pvalue'],Wilcoxon_nonzero_n=r['exact_Wilcoxon']['nonzero_n'],
        family_size=r['family_size'],paired_t_status=r['paired_t_status'])
