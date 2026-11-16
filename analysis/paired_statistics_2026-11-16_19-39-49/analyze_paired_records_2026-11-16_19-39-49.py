"""Exploratory paired statistics from explicitly bound CSV records only.

No tensor, image, checkpoint, scientific runner or GPU is loaded. The JSON
configuration fixes inputs, arms, matched seeds and multiplicity families.
New studies must use a new plan/tag/output; prior deliverables are never replaced.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


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


def load_records(config: dict, config_path: Path) -> tuple[dict, list[dict]]:
    tables: dict = {}
    sources = []
    needed = {'Baseline'} | {c[k] for c in config['contrasts'] for k in ('left', 'right')}
    seeds = set(config['seeds'])
    for dataset in config['datasets']:
        name = dataset['name']
        records: dict[tuple[str, int, str], dict] = {}
        for source in dataset['sources']:
            path = (config_path.parent / source['path']).resolve()
            assert sha256(path) == source['sha256'], 'Input bytes changed: ' + str(path)
            sources.append(dict(dataset=name, role=source['role'], path=path.name,
                                sha256=source['sha256']))
            mapping = dict(source['arms'], baseline='Baseline')
            with path.open(encoding='utf-8-sig', newline='') as stream:
                for row in csv.DictReader(stream):
                    if row['output'] != 'raw' or row['arm'] not in mapping:
                        continue
                    arm = mapping[row['arm']]
                    if arm not in needed:
                        continue
                    seed = int(row['refiner_seed'])
                    assert seed == -1 if arm == 'Baseline' else seed in seeds
                    image = row['image']
                    value = float(row['dice'])
                    assert np.isfinite(value) and 0 <= value <= 1
                    subject = row.get('subject', image)
                    key = arm, seed, image
                    record = dict(dataset=name, arm=arm, refiner_seed=seed, image=image,
                                  subject=subject, dice=value)
                    if key in records:
                        assert arm == 'Baseline', 'Duplicate learned record'
                        assert records[key] == record, 'Conflicting baseline across phase sources'
                    else:
                        records[key] = record
        images = sorted(key[2] for key in records if key[:2] == ('Baseline', -1))
        assert len(images) == len(set(images)) == dataset['expected_images']
        for arm in needed:
            expected_seeds = [-1] if arm == 'Baseline' else config['seeds']
            expected = {(arm, seed, image) for seed in expected_seeds for image in images}
            assert {key for key in records if key[0] == arm} == expected, (name, arm, 'incomplete matching')
        subjects = {image: records['Baseline', -1, image]['subject'] for image in images}
        if name == 'CHASE':
            grouped = defaultdict(list)
            for image, subject in subjects.items():
                grouped[subject].append(image)
            assert len(grouped) == 4 and all(len(v) == 2 for v in grouped.values())
            assert all({i[-1] for i in pair} == {'L', 'R'} for pair in grouped.values())
        for key, record in records.items():
            assert record['subject'] == subjects[record['image']], 'Subject inventory differs'
        tables[name] = dict(records=records, images=images, subjects=subjects, unit=dataset['unit'])
    return tables, sources


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(rows[0])
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def number(value: float | None, p=False) -> str:
    if value is None:
        return '--'
    if p and value != 0 and abs(value) < 0.0001:
        mantissa, exponent = f'{value:.3e}'.split('e')
        return f'${mantissa}\\times10^{{{int(exponent)}}}$'
    return f'{value:.6f}'


def create_latex(output: Path, tag: str, image_rows: list[dict], tests: list[dict]) -> None:
    fragments = []
    arms = ['Baseline', 'Direct', 'ESFM', 'O1', 'O2', 'O3']
    for dataset in dict.fromkeys(r['dataset'] for r in image_rows):
        rows = [r for r in image_rows if r['dataset'] == dataset]
        for kind in ('means', 'differences'):
            label = f'tab:completion-{dataset.lower()}-image-{kind}'
            caption = (f'Per-image raw Dice on {dataset}. Learned-model values average the five fitted refiner seeds; the baseline is one fixed model.'
                       if kind == 'means' else f'Per-image raw Dice changes on {dataset}, averaged over the five fitted refiner seeds. B denotes the fixed baseline.')
            columns = arms if kind == 'means' else ['O1-B', 'O2-B', 'O3-B', 'O1-Direct', 'O2-ESFM', 'O3-Direct']
            fragments += [r'\begin{table}[p]', '\\caption{' + caption + '}\\label{' + label + '}',
                          r'\centering\small\setlength{\tabcolsep}{4pt}',
                          r'\begin{tabular}{lrrrrrr}', r'\toprule',
                          'Image & ' + ' & '.join(columns) + r' \\', r'\midrule']
            for row in rows:
                values = ([row[a + '_mean_Dice'] for a in arms] if kind == 'means'
                          else [row['delta_' + c.replace('-B', '-Baseline').replace('-', '_minus_')] for c in columns])
                fragments.append(latex_escape(row['image']) + ' & ' + ' & '.join(number(v) for v in values) + r' \\')
            fragments += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    with (output / f'per_image_tables_{tag}.tex').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(fragments))
    fragments = []
    for level in ('refiner_seed', 'patient_or_image_sensitivity'):
        for family in ('primary', 'baseline_secondary'):
            selected = [r for r in tests if r['analysis_level'] == level and r['family'] == family]
            caption = (f'Exploratory conditional paired Dice comparisons ({level.replace("_", " ")}; {family.replace("_", " ")}). '
                       'Holm correction covers the nine dataset--contrast tests within this family, separately for t and exact signed-rank tests. '
                       'The interval is an unadjusted two-sided t interval for the stated unit, distinct from the previous bootstrap interval.')
            fragments += [r'\begin{table}[p]', '\\caption{' + caption + '}\\label{tab:completion-' + level + '-' + family + '}',
                          r'\centering\scriptsize\setlength{\tabcolsep}{3pt}',
                          r'\begin{tabular}{llrrrrrr}', r'\toprule',
                          r'Dataset & Contrast & $n$ & Mean $\Delta$ & t 95\% CI & $p_t^{\mathrm{Holm}}$ & $p_W$ exact & $p_W^{\mathrm{Holm}}$ \\', r'\midrule']
            for row in selected:
                lo, hi = row['t95ci']
                fragments.append(' & '.join([row['dataset'], row['contrast'].replace('_minus_', ' -- '), str(row['n']),
                    number(row['mean_difference']), f'[{number(lo)}, {number(hi)}]',
                    number(row['paired_t_Holm_pvalue'], True), number(row['exact_Wilcoxon']['pvalue'], True),
                    number(row['exact_Wilcoxon_Holm_pvalue'], True)]) + r' \\')
            fragments += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    with (output / f'paired_test_tables_{tag}.tex').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(fragments))


def analyze(config_path: Path, override_output: Path | None = None) -> Path:
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    tag = config['tag']
    output = override_output or (config_path.parent / config.get('output', '.')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / f'paired_statistics_{tag}.json').exists(), 'Never overwrite an earlier analysis'
    tables, sources = load_records(config, config_path)
    seeds = config['seeds']
    arms = ['Baseline', 'Direct', 'ESFM', 'O1', 'O2', 'O3']
    arms += sorted({c[k] for c in config['contrasts'] for k in ('left', 'right')} - set(arms))
    image_rows, seed_rows, summary, joined = [], [], [], []
    for dataset, table in tables.items():
        records, images = table['records'], table['images']
        for arm in arms:
            if not any(key[0] == arm for key in records):
                continue
            arm_seeds = [-1] if arm == 'Baseline' else seeds
            means = [float(np.mean([records[arm, seed, image]['dice'] for image in images])) for seed in arm_seeds]
            for seed, mean in zip(arm_seeds, means):
                seed_rows.append(dict(dataset=dataset, arm=arm, refiner_seed=seed, mean_Dice=mean,
                                      meaning='single frozen baseline' if arm == 'Baseline' else 'one independently fitted refiner'))
            summary.append(dict(dataset=dataset, arm=arm, mean_Dice=float(np.mean(means)),
                                sample_SD_training_seed_means=None if arm == 'Baseline' else float(np.std(means, ddof=1)),
                                n_fitted_refiners=0 if arm == 'Baseline' else len(seeds), n_baselines=1))
        for image in images:
            row = dict(dataset=dataset, image=image,
                       evaluation_unit=table['subjects'][image] if dataset == 'CHASE' else image)
            for arm in arms:
                if not any(key[0] == arm for key in records):
                    continue
                values = [records[arm, seed, image]['dice'] for seed in ([-1] if arm == 'Baseline' else seeds)]
                row[arm + '_mean_Dice'] = float(np.mean(values))
                row[arm + '_SD_between_refiners'] = None if arm == 'Baseline' else float(np.std(values, ddof=1))
            for contrast in config['contrasts']:
                row['delta_' + contrast['id']] = row[contrast['left'] + '_mean_Dice'] - row[contrast['right'] + '_mean_Dice']
            image_rows.append(row)
        joined += sorted(records.values(), key=lambda r: (r['arm'], r['refiner_seed'], r['image']))
    tests = []
    for dataset, table in tables.items():
        records, images = table['records'], table['images']
        for contrast in config['contrasts']:
            left, right = contrast['left'], contrast['right']
            seed_differences = []
            for seed in seeds:
                left_values = [records[left, -1 if left == 'Baseline' else seed, image]['dice'] for image in images]
                right_values = [records[right, -1 if right == 'Baseline' else seed, image]['dice'] for image in images]
                seed_differences.append(float(np.mean(np.asarray(left_values) - np.asarray(right_values))))
            selected_images = [r for r in image_rows if r['dataset'] == dataset]
            unit_values = defaultdict(list)
            for row in selected_images:
                unit_values[row['evaluation_unit']].append(row['delta_' + contrast['id']])
            units = sorted(unit_values)
            unit_differences = [float(np.mean(unit_values[key])) for key in units]
            for level, differences, ids in [('refiner_seed', seed_differences, seeds),
                                             ('patient_or_image_sensitivity', unit_differences, units)]:
                result = dict(dataset=dataset, contrast=contrast['id'], left=left, right=right,
                              family=contrast['family'], analysis_level=level, unit_ids=ids,
                              evaluation_unit='refiner seed' if level == 'refiner_seed' else table['unit'],
                              **describe_differences(np.asarray(differences)))
                tests.append(result)
            assert np.isclose(np.mean(seed_differences), np.mean(unit_differences), rtol=1e-11, atol=1e-14)
    for level in dict.fromkeys(r['analysis_level'] for r in tests):
        for family in dict.fromkeys(r['family'] for r in tests):
            members = [r for r in tests if r['analysis_level'] == level and r['family'] == family]
            for pkey, outkey, getter in [('paired_t_pvalue', 'paired_t_Holm_pvalue', lambda r: r['paired_t_pvalue']),
                                       ('exact_Wilcoxon', 'exact_Wilcoxon_Holm_pvalue', lambda r: r['exact_Wilcoxon']['pvalue'])]:
                adjusted = holm([getter(r) for r in members])
                for row, p in zip(members, adjusted):
                    row[outkey] = p
                    row['Holm_family_size'] = len(members)
                    row['Holm_family_id'] = level + '/' + family
    write_csv(output / f'per_image_dice_and_differences_{tag}.csv', image_rows)
    write_csv(output / f'raw_seed_means_{tag}.csv', seed_rows)
    write_csv(output / f'raw_mean_SD_{tag}.csv', summary)
    write_csv(output / f'joined_raw_records_{tag}.csv', joined)
    flat = []
    for r in tests:
        flat.append({k: r[k] for k in ('dataset', 'contrast', 'analysis_level', 'evaluation_unit', 'family', 'n',
            'mean_difference', 'sample_SD_difference', 'paired_t_statistic', 'paired_t_df', 'paired_t_pvalue',
            'paired_t_Holm_pvalue', 'paired_t_status', 'Holm_family_size', 'positive_differences', 'negative_differences', 'zero_differences')} |
            dict(t95ci_lower=r['t95ci'][0], t95ci_upper=r['t95ci'][1], exact_Wilcoxon_statistic=r['exact_Wilcoxon']['statistic'],
                 exact_Wilcoxon_pvalue=r['exact_Wilcoxon']['pvalue'], exact_Wilcoxon_Holm_pvalue=r['exact_Wilcoxon_Holm_pvalue']))
    write_csv(output / f'paired_tests_{tag}.csv', flat)
    result = dict(tag=tag, source_plan=config_path.name, source_plan_sha256=sha256(config_path),
                  source_manifest=sources, design=config['policies'], descriptive_summary=summary, tests=tests,
                  source_bytes_unchanged=True, raw_array_model_or_GPU_loaded=False,
                  prior_bootstrap_intervals_recomputed=False,
                  uncertainty='All new inference is exploratory and conditional. Baseline/OOF maps/split are fixed; cohorts and outcomes were already inspected. Image independence is unverified outside CHASE subject grouping.')
    for source in config['datasets']:
        for entry in source['sources']:
            assert sha256(config_path.parent / entry['path']) == entry['sha256']
    with (output / f'paired_statistics_{tag}.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    create_latex(output, tag, image_rows, tests)
    lines = ['# Additional exploratory paired analysis', '',
             'The family plan was registered before this additional calculation, after prior outcome inspection. Only unchanged raw-output CSV records were analyzed. No existing manuscript, previous result, prediction, checkpoint or bootstrap interval was modified.', '',
             'Five refiner fits share one fixed baseline and OOF-map set. Seed-level tests concern refiner variation conditional on that source; they do not estimate full-pipeline variability. Baseline contrasts use a fixed reference, not five independently trained baseline models.', '',
             'Two-sided exact signed-rank tests use the exhaustive random-sign distribution with average ranks and dropped zero differences. With only five nonzero paired refiner differences their minimum two-sided p-value is 0.0625. Paired t tests assume independent approximately normal difference means. Differences after matching seed labels can reduce some variability, but matching labels alone is not proof of a broader randomization design.', '',
             'Holm correction is separate for the nine primary tests and nine baseline-secondary tests, for each analysis level and separately for t and Wilcoxon. This does not control one pooled 18-test family, all metrics, model development or the complete research sequence.', '',
             'CHASE sensitivity uses four two-eye subject means. DRIVE and HRF use 20 and 12 image means respectively because independent patient identities are not established. Their p-values require the stated unit assumptions and do not resolve selection history, annotation provenance, baseline-seed variation or source-data redistribution restrictions.', '',
             '| Level | Dataset | Contrast | Mean difference | t95CI | t p Holm | Exact W p | W p Holm |',
             '|---|---|---|---:|---|---:|---:|---:|']
    for r in tests:
        lines.append(f'|{r["analysis_level"]}|{r["dataset"]}|{r["contrast"]}|{r["mean_difference"]:.9f}|{r["t95ci"]}|{r["paired_t_Holm_pvalue"]}|{r["exact_Wilcoxon"]["pvalue"]}|{r["exact_Wilcoxon_Holm_pvalue"]}|')
    with (output / f'analysis_report_{tag}.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(analyze(args.config.resolve(), args.output.resolve() if args.output else None))
