"""Synthetic scalar fixtures only. No private metadata, actual result or model."""
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reanalysis',HERE/'reanalyze_scalar_tables_2026-11-16_21-47-36.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
CORE,BINDINGS=R.load_core()
PLAN=R.read_json(HERE/'public_analysis_plan_2026-11-16_21-47-36.json')

def write_fixture(root):
    plan=root/'public_plan.json';plan.write_text(json.dumps(PLAN),encoding='utf-8')
    manifest=dict(schema_version=1,ready=True,numerical_table_reanalysis_only=True,
        plan=dict(path=plan.name,sha256=R.sha(plan)),descriptive_metrics=['dice','cldice'],datasets=[])
    needed=sorted({'Baseline'}|{a for f in ('ablation_contrasts','external_contrasts') for c in PLAN[f] for a in c['coefficients']})
    records={};tables={}
    for dataset in R.DATASETS:
        images=([f'case{s:02d}{eye}' for s in range(4) for eye in 'LR'] if dataset=='CHASE'
                else [f'image{i:02d}' for i in range(R.N_IMAGES[dataset])])
        rows=[];groups={};subjects={}
        for j,arm in enumerate(needed):
            for k,seed in enumerate((-1,) if arm=='Baseline' else R.SEEDS):
                for n,image in enumerate(images):
                    value=.7+j*.001+k*.00002*(j+1)+n*.000001*(j+1)
                    subject=image[:-1] if dataset=='CHASE' else image
                    rows.append(dict(arm=arm,refiner_seed=seed,image=image,subject=subject,output='raw',dice=value,cldice=''))
                    records[dataset,arm,'raw',seed,image]=dict(dataset=dataset,arm=arm,output='raw',seed=seed,image=image,
                        subject=subject,metrics=dict(dice=value,cldice=None))
                    subjects[image]=subject
        for image,subject in subjects.items():groups.setdefault(subject,[]).append(image)
        tables[dataset]=dict(images=images,subjects=subjects,groups=groups)
        path=root/f'{dataset}.csv';R.write_csv(path,rows)
        manifest['datasets'].append(dict(name=dataset,expected_images=len(images),image_ids=images,
            csv=dict(path=path.name,sha256=R.sha(path),size_bytes=path.stat().st_size)))
    path=root/'ready_inputs.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return path,manifest,records,tables

class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path,self.manifest,self.records,self.tables=write_fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def inputs(self):return R.validate_manifest(self.manifest,self.root,BINDINGS)

    def test_default_is_inert_without_opening_CSV(self):
        with patch.object(R,'execute',side_effect=AssertionError('No execution')),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(R.main(['--manifest',str(self.path)]),0)
        self.assertIn('INERT_NO_CSV_READ',out.getvalue())

    def test_incomplete_template_blocks_before_CSV_loading(self):
        template=R.read_json(HERE/'scalar_inputs_TEMPLATE_INCOMPLETE_2026-11-16_21-47-36.json')
        with self.assertRaisesRegex(ValueError,'deliberately incomplete'):
            R.validate_manifest(template,HERE,BINDINGS)

    def test_numeric_join_matches_original_scalar_values_exactly(self):
        plan,sources,_=self.inputs();records,tables=R.load_scalar_records(self.manifest,plan,sources)
        self.assertEqual(records,self.records);self.assertEqual(tables,self.tables)
        expected=CORE.analyze(self.records,self.tables,PLAN,['dice','cldice'])
        actual=CORE.analyze(records,tables,plan,['dice','cldice'])
        self.assertEqual(actual,expected);self.assertEqual(len(actual['tests']),144)
        for unit in ('refiner_seed','patient_or_image_sensitivity'):
            for family,count in [('ablation',57),('external',15)]:
                self.assertEqual(len([r for r in actual['tests'] if r['family']==family and r['analysis_level']==unit]),count)

    def test_signed_interaction_and_known_paired_math(self):
        dataset='CHASE'
        for seed in R.SEEDS:
            for image in self.tables[dataset]['images']:
                for arm,value in [('Direct',.7),('thin',.71),('anchor',.72),('O1',.75)]:
                    self.records[dataset,arm,'raw',seed,image]['metrics']['dice']=value
        v=CORE.contrast_vectors(self.records,self.tables[dataset],dataset,dict(O1=1,thin=-1,anchor=-1,Direct=1))
        np.testing.assert_allclose(v['seed'],.02,atol=1e-14)
        stats=CORE.describe_differences(np.arange(1.,6.))
        self.assertAlmostEqual(stats['paired_t_pvalue'],.013235599563682695)
        self.assertEqual(CORE.exact_signed_rank(np.arange(1.,6.))['pvalue'],.0625)
        np.testing.assert_allclose(CORE.holm([.01,.04,.03]),[.03,.06,.06])

    def test_changed_CSV_hash_blocks(self):
        p=self.root/'CHASE.csv'
        with p.open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'size/SHA256 differs'):self.inputs()

    def test_missing_arm_seed_row_blocks_even_with_rebound_file_hash(self):
        p=self.root/'CHASE.csv'
        with p.open(newline='') as f:rows=list(csv.DictReader(f))
        rows.pop();p.unlink();R.write_csv(p,rows)
        self.manifest['datasets'][0]['csv'].update(sha256=R.sha(p),size_bytes=p.stat().st_size)
        plan,sources,_=self.inputs()
        with self.assertRaisesRegex(ValueError,'Full matched'):R.load_scalar_records(self.manifest,plan,sources)

    def test_changed_family_is_rejected(self):
        plan=deepcopy(PLAN);plan['ablation_contrasts'].pop()
        p=self.root/'public_plan.json';p.write_text(json.dumps(plan))
        self.manifest['plan']['sha256']=R.sha(p)
        with self.assertRaisesRegex(ValueError,'families changed'):self.inputs()

    def test_relative_path_guard_rejects_escape_and_machine_paths(self):
        for value in ('../secret.csv','C:/private.csv','/tmp/source.csv','folder\\file.csv'):
            with self.assertRaises(ValueError):R.relative_file(self.root,value)

    def test_before_after_hash_guard_blocks_mutation_during_analysis(self):
        original=CORE.analyze
        def mutating(*args):
            result=original(*args)
            with (self.root/'DRIVE.csv').open('a') as f:f.write('\n')
            return result
        with patch.object(R,'load_core',return_value=(CORE,BINDINGS)),patch.object(CORE,'analyze',side_effect=mutating):
            with self.assertRaisesRegex(ValueError,'changed during'):
                R.execute(self.path,self.root/'out_2026-11-16_21-47-36')
        self.assertFalse((self.root/'out_2026-11-16_21-47-36').exists())

    def test_end_to_end_is_explicit_numerical_reanalysis_without_private_paths(self):
        out=self.root/'out_2026-11-16_21-47-36'
        with redirect_stdout(io.StringIO()):report=R.execute(self.path,out)
        self.assertTrue(report['numerical_table_reanalysis_only'])
        self.assertFalse(report['predictions_evaluated']);self.assertFalse(report['dataset_source_or_execution_provenance_verified'])
        self.assertFalse(report['private_authorization_records_required'])
        self.assertEqual(len(list(out.glob('paired_tests_*.csv'))),1)
        self.assertNotIn(str(self.root),json.dumps(report))
        self.assertEqual(len(report['tests']),144)

if __name__=='__main__':unittest.main(verbosity=2)
