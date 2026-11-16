"""Synthetic path/gate checks only: no retinal input, model run or CUDA work."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,os,subprocess,sys,tempfile,unittest
from unittest.mock import patch
from official_comparisons import portable_api as api

PACKAGE=Path(__file__).resolve().parents[1]
FIXED=json.loads((PACKAGE/'official_comparisons/FIXED_SPLITS.json').read_text())

class PortableTests(unittest.TestCase):
    def tearDown(self):api._CONFIG=None

    def fixture(self,temp):
        workspace=Path(temp)/'relocated_artifacts';workspace.mkdir();inputs=workspace/'private_prepared';inputs.mkdir()
        output=Path(temp)/'separate_runs';manifest=dict(schema_version=1,input_root='private_prepared',datasets={})
        for dataset in api.DATASETS:
            spec=dict(split=copy.deepcopy(FIXED[dataset]),samples={},baseline_references={},thin_cutoff=1.,ground_truth_provenance='Synthetic test fixture only; no retinal data')
            for name in sum(spec['split'].values(),[]):
                relative=f'private_prepared/{dataset}_{name}.npz';path=workspace/relative;path.write_bytes(b'synthetic placeholder; never a model or real image')
                record=dict(path=relative,sha256=api.sha(path))
                if dataset=='CHASE':record['subject']=int(name.split('_')[1][:2])
                spec['samples'][name]=record
            for name in spec['split']['test']:
                relative=f'private_prepared/{dataset}_{name}_reference.npz';path=workspace/relative;path.write_bytes(b'synthetic baseline placeholder')
                spec['baseline_references'][name]=dict(path=relative,sha256=api.sha(path))
            manifest['datasets'][dataset]=spec
        path=workspace/'prepared_inputs.json';api_path=path
        path.write_text(json.dumps(manifest));return workspace,output,path,manifest

    def configure_fixture(self,temp):
        workspace,output,path,manifest=self.fixture(temp)
        config=api.configure(workspace,output,path.name);return config,manifest

    def make_global_lock(self,config):
        tag='2026-11-16_20-40-37';run=config.output_root/f'synthetic_{tag}';run.mkdir(parents=True);decisions={};configs={}
        for dataset in api.DATASETS:
            decisions[dataset]=[];folder=run/dataset;folder.mkdir();cfg=folder/f'configuration_before_dataset_final_fitting_{tag}.json';cfg.write_text('{}');configs[dataset]=api.sha(cfg)
            for seed in api.SEEDS:
                dest=folder/'final'/f'seed_{seed}';dest.mkdir(parents=True);checkpoint=dest/'synthetic.pt';checkpoint.write_bytes(b'synthetic bytes; never deserialized')
                choice=dict(seed=seed,checkpoint=checkpoint.name,checkpoint_sha256=api.sha(checkpoint));decisions[dataset].append(choice)
                (dest/f'selection_locked_{tag}.json').write_text(json.dumps(choice))
        lock=run/f'all_selections_before_test_{tag}.json';lock.write_text(json.dumps(dict(decisions=decisions,dataset_configuration_locks_sha256=configs)))
        return run,tag,lock

    def test_module_imports_are_inert_without_torch_numpy_or_CUDA(self):
        code="import sys; import official_comparisons.segflow; import official_comparisons.frunet; from official_comparisons import portable_api; assert portable_api._CONFIG is None; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-c',code],cwd=PACKAGE,env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_paths_relocate_and_output_is_confined(self):
        with tempfile.TemporaryDirectory() as temp:
            config,_=self.configure_fixture(temp)
            self.assertEqual(api.checked_output('run_2026-11-16_20-40-37'),config.output_root/'run_2026-11-16_20-40-37')
            with self.assertRaises(ValueError):api.checked_output('../escape')
            with self.assertRaises(ValueError):api.checked_output(config.workspace_root/'private_prepared/new_output')
            with self.assertRaises(ValueError):api.checked_output(config.output_root)
            self.assertFalse(config.output_root.exists())

    def test_CLI_without_execute_does_not_access_missing_inputs_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'not_created';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
            for method in ['segflow','frunet']:
                args=[sys.executable,'-m','official_comparisons.'+method,'--workspace-root',str(Path(temp)/'missing_workspace'),'--output-root',str(output),'--input-manifest','missing.json','--output','run_2026-11-16_20-40-37']
                result=subprocess.run(args,cwd=PACKAGE,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr);self.assertIn('No experiment',result.stdout)
            self.assertFalse(output.exists())

    def test_absolute_and_traversal_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for invalid in ['../outside','/absolute/path','C:/machine/path','F:\\machine\\path','C:drive_relative']:
                with self.assertRaises(ValueError,msg=invalid):api.local_path(root,invalid)

    def test_output_root_must_not_overlap_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace,_,path,_=self.fixture(temp)
            with self.assertRaisesRegex(ValueError,'overlaps'):api.configure(workspace,workspace/'private_prepared/results',path.name)

    def test_fixed_partition_order_and_subject_groups_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace,output,path,manifest=self.fixture(temp)
            manifest['datasets']['CHASE']['split']['train'].reverse();path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'IDs/order'):api.configure(workspace,output,path.name)

    def test_late_test_IO_is_held_without_loading_array(self):
        with tempfile.TemporaryDirectory() as temp:
            config,_=self.configure_fixture(temp);ref=api.load_data('CHASE');name=ref.manifest['test'][0]
            with patch.object(api,'_verified_record',side_effect=AssertionError('Input file was accessed')):
                with self.assertRaisesRegex(RuntimeError,'Test input/GT access held'):ref.data[name]
                with self.assertRaisesRegex(RuntimeError,'reference access is held'):api.baseline_archive('CHASE',name)

    def test_global15choice_lock_and_explicit_finalize_required(self):
        with tempfile.TemporaryDirectory() as temp:
            config,_=self.configure_fixture(temp);run,tag,lock=self.make_global_lock(config)
            with self.assertRaisesRegex(RuntimeError,'explicit finalization'):api.authorize_test_inputs(run,tag,api.DATASETS,api.SEEDS,False)
            doc=json.loads(lock.read_text());doc['decisions']['DRIVE'].pop();lock.write_text(json.dumps(doc))
            with self.assertRaisesRegex(RuntimeError,'inventory incomplete'):api.authorize_test_inputs(run,tag,api.DATASETS,api.SEEDS,True)
            self.assertFalse(config.test_access_allowed)

    def test_complete_lock_unlocks_only_after_hashes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            config,_=self.configure_fixture(temp);run,tag,_=self.make_global_lock(config)
            api.authorize_test_inputs(run,tag,api.DATASETS,api.SEEDS,True);self.assertTrue(config.test_access_allowed)
        with tempfile.TemporaryDirectory() as temp:
            config,_=self.configure_fixture(temp);run,tag,_=self.make_global_lock(config)
            (run/'HRF/final/seed_3001/synthetic.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'checkpoint differs'):api.authorize_test_inputs(run,tag,api.DATASETS,api.SEEDS,True)
            self.assertFalse(config.test_access_allowed)

    def test_input_manifest_cannot_change_before_test_unlock(self):
        with tempfile.TemporaryDirectory() as temp:
            config,_=self.configure_fixture(temp);run,tag,_=self.make_global_lock(config)
            config.manifest_path.write_text(config.manifest_path.read_text()+'\n')
            with self.assertRaisesRegex(RuntimeError,'manifest changed'):api.authorize_test_inputs(run,tag,api.DATASETS,api.SEEDS,True)

    def test_snapshot_verifier_detects_changed_bytes_without_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code=root/'model.py';code.write_text('# synthetic source')
            (root/'SOURCE.json').write_text(json.dumps(dict(commit='pinned',files={'model.py':api.sha(code)})))
            self.assertEqual(api.verify_snapshot(root,'pinned')['commit'],'pinned')
            code.write_text('# changed')
            with self.assertRaisesRegex(RuntimeError,'snapshot bytes changed'):api.verify_snapshot(root,'pinned')

    def test_validation_score_matches_known_FOV_bookkeeping(self):
        import torch
        p=torch.tensor([.9,.8,.2,.1]);g=torch.tensor([1.,0.,1.,0.]);fov=torch.tensor([True,False,True,True])
        full=api._score(p,g);masked=api._score(p,g,fov)
        self.assertEqual(full['dice'],.5);self.assertEqual(masked['tp'],1);self.assertEqual(masked['fp'],0);self.assertEqual(masked['fn'],1);self.assertEqual(masked['tn'],1)
        self.assertAlmostEqual(masked['dice'],2/3);self.assertFalse(torch.cuda.is_initialized())

    def test_export_contains_no_data_weights_or_correspondence(self):
        forbidden={'.npz','.npy','.pth','.pt','.pdf','.docx','.png','.jpg','.gif','.tif','.tiff','.eml'}
        self.assertFalse([p for p in PACKAGE.rglob('*') if p.is_file() and p.suffix.lower() in forbidden])

if __name__=='__main__':unittest.main(verbosity=2)
