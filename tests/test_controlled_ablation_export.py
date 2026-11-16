"""Synthetic path, inert-import and scientific-AST checks; no model execution."""
from pathlib import Path
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/"flow_matching_refiner/controlled_ablations"
SNAPSHOT=next((ROOT/"provenance").glob("original_controlled_ablations_*"))


def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


class ExportASTTests(unittest.TestCase):
    def test_operations_and_specs_bytes_preserved(self):
        for name in ("operations.py","specs.py"):
            self.assertEqual((PACKAGE/name).read_bytes(),(SNAPSHOT/name).read_bytes())
    def test_all_scientific_runner_functions_AST_preserved(self):
        old,new=functions(SNAPSHOT/"runner.py"),functions(PACKAGE/"runner.py")
        for name in old.keys()-{"main","protected_inventory"}:
            self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False),name)
    def test_main_only_output_root_guard_changed(self):
        old,new=functions(SNAPSHOT/"runner.py")["main"],functions(PACKAGE/"runner.py")["main"]
        old.body=[node for node in old.body if not (isinstance(node,ast.Assert) and "outputs/reviewer_completion_2026-11-16_19-37-22" in ast.dump(node))]
        new.body=[node for node in new.body if not (isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and getattr(node.value.func,"id",None)=="assert_output_path")]
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
    def test_adapted_code_has_no_private_machine_root(self):
        for name in ("runner.py","cli.py"):
            text=(PACKAGE/name).read_text()
            self.assertNotIn("F:/",text);self.assertNotIn("F:\\",text)
            self.assertNotIn("current_three_options_MDPI_delivery",text)
    def test_export_contains_no_data_models_manuscripts_or_responses(self):
        bad_extensions={".npz",".npy",".png",".tif",".tiff",".jpg",".pt",".pth",".pdf",".tex",".docx"}
        # Scope this guard to the added export. An integrated repository may
        # separately contain an authorized paper PDF or LaTeX statistics tables.
        additions=list(PACKAGE.rglob("*"))+list(SNAPSHOT.rglob("*"))
        additions += list((ROOT/"configs").glob("controlled_ablation_registered_plan_*.json"))
        additions += list((ROOT/"docs").glob("CONTROLLED_ABLATIONS_*.md"))
        additions += list((ROOT/"provenance").glob("public_ablation_export_manifest_*.json"))
        for path in additions:
            if not path.is_file():continue
            self.assertNotIn(path.suffix.lower(),bad_extensions,str(path))
            self.assertNotIn("responses_to_reviewers",path.name.lower())
            self.assertNotIn("template261006",path.name.lower())


class PortableGateTests(unittest.TestCase):
    def child(self,code):
        environment=os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"]="1"
        environment["CUDA_VISIBLE_DEVICES"]="-1"
        environment["PYTHONPATH"]=str(ROOT)
        result=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=environment,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr or result.stdout)
    def test_inert_runner_and_CLI_import(self):
        self.child("import sys; from flow_matching_refiner.controlled_ablations import runner,cli; assert 'torch' not in sys.modules; assert 'flow_matching_refiner.data' not in sys.modules")
    def test_default_description_loads_no_inputs_or_models(self):
        self.child("import sys; from flow_matching_refiner.controlled_ablations.cli import main; r=main([]); assert r['status']=='description_only'; assert r['plan']['new_final_fits']==240; assert 'torch' not in sys.modules; assert 'flow_matching_refiner.data' not in sys.modules")
    def configuration(self,folder):
        folder=Path(folder)
        path=folder/"paths.json"
        path.write_text(json.dumps(dict(artifact_root=str(folder/"retained"),output_root=str(folder/"runs"))))
        return path
    def configured_child(self,folder,body):
        config=self.configuration(folder)
        self.child("from pathlib import Path\nimport json,io\nfrom contextlib import redirect_stdout\nfrom unittest.mock import patch\n"
            "from flow_matching_refiner.controlled_ablations import cli\n"
            f"config=Path({str(config)!r})\nfolder=Path({str(folder)!r})\n"+body)
    def test_registration_is_data_free_and_records_all16arms(self):
        with tempfile.TemporaryDirectory() as folder:
            self.configured_child(folder,"""
out=folder/"runs/ablations_2026-11-16_00-00-00"
with redirect_stdout(io.StringIO()):result=cli.main(["--config",str(config),"--register","--output",str(out)])
assert not result["training_performed"]
design=json.loads(next(out.glob("prespecified_design_*.json")).read_text())
assert len(design["arms"])==16 and design["new_final_fits"]==240
assert not (folder/"retained").exists()
""")
    def test_registration_rejects_existing_result_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            self.configured_child(folder,"""
out=folder/"runs/ablations_2026-11-16_00-00-00"
out.mkdir(parents=True);marker=out/"original.txt";marker.write_text("unchanged")
try:cli.main(["--config",str(config),"--register","--output",str(out)])
except FileExistsError:pass
else:raise AssertionError("Existing run was accepted")
assert marker.read_text()=="unchanged"
""")
    def test_registration_rejects_missing_timestamp(self):
        with tempfile.TemporaryDirectory() as folder:
            self.configured_child(folder,"""
out=folder/"runs/undated"
try:cli.main(["--config",str(config),"--register","--output",str(out)])
except ValueError:pass
else:raise AssertionError("Undated output was accepted")
assert not out.exists()
""")
    def test_output_may_not_escape_configured_output_root(self):
        with tempfile.TemporaryDirectory() as folder:
            self.configured_child(folder,"""
out=folder/"retained/ablations_2026-11-16_00-00-00"
try:cli.main(["--config",str(config),"--register","--output",str(out)])
except ValueError:pass
else:raise AssertionError("Retained-input output was accepted")
assert not out.exists()
""")
    def test_execute_cannot_skip_strict_preflight_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            self.configured_child(folder,"""
with patch.object(cli,"preflight",side_effect=FileNotFoundError("missing original bytes")) as guard:
 try:cli.main(["--config",str(config),"--execute","--output",str(folder/"runs/missing")])
 except FileNotFoundError as error:assert "missing original bytes" in str(error)
 else:raise AssertionError("Strict preflight failure was ignored")
 guard.assert_called_once()
""")


if __name__=="__main__":unittest.main(verbosity=2)
