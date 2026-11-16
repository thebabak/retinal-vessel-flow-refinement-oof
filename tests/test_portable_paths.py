"""Synthetic portability/dispatch checks; never execute an experiment.

Real path-configuration checks run in fresh child interpreters so they cannot
inherit a previously imported scientific loader. Fixtures contain only small
text/JSON files. CLI dispatch uses mocks, including when --execute is supplied.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[1]


class PortablePathTests(unittest.TestCase):
    def fixture(self, folder, **changes):
        folder = Path(folder)
        artifact = folder / "retained_artifacts"
        artifact.mkdir()
        document = {
            "artifact_root": "retained_artifacts",
            "output_root": "new_runs",
            "raw_roots": {name: "raw/" + name for name in ("CHASE", "DRIVE", "DRIVE_TEST_GT", "HRF")},
        }
        document.update(changes)
        path = folder / "paths.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path, artifact

    def child(self, config, assertions):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("RETINAL_")}
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
        code = (
            "import sys\nfrom pathlib import Path\n"
            "from flow_matching_refiner import portable_paths as paths\n"
            "configuration = Path(sys.argv[1])\n" + assertions
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(config)], cwd=REPOSITORY,
            env=environment, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)

    def test_longest_windows_alias_case_and_path_boundary(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, artifact = self.fixture(folder, recorded_path_aliases={
                "R:\\Legacy": "original",
                "R:\\Legacy\\Project": "original/long_match",
            })
            for relative, content in (
                ("original/Project/payload.bin", b"wrong shorter alias"),
                ("original/long_match/payload.bin", b"original longest alias"),
                ("original/ProjectSibling/payload.bin", b"sibling boundary"),
            ):
                path = artifact / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            self.child(config, r'''
cfg = paths.configure(configuration)
assert paths.current() is cfg
assert cfg.artifact_root == (configuration.parent / "retained_artifacts").resolve()
resolved = paths.resolve_recorded_path(r"r:\legacy\pRoJeCt\payload.bin")
assert resolved == cfg.artifact_root / "original/long_match/payload.bin"
assert resolved.read_bytes() == b"original longest alias"
sibling = paths.resolve_recorded_path(r"R:\Legacy\ProjectSibling\payload.bin")
assert sibling == cfg.artifact_root / "original/ProjectSibling/payload.bin"
assert sibling.read_bytes() == b"sibling boundary"
''')

    def test_list_alias_reads_original_snapshot_not_adapted_code(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, artifact = self.fixture(folder, recorded_path_aliases=[{
                "recorded_prefix": "R:\\Legacy\\Project\\flow_matching_refiner",
                "target": "original_source/flow_matching_refiner",
            }])
            source = artifact / "original_source/flow_matching_refiner/data.py"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"ORIGINAL RETAINED SOURCE BYTES\n")
            self.child(config, r'''
cfg = paths.configure(configuration)
resolved = paths.resolve_recorded_path(r"R:\Legacy\Project\flow_matching_refiner\data.py")
assert resolved == cfg.artifact_root / "original_source/flow_matching_refiner/data.py"
assert resolved != paths.CODE_ROOT / "flow_matching_refiner/data.py"
assert resolved.read_bytes() == b"ORIGINAL RETAINED SOURCE BYTES\n"
''')

    def test_missing_alias_and_missing_artifact_fail_strictly(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, artifact = self.fixture(folder)
            (artifact / "fixture").mkdir()
            (artifact / "fixture/record.bin").write_bytes(b"retained fixture")
            self.child(config, r'''
cfg = paths.configure(configuration)
for value in (r"R:\UNMAPPED_RETAINED_UNIT_TEST\missing.bin", "missing.bin"):
    try:
        paths.resolve_recorded_path(value)
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError("Missing historical path was silently accepted")
resolved = paths.resolve_recorded_path("record.bin", base=cfg.artifact_root / "fixture")
assert resolved.read_bytes() == b"retained fixture"
assert paths.resolve_recorded_path("planned.bin", must_exist=False) == cfg.artifact_root / "planned.bin"
''')

    def test_alias_parent_traversal_cannot_leave_original_target(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, artifact = self.fixture(folder, recorded_path_aliases={
                "R:\\Legacy\\Project": "original/target",
            })
            (artifact / "original/target").mkdir(parents=True)
            (artifact / "original/outside.bin").write_bytes(b"outside the aliased snapshot")
            self.child(config, r'''
paths.configure(configuration)
try:
    paths.resolve_recorded_path(r"R:\Legacy\Project\..\outside.bin")
except ValueError:
    pass
else:
    raise AssertionError("Alias traversal escaped into another retained subtree")
''')

    def test_alias_symlink_cannot_leave_original_target(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, artifact = self.fixture(folder, recorded_path_aliases={
                "R:\\Legacy\\Project": "original/target",
            })
            target = artifact / "original/target"
            target.mkdir(parents=True)
            outside = artifact / "different_snapshot"
            outside.mkdir()
            (outside / "payload.bin").write_bytes(b"outside the aliased snapshot")
            try:
                (target / "link").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Directory symlinks are unavailable for this test account")
            self.child(config, r'''
paths.configure(configuration)
try:
    paths.resolve_recorded_path(r"R:\Legacy\Project\link\payload.bin")
except ValueError:
    pass
else:
    raise AssertionError("Alias-internal symlink escaped the original target")
''')

    def test_output_cannot_enter_artifact_or_raw_roots(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, _ = self.fixture(folder, output_root="retained_artifacts/new_runs")
            self.child(config, '''
try:
    paths.configure(configuration)
except ValueError:
    pass
else:
    raise AssertionError("Output inside retained artifacts was accepted")
''')
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            raw = Path(folder) / "separate_raw"
            raw.mkdir()
            config, _ = self.fixture(folder, output_root=str(raw / "new_runs"),
                                     raw_roots={"CHASE": str(raw)})
            self.child(config, '''
try:
    paths.configure(configuration)
except ValueError:
    pass
else:
    raise AssertionError("Output inside a raw-data root was accepted")
''')

    def test_historical_alias_cannot_substitute_adapted_source(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, _ = self.fixture(folder, recorded_path_aliases={
                "R:\\OriginalSource": str(REPOSITORY / "flow_matching_refiner"),
            })
            self.child(config, '''
try:
    paths.configure(configuration)
except ValueError:
    pass
else:
    raise AssertionError("Adapted source was accepted as original artifact bytes")
''')

    def test_output_cannot_enter_controls_or_v3_inside_output_root(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            output = Path(folder) / "new_runs"
            control = output / "retained_control"
            v3 = output / "retained_v3"
            control.mkdir(parents=True)
            v3.mkdir()
            config, _ = self.fixture(folder, controls={"CHASE": str(control)}, v3_study=str(v3))
            self.child(config, '''
cfg = paths.configure(configuration)
for retained in (cfg.control("CHASE"), cfg.v3_root):
    try:
        paths.assert_output_path(retained / "new_2030-01-01_00-00-00")
    except ValueError:
        pass
    else:
        raise AssertionError("Output entered an input archive nested inside output_root")
fresh = cfg.output_root / "fresh_2030-01-01_00-00-00"
assert paths.assert_output_path(fresh) == fresh
assert not fresh.exists()
''')

    def test_configure_refuses_after_scientific_loader_import(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, _ = self.fixture(folder)
            self.child(config, '''
from types import ModuleType
sys.modules["flow_matching_refiner.data"] = ModuleType("flow_matching_refiner.data")
try:
    paths.configure(configuration)
except RuntimeError:
    pass
else:
    raise AssertionError("Configuration changed after scientific-loader import")
assert not ({"torch", "numpy", "cv2", "PIL"} & set(sys.modules))
''')

    def test_missing_preflight_is_inert_and_does_not_make_outputs(self):
        with tempfile.TemporaryDirectory(prefix="portable_paths_fixture_") as folder:
            config, _ = self.fixture(folder)
            self.child(config, '''
cfg = paths.configure(configuration)
assert not ({"torch", "numpy", "cv2", "PIL"} & set(sys.modules))
try:
    paths.preflight()
except FileNotFoundError:
    pass
else:
    raise AssertionError("Incomplete retained archives passed preflight")
assert not ({"torch", "numpy", "cv2", "PIL"} & set(sys.modules))
assert not cfg.output_root.exists()
''')


class PortableCLITests(unittest.TestCase):
    @staticmethod
    def checked():
        return dict(status="passed", required_files=0, historical_digest_bindings_verified=0,
                    experiment_evaluation_performed=False, training_performed=False,
                    data_or_tensor_loaded=False)

    def test_default_cli_never_dispatches_scientific_code(self):
        from flow_matching_refiner import portable_cli as cli
        for phase in ("objectives", "native", "controls", "report"):
            with self.subTest(phase=phase), patch.object(cli, "configure") as configure, \
                 patch.object(cli, "preflight", return_value=self.checked()), \
                 patch.object(cli.importlib, "import_module") as import_module, redirect_stdout(io.StringIO()):
                result = cli.main(["--phase", phase])
                self.assertEqual(result["status"], "passed")
                configure.assert_called_once_with(None)
                import_module.assert_not_called()

    def test_phase_wide_lock_rejects_subset_before_dispatch(self):
        from flow_matching_refiner import portable_cli as cli
        for phase in ("objectives", "native"):
            with self.subTest(phase=phase), patch.object(cli, "configure"), \
                 patch.object(cli, "preflight", return_value=self.checked()), \
                 patch.object(cli.importlib, "import_module") as import_module, redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    cli.main(["--phase", phase, "--dataset", "CHASE", "--execute"])
                import_module.assert_not_called()

    def test_execute_dispatch_is_mocked_and_preserves_arguments(self):
        from flow_matching_refiner import portable_cli as cli
        for phase, expected_module in (
            ("objectives", "flow_matching_refiner.run_refinement_v3"),
            ("native", "flow_matching_refiner.run_native_patch"),
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory(prefix="portable_cli_fixture_") as folder:
                output_root = Path(folder) / "new_runs"
                output_root.mkdir()
                output = output_root / (phase + "_2030-01-01_00-00-00")
                calls = []
                def fake_main():
                    calls.append(sys.argv[1:])
                    return "mock-only"
                fake_module = SimpleNamespace(main=fake_main)
                with patch.object(cli, "configure"), patch.object(cli, "preflight", return_value=self.checked()), \
                     patch.object(cli, "current", return_value=SimpleNamespace(output_root=output_root)), \
                     patch.object(cli, "assert_output_path", side_effect=lambda value, **kwargs: Path(value).resolve()), \
                     patch.object(cli.importlib, "import_module", return_value=fake_module) as import_module, \
                     redirect_stdout(io.StringIO()):
                    result = cli.main(["--phase", phase, "--execute", "--output", str(output)])
                self.assertEqual(result, "mock-only")
                import_module.assert_called_once_with(expected_module)
                self.assertEqual(calls, [["--execute", "--output", str(output.resolve())]])
                self.assertFalse(output.exists())

    def test_report_cannot_write_into_retained_archives(self):
        from flow_matching_refiner import portable_cli as cli
        with tempfile.TemporaryDirectory(prefix="portable_cli_fixture_") as folder:
            output_root = Path(folder) / "new_runs"
            retained = Path(folder) / "retained_artifacts" / "prior_2030-01-01_00-00-00"
            retained.mkdir(parents=True)
            with patch.object(cli, "configure"), patch.object(cli, "preflight", return_value=self.checked()), \
                 patch.object(cli, "current", return_value=SimpleNamespace(output_root=output_root)), \
                 patch.object(cli.importlib, "import_module") as import_module, redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    cli.main(["--phase", "report", "--execute", "--experiment", str(retained)])
                import_module.assert_not_called()
            self.assertEqual(list(retained.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
