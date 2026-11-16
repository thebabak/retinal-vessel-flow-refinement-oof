"""Synthetic archive/downloader checks; no retinal inputs or models are loaded."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

helper_path = Path(__file__).with_name("fetch_artifacts.py")
if not helper_path.is_file(): helper_path = Path(__file__).resolve().parents[1] / "tools/fetch_artifacts.py"
spec = importlib.util.spec_from_file_location("artifact_helper", helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def bundle(self, members=None):
        members = members or {"weights/fake-checkpoint.bin": b"synthetic bytes only"}
        path = self.root / "synthetic.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items(): archive.writestr(name, content)
        asset = {"name": path.name, "url": "https://github.com/" + helper.REPOSITORY + "/releases/download/artifacts-2026-11-16_18-53-15/" + path.name,
                 "sha256": helper.sha256_file(path), "size_bytes": path.stat().st_size,
                 "files": [{"path": name, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()} for name, value in members.items()]}
        document = {"schema_version": 1, "repository": helper.REPOSITORY,
                    "release_tag": "artifacts-2026-11-16_18-53-15", "assets": [asset],
                    "provider_required": [{"dataset": "synthetic", "reason": "not a real dataset"}],
                    "raw_roots": {"CHASE": "raw/CHASE"},
                    "recorded_path_aliases": {"/recorded/study": "."}}
        return path, asset, document

    def test_install_and_verify_byte_exact_new_tree(self):
        path, asset, document = self.bundle({"sources/a.py": b"# synthetic\n", "maps/only.bin": b"123"})
        helper.validate_manifest(document)
        destination = self.root / "installed"
        result = helper.install_archives(destination, document, {asset["name"]: path})
        self.assertEqual(result["published_files_verified"], 2)
        self.assertFalse(result["training_or_evaluation_performed"])
        self.assertEqual((destination / "maps/only.bin").read_bytes(), b"123")
        self.assertFalse((destination / ".installation_incomplete").exists())
        config = json.loads(Path(result["generated_local_config"]).read_text())
        self.assertEqual(config["artifact_root"], str(destination.resolve()))
        self.assertFalse(Path(config["output_root"]).is_relative_to(destination))
        self.assertEqual(config["recorded_path_aliases"], document["recorded_path_aliases"])
        self.assertEqual(helper.verify_installation(destination, document)["published_files_verified"], 2)

    def test_existing_directory_is_never_overwritten(self):
        path, asset, document = self.bundle()
        destination = self.root / "old"
        destination.mkdir(); (destination / "prior.bin").write_bytes(b"retain")
        with self.assertRaises(FileExistsError): helper.install_archives(destination, document, {asset["name"]: path})
        self.assertEqual((destination / "prior.bin").read_bytes(), b"retain")

    def test_bad_archive_digest_creates_no_destination(self):
        path, asset, document = self.bundle()
        asset["sha256"] = "0" * 64
        destination = self.root / "new"
        with self.assertRaises(ValueError): helper.install_archives(destination, document, {asset["name"]: path})
        self.assertFalse(destination.exists())

    def test_member_digest_failure_marks_partial_tree(self):
        path, asset, document = self.bundle()
        asset["files"][0]["sha256"] = "0" * 64
        destination = self.root / "new"
        with self.assertRaises(ValueError): helper.install_archives(destination, document, {asset["name"]: path})
        self.assertTrue((destination / ".installation_incomplete").exists())
        with self.assertRaises(ValueError): helper.verify_installation(destination, document)

    def test_unsafe_paths_rejected(self):
        for value in ("../x", "/x", "C:/x", "a\\b", "a/../b", "a//b", "a/./b", ".", "NUL.txt", "a/COM1", "a/b.", "a/b ", "a:b"):
            with self.subTest(value=value), self.assertRaises(ValueError): helper.member_path(value)

    def test_noncanonical_repo_url_rejected(self):
        _, asset, document = self.bundle()
        asset["url"] = "https://example.org/synthetic.zip"
        with self.assertRaises(ValueError): helper.validate_manifest(document)

    def test_cross_zip_file_parent_collision_rejected(self):
        _, asset, document = self.bundle()
        asset["files"] = [{"path": "a", "sha256": "0" * 64, "size_bytes": 0}, {"path": "a/b", "sha256": "0" * 64, "size_bytes": 0}]
        with self.assertRaises(ValueError): helper.validate_manifest(document)

    def test_manifest_case_collision_rejected(self):
        _, asset, document = self.bundle()
        asset["files"] += [{"path": asset["files"][0]["path"].upper(), "sha256": "0" * 64, "size_bytes": 0}]
        with self.assertRaises(ValueError): helper.validate_manifest(document)

    def test_unlisted_zip_member_rejected(self):
        path, asset, document = self.bundle()
        with zipfile.ZipFile(path, "a") as archive: archive.writestr("unexpected.bin", b"x")
        with self.assertRaises(ValueError): helper.archive_members(path, asset)

    def test_duplicate_zip_entries_rejected(self):
        path, asset, document = self.bundle()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "a") as archive: archive.writestr(asset["files"][0]["path"], b"x")
        with self.assertRaises(ValueError): helper.archive_members(path, asset)

    def test_symlink_zip_entry_rejected(self):
        path, asset, document = self.bundle()
        info = zipfile.ZipInfo(asset["files"][0]["path"])
        info.create_system = 3; info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(path, "w") as archive: archive.writestr(info, b"outside")
        with self.assertRaises(ValueError): helper.archive_members(path, asset)

    def test_default_cli_is_metadata_only(self):
        _, _, document = self.bundle()
        manifest = self.root / "manifest.json"; manifest.write_text(json.dumps(document))
        with patch.object(helper.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")), contextlib.redirect_stdout(io.StringIO()) as output:
            helper.main(["--manifest", str(manifest)])
        self.assertIn("provider_required", output.getvalue())

    def test_corrupt_cached_asset_refused_without_network(self):
        path, asset, _ = self.bundle()
        path.write_bytes(b"corrupt")
        with patch.object(helper.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(ValueError): helper.download_asset(asset, self.root)
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_download_checks_digest_and_reuses_valid_cache(self):
        path, asset, _ = self.bundle(); raw = path.read_bytes()
        cache = self.root / "cache"
        with patch.object(helper.urllib.request, "urlopen", return_value=io.BytesIO(raw)) as call:
            saved = helper.download_asset(asset, cache)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(saved.read_bytes(), raw)
        with patch.object(helper.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")):
            self.assertEqual(helper.download_asset(asset, cache), saved)
        self.assertEqual(list(cache.glob(".retinal-part-*")), [])

    def test_oversized_download_leaves_no_cached_complete_file(self):
        path, asset, _ = self.bundle()
        cache = self.root / "cache"
        with patch.object(helper.urllib.request, "urlopen", return_value=io.BytesIO(path.read_bytes() + b"too-long")):
            with self.assertRaises(ValueError): helper.download_asset(asset, cache)
        self.assertFalse((cache / asset["name"]).exists())
        self.assertEqual(list(cache.glob(".retinal-part-*")), [])

    def test_tampered_installed_file_detected(self):
        path, asset, document = self.bundle()
        destination = self.root / "new"
        helper.install_archives(destination, document, {asset["name"]: path})
        (destination / asset["files"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaises(ValueError): helper.verify_installation(destination, document)


if __name__ == "__main__": unittest.main()
