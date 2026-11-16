"""Download and verify published research artifacts; never evaluate a model.

Uses only Python's standard library. With no action flag, print the manifest.
Installation creates a NEW destination and never overwrites retained inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tempfile
from urllib.parse import quote
import urllib.request
import zipfile

REPOSITORY = "thebabak/retinal-vessel-flow-refinement-oof"
DEFAULT_MANIFEST = "artifacts_manifest_2026-11-16_18-53-15.json"
BLOCK = 1024 * 1024
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{stem}{i}" for stem in ("COM", "LPT") for i in range(1, 10)}


def member_path(value: str) -> str:
    """Reject traversal, absolute paths and Windows filename ambiguities."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("Archive paths must be nonempty POSIX relative paths")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or PureWindowsPath(value).drive or value != path.as_posix():
        raise ValueError("Archive path must be canonical and relative: " + value)
    if any(part in (".", "..") or part.endswith((".", " ")) or ":" in part
           or part.split(".", 1)[0].upper() in RESERVED for part in path.parts):
        raise ValueError("Unsafe archive member path: " + value)
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A lowercase SHA256 digest is required")
    return value


def _size(value):
    if type(value) is not int or value < 0:
        raise ValueError("size_bytes must be a nonnegative integer")
    return value


def validate_manifest(document: dict) -> dict:
    if document.get("schema_version") != 1 or document.get("repository") != REPOSITORY:
        raise ValueError("Unsupported artifact manifest schema/repository")
    tag = document.get("release_tag")
    if not isinstance(tag, str) or not re.fullmatch(r"artifacts-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", tag):
        raise ValueError("Expected a timestamped artifact release tag")
    assets = document.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("The manifest must list release assets")
    asset_names, file_names = set(), set()
    for asset in assets:
        name = asset["name"]
        if member_path(name) != PurePosixPath(name).name or not name.endswith(".zip"):
            raise ValueError("Assets must be ZIP basenames")
        if name.casefold() in asset_names:
            raise ValueError("Duplicate asset name")
        asset_names.add(name.casefold())
        _sha(asset["sha256"]); _size(asset["size_bytes"])
        expected_url = "https://github.com/" + REPOSITORY + "/releases/download/" + quote(tag, safe="") + "/" + quote(name, safe="")
        if asset["url"] != expected_url:
            raise ValueError("Asset URL is outside the declared public GitHub release")
        files = asset["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("Each ZIP requires a complete member inventory")
        for item in files:
            path = member_path(item["path"])
            normalized = path.casefold()
            if normalized in file_names:
                raise ValueError("Duplicate/case-colliding artifact path: " + path)
            file_names.add(normalized)
            _sha(item["sha256"]); _size(item["size_bytes"])
    # File/directory prefix collisions are invalid even across different ZIPs.
    for name in file_names:
        if any(parent.as_posix().casefold() in file_names for parent in PurePosixPath(name).parents if parent.as_posix() != "."):
            raise ValueError("An artifact file is also used as a parent directory")
    for key, target in document.get("recorded_path_aliases", {}).items():
        if not (PureWindowsPath(key).is_absolute() or PurePosixPath(key).is_absolute()):
            raise ValueError("Historical alias prefix must be absolute")
        if target != ".": member_path(target)
    for value in document.get("raw_roots", {}).values():
        member_path(value)
    reserved_outputs = {".installation_incomplete", "paths.downloaded_" + tag.removeprefix("artifacts-") + ".local.json"}
    if file_names & {name.casefold() for name in reserved_outputs}:
        raise ValueError("Archive collides with an installer-owned marker/configuration")
    return document


def check_asset(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["size_bytes"] or sha256_file(path) != asset["sha256"]:
        raise ValueError("Downloaded asset size/SHA256 mismatch: " + asset["name"])


def download_asset(asset: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / asset["name"]
    if destination.exists():
        if destination.is_symlink():
            raise ValueError("Cached assets may not be symlinks")
        check_asset(destination, asset)
        return destination
    # Temporary output is owned by this invocation. Cached complete files are immutable.
    handle, temporary_name = tempfile.mkstemp(prefix=".retinal-part-", dir=cache)
    temporary = Path(temporary_name)
    try:
        count = 0
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "retinal-flow-artifact-downloader"})
        with os.fdopen(handle, "wb") as out, urllib.request.urlopen(request, timeout=60) as response:
            for block in iter(lambda: response.read(BLOCK), b""):
                count += len(block)
                if count > asset["size_bytes"]:
                    raise ValueError("Response exceeded declared asset length")
                out.write(block)
        check_asset(temporary, asset)
        # Hard-link installation is atomic and refuses to replace a concurrent file.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            check_asset(destination, asset)
        return destination
    finally:
        if temporary.exists(): temporary.unlink()


def archive_members(path: Path, asset: dict) -> dict[str, zipfile.ZipInfo]:
    expected = {item["path"]: item for item in asset["files"]}
    found, names = {}, set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            # Directory records are unnecessary and deliberately unsupported.
            name = member_path(info.filename)
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1:
                raise ValueError("Directory/symlink/encrypted ZIP entry is not supported")
            if name.casefold() in names:
                raise ValueError("Duplicate/case-colliding ZIP member")
            names.add(name.casefold())
            if name not in expected or info.file_size != expected[name]["size_bytes"]:
                raise ValueError("ZIP inventory/size differs from the manifest: " + name)
            found[name] = info
        if set(found) != set(expected):
            raise ValueError("ZIP member inventory is incomplete")
    return found


def verify_installation(destination: Path, document: dict, *, _installing=False) -> dict:
    if not _installing and (destination / ".installation_incomplete").exists():
        raise ValueError("Installation is marked incomplete; use a fresh destination")
    checked = 0
    for asset in document["assets"]:
        for item in asset["files"]:
            path = destination / item["path"]
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Missing/escaped artifact file: " + item["path"])
            if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError("Installed member SHA256/size mismatch: " + item["path"])
            checked += 1
    return {"published_files_verified": checked, "training_or_evaluation_performed": False}


def install_archives(destination: Path, document: dict, archive_paths: dict[str, Path]) -> dict:
    if destination.exists():
        raise FileExistsError("Installation requires a NEW destination; existing retained inputs are never overwritten")
    # Verify archive hashes and all member inventories before creating the output tree.
    for asset in document["assets"]:
        path = archive_paths[asset["name"]]
        check_asset(path, asset)
        archive_members(path, asset)
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / ".installation_incomplete"
    marker.write_text("Do not use this directory until installation completes.\n", encoding="utf-8")
    for asset in document["assets"]:
        expected = {item["path"]: item for item in asset["files"]}
        with zipfile.ZipFile(archive_paths[asset["name"]]) as archive:
            for info in archive.infolist():
                item = expected[info.filename]
                target = destination / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Extraction path escapes the new destination")
                h, count = hashlib.sha256(), 0
                with archive.open(info) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(BLOCK), b""):
                        count += len(block)
                        if count > item["size_bytes"]:
                            raise ValueError("Member exceeded declared length")
                        h.update(block); output.write(block)
                if count != item["size_bytes"] or h.hexdigest() != item["sha256"]:
                    raise ValueError("Extracted member size/SHA256 mismatch: " + info.filename)
    result = verify_installation(destination, document, _installing=True)
    config_name = "paths.downloaded_" + document["release_tag"].removeprefix("artifacts-") + ".local.json"
    config = {
        "artifact_root": str(destination.resolve()),
        "output_root": str(destination.resolve().parent / (destination.name + "_new_runs")),
        "raw_roots": document.get("raw_roots", {}),
        "recorded_path_aliases": document.get("recorded_path_aliases", {}),
    }
    for key in ("controls", "v3_study"):
        if key in document: config[key] = document[key]
    config_path = destination / config_name
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    marker.unlink()
    result["generated_local_config"] = str(config_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "configs" / DEFAULT_MANIFEST)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true", help="Download and SHA256-verify all published ZIP assets")
    action.add_argument("--install", action="store_true", help="Download and extract all assets into a NEW artifact directory")
    action.add_argument("--verify", action="store_true", help="Check existing installed published files without network/extraction")
    parser.add_argument("--destination", type=Path, help="Required for installation/verification")
    parser.add_argument("--cache", type=Path, default=Path(__file__).resolve().parents[1] / ".artifact-download-cache")
    args = parser.parse_args(argv)
    document = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8-sig")))
    if (args.install or args.verify) and args.destination is None:
        parser.error("--destination is required for --install or --verify")
    print(json.dumps({"repository": document["repository"], "release_tag": document["release_tag"],
        "assets": [{k: a[k] for k in ("name", "size_bytes", "sha256")} for a in document["assets"]],
        "provider_required": [{"dataset": item.get("dataset"), "provider_url": item.get("provider_url"),
            "reason": item.get("reason"), "required_path_count": len(item.get("required_paths", []))}
            for item in document.get("provider_required", [])],
        "notice": "Archive verification is not model evaluation. Provider-required inputs are not supplied by these downloads."}, indent=2))
    if args.verify:
        print(json.dumps(verify_installation(args.destination.resolve(), document), indent=2))
    elif args.download or args.install:
        paths = {a["name"]: download_asset(a, args.cache.resolve()) for a in document["assets"]}
        for asset in document["assets"]:
            archive_members(paths[asset["name"]], asset)
        if args.install:
            print(json.dumps(install_archives(args.destination.resolve(), document, paths), indent=2))
        else:
            print("All published ZIP assets passed size, SHA256 and member-inventory checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
