> Current study: see [THREE_DATASET_COMPLETION_2026-11-19_12-03-07.md](THREE_DATASET_COMPLETION_2026-11-19_12-03-07.md). This document records the earlier initial scope; dated progress and release statements below are historical.

# Downloading and checking research artifacts

The timestamped artifact manifest lists the public GitHub Release ZIPs, their exact sizes and SHA256 digests, and every extracted member's relative path, size and SHA256. It also declares provider-required inputs that are not contained in those ZIPs. The manifest does not grant a new license to source datasets or third-party annotations.

The [artifact release](https://github.com/thebabak/retinal-vessel-flow-refinement-oof/releases/tag/artifacts-2026-11-16_18-53-15) supplies 413 retained checkpoint files, 14 merged/per-fold OOF probability containers, control/objective/native-patch outputs and records, and licensed CHASE/HRF source data. See [data notices](DATA_LICENSES.md). DRIVE source RGB/GT/FOV and original containers embedding them are omitted; separately named prediction-only copies preserve the retained model-output NPY members. Forty-five public HRF annotation JSONs remove `labelerLogin` metadata only, preserving bitmap payloads and all remaining fields. Their public hashes differ from the original historical hashes. The strict all-dataset historical preflight consequently **does not pass with these public downloads alone**. No digest bypass is implemented.

Inspect the release inventory and provider requirements with Python's standard library:

```bash
python -S -m tools.fetch_artifacts
```

The default command reads the local manifest only. It does not download archives, import PyTorch, open retinal images, deserialize checkpoints, run predictions, train models or calculate segmentation statistics.

Download and check all published assets, without extraction:

```bash
python -S -m tools.fetch_artifacts --download --cache /path/to/download-cache
```

Install all published assets into a **new**, dedicated retained-input directory:

```bash
python -S -m tools.fetch_artifacts --install --destination /path/to/new-retained-inputs --cache /path/to/download-cache
```

The destination must not already exist. Existing archives in the cache are reused only after their size and SHA256 match the manifest; a corrupt cached file is refused. Each ZIP's complete member inventory is checked before an installation directory is created. Extraction rejects absolute/traversing paths, duplicate and case-colliding members, symlinks, encrypted entries, file/directory conflicts and unlisted members. Each extracted file is opened exclusively and checked against its size and SHA256. An interrupted or failed installation keeps an `.installation_incomplete` marker; do not use that partial tree. Choose a new destination for a fresh attempt. The helper does not automatically erase research inputs.

The helper creates `paths.downloaded_2026-11-16_18-53-15.local.json` inside the new retained-input directory. It maps historical producer path prefixes to the downloaded original bytes and selects a separate sibling directory for future outputs. Review its raw-data locations, supply every provider-required file in the documented layout, then use the canonical preflight:

```bash
python -S -m flow_matching_refiner.portable_cli --config /path/to/new-retained-inputs/paths.downloaded_2026-11-16_18-53-15.local.json --phase objectives
```

Artifact-download verification proves only that the published files were downloaded intact. It does **not** assert that unpublished provider inputs are present, that source annotations have established upstream provenance, that the entire historical archive is available, or that the model results have been rerun. The strict preflight still fails if a required original file is missing. A prediction-only public export, where supplied, remains separately named; it does not replace a historical prediction archive containing GT/FOV fields or satisfy that archive's original digest binding.

Historical metadata and original source snapshots retain their original bytes to preserve the recorded hashes. Some files therefore retain producer-computer path strings. These are locations in the experiment record, not locations that must exist on a reader's computer. The local aliases resolve them without editing those original files. Do not point historical aliases at the adapted public Python source tree.

Verify the installed published files later, without network or extraction:

```bash
python -S -m tools.fetch_artifacts --verify --destination /path/to/new-retained-inputs
```

The `--verify` action checks only files listed in the public artifact manifest. Additional provider files can be present without being treated as published assets. The helper never deserializes a `.pt` file or evaluates a model. Explicit experimental execution, when all exact retained inputs and the appropriate environment are available, is documented in [PORTABILITY.md](PORTABILITY.md).

The downloader has standard-library synthetic safety tests; exact scientific replay remains Windows-tested. Historical relative paths containing Windows backslashes have not been qualified for exact Linux replay.
