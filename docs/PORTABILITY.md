# Portable configuration and retained-artifact replay

This public copy keeps the executed scientific recipes and adds path isolation, strict input checks and guarded commands. No model was fitted and no stored metric or confidence interval was recomputed while preparing the repository. The original executed source snapshots remain the reference for historical SHA256 checks; the adapted public modules are not replacements for them.

## Scope

A clone supports reading the small result tables and running the synthetic software tests. Full computational replay additionally requires the exact retained baseline/OOF/control archives and raw annotation versions described below. They are not downloaded or manufactured by the adapter. There is no dummy dataset, automatic checkpoint training fallback, hash bypass or substitute result source.

The canonical entry point is:

```bash
python -m flow_matching_refiner.portable_cli --config configs/paths.local.json --phase objectives
```

Without `--execute`, every phase performs the same standard-library-only filesystem, JSON and historical-digest preflight. It neither imports the scientific runners nor loads image pixels, prediction tensors or checkpoint tensors. It does read JSON and streams bound files to calculate SHA256. Missing inputs raise `FileNotFoundError`; an invalid configuration, inconsistent split or incorrect hash raises an explicit error. Preflight is necessary but is not a replacement for the deeper checkpoint/OOF replay and prediction verification performed during explicitly requested execution.

## Three separate roots

- **CODE_ROOT** is this repository. It supplies the adapted Python source and is never used as an implicit historical-artifact directory.
- **ARTIFACT_ROOT** contains the original retained experiment tree and byte-identical original source snapshots. It is read-only input.
- **OUTPUT_ROOT** is a separate directory for new timestamped audits, training outputs and reports. It must not lie inside any retained artifact or raw-data root. Every generated destination is also checked against configured control/V3/raw roots, including a retained input that happens to be nested below OUTPUT_ROOT.

Raw dataset roots may live elsewhere. The default ignored staging layout is `artifacts/` and the default ignored generated layout is `runs/`. Both are distinct from the source packages. If an external ARTIFACT_ROOT is an ancestor that contains this public checkout, OUTPUT_ROOT must be outside that ancestor. The snapshotter excludes the adapted checkout subtree from that historical ancestor, then binds the public Python code separately. It includes staged artifacts below CODE_ROOT when that staging directory is the configured ARTIFACT_ROOT.

Copy `configs/paths.example.json` to the ignored `configs/paths.local.json` and replace its placeholder historical prefixes and locations. The example has no personal machine paths. Do not commit the local configuration, raw data, checkpoint archives or unsanitized historical JSON.

### JSON schema and relative paths

| Key | Meaning | Relative paths resolve against |
| --- | --- | --- |
| `artifact_root` | Complete retained-input tree | Configuration file's directory |
| `output_root` | Dedicated generated-output tree | Configuration file's directory |
| `raw_roots.CHASE` | CHASE JPG and first-human-observer PNG directory | ARTIFACT_ROOT |
| `raw_roots.DRIVE` | DRIVE root containing the retained training/test image and FOV folders | ARTIFACT_ROOT |
| `raw_roots.DRIVE_TEST_GT` | Retained native-size local DRIVE test-label PNG directory | ARTIFACT_ROOT |
| `raw_roots.HRF` | Retained Supervisely dataset root containing `img/` and `ann/` | ARTIFACT_ROOT |
| `controls.CHASE`, `.DRIVE`, `.HRF` (optional) | Individual completed earlier five-arm experiment directories | ARTIFACT_ROOT |
| `v3_study` (optional) | Completed earlier two-option study parent | ARTIFACT_ROOT |
| `recorded_path_aliases` | Historical absolute-prefix to original retained location map | Alias targets resolve against ARTIFACT_ROOT |

Without a configuration file, the default roots are `CODE_ROOT/artifacts` and `CODE_ROOT/runs`. Default raw subdirectories are `ARTIFACT_ROOT/raw/{CHASE,DRIVE,DRIVE_TEST_GT,HRF}`. Optional controls and V3 paths default to the exact archived relative layout below.

Environment overrides are `RETINAL_PATHS_CONFIG`, `RETINAL_ARTIFACT_ROOT`, `RETINAL_OUTPUT_ROOT`, `RETINAL_RAW_CHASE`, `RETINAL_RAW_DRIVE`, `RETINAL_RAW_DRIVE_TEST_GT` and `RETINAL_RAW_HRF`. The explicit `--config` path takes precedence over `RETINAL_PATHS_CONFIG`; other environment overrides take precedence over their corresponding JSON values. Use a fresh Python process when changing configuration. Configure before importing loaders or scientific runners; late configuration is rejected so module-level retained paths cannot silently remain stale.

### Historical path aliases

Original metadata may contain paths from the producing computer. Do not edit those historical JSON files: doing so invalidates their hashes. Configure aliases from the exact recorded prefix to the directory containing the original bytes on the current computer. Both dictionary syntax and a list of `{"recorded_prefix": "...", "target": "..."}` objects are accepted.

Aliases use the longest matching prefix, compare case-insensitively and require a slash boundary. Prefixes must be complete absolute directory or file paths; bare drive-root aliases are not supported. Duplicate prefixes after slash/case normalization are rejected. Resolving an alias is read-only. A suffix containing parent traversal or a symlink cannot escape its target. Alias targets must be inside retained input roots; adapted public source is rejected, except for a distinct ignored artifact staging directory under CODE_ROOT.

For example, `/recorded/study` may map to ARTIFACT_ROOT. Its historical `paired_reproduction/run.py` then resolves to the **original** `ARTIFACT_ROOT/paired_reproduction/run.py`. If original flow source bytes are stored separately, a longer `/recorded/study/flow_matching_refiner` alias may map to `ARTIFACT_ROOT/source_snapshots/flow_matching_refiner`. Those files must match every original SHA256 binding. Do not point this alias at the modified public `flow_matching_refiner/` directory. A different dataset prefix can map to its configured raw root.

## Required retained input layout

The adapter deliberately uses a full retained-archive mode. File names, split identities and original digest bindings are part of the protocol, rather than inferred from a newly downloaded dataset.

```text
ARTIFACT_ROOT/
  paired_reproduction/
    [original producer Python source files, unchanged]
    all_datasets_five_full_runs_2026-10-08_15-38-24/
      {CHASE,DRIVE,HRF}/train_seed_42/
        baseline.pt
        autoencoder.pt
        protocol.json
        file_provenance.json              # DRIVE/HRF
        <test-image>_reference.npz
  revise/oof_comparison_2026-10-09_23-30-40/
    {CHASE,DRIVE,HRF}/seed_42/
      oof_protocol_2026-10-09_23-30-40.json
      oof_training_maps_2026-10-09_23-30-40.npz
      fold_<i>/
        baseline.pt
        fold_inventory_2026-10-09_23-30-40.json
        held_out_predictions_2026-10-09_23-30-40.npz
  outputs/
    esfm_oof_chase_2026-11-15_01-59-17/
    esfm_all_datasets_2026-11-15_02-35-09/
      DRIVE_2026-11-15_02-35-09/
      HRF_2026-11-15_02-35-09/
    refinement_v3_2026-11-15_23-28-26/
      [parent configuration/global selection locks and study plan]
      {CHASE,DRIVE,HRF}_2026-11-15_23-28-26/
  source_snapshots/                        # optional reorganized original source
  raw/                                    # optional raw-data staging
```

Each completed control or V3 child must retain its full original metadata, model checkpoints, prediction NPZ files, training/validation logs, thin-definition file, completion/selection locks, test-record CSV, summary and verified-report JSON. Keep parent lock/study-plan files referenced by child metadata. Preserve every additional original file referenced in `verified_input_sha256` or source-hash maps, not just the abbreviated tree above. Preflight verifies existing recorded bindings; real reporters additionally verify selections, checkpoints and arrays. The controls/native/V3 runners snapshot the input trees and public code before and after execution.

There are four CHASE, four DRIVE and three HRF seed-42 OOF producer folds. No new fold baseline is fitted by the refiner loaders. They replay each retained checkpoint and require exact agreement of merged/per-fold probabilities, full-baseline validation/test fields and fixed partitions. All five refiners share the same frozen full/fold baseline inputs. The retained `autoencoder.pt` files remain mandatory compatibility/provenance artifacts of the original loader contract; these direct/flow recipes do not perform autoencoder inference.

CHASE requires the original RGB JPGs and `_1stHO.png` labels. DRIVE requires the retained manual-1/FOV conversions and local native-size test PNG labels. HRF requires the retained Supervisely indexed bitmap JSON and source images. Native loading verifies the recorded conversion to 512 against the retained RGB/GT/FOV fields; it never upsamples a 512 GT image and calls it a native annotation. Exact replay can fail on different floating-point/backend implementations; there is no tolerance relaxation or retraining fallback. The incomplete upstream DRIVE test-label acquisition and HRF annotation-conversion history remain limitations even if internal replay succeeds.

## Explicit execution

Only add `--execute` after complete inputs and an appropriate environment are available. `--output`, when supplied, must be a **new** directory inside OUTPUT_ROOT whose basename ends in Tehran local `YYYY-MM-DD_HH-mm-ss`; existing experiment directories are refused. Commands below are examples, not evidence that new runs were performed during repository preparation.

```bash
# O1/O2: all three datasets, 12 pilots and 30 final refiner fits.
python -m flow_matching_refiner.portable_cli --config configs/paths.local.json --phase objectives --execute

# O3: all three datasets, 15 final native-patch refiner fits.
python -m flow_matching_refiner.portable_cli --config configs/paths.local.json --phase native --execute

# Earlier matched five-arm controls; ALL is the default.
python -m flow_matching_refiner.portable_cli --config configs/paths.local.json --phase controls --dataset ALL --execute
```

Objectives/native phases reject a single-dataset request because their original selection policy locks the entire phase across datasets. The controls phase also supports a single `--dataset CHASE`, `DRIVE` or `HRF`. It fits all five arms and five refiner seeds for that dataset; it does not fit the retained baseline/OOF producers. All phases preserve 120 epochs, seeds 42/123/2026/3000/3001, fixed seed-42 retained inputs and the original objectives/inference settings. Objectives preserve the separate pilot seed 91011, anchor candidates and validation-only configuration selection; all 30 final choices precede that phase's tests. Native preserves its own later 15-model selection lock, four 256-patch draws per image per epoch and batch 16. These are separate phases, not a single 45-model pre-test lock.

The optional `controls` and `v3_study` configuration keys select the original input archives used for contextual comparisons; launching a new phase does not silently replace those inputs with a freshly generated output. If deliberately chaining a newly completed archive into a subsequent run, configure that exact completed archive in a fresh process and keep all referenced hashes/source snapshots.

### Reports over newly generated outputs

```bash
python -m flow_matching_refiner.portable_cli --config configs/paths.local.json --phase report --execute --experiment /path/to/OUTPUT_ROOT/study_YYYY-MM-DD_HH-mm-ss/CHASE_YYYY-MM-DD_HH-mm-ss
```

Point `--experiment` to one completed dataset child directory. The wrapper selects its matching reporter from the recorded arm inventory. Report execution loads saved predictions and calculates metric/statistical reports, so it is intentionally guarded by `--execute`. It is restricted to new output trees and rejects retained input archives; it never writes reports into a historical control/V3 source tree. The default `--phase report` without `--execute` performs preflight only. Existing published small table files can be inspected without report execution through `python -m tools.show_results`.

## Python API and legacy drivers

For explicit programmatic loading, call `portable_paths.configure(local_config)` before importing `data`, `oof_data_all` or `native_patch_data`. Their explicit APIs retain `load_reference(dataset, baseline_seed, workdir, device)`, `load_dataset_oof(dataset, workdir, device)` and `load_native_dataset(dataset, workdir, device)`. All audit workdirs must be under OUTPUT_ROOT and outside retained inputs. Importing the adapters alone is inert; calling OOF/native loaders explicitly loads real data and replays retained baselines, but never fits a baseline or computes segmentation metrics.

The copied `paired_reproduction/run.py`, `run_drive.py` and `run_hrf_baseline.py` are legacy source producers for the exact preprocessing-prefix composer. They are guarded against ordinary direct execution/import, which formerly executed training at module scope. The audited loader executes only the extracted safe prefix under its dedicated internal module name, removes original metadata writes and redirects audit output. Use the canonical wrapper for experiments; do not import a legacy raw driver as a training API.

## Adaptation and validation boundaries

The adaptation manifest records original/adapted source hashes using repository-relative names and invariant scientific-function checks. Model/loss definitions, training-step bodies, checkpoint selection and prediction functions remain those of the executed studies. Only path resolution, input/output isolation, legacy execution guards, protection snapshots and command dispatch were adapted. Source-byte hashes necessarily differ for those path-only modules; all historical hashes continue to be checked against original retained bytes.

Software validation uses synthetic CPU tensors, temporary missing-input fixtures, mocked command dispatch, inert imports and static/source-prefix checks. It is not a fresh experimental reproduction, GPU qualification or cross-platform bitwise-replay guarantee. The public copy contains no original private correspondence or manuscript submission files.
