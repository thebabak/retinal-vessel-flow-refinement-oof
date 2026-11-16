# Official-code retinal comparisons: staged portable export

This source-only package adapts the official **FMS² SegFlow stage** and **FR-UNet network** to the paper's fixed retinal protocol. It contains no dataset, ground truth, pretrained weights, private correspondence, or experiment result. This portable packaging layer has **not** produced the recorded experiment results: the original executed wrappers are preserved byte-for-byte under `official_comparisons/executed_reference/` and the source/AST audit identifies every packaging change.

The comparisons are controlled-protocol adaptations, not reproductions of the original publications' benchmark scores. SegFlow uses RGB and does not consume a coarse probability map. SynFlow synthesis is not run. FR-UNet uses its raw sigmoid output; the upstream DTI helpers are not executed or silently repaired.

## Sources and adaptations

| Source | Pinned commit | Preserved core | Explicit adaptations |
|---|---|---|---|
| [FMS²](https://github.com/BabakAsadi94/FMS2) | `ab18a8f33c7aa27425d7265c3dc9a69bf1d8c5a4` | Official 2,963,907-parameter UNet, sigma-zero CFM, velocity MSE, 0.5 self-conditioning, official eleven-point cyclic training time schedule and ten-step self-conditioned Euler inference | Retinal RGB256/GT256 with shared512 comparison grid;120epochs;logicalbatch4/microbatch1;validation-only LR selection;5%warmup/cosine;EMA.99;FP32;gradientclip1 |
| [FR-UNet](https://github.com/lseventeen/FR-UNet) | `1856bc206055fdfccde184d0ebf9ac8c3207387d` | Official 7,374,347-parameter one-channel network, five-head logit averaging, initialization and BCEWithLogits(pos_weight1);aligned flips/right-angle augmentation | Retained green/CLAHE512 whole images;120epochs;logicalbatch4/microbatch1;validation-only LR/checkpoint selection;FP32;noDTI |

FR BatchNorm observes microbatch1. Gradient accumulation does not reproduce BatchNorm batch4. Upstream pretrained models are not used. The required official source snapshots, selected documentation/configuration, and full MIT notices are byte-preserved. FR's unused author CLI/data scripts with machine-specific defaults are omitted; `SOURCE.json` identifies the exact included subset. The portable entrypoints do not execute unused author training/data scripts and do not require their paths or `.git` directories.

## Local inputs

Install the dependencies in `requirements.txt`, using a PyTorch build compatible with your own CUDA hardware. Run from the export root with `python -m official_comparisons.segflow` or `python -m official_comparisons.frunet`. Importing either module is inert: no data/model/checkpoint loading, CUDA initialization, fitting, scoring or output creation occurs at import.

Supply `--workspace-root` pointing to a local artifact workspace, `--input-manifest` as a **workspace-relative** JSON filename, and `--output-root` pointing to a separate output tree. No drive letter or original author's directory layout is required. The output run must lie strictly inside `--output-root`; private input/package-source overlap and path traversal are rejected. Existing executed-original runs cannot be adopted silently by these portable wrappers.

`examples/prepared_inputs_manifest_TEMPLATE.json` declares the exact retained image IDs and ordering for all three datasets. Fill its local relative paths, per-file SHA256, training-derived thin cutoffs and truthful annotation-source/provenance descriptions. All prepared records must remain in a dedicated private `input_root`. The template contains no numeric results and is deliberately invalid until filled. Do not invent a cutoff or a missing test annotation.

Each image NPZ contains these arrays, using `allow_pickle=False`:

| Key | Shape/type | Meaning |
|---|---|---|
| `x` | `1×512×512`,FP32 `[0,1]` | Retained green/CLAHE/inverse-gamma input |
| `rgb` | `3×512×512`,FP32 `[0,1]` | Retained RGB bilinear512 tensor |
| `g` | `1×512×512`,binary | Observer-labelled segmentation on the comparison grid |
| `fov` | `1×512×512`,binary | Valid evaluation field; all ones for this CHASE protocol |

Paired baseline reference NPZ files, required only after the global test gate, contain `coarse`, `ground_truth`, and `fov` with the exact corresponding512 tensors. Inference does not accept or use a GT argument; GT is used only in authorized training, validation and final scoring. Test image/GT and paired references are loaded lazily and remain inaccessible until all intended final checkpoints are globally locked.

This package does not reconstruct prepared records from redistributable raw datasets automatically. Use the parent project's audited data-preparation/baseline pipeline or equivalent explicitly documented preparation; check the declared hashes. Conditional provenance limitations remain: the local DRIVE test labels' exact upstream provider/observer are unverified. The [DRIVE provider](https://drive.grand-challenge.org/) and user-reported [Kaggle mirror](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction) are attribution/access links; **we do not claim that the Kaggle archive supplies the retained20test labels**. Raw DRIVE images/masks are not included or authorized for redistribution.

## New execution and selection gates

Example shell arguments (replace local paths; a fresh timestamped run name is required):

```text
python -m official_comparisons.segflow --workspace-root /local/artifacts --input-manifest prepared_inputs.json --output-root /local/new_runs --output segflow_YYYY-MM-DD_HH-MM-SS --epochs 120 --microbatch 1 --memory-fraction .35 --datasets CHASE DRIVE HRF --execute
python -m official_comparisons.frunet --workspace-root /local/artifacts --input-manifest prepared_inputs.json --output-root /local/new_runs --output frunet_YYYY-MM-DD_HH-MM-SS --epochs 120 --microbatch 1 --memory-fraction .35 --datasets CHASE DRIVE HRF --execute
```

These commands fit only. `--finalize-tests` is a separate explicit flag and still cannot access test records until **all15choices** exist (three datasets × training seeds42,123,2026,3000,3001). Three LR pilots per dataset use seed91011 and candidates1e-5,1e-4,3e-4. The dataset LR locks before its five fresh final fits; every final checkpoint uses validation only. Do not claim all three LR configurations preceded the first final fit. `PAUSE_REQUESTED` stops safely after the current fit; a resumed identical portable run checks existing checkpoint hashes. The input-manifest hash must remain unchanged across resume.

No published score, result table or timing is supplied by this package. Run synthetic checks without fitting or GPU work:

```text
python -m unittest discover -s tests -v
```

The source/AST proof distinguishes unchanged fitting/inference math, snapshot/path/metric namespace changes, and the new controller/manifest adapter. Byte-preserved executed originals are evidence, not the portable CLI entrypoints. Do not use them to bypass the portable path or test gates.

## Attribution

Keep `THIRD_PARTY_NOTICES.txt` and both vendor `LICENSE` files when redistributing. FMS²'s license includes the full Alexander Tong conditional-flow-matching MIT notice. Cite the original FMS²/FR-UNet articles and TorchCFM as listed in the notices. The parent project must separately specify the license for its own adaptation code; upstream MIT notices do not establish authorship/licensing of unrelated project files.
