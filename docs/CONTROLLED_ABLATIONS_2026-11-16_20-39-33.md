# Fixed 16-arm controlled ablations

This export adds the implementation and the historical registered plan for 16
ablation arms, three datasets and five refiner seeds (240 fits; 120 epochs).
It does not contain completed new experimental results. The separately running
author experiment must finish and pass prediction verification before its new
outcomes can be reported. Previous studies and published files remain unchanged.

The arms isolate the O1 recovery/anchor factors, O2 endpoint/anchor factors,
Direct Dice coefficients, RGB conditioning, augmentation, a 512-grid patch
control, native uniform sampling, and native RGB/GT information degradation.
The retained controls are reused; they are not fresh independent baseline fits.
All 240 selected model choices must be locked before new test evaluation.

## Portable invocation

Merge this export into the public repository and use its existing
`flow_matching_refiner.portable_paths` configuration. Put machine-specific roots
in a local ignored JSON based on `configs/paths.example.json`. No private drive
letter is required by the adapted implementation.

Describe the plan without loading data or models:

```console
python -m flow_matching_refiner.controlled_ablations.cli
```

Register a **new** local output under the configured output root, without
training, inference, or reading scientific arrays:

```console
python -m flow_matching_refiner.controlled_ablations.cli --config paths.local.json --register --output runs/controlled_ablations_YYYY-MM-DD_HH-mm-ss
```

Check retained original bytes separately:

```console
python -m flow_matching_refiner.controlled_ablations.cli --config paths.local.json --preflight
```

Explicitly execute a registered run only when its required inputs are available:

```console
python -m flow_matching_refiner.controlled_ablations.cli --config paths.local.json --output runs/controlled_ablations_YYYY-MM-DD_HH-mm-ss --execute --phase all
```

`--phase train` and `--stop-after-dataset CHASE` allow the registered training
gate; `--phase evaluate` still requires the complete 240-choice lock. Resuming
an unfinished fit archives that attempt and restarts the same seed from its
initialization. Completed, SHA-verified selections are reused. Use a fresh
registration on another machine; do not relabel an old experiment as a new run.

## What is preserved

`operations.py` and `specs.py` are byte-identical to the registered author code.
Every scientific runner function has the same AST as its original snapshot,
including input preparation, patch pools, fitting, validation, inference,
prediction storage and metric calculation. Only root resolution, protected-file
inventory and the output-root check differ in the adapted runner. The added CLI
configures paths before scientific imports and records a fresh local source lock.
The historical plan is preserved byte for byte; its historical provenance-limit
wording is not a new claim that later HRF correspondence work was never done.

Original source snapshots and SHA/AST correspondence are in `provenance/`.
These snapshots document the author code. Use the portable CLI for new runs.

## Required data and limitations

This export contains no retinal source images, GT/FOV arrays, checkpoints,
prediction arrays, manuscripts, reviewer responses or correspondence. Obtain
datasets from their providers and configure retained artifacts explicitly.
DRIVE images and training annotations were attributed by the authors to
https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction.
The acquisition of the retained local DRIVE test manual-1 PNGs is independently
unverified; that attribution is not proof of canonical hidden-test labels.

The public artifact release omits DRIVE source data and removes unnecessary
personal HRF annotation metadata. Consequently it does **not**, by itself,
satisfy the historical strict original-byte preflight. This CLI retains those
checks and fails when required original bytes or aliases are absent. It does
not manufacture masks, silently ignore digest failures or equate source-pixel
correspondence with original-byte identity. A separately documented scientific
replay adapter, if provided, must remain distinct from this strict mode.

The synthetic scientific test file is retained in `tests/`. Export-specific
tests cover inert imports, path guards, registration and AST preservation only;
they do not execute dataset training or test evaluation.
