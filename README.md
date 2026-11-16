# Retinal Vessel Flow Refinement with OOF Inputs

Research code and verified numerical tables for **Structural Objectives and Native-Resolution Patches for Direct and Flow-Matching Retinal Vessel Refinement**.

**Authors:** Babak Bayanian, Hanieh Naderi and Chinthaka Dinesh. The associated manuscript is unpublished; no manuscript DOI is supplied here.

## Three separately evaluated options

| Option | Recipe | Inference |
| --- | --- | --- |
| O1 | Direct refinement with missed-thin-vessel recovery and correct-pixel anchoring | One residual update |
| O2 | Error-selective flow matching with full local endpoint supervision and anchoring | Eight Euler velocity updates and one gate evaluation |
| O3 | Direct refinement trained on native-resolution RGB/annotation patches, using the original Direct loss | Overlapping native tiles; Hann-weighted logit-residual fusion and residual-only projection to the comparison grid |

O1 and O3 are deterministic Direct recipes. O2 is the new flow-matching recipe. They are separate experiments, rather than successive stages or additive components of one model.

Core code map:

- [Shared refiner and inference](flow_matching_refiner/esfm_model.py).
- [O1/O2 objective recipes](flow_matching_refiner/refinement_objectives_v3.py).
- [Native patch sampling and residual fusion](flow_matching_refiner/native_patch_model.py).
- [Guarded portable entry point](flow_matching_refiner/portable_cli.py).

## Recorded raw Dice results

Values are copied from the completed verified analyses: mean ± sample SD across five refiner-training means. Each dataset uses **one shared frozen baseline**, so the baseline has no refiner-seed SD.

| Dataset | Frozen baseline | O1 | O2 | O3 |
| --- | ---: | ---: | ---: | ---: |
| CHASE-DB1 | 0.810568 | 0.805423 ± 0.003107 | 0.809365 ± 0.000500 | 0.812398 ± 0.000327 |
| DRIVE | 0.797975 | 0.791415 ± 0.002721 | 0.801119 ± 0.000498 | 0.804570 ± 0.000534 |
| HRF | 0.733117 | 0.721827 ± 0.005688 | 0.736088 ± 0.001040 | 0.755753 ± 0.003517 |

O3 has the largest raw Dice point mean among the nine compared methods on these cohorts. Its precision is lower than the original Direct control on all three datasets, while O1 has higher thin-skeleton recall than O3. On DRIVE, O3-minus-baseline clDice and thin-recall intervals include zero. These conditional results do not establish state-of-the-art performance, general superiority, or a causal effect of resolution.

See [results and interpretation](docs/RESULTS.md), the [experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and the [saved tables](results/).

## Scope and reproducibility

- Training coarse maps are retained out-of-fold (OOF) predictions; validation/test maps come from the frozen full-training baseline, seed 42.
- The five refiner seeds are 42, 123, 2026, 3000 and 3001. They are independent refiner fits conditional on retained inputs, rather than five repetitions of the full segmentation pipeline.
- The two-objective phase used 12 pilot fits and 30 final fits. The later native-patch phase used 15 final fits. Seventy-five earlier control fits are reused. The 30-model and 15-model selection locks belong to separate phases; there was no single 45-model lock preceding all test exposure.
- Every reported comparison uses the original **512 × 512 GT/FOV fields**. Native coarse inputs are bilinear projections of existing 512-grid predictions. Only the learned native residual is projected back and applied to the original coarse map. No native-grid segmentation metric is claimed.
- Confidence intervals are exploratory, unadjusted and conditional on the fitted models. The evaluation cohorts had already been examined, and independent patient IDs are unverified for DRIVE and HRF.

The repository includes the three-option implementation, required earlier-control and OOF producer dependencies, software tests and numerical tables. Retained model weights, OOF probability maps, predictions, protocol/source records and licensed CHASE/HRF data are downloadable from the [research-artifact release](https://github.com/thebabak/retinal-vessel-flow-refinement-oof/releases/tag/artifacts-2026-11-16_18-53-15). Private correspondence and manuscript submission files are excluded.

Read [artifact downloads and integrity checks](docs/ARTIFACT_DOWNLOADS.md) and [data attribution and access boundaries](docs/DATA_LICENSES.md). DRIVE source RGB/GT/FOV are excluded because redistribution permission is unavailable; DRIVE prediction-only exports retain model outputs and omit source fields. HRF public annotation JSONs remove third-party labeler-login metadata while preserving bitmap payloads. Those privacy derivatives have distinct hashes. **The public downloads do not satisfy the complete original-byte historical preflight by themselves:** original licensed DRIVE containers and original HRF annotation metadata remain required by that strict check. No hash bypass or fresh experimental reproduction is claimed.

Inspect the published artifact inventory without downloading or evaluating a model:

```bash
python -S -m tools.fetch_artifacts
```

Display saved values using the Python standard library, without installing training packages:

```bash
python -m tools.show_results
```

### Setup

Use a separate Python environment and install a PyTorch build appropriate for the available hardware, then the listed dependencies:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -r requirements.txt
```

[Tested package versions and validation scope](docs/ENVIRONMENT.md) are recorded separately from the dependency requirements.

CPU software tests use synthetic inputs and do not fit a retinal model:

```bash
python -m unittest discover -s flow_matching_refiner -p "test_*.py"
python -m unittest discover -s tests -p "test_*.py"
```

### Required research inputs

Cloning the repository and installing packages is sufficient to inspect tables and run the software tests. The separate release supplies the permitted retained artifacts described above; these large assets are not part of a Git clone. Full computational replay additionally requires every exact dataset/annotation version and original hash-bound container expected by the reference adapters, including the omitted DRIVE inputs and original HRF annotation metadata. Download verification checks the published copies; it is distinct from the stricter historical input preflight.

See [portable configuration and commands](docs/PORTABILITY.md) before launching an experiment. Create a private local configuration, for example `configs/paths.local.json`, using its documented schema:

```bash
python -m flow_matching_refiner.portable_cli --config configs/paths.local.json --phase objectives
```

Without `--execute`, this performs standard-library path/hash preflight only. It does not train, load image pixels, evaluate predictions or recalculate scores. Supported phases are `objectives`, `native`, `controls` and `report`; real execution requires explicitly adding `--execute` after the preflight succeeds. Configure a complete retained-artifact archive, separate output directory and raw dataset roots. Original source snapshots are required for historical hash checks; adapted public code is not a substitute for those snapshots.

Missing real inputs must cause an explicit failure; no replacement dataset or automatic training substitute is supplied. Obtain source datasets through their providers and follow their terms. Local DRIVE test-label acquisition and the upstream HRF annotation conversion remain incompletely documented; exact internal replay does not independently resolve that source limitation.

Repository preparation did not retrain models or recompute the saved experimental metrics or confidence intervals. New path/command adapters are documented separately from the executed study sources.

## Citation and licensing

Use [CITATION.cff](CITATION.cff) to cite this software repository. The associated unpublished manuscript title and author list are given above; a DOI for that new manuscript has not been added.

A code license has not yet been selected by the authors, and this repository currently contains no `LICENSE` file. No MIT/GPL or other license grant is asserted. Source-dataset access and redistribution remain subject to the original providers' terms.

See [method attribution and retained-source provenance](docs/ATTRIBUTIONS.md) for the clDice and baseline references.
