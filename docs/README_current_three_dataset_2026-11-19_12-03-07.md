# Retinal Vessel Flow Refinement with OOF Inputs

Code, manuscript and verified tables for **A Three-Dataset Evaluation of Flow Matching for Retinal Vessel Refinement with Native Patches and Nested Out-of-Fold Inputs**.

**Authors:** Babak Bayanian, Hanieh Naderi and Chinthaka Dinesh. The new manuscript is unpublished; no new manuscript DOI is asserted. The earlier published diffusion article is a separate study ([DOI](https://doi.org/10.3390/jimaging12110558)).

## Latest completed study

- [35-page MDPI manuscript PDF](manuscripts/three_dataset_2026-11-19_07-40-21/retinal_flow_all_datasets_2026-11-19_07-40-21.pdf)
- [LaTeX source](manuscripts/three_dataset_2026-11-19_07-40-21/retinal_flow_all_datasets_2026-11-19_07-40-21.tex) and [complete Overleaf/results ZIP](manuscripts/three_dataset_2026-11-19_07-40-21/PUBLIC_MDPI_PDF_LaTeX_Overleaf_results_2026-11-19_12-03-07.zip)
- [Completed scope, findings and reproducibility limits](docs/THREE_DATASET_COMPLETION_2026-11-19_12-03-07.md)
- [Current mean/SD tables](results/three_dataset_2026-11-19_07-40-21/mean_SD_2026-11-19_07-12-53.csv), [paired tests/CIs](results/three_dataset_2026-11-19_07-40-21/paired_tests_2026-11-19_07-12-53.csv) and [per-image records](results/three_dataset_2026-11-19_07-40-21/all_verified_per_image_2026-11-18_13-56-33.csv)
- [Recorded code for the completed extension](research_source/three_dataset_2026-11-19_12-03-07/README_2026-11-19_12-03-07.txt)

All 36 planned extension phases are complete. Main comparisons use five correction-training seeds (42, 123, 2026, 3000, 3001); pilots and shared upstream fits retain their separate uncertainty units. The 482 fitting calls include upstream fits, auxiliaries and pilots, rather than 482 repetitions of one comparison.

**Result:** at threshold 0.5, fresh-CV selected Flow and matched Direct preserve the O3 probabilities within the valid FOV on CHASE, DRIVE and HRF. Validation-only calibration passes its pilot gate on DRIVE/HRF and improves mean Dice relative to separately calibrated O3 by 0.000284/0.004872. Matched Direct attains slightly higher mean Dice; a Flow-specific advantage is not established. The DRIVE clDice interval and both precision intervals include zero. Repeated validation use, shared upstream estimators and data-provenance limits keep the conclusions exploratory.

These files supersede the old paused/incomplete progress statements for the completed scopes. Earlier dated documents, result tables, commits and the initial artifact release remain preserved as history. FR-UNet still has three validation learning-rate pilots per dataset, not a five-seed final test study. No unexecuted CHASE calibration-test result is supplied.

## Inspect results without training

```bash
python -S -m tools.show_latest_results
```

This standard-library command reads saved scalar CSV values. It does not load predictions, fit models or recalculate metrics, tests or intervals.

The complete new implementation is an exact historical snapshot in `research_source/`, separate from the earlier public portable adapters. Recorded paths, exact datasets, upstream checkpoints, OOF maps and locks remain required for full execution. New extension weights/maps are not included in this Git publication. The [earlier artifact release](https://github.com/thebabak/retinal-vessel-flow-refinement-oof/releases/tag/artifacts-2026-11-16_18-53-15) covers its original dated initial experiments only.

## Earlier initial recipes and portable code

O1 is Direct refinement with thin-vessel recovery/anchoring; O2 is error-selective Flow with full endpoint supervision; O3 is native-patch Direct refinement. They are separately evaluated recipes. Their earlier frozen-cohort results remain in [the original tables](results/) and [historical interpretation](docs/RESULTS.md). They must not be mixed with the new fresh-CV cohort means.

The existing [portable configuration and commands](docs/PORTABILITY.md) support the initial `objectives`, `native`, `controls` and `report` phases only. The new source snapshot does not silently extend that CLI. Missing real inputs must fail explicitly. Install dependencies in a separate environment using `requirements.txt`; see [recorded environment](docs/ENVIRONMENT.md). Existing synthetic software tests remain:

```bash
python -m unittest discover -s flow_matching_refiner -p "test_*.py"
python -m unittest discover -s tests -p "test_*.py"
```

## Data, provenance and licensing

Raw DRIVE images/GT/FOV are excluded because redistribution permission is unavailable. [Official dataset access and attribution](docs/DATA_LICENSES.md) remain authoritative. Retained local DRIVE test-label acquisition provenance is independently unverified. HRF target pixels match official foreground luminance greater than 254, while historical converter/acquisition history remains distinct. Exact internal replay does not resolve acquisition provenance.

The public manuscript is a research draft; author declarations need confirmation before submission. Private reviewer responses/correspondence are excluded. No model training, prediction evaluation or statistical recalculation was performed while publishing these files.

[CITATION.cff](CITATION.cff) describes this software. No author-code license has been selected; third-party notices remain attached to their respective snapshots. Read [the full completion record](docs/THREE_DATASET_COMPLETION_2026-11-19_12-03-07.md) for uncertainty units, current statistical products, artifact boundaries and remaining limitations.
