# Completed three-dataset study (2026-11-19_12-03-07)

Current manuscript: **A Three-Dataset Evaluation of Flow Matching for Retinal Vessel Refinement with Native Patches and Nested Out-of-Fold Inputs**. Authors: Babak Bayanian, Hanieh Naderi and Chinthaka Dinesh. This is an unpublished research draft.

- [PDF](../manuscripts/three_dataset_2026-11-19_07-40-21/retinal_flow_all_datasets_2026-11-19_07-40-21.pdf)
- [Standalone LaTeX](../manuscripts/three_dataset_2026-11-19_07-40-21/retinal_flow_all_datasets_2026-11-19_07-40-21.tex)
- [Complete Overleaf project](../manuscripts/three_dataset_2026-11-19_07-40-21/Overleaf/main.tex)
- [Public Overleaf/results ZIP](../manuscripts/three_dataset_2026-11-19_07-40-21/PUBLIC_MDPI_PDF_LaTeX_Overleaf_results_2026-11-19_12-03-07.zip)
- [Recorded implementation snapshot](../research_source/three_dataset_2026-11-19_12-03-07/README_2026-11-19_12-03-07.txt)
- [Current mean/SD CSV](../results/three_dataset_2026-11-19_07-40-21/mean_SD_2026-11-19_07-12-53.csv)
- [Current paired tests and confidence intervals](../results/three_dataset_2026-11-19_07-40-21/paired_tests_2026-11-19_07-12-53.csv)
- [All 15,070 per-image records](../results/three_dataset_2026-11-19_07-40-21/all_verified_per_image_2026-11-18_13-56-33.csv)
- [Independent verification report](../results/three_dataset_2026-11-19_07-40-21/independent_verification_2026-11-18_13-56-33.json)

## Completed scope

All 36 planned missing-dataset phases are complete: controlled recipes; encoded and mask-grid pilots; native Flow; anchored, strictly OOF, topology and retention variants; local acceptance; rollout and whole-image context; fresh nested source fits and outer CV; validation-only calibration, precision, solver, boundary, connection and strength/bias diagnostics; the DRIVE FR-UNet pilot scope; and frozen HRF SegFlow test inference.

The extension contains 482 new fitting calls, including upstream/source fits, pilots, auxiliary models and final refiners. This is not 482 independent repetitions of one experiment. Main comparisons use correction-training seeds 42, 123, 2026, 3000 and 3001. CHASE single-fit geometry and encoded prototypes remain single-fit scopes on the other datasets. FR-UNet remains three learning-rate pilots per dataset; no five-seed FR-UNet test benchmark is claimed. Conditional gates are preserved rather than inventing unexecuted results.

Independent replay checked 15,070 per-image rows, 60,280 scalar metrics, 482 consecutive training logs and coverage of every planned test image/seed/output group. Earlier papers and stored predictions were not replaced.

## Findings copied from the final saved tables

At threshold 0.5, the selected Flow and matched Direct fresh-CV outputs exactly preserve O3 probabilities within the valid FOV on all three datasets. Dice means are CHASE 0.815807, DRIVE 0.800042 and HRF 0.759100. Their zero correction-stage SD is conditional on shared upstream fits and identity selection; it is not zero population or full-pipeline uncertainty.

The prespecified validation-only calibration pilot failed on CHASE; no CHASE calibrated test result is inferred. The gate passed on DRIVE and HRF. Relative to separately calibrated O3:

| Dataset | Flow Dice change | clDice change | Thin-skeleton recall change | Flow minus matched Direct Dice |
|---|---:|---:|---:|---:|
| DRIVE | 0.000284 | 0.000282 | 0.013709 | -0.000118 |
| HRF | 0.004872 | 0.006520 | 0.033165 | -0.000435 |

The DRIVE clDice interval and both precision intervals include zero. All eight Holm-corrected Flow-versus-Direct four-endpoint tests are nonsignificant. Direct has slightly higher mean Dice. These results do not establish a Flow-specific advantage, equivalence or a joint four-metric improvement. CIs remain exploratory and conditional on the retained cohorts and shared source models; five correction seeds are not five whole-pipeline repetitions.

## Corrected statistical products

Use only the current mean/SD and paired CSVs linked above. Exact-constant seed vectors previously produced tiny floating-point SD residuals: 12 summary SD cells and two t/CI cases were corrected before this manuscript was compiled. Nonzero constant differences have undefined t/CI rather than fabricated significance. All measured means, mean differences, per-image metrics and FP32 predictions are unchanged. Earlier published tables remain historical and are not substituted for these current products.

For public per-image and identity CSVs, only the absolute local path column was omitted. Every other field, including all numerical strings and prediction hashes, was compared unchanged with its source. The export audit records both source and public SHA256 digests. No training, prediction evaluation, bootstrap or statistical recalculation was performed during publication.

## Access and limits

The exact source snapshot is distinct from the earlier guarded portable CLI. The latter still supports only its documented initial phases. A clone can inspect the saved results and manuscript; complete new execution requires exact retained inputs and upstream artifacts. New extension checkpoints/maps are not being uploaded by this publication. The earlier public artifact release remains available for its dated original scope, not the complete extension.

Raw DRIVE files and private reviewer correspondence are excluded. DRIVE local test-label acquisition provenance remains independently unverified. HRF source-pixel correspondence was verified with official foreground luminance greater than 254; historical conversion/acquisition history is a distinct unresolved issue. Settings use validation, but repeated validation/cohort exposure keeps the study exploratory. Study-specific author declarations still require author confirmation before submission.

Third-party licenses remain attached to their original sources. No new license for unrelated author code is implied.
