# Interpretation and integration constraints

This is an additional **post-hoc exploratory** analysis of already inspected outcomes. The family plan was recorded before the new computation, not prospectively before training, cohort inspection or hypothesis generation. It does not turn the original development history into a confirmatory study.

The five seed-level observations are fitted-refiner means conditional on one frozen coarse baseline, one retained OOF-map source and one fixed cohort per dataset. Baseline contrasts use that one fixed reference; no independent baseline replicate is invented. Matching numeric training-seed labels supplies explicit blocks but does not prove equivalence of all random-number streams or a fully randomized block experiment.

Paired t tests assume independent approximately normal difference observations. Exact Wilcoxon signed-rank tests assume independent differences with a symmetric null distribution; exhaustive enumeration removes an asymptotic approximation but does not establish these assumptions. Zeros are dropped and tied absolute differences use average ranks. With five nonzero differences, the minimum attainable two-sided exact signed-rank p-value is 0.0625; no seed-level Wilcoxon result can reach 0.05 even before Holm correction.

The primary nine-comparison family and secondary nine baseline comparisons are corrected separately; t and signed-rank diagnostics and the two uncertainty units also have separate families. This does not control a single family spanning all 36 comparisons, all metrics, all prior model searches or both test types. Report both tests instead of selecting whichever yields a favorable decision.

The new unadjusted t intervals identify their unit explicitly. They are **not** replacements for the previously published patient/image-bootstrap intervals. A seed interval and a cohort interval can give different positions relative to zero because they quantify different variation. No previous bootstrap was rerun.

Patient/image sensitivity averages the five fixed outputs within an image before testing. CHASE averages its two eyes per subject and has only four subject observations. DRIVE and HRF use image observations with unverified patient independence. Neither level integrates baseline training, split selection, prior cohort exposure, annotation-source uncertainty or external institutions.

All per-image Dice values and matched differences derive from the nine SHA256-bound CSV files. No prediction tensor, raw image, checkpoint or GPU model was loaded, and no old result file was changed.
