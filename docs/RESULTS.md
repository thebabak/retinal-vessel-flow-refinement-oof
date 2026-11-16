# Saved results and interpretation

These tables reproduce completed, verified summaries. Formatting to six decimals does not change the underlying values; the [published CSV/JSON records](../results/) retain the saved precision. No model fitting, prediction evaluation or confidence-interval calculation was performed while preparing this repository.

## Raw results on the preserved comparison grid

All metrics use the original **512 × 512 GT/FOV fields**. Each learned entry is mean ± sample SD across the five fitted refiner-training means. The shared frozen baseline has no refiner-seed SD. The five earlier learned controls are reused, not retrained.

### CHASE-DB1

| Method | Dice | clDice | Thin-skeleton recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline (B) | 0.810568 | 0.830638 | 0.564885 | 0.814884 |
| Direct (D) | 0.809555 ± 0.000664 | 0.826952 ± 0.000830 | 0.545020 ± 0.007400 | 0.832526 ± 0.004258 |
| Endpoint-iterative (EI) | 0.809063 ± 0.000319 | 0.827623 ± 0.000813 | 0.558699 ± 0.009555 | 0.825878 ± 0.001870 |
| Full FM (FM) | 0.808678 ± 0.000204 | 0.827763 ± 0.000150 | 0.553691 ± 0.000888 | 0.828684 ± 0.000472 |
| Selective, no gate (SG) | 0.808812 ± 0.001368 | 0.830509 ± 0.001177 | 0.592742 ± 0.029757 | 0.812784 ± 0.011424 |
| ESFM (ES) | 0.809664 ± 0.000282 | 0.830512 ± 0.001642 | 0.579056 ± 0.018569 | 0.817808 ± 0.006619 |
| Thin/anchor Direct (O1) | 0.805423 ± 0.003107 | 0.835866 ± 0.000758 | 0.640331 ± 0.017189 | 0.786709 ± 0.009935 |
| Endpoint/anchor ESFM (O2) | 0.809365 ± 0.000500 | 0.829004 ± 0.000974 | 0.570239 ± 0.016370 | 0.821639 ± 0.003980 |
| Native-patch Direct (O3) | 0.812398 ± 0.000327 | 0.832913 ± 0.000653 | 0.597733 ± 0.003634 | 0.815997 ± 0.001154 |

### DRIVE

| Method | Dice | clDice | Thin-skeleton recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline (B) | 0.797975 | 0.825170 | 0.614437 | 0.763946 |
| Direct (D) | 0.801207 ± 0.000388 | 0.818733 ± 0.001577 | 0.582301 ± 0.008069 | 0.791603 ± 0.006504 |
| Endpoint-iterative (EI) | 0.801807 ± 0.000354 | 0.819781 ± 0.001407 | 0.592314 ± 0.005552 | 0.791661 ± 0.005584 |
| Full FM (FM) | 0.800501 ± 0.000511 | 0.821514 ± 0.001198 | 0.607468 ± 0.012877 | 0.784969 ± 0.005524 |
| Selective, no gate (SG) | 0.798477 ± 0.002902 | 0.822808 ± 0.004145 | 0.634144 ± 0.024629 | 0.769758 ± 0.013987 |
| ESFM (ES) | 0.799946 ± 0.001321 | 0.821275 ± 0.004360 | 0.615505 ± 0.015721 | 0.779821 ± 0.013022 |
| Thin/anchor Direct (O1) | 0.791415 ± 0.002721 | 0.831304 ± 0.000964 | 0.675335 ± 0.015389 | 0.739298 ± 0.007846 |
| Endpoint/anchor ESFM (O2) | 0.801119 ± 0.000498 | 0.818803 ± 0.002883 | 0.597525 ± 0.009938 | 0.790082 ± 0.008467 |
| Native-patch Direct (O3) | 0.804570 ± 0.000534 | 0.825299 ± 0.000496 | 0.615358 ± 0.003817 | 0.783800 ± 0.001245 |

### HRF

| Method | Dice | clDice | Thin-skeleton recall | Precision |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline (B) | 0.733117 | 0.719941 | 0.657610 | 0.684823 |
| Direct (D) | 0.741577 ± 0.003029 | 0.722660 ± 0.002856 | 0.627416 ± 0.008103 | 0.725491 ± 0.010795 |
| Endpoint-iterative (EI) | 0.737538 ± 0.000286 | 0.717764 ± 0.002790 | 0.645873 ± 0.015610 | 0.714677 ± 0.006670 |
| Full FM (FM) | 0.736348 ± 0.000658 | 0.717889 ± 0.002626 | 0.652142 ± 0.016427 | 0.704730 ± 0.006673 |
| Selective, no gate (SG) | 0.733350 ± 0.002883 | 0.716578 ± 0.002230 | 0.675752 ± 0.029032 | 0.687456 ± 0.016922 |
| ESFM (ES) | 0.735179 ± 0.001606 | 0.717397 ± 0.001722 | 0.657112 ± 0.023914 | 0.703234 ± 0.018569 |
| Thin/anchor Direct (O1) | 0.721827 ± 0.005688 | 0.718836 ± 0.004740 | 0.743081 ± 0.020767 | 0.640066 ± 0.014299 |
| Endpoint/anchor ESFM (O2) | 0.736088 ± 0.001040 | 0.716763 ± 0.002550 | 0.657325 ± 0.024265 | 0.705002 ± 0.015398 |
| Native-patch Direct (O3) | 0.755753 ± 0.003517 | 0.752433 ± 0.004938 | 0.715623 ± 0.009706 | 0.707885 ± 0.002634 |

## Existing planned raw Dice contrasts

O1 is compared with retained Direct (D), O2 with retained ESFM (ES), and O3 with retained Direct (D). These roles were defined within separate experimental phases. The intervals below are copied from those analyses; they are **exploratory, unadjusted percentile intervals conditional on the fitted models**, not a newly pooled confirmatory test family.

| Dataset | Contrast | Mean Dice difference | Saved 95% interval |
| --- | --- | ---: | --- |
| CHASE-DB1 | O1 − D | -0.004132 | [-0.005339, -0.002747] |
| CHASE-DB1 | O2 − ES | -0.000299 | [-0.000506, -0.000089] |
| CHASE-DB1 | O3 − D | +0.002843 | [0.001565, 0.004405] |
| DRIVE | O1 − D | -0.009791 | [-0.013319, -0.006415] |
| DRIVE | O2 − ES | +0.001174 | [0.000340, 0.002019] |
| DRIVE | O3 − D | +0.003363 | [0.002478, 0.004233] |
| HRF | O1 − D | -0.019750 | [-0.025263, -0.014120] |
| HRF | O2 − ES | +0.000909 | [0.000413, 0.001462] |
| HRF | O3 − D | +0.014176 | [0.008405, 0.020247] |

CHASE-DB1 intervals resample its four paired two-eye patient clusters. DRIVE/HRF intervals resample images because independent patient IDs were not verified. The five fitted outputs stay together within each resampling unit; pixels and refiner seeds are not independently resampled. Full structural contrasts are available in the saved tables.

## What the comparisons support

- **O1 improves thin-skeleton recall relative to Direct on all three retained cohorts, while reducing raw Dice and precision.** It also exceeds O3 in thin recall on all three. Its clDice change relative to Direct is positive on CHASE-DB1 and DRIVE; the HRF clDice interval includes zero. Recovery, overlap and precision are different outcomes.
- **O2 produces small, dataset-dependent Dice changes relative to ESFM.** Its planned Dice difference is negative on CHASE-DB1 and positive on DRIVE/HRF. Its clDice changes relative to ESFM are negative on all three datasets. Full local endpoint supervision and anchoring do not deliver a consistent structural improvement in this experiment.
- **O3 has the largest raw Dice point mean among all nine listed methods on each cohort.** Its precision remains below retained Direct on all three, and it does not have the highest thin recall. DRIVE O3-minus-baseline clDice is +0.000130 with interval [−0.000621, 0.000853]; the thin-recall difference is +0.000921 with interval [−0.002875, 0.004560]. Those structural intervals include zero.

A favorable point estimate or an interval on one side of zero is not a multiplicity-adjusted general-superiority decision. The cohorts were previously inspected, the baseline/OOF inputs are fixed, and the native recipe changes several factors together. No state-of-the-art, clinical-benefit, isolated-resolution or external-generalization claim follows from these records.

## Secondary blends and seed variation

Raw learned outputs remain primary. Validation also selected a secondary epoch/blend policy that could preserve the baseline exactly. O1 on DRIVE chose identity for all five seeds; its blended Dice therefore equals the baseline with zero variation across those identical outputs. That is a conservative validation choice, not five improved learned refinements.

All fifteen native-patch choices selected blend coefficient one and the same epoch as their raw model. Their raw/blended scores are identical by construction, not independent replications. The saved blend table preserves the other options' identity counts and coefficients in seed order 42, 123, 2026, 3000, 3001.

The five-seed SDs quantify fitted-refiner variation conditional on one shared baseline and one retained coarse-map set. They do not include baseline-training, alternative-split, development-search or external-cohort uncertainty.

## Recorded native inference work

These are metadata for the fixed first image at seed 42 in each dataset, with tile size 256, stride 128 and inference batch 16. They are **not measured milliseconds**.

| Dataset | Fixed image | Native height × width | Logical tile evaluations | Batched forwards |
| --- | --- | --- | ---: | ---: |
| CHASE-DB1 | `Image_02L` | 960 × 999 | 49 | 4 |
| DRIVE | `01_test` | 584 × 565 | 16 | 1 |
| HRF | `01_g` | 2336 × 3504 | 486 | 31 |

Each native tile uses one Direct velocity evaluation and no gate. Whole-grid Direct/O1 use one velocity evaluation per image; ESFM/O2 use eight velocity evaluations and one gate. Equal training parameter counts do not establish equal inference cost, speed or memory use.

## Record locations

- [Path-free verified summary](../results/verified_paper_results.json): saved means, SDs, comparisons and source digests.
- [Paper table CSVs](../results/paper_tables/): all nine raw methods, planned comparisons, secondary blends, seed means, checkpoint choices and recorded work counts.
- [Protocol](EXPERIMENT_PROTOCOL.md): partitions, fitting, phase-specific locks, outcome definitions and uncertainty limits.
- [Portability and required inputs](PORTABILITY.md): configuration, retained artifacts and executable entry points.

The source-image/annotation versions, large weights and raw predictions are not redistributed in this code repository. Incomplete local DRIVE test-label acquisition and upstream HRF conversion provenance remain disclosed limitations.
