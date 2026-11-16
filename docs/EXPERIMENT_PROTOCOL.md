# Recorded experiment protocol

This document describes the completed studies underlying the associated manuscript. It records the executed recipes and their limits; preparing this repository did not run a new experiment. Public path/entry-point adaptations are described in [PORTABILITY.md](PORTABILITY.md).

## Inputs and fixed partitions

| Dataset | Training / validation / test images | Retained OOF folds | Evaluation unit |
| --- | --- | --- | --- |
| CHASE-DB1 | 16 / 4 / 8 | Four subject-grouped folds | Four test patients, two eyes each |
| DRIVE | 16 / 4 / 20 | Four image folds | Image; independent patient IDs unverified |
| HRF | 27 / 6 / 12 | Three pathology-stratified image folds | Image; independent patient IDs unverified |

Each dataset uses one frozen full-training baseline checkpoint with seed 42. Training coarse maps come from retained seed-42 fold baselines; validation/test coarse maps come from the full-training baseline. The current inputs therefore reuse eleven fold checkpoints across the three datasets, rather than fitting a new coarse model for every refiner seed.

Fold fitting excludes the held-out training fold. Historical fold checkpoint selection used the original validation partition, so the design is not a fully nested validation study. The smaller fold fitting sets and the full-training baseline can produce different input distributions; these records do not establish absence of that shift.

All comparative metrics use the original 512 × 512 ground truth (GT), field of view (FOV), threshold and image memberships. CHASE uses the full image; DRIVE and HRF use their retained FOV masks. CHASE/DRIVE label conversion uses Pillow nearest-neighbour sampling; HRF uses its recorded bitmap reconstruction and OpenCV nearest-neighbour sampling. Native source checks reproduce those retained comparison fields.

The local DRIVE test-label acquisition history and the upstream HRF annotation conversion are incompletely documented. Internal tensor/prediction consistency does not independently verify canonical upstream benchmark provenance. Raw images, annotations and FOV files must be obtained separately under their providers' terms.

## Shared refiner and controls

The shared spatial U-Net receives the current one-channel state, the one-channel coarse probability map and three RGB channels. Encoder widths are 12, 24, 48 and 96, with four-group normalization and SiLU activations. A time embedding modulates the blocks. The zero-initialized output head is scaled by six.

All recipes have 315,974 training parameters, including the auxiliary error classifier. Direct inference uses the 314,185-parameter velocity network without the classifier. Gated ESFM additionally uses the 1,789-parameter classifier, with gate `0.05 + 0.95 * sigmoid(logits)`. The new recipes do not use an autoencoder.

Five earlier learned controls are retained: ordinary Direct, endpoint-supervised iterative refinement without an FM term, full-target ungated FM, selective-target ungated FM, and gated error-selective FM (ESFM). Their completed models/results are reused, rather than retrained for this manuscript. With the frozen baseline and O1/O2/O3, each dataset's complete raw table has nine methods.

The shared FOV-aware objective contains BCE (weight 0.5), Dice (0.5), soft-clDice (0.1) and auxiliary error-classification BCE (0.1). Soft-clDice uses the same ten-iteration pooling skeleton approximation. A soft skeleton objective is not a guarantee of topological preservation. FM recipes additionally use the recorded thin-weighted velocity MSE, normalized by 36 and by its weighted valid support.

## The three options

### O1: thin recovery and correct-pixel anchoring

O1 retains the original Direct architecture/inference and adds:

- A fixed-weight 0.5 positive loss on training GT thin-skeleton pixels missed by the coarse prediction. Empty support contributes zero.
- A squared probability-displacement anchor on baseline-correct valid pixels. Nonempty correct foreground/background classes receive equal weight; this avoids background domination.

The selected anchor weight is 1.0 on all three datasets. GT-derived thin support and correctness are training supervision, not inference inputs. The thin term and anchor changed together, so the study does not isolate either term's causal effect.

### O2: full local endpoint supervision and anchoring

O2 retains the selective target, thin-weighted FM loss and learned gate of ESFM. Its segmentation terms supervise the complete local endpoint inside FOV, instead of a GT-conditioned composite that fixes initially correct pixels within the auxiliary loss. It also adds the same correct-pixel anchor, with selected weight 0.1 on all datasets. There is no additional O1 thin-recovery term.

The auxiliary endpoint comes from a sampled reference-conditioned linear bridge state. Training does not backpropagate through the full eight-step inference trajectory. Deployment starts from the coarse logits, uses eight Euler velocity updates and evaluates its gate once, without GT, error masks or reference skeletons.

### O3: native-input patch Direct refinement

O3 uses the original Direct objective, with neither FM nor the O1/O2 extra losses. It reads native RGB, GT and FOV. Its coarse native input is a bilinear projection of the retained 512-grid coarse probability map, not a fresh native-resolution baseline prediction.

Each training image supplies four 256 × 256 crops per epoch. Crop centres are drawn equally from training vessel pixels within FOV and uniformly from FOV; no test GT guides prediction. RGB/coarse/GT/FOV receive aligned right-angle rotations and flips. Small-image padding replicates RGB/coarse and excludes zero-padded GT/FOV from supervision.

Inference uses tile size 256, stride 128 and final-edge anchoring. A nonperiodic one-dimensional Hann window is floored at `1e-3` before forming positive separable weights. The model's **logit residuals**, rather than probabilities, are fused in FP32. Only the fused residual is projected to 512 using the dataset-specific nearest-neighbour rule, then applied to the **original** coarse logits. An exact-zero residual returns the original coarse probability without a logit/sigmoid round trip.

Pillow float-image nearest sampling is used for CHASE/DRIVE; OpenCV `INTER_NEAREST` is used for HRF. Native probability arrays are not downsampled to construct the comparison output. Native-grid segmentation accuracy is not reported.

Native inputs change resolution, crop sampling, spatial context, GroupNorm support, overlap fusion and validation workload together. No matched 512-patch, sampling-only or native whole-image control was run. The comparison concerns the complete native-patch recipe, not an isolated resolution effect.

## Fitting and sequential selection

Final refiner seeds are **42, 123, 2026, 3000 and 3001**. All final fits use 120 epochs, FP32, AdamW with initial learning rate `3e-4`, weight decay `1e-4`, cosine annealing and separate gradient-norm clipping at one for the velocity/classifier branches. No learning-rate warm-up is used.

Whole-grid recipes use batches of four 512-grid images. O3 uses batches of sixteen native patches and four patches per training image, matching the prior sampled-pixel/update budget. That match does not make coverage, context or inference work equivalent.

The O1/O2 phase first evaluates anchor candidates 0.1 and 1.0 with pilot seed 91011, separately for each option/dataset: twelve pilot fits in total. Selection uses learned raw validation Dice; ties within `1e-12` prefer the smaller anchor and then earlier epoch. Six chosen configurations are locked before thirty fresh final fits.

For each final fit, raw validation mean image Dice selects among learned epochs 10–120, evaluated every ten epochs; epoch-zero identity is excluded. A secondary policy independently selects epoch and blend coefficient from `0, 0.25, 0.5, 0.75, 1`, allowing identity. All thirty objective-phase final choices are locked across datasets before that phase's new test outputs are scored.

O3 is a later follow-up, after earlier cohort results had been inspected, with fixed patch settings and no additional pilot search. Its fifteen choices are locked before that phase's test outputs. There was **no single 45-model lock before all phases**. The completed inventory is twelve pilots plus forty-five final refiners; seventy-five earlier control fits are reused.

An initial native attempt was interrupted after 28 completed CHASE seed-42 epochs, without completing a fit or evaluating test data. The final native phase restarted all fifteen fits from initialization after an exact-zero-residual numerical-identity correction. Partial models did not supply final weights/results and are outside the completed-fit inventory.

## Outcomes and uncertainty

Saved per-image outcomes include Dice, IoU, precision, recall, hard clDice, thin-skeleton recall, new/fixed pixel-error counts and rates, and absolute eight-connected component-count error. Thin support is defined from training GT skeleton radii at the pooled one-third quantile; the retained cutoff is one comparison-grid pixel for each dataset. It does not measure native anatomical vessel diameter. Component counts and clDice do not independently classify false bridges or establish anatomical correctness.

An image receives equal weight within each fitted refiner; reported means then average the five refiner means. Reported SD is their sample SD. Five refiners share one baseline and retained coarse-map set, so this dispersion excludes baseline-training, partition and external-cohort uncertainty.

The original planned raw Dice comparisons are O1 minus Direct, O2 minus ESFM, and O3 minus Direct. Structural outcomes, blending, baseline contrasts and other retained-control contrasts are secondary. CHASE intervals enumerate all 256 ordered resamples of its four paired two-eye patient clusters. DRIVE/HRF use 50,000 paired image-bootstrap draws with seed 20261115. All five fitted outputs remain together within a resampling unit; pixels and training seeds are not independently resampled.

Intervals are saved exploratory, unadjusted percentile intervals, conditional on the fitted models and retained inputs. They are not a newly pooled confirmatory family, a general superiority test, an equivalence result or a confidence-coverage guarantee. Prior cohort inspection and unequal historical pilot-search budgets remain limitations.

## Computational work and artifact release

The fixed native examples record 49, 16 and 486 tile evaluations for CHASE, DRIVE and HRF, corresponding to four, one and 31 batched forwards at batch 16. Those are operation counts, not measured latency. Equal parameter counts do not establish equal runtime, memory use or clinical throughput.

Small saved numerical tables are published under [results](../results/). Large weights, raw prediction arrays and source photographs/annotations are excluded. Full replay requires the matched artifacts described in [PORTABILITY.md](PORTABILITY.md); a code clone alone does not reproduce the completed experiments.
