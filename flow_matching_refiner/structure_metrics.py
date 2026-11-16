"""CPU structural metrics and a differentiable 2-D soft-clDice loss.

Hard clDice uses the original centreline precision/sensitivity definition:
https://github.com/jocpae/clDice
Shit et al., clDice, CVPR 2021, doi:10.1109/CVPR46437.2021.01629.
The differentiable pooling construction follows the same paper's soft skeleton
idea; finite iterations are an approximation, not a topology guarantee.

Binary masks use >= 0.5. Hard skeletons use skimage.morphology.skeletonize.
Betti-0 here is only the foreground component count with 8-connectivity.
It must not be relabelled as a measured false-bridge count.

Thin pixels are GT skeleton locations whose padded-background Euclidean
distance-transform radius is <= a cutoff fixed from TRAINING GT masks only.
Because radii are discrete and ties are included, the resulting thin subset
can contain more than one third of skeleton pixels.

Empty conventions: Dice/IoU/clDice and thin recall are 1 for empty matching
supports. With one nonempty segmentation and one empty segmentation, Dice,
IoU, precision, recall and clDice are 0. A zero opportunity count gives a zero
new/fixed-error rate. Counts are always returned so empty cases remain visible.

No data, predictions, model, or experiment is loaded at import time.
Torch is imported only by differentiable helper calls.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import scipy
from scipy import ndimage
import skimage
from skimage.morphology import skeletonize


DEFINITION = {
    "binary_threshold": 0.5,
    "threshold_operator": ">=",
    "hard_skeleton": "skimage.morphology.skeletonize, default 2-D Zhang algorithm",
    "radius": "scipy.ndimage.distance_transform_edt of one-pixel zero-padded GT, cropped back",
    "thin_cutoff": "pooled training-GT skeleton radii, q=1/3, numpy.quantile method=linear",
    "thin_selection": "GT skeleton AND radius <= fixed training-derived cutoff",
    "new_error_rate": "baseline-correct pixels made wrong / baseline-correct pixels",
    "fixed_error_rate": "baseline-wrong pixels corrected / baseline-wrong pixels",
    "betti0_connectivity": 8,
    "betti0_limitation": "Component-count difference only; not false-bridge classification or topology guarantee",
}


def dependency_versions():
    """Versions of libraries implementing the hard metrics; no Torch import."""
    return {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__}


def _mask(array, name):
    value = np.asarray(array)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2-D array")
    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise TypeError(f"{name} must be numeric or boolean")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(value >= .5)


def _radii(mask):
    return ndimage.distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]


def derive_thin_cutoff(training_GT_masks: Iterable[np.ndarray]):
    """Return cutoff metadata from the supplied TRAINING masks only.

    This function accepts no validation/test arrays and never consults model
    predictions. The caller is responsible for supplying its recorded training
    partition and preserving this result unchanged for validation/test scoring.
    All pooled skeleton pixels have equal weight, not all images equal weight.
    """
    radius_sets = []
    images = 0
    nonempty = 0
    for images, array in enumerate(training_GT_masks, start=1):
        mask = _mask(array, f"training_GT_masks[{images-1}]")
        centreline = skeletonize(mask)
        if centreline.any():
            radius_sets.append(_radii(mask)[centreline])
            nonempty += 1
    if not radius_sets:
        raise ValueError("No training-GT skeleton pixels; a thin cutoff is undefined")
    pooled = np.concatenate(radius_sets)
    cutoff = float(np.quantile(pooled, 1/3, method="linear"))
    return {
        "cutoff": cutoff,
        "thin_cutoff_px": cutoff,
        "quantile": 1/3,
        "quantile_method": "linear",
        "method": DEFINITION["thin_cutoff"],
        "radius_definition": DEFINITION["radius"],
        "training_images": images,
        "nonempty_training_skeleton_images": nonempty,
        "pooled_training_skeleton_pixels": int(pooled.size),
        "training_skeleton_pixels_at_or_below_cutoff": int((pooled <= cutoff).sum()),
        "source_scope": "caller-supplied training GT only; no validation/test or prediction input",
        "dependency_versions": dependency_versions(),
    }


def _fraction(numerator, denominator, empty=0.):
    return float(numerator / denominator) if denominator else float(empty)


def metrics(prob, gt, baseline, thin_cutoff):
    """Score one 2-D image; return scalar metrics and interpretable counts.

    ``thin_cutoff`` must be the numeric cutoff from derive_thin_cutoff, frozen
    before model validation/test evaluation. This function does not derive or
    tune it from the evaluated image. No image registration, tolerance dilation,
    or FOV crop is applied. CHASE whole-image evaluation is therefore explicit.
    """
    cutoff = float(thin_cutoff)
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("thin_cutoff must be a finite, nonnegative training-derived radius")
    predicted, truth, coarse = (_mask(x, name) for x, name in
                                 ((prob, "prob"), (gt, "gt"), (baseline, "baseline")))
    if predicted.shape != truth.shape or coarse.shape != truth.shape:
        raise ValueError("Prediction, GT and baseline shapes must match")
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    both_empty = not predicted.any() and not truth.any()
    pred_skeleton, gt_skeleton = skeletonize(predicted), skeletonize(truth)
    pred_count, gt_count = int(pred_skeleton.sum()), int(gt_skeleton.sum())
    topology_precision = _fraction(int((pred_skeleton & truth).sum()), pred_count, float(both_empty))
    topology_sensitivity = _fraction(int((gt_skeleton & predicted).sum()), gt_count, float(both_empty))
    cldice = _fraction(2*topology_precision*topology_sensitivity,
                      topology_precision+topology_sensitivity, float(both_empty))
    thin = gt_skeleton & (_radii(truth) <= cutoff)
    thin_count = int(thin.sum())
    thin_hit = int((thin & predicted).sum())
    baseline_correct, prediction_correct = coarse == truth, predicted == truth
    new_error = int((baseline_correct & ~prediction_correct).sum())
    fixed_error = int((~baseline_correct & prediction_correct).sum())
    base_correct_count = int(baseline_correct.sum())
    base_error_count = int((~baseline_correct).sum())
    connectivity = np.ones((3, 3), dtype=bool)
    pred_components = int(ndimage.label(predicted, structure=connectivity)[1])
    gt_components = int(ndimage.label(truth, structure=connectivity)[1])
    baseline_components = int(ndimage.label(coarse, structure=connectivity)[1])
    return {
        "dice": _fraction(2*tp, 2*tp+fp+fn, 1.),
        "iou": _fraction(tp, tp+fp+fn, 1.),
        "precision": _fraction(tp, tp+fp, float(both_empty)),
        "recall": _fraction(tp, tp+fn, float(both_empty)),
        "sensitivity": _fraction(tp, tp+fn, float(both_empty)),
        "specificity": _fraction(tn, tn+fp, float(not predicted.any())),
        "accuracy": (tp+tn)/truth.size,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "cldice": cldice,
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
        "predicted_skeleton_count": pred_count,
        "gt_skeleton_count": gt_count,
        "thin_skeleton_recall": _fraction(thin_hit, thin_count, 1.),
        "thin_skeleton_gt_count": thin_count,
        "thin_skeleton_hit": thin_hit,
        "thin_skeleton_missed": thin_count-thin_hit,
        "thin_cutoff_px": cutoff,
        "baseline_correct_pixels": base_correct_count,
        "baseline_wrong_pixels": base_error_count,
        "new_error_count": new_error,
        "new_error_rate": _fraction(new_error, base_correct_count),
        "fixed_error_count": fixed_error,
        "fixed_error_rate": _fraction(fixed_error, base_error_count),
        "predicted_betti0": pred_components,
        "gt_betti0": gt_components,
        "baseline_betti0": baseline_components,
        "betti0_absolute_error": abs(pred_components-gt_components),
        "baseline_betti0_absolute_error": abs(baseline_components-gt_components),
        "betti0_connectivity": 8,
    }


def soft_skeletonize(probability, iterations=10):
    """Differentiable pooling approximation for [B,1,H,W] probabilities.

    Cross-shaped 3-pixel erosion uses min(vertical-min, horizontal-min).
    Opening applies that erosion then a 3x3 max-pool dilation. PyTorch pooling
    padding ignores positions outside the image (max-pool implicit -infinity;
    negated max-pool erosion therefore implicit +infinity). This matches the
    reference soft-skeleton construction, not the hard EDT boundary convention.
    """
    import torch
    import torch.nn.functional as F
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("Expected a binary foreground probability tensor [B,1,H,W]")
    if not probability.is_floating_point():
        raise TypeError("Soft skeleton needs floating point probabilities")
    def erode(x):
        vertical = -F.max_pool2d(-x, (3,1), stride=1, padding=(1,0))
        horizontal = -F.max_pool2d(-x, (1,3), stride=1, padding=(0,1))
        return torch.minimum(vertical, horizontal)
    def opening(x):
        return F.max_pool2d(erode(x), 3, stride=1, padding=1)
    image = probability
    skeleton = F.relu(image-opening(image))
    for _ in range(iterations):
        image = erode(image)
        residual = F.relu(image-opening(image))
        skeleton = skeleton + F.relu(residual-skeleton*residual)
    return skeleton


def soft_cldice_loss(prob, gt, iterations=10, smooth=1., reduction="mean"):
    """Per-image soft clDice loss; no cross-image pooling or topology guarantee.

    Inputs are foreground probabilities and binary/soft GT of shape [B,1,H,W].
    The positive smoothing constant is applied to numerator and denominator of
    both topology ratios, as in the published loss implementation. No threshold
    or skeleton from a held-out image is used to change training hyperparameters.
    """
    import torch
    if prob.shape != gt.shape:
        raise ValueError("prob and gt must have identical shapes")
    if not math.isfinite(float(smooth)) or smooth <= 0:
        raise ValueError("smooth must be positive and finite")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be mean, sum or none")
    truth = gt.to(dtype=prob.dtype).detach()
    pred_skeleton = soft_skeletonize(prob, iterations)
    gt_skeleton = soft_skeletonize(truth, iterations)
    dimensions = (1,2,3)
    precision = ((pred_skeleton*truth).sum(dimensions)+smooth)/(pred_skeleton.sum(dimensions)+smooth)
    sensitivity = ((gt_skeleton*prob).sum(dimensions)+smooth)/(gt_skeleton.sum(dimensions)+smooth)
    loss = 1 - 2*precision*sensitivity/(precision+sensitivity).clamp_min(torch.finfo(prob.dtype).eps)
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss
