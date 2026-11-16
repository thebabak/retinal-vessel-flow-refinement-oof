"""FOV-aware structural scoring for the new DRIVE/HRF ESFM extensions.

Existing CHASE metrics and experiment files are not modified. All foreground
structures are intersected with FOV before hard skeletonization, distance
transforms and component counting. Pixel confusion counts, accuracy and error
opportunities exclude positions outside FOV. Thus an FOV boundary may truncate
a vessel and change its radius/skeleton: this convention is explicit and is
applied identically to training-cutoff construction and evaluation.

The new FOV-aware training objective is defined separately in
esfm_objective_all.py; this module defines evaluation and thin-weight support.
Betti-0 remains an 8-connected component-count discrepancy, not a false-bridge
classifier. No topology guarantee or patient independence is claimed.
"""
from __future__ import annotations

from itertools import zip_longest
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from .structure_metrics import (
    DEFINITION as CHASE_DEFINITION,
    dependency_versions,
    derive_thin_cutoff as _derive_cutoff,
    metrics as _whole_grid_metrics,
)


DEFINITION = {
    **CHASE_DEFINITION,
    "region": "pixel metrics restricted to valid FOV; structures zeroed outside FOV before skeletonization/EDT/components",
    "radius": "Euclidean EDT of GT intersect FOV, one-pixel zero padded then cropped",
    "new_error_rate": "baseline-correct valid-FOV pixels made wrong / baseline-correct valid-FOV pixels",
    "fixed_error_rate": "baseline-wrong valid-FOV pixels corrected / baseline-wrong valid-FOV pixels",
    "boundary_limitation": "FOV truncation changes local vessel radii and skeletons; one consistent rule is used throughout",
}


def _array(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a nonempty 2-D numeric array")
    if not np.issubdtype(array.dtype,np.number) and array.dtype != np.bool_:
        raise TypeError(f"{name} must be numeric or boolean")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _valid_fov(fov, shape):
    if fov is None:
        return np.ones(shape,dtype=bool)
    valid = _array(fov,"fov") >= .5
    if valid.shape != shape or not valid.any():
        raise ValueError("FOV shape must match and contain at least one valid pixel")
    return valid


def derive_thin_cutoff(training_GT_masks, training_FOV_masks=None):
    """Derive a fixed pooled q=1/3 radius cutoff from training GT/FOV only.

    With FOV supplied, its iterable must have exactly one mask per training GT.
    No prediction or validation/test array is accepted by this API.
    """
    counter = {"images":0,"valid_pixels":0,"image_pixels":0}
    sentinel = object()
    pairs = ((gt,None) for gt in training_GT_masks) if training_FOV_masks is None else zip_longest(
        training_GT_masks,training_FOV_masks,fillvalue=sentinel)
    def masked_training():
        for gt,fov in pairs:
            if gt is sentinel or fov is sentinel:
                raise ValueError("Training GT and FOV iterables have different lengths")
            truth = _array(gt,"training_gt") >= .5
            valid = _valid_fov(fov,truth.shape)
            counter["images"] += 1
            counter["valid_pixels"] += int(valid.sum())
            counter["image_pixels"] += int(valid.size)
            yield truth & valid
    result = _derive_cutoff(masked_training())
    result.update(
        region=DEFINITION["region"],radius_definition=DEFINITION["radius"],
        fov_supplied=training_FOV_masks is not None,
        pooled_training_valid_pixels=counter["valid_pixels"],
        pooled_training_image_pixels=counter["image_pixels"],
        fov_boundary_limitation=DEFINITION["boundary_limitation"],
    )
    return result


def thin_skeleton_mask(gt,thin_cutoff,fov=None):
    """Return fixed-cutoff thin GT skeleton support inside FOV, without tuning."""
    cutoff = float(thin_cutoff)
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("thin_cutoff must be finite and nonnegative")
    truth = _array(gt,"gt") >= .5
    valid = _valid_fov(fov,truth.shape)
    masked = truth & valid
    radius = distance_transform_edt(np.pad(masked,1,constant_values=False))[1:-1,1:-1]
    return skeletonize(masked) & (radius<=cutoff)


def metrics(prob,gt,baseline,thin_cutoff,fov=None):
    """Return hard structural metrics plus exact valid-FOV pixel bookkeeping."""
    probability,truth,coarse = (_array(value,name) for value,name in
                               ((prob,"prob"),(gt,"gt"),(baseline,"baseline")))
    if probability.shape != truth.shape or coarse.shape != truth.shape:
        raise ValueError("Prediction, GT and baseline shapes must match")
    valid = _valid_fov(fov,truth.shape)
    result = _whole_grid_metrics(np.where(valid,probability,0),np.where(valid,truth,0),
                                 np.where(valid,coarse,0),thin_cutoff)
    p,g,b = probability>=.5,truth>=.5,coarse>=.5
    tp = int((p & g & valid).sum())
    fp = int((p & ~g & valid).sum())
    fn = int((~p & g & valid).sum())
    tn = int((~p & ~g & valid).sum())
    n = int(valid.sum())
    baseline_correct = valid & (b==g)
    baseline_wrong = valid & (b!=g)
    new_errors = int((baseline_correct & (p!=g)).sum())
    fixed_errors = int((baseline_wrong & (p==g)).sum())
    correct_count,wrong_count = int(baseline_correct.sum()),int(baseline_wrong.sum())
    result.update(
        tp=tp,fp=fp,fn=fn,tn=tn,accuracy=(tp+tn)/n,
        specificity=tn/(tn+fp) if tn+fp else float(not (p&valid).any()),
        baseline_correct_pixels=correct_count,baseline_wrong_pixels=wrong_count,
        new_error_count=new_errors,new_error_rate=new_errors/correct_count if correct_count else 0.,
        fixed_error_count=fixed_errors,fixed_error_rate=fixed_errors/wrong_count if wrong_count else 0.,
        valid_pixels=n,fov_fraction=n/valid.size,
    )
    assert tp+fp+fn+tn == n
    assert fp+fn-wrong_count == new_errors-fixed_errors
    return result


def _self_test():
    truth = np.zeros((9,9),dtype=bool)
    truth[2:7,4] = True
    fov = np.zeros_like(truth)
    fov[1:8,2:7] = True
    outside_changed = truth.copy();outside_changed[0,:] = True
    base = truth.copy();base[2,4] = False
    first,second = metrics(truth,truth,base,1.,fov),metrics(outside_changed,truth,base,1.,fov)
    assert first == second
    assert first["valid_pixels"] == 35 and first["tn"] == 30
    assert first["fixed_error_count"] == 1 and first["baseline_correct_pixels"] == 34
    full = metrics(truth,truth,base,1.)
    old = _whole_grid_metrics(truth,truth,base,1.)
    assert all(full[key]==value for key,value in old.items())
    cutoff = derive_thin_cutoff([outside_changed],[fov])
    assert cutoff["cutoff"] == 1. and cutoff["pooled_training_valid_pixels"] == 35
    assert not (thin_skeleton_mask(outside_changed,1.,fov)&~fov).any()
    try:
        derive_thin_cutoff([truth,truth],[fov])
    except ValueError:
        pass
    else:
        raise AssertionError("Unequal GT/FOV inventories were accepted")
    print("Synthetic FOV metric checks passed; no real data accessed.")


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:_self_test()
