"""Synthetic structural-metric checks; no retinal data or models are evaluated."""
import unittest

import numpy as np

from flow_matching_refiner.structure_metrics import (
    derive_thin_cutoff, metrics, soft_cldice_loss,
)


class HardStructureChecks(unittest.TestCase):
    def test_perfect_and_empty_segmentations(self):
        truth = np.zeros((11,11), dtype=bool)
        truth[2:9,5] = True
        for mask in (truth, np.zeros_like(truth)):
            score = metrics(mask, mask, mask, 1.)
            for key in ("dice", "iou", "precision", "recall", "cldice", "thin_skeleton_recall"):
                self.assertEqual(score[key], 1., key)
            for key in ("fp", "fn", "new_error_count", "fixed_error_count", "betti0_absolute_error"):
                self.assertEqual(score[key], 0, key)
        empty_pred = metrics(np.zeros_like(truth), truth, truth, 1.)
        self.assertEqual(empty_pred["cldice"], 0.)
        self.assertEqual(empty_pred["thin_skeleton_missed"], int(truth.sum()))
        self.assertEqual(empty_pred["thin_skeleton_recall"], 0.)

    def test_false_bridge_harms_cldice_and_components(self):
        truth = np.zeros((15,15), dtype=bool)
        truth[2:13,4] = True
        truth[2:13,10] = True
        bridged = truth.copy()
        bridged[7,4:11] = True
        score = metrics(bridged, truth, truth, 1.)
        self.assertEqual(score["gt_betti0"], 2)
        self.assertEqual(score["predicted_betti0"], 1)
        self.assertEqual(score["betti0_absolute_error"], 1)
        self.assertGreater(score["cldice"], 0.)
        self.assertLess(score["cldice"], 1.)
        self.assertEqual(score["new_error_count"], 5)
        self.assertNotIn("false_bridges", score)

    def test_new_and_fixed_error_opportunities(self):
        truth = np.zeros((9,9), dtype=bool)
        truth[2:7,4] = True
        baseline = truth.copy()
        baseline[2,4] = False  # one baseline error to fix
        revised = truth.copy()
        revised[0,0] = True   # one new error
        score = metrics(revised, truth, baseline, 1.)
        self.assertEqual(score["baseline_wrong_pixels"], 1)
        self.assertEqual(score["fixed_error_count"], 1)
        self.assertEqual(score["new_error_count"], 1)
        self.assertEqual(score["fixed_error_rate"], 1.)
        self.assertEqual(score["new_error_rate"], 1/80)
        # Hamming error change = new errors minus corrected errors.
        self.assertEqual(score["fp"]+score["fn"]-score["baseline_wrong_pixels"],
                         score["new_error_count"]-score["fixed_error_count"])

    def test_thin_cutoff_uses_only_supplied_training_masks(self):
        training = np.zeros((21,21), dtype=bool)
        training[2:19,10] = True
        cutoff = derive_thin_cutoff([training])
        self.assertEqual(cutoff["cutoff"], 1.)
        self.assertEqual(cutoff["training_images"], 1)
        self.assertEqual(cutoff["pooled_training_skeleton_pixels"], 17)
        before = cutoff.copy()
        heldout = np.zeros_like(training)
        heldout[3:18,3:8] = True
        score = metrics(heldout, heldout, heldout, cutoff["cutoff"])
        self.assertEqual(score["thin_cutoff_px"], 1.)
        self.assertEqual(cutoff, before)
        no_thin = metrics(training, training, training, .5)
        self.assertEqual(no_thin["thin_skeleton_gt_count"], 0)
        self.assertEqual(no_thin["thin_skeleton_recall"], 1.)
        with self.assertRaises(ValueError):
            derive_thin_cutoff([np.zeros_like(training)])

    def test_threshold_shape_and_finite_checks(self):
        empty = np.zeros((5,5))
        equal = empty.copy(); equal[2,2] = .5
        score = metrics(equal, equal, empty, 1.)
        self.assertEqual(score["tp"], 1)
        with self.assertRaises(ValueError):
            metrics(np.zeros((1,5,5)), empty, empty, 1.)
        with self.assertRaises(ValueError):
            metrics(empty, empty, empty, float("nan"))
        bad = empty.copy(); bad[0,0] = np.nan
        with self.assertRaises(ValueError):
            metrics(bad, empty, empty, 1.)


class SoftStructureChecks(unittest.TestCase):
    def test_finite_nonzero_gradient_and_batch_reduction(self):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(17)
        logits = torch.randn(2,1,17,17, requires_grad=True)
        truth = torch.zeros_like(logits)
        truth[:,:,3:14,8] = 1.
        per_image = soft_cldice_loss(logits.sigmoid(), truth, reduction="none")
        self.assertEqual(tuple(per_image.shape), (2,))
        self.assertTrue(bool(torch.isfinite(per_image).all()))
        per_image.mean().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(logits.grad.abs().sum().item(), 0.)

    def test_perfect_binary_and_empty_soft_loss(self):
        import torch
        truth = torch.zeros(1,1,15,15)
        truth[:,:,2:13,7] = 1.
        self.assertAlmostEqual(soft_cldice_loss(truth, truth).item(), 0., places=7)
        empty = torch.zeros_like(truth)
        self.assertAlmostEqual(soft_cldice_loss(empty, empty).item(), 0., places=7)


if __name__ == "__main__":
    unittest.main()
