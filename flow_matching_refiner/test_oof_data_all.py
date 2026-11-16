"""Synthetic inventory checks; no real image loading, inference or training."""
import copy
import unittest

from flow_matching_refiner.oof_data_all import validate_membership


def drive_fixture():
    split = dict(train=[f"train_{i}" for i in range(16)],
                 validation=[f"validation_{i}" for i in range(4)],
                 test=[f"test_{i}" for i in range(20)])
    folds = [split["train"][start:start+4] for start in range(0, 16, 4)]
    return split, folds


def hrf_fixture():
    split = {key: [f"{index:02}_{category}" for category in ("h", "dr", "g") for index in indices]
             for key, indices in (("train", range(1, 10)), ("validation", range(10, 12)), ("test", range(12, 16)))}
    folds = [[f"{index:02}_{category}" for category in ("h", "dr", "g")
              for index in range(start, start+3)] for start in (1, 4, 7)]
    return split, folds


class InventoryChecks(unittest.TestCase):
    def test_valid_image_folds(self):
        validate_membership("DRIVE", *drive_fixture())
        validate_membership("HRF", *hrf_fixture())

    def test_outer_test_overlap_rejected(self):
        split, folds = drive_fixture()
        split["test"][0] = split["train"][0]
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            validate_membership("DRIVE", split, folds)

    def test_duplicate_held_image_rejected(self):
        split, folds = hrf_fixture()
        broken = copy.deepcopy(folds)
        broken[1][0] = broken[0][0]
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            validate_membership("HRF", split, broken)

    def test_pathology_imbalance_rejected(self):
        split, folds = hrf_fixture()
        folds[0][0], folds[1][3] = folds[1][3], folds[0][0]
        with self.assertRaisesRegex(RuntimeError, "pathology-stratified"):
            validate_membership("HRF", split, folds)


if __name__ == "__main__":
    unittest.main()
