"""Synthetic native-loader checks; no real source, checkpoint or inference access."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib


class NativeSourceTests(unittest.TestCase):
    def test_import_is_inert(self):
        code = ("import sys; import flow_matching_refiner.native_patch_data; "
                "assert not ({'torch','numpy','cv2','PIL'} & set(sys.modules)); print('inert')")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "inert")

    @staticmethod
    def bitmap(values, origin):
        import numpy as np
        from PIL import Image
        stream = io.BytesIO()
        Image.fromarray(np.asarray(values, dtype=np.uint8)).save(stream, format="PNG")
        return dict(bitmap=dict(data=base64.b64encode(zlib.compress(stream.getvalue())).decode(), origin=origin))

    def test_bitmap_uses_full_canvas_and_origin(self):
        import numpy as np
        from flow_matching_refiner.native_patch_data import _bitmap_canvas
        canvas, info = _bitmap_canvas(self.bitmap([[0, 1], [2, 0]], [3, 1]), dict(width=8, height=5))
        expected = np.zeros((5, 8), dtype=np.uint8)
        expected[1, 4] = expected[2, 3] = 1
        self.assertTrue(np.array_equal(canvas, expected))
        self.assertEqual(info["origin_xy"], [3, 1])
        self.assertEqual(info["positive_patch_pixels"], 2)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            _bitmap_canvas(self.bitmap([[1, 1]], [7, 0]), dict(width=8, height=5))

    def test_non_native_test_label_is_not_upscaled(self):
        import numpy as np
        from PIL import Image
        from flow_matching_refiner.native_patch_data import _read_native_arrays
        with tempfile.TemporaryDirectory(prefix="native_loader_test_") as folder:
            folder = Path(folder)
            Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(folder / "rgb.tif")
            Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(folder / "gt.png")
            Image.fromarray(np.full((12, 16), 255, dtype=np.uint8)).save(folder / "fov.gif")
            record = dict(image_path=str(folder / "rgb.tif"), gt_path=str(folder / "gt.png"), fov_path=str(folder / "fov.gif"))
            arrays, info = _read_native_arrays("DRIVE", record, "test")
            self.assertIsNone(arrays["g"])
            self.assertFalse(info["native_gt_available"])
            self.assertEqual(arrays["small_g"].shape, (512, 512))
            self.assertEqual(arrays["fov"].shape, (12, 16))
            with self.assertRaisesRegex(RuntimeError, "training/validation"):
                _read_native_arrays("DRIVE", record, "train")

    def test_hrf_native_bitmap_matches_retained_nearest_conversion(self):
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from flow_matching_refiner.native_patch_data import _native_tensors, _read_native_arrays, _verify_reference_512
        with tempfile.TemporaryDirectory(prefix="native_loader_test_") as folder:
            folder = Path(folder)
            rgb = np.zeros((12, 16, 3), dtype=np.uint8)
            rgb[:, :, 1] = np.arange(16, dtype=np.uint8)[None] * 11
            Image.fromarray(rgb).save(folder / "rgb.png")
            vessel = self.bitmap([[1, 0], [0, 1]], [4, 3]); vessel["classTitle"] = "vessels"
            fov = self.bitmap(np.ones((10, 12), dtype=np.uint8), [2, 1]); fov["classTitle"] = "field of view"
            annotation = folder / "ann.json"
            annotation.write_text(json.dumps(dict(size=dict(width=16, height=12), objects=[vessel, fov])))
            record = dict(image_path=str(folder / "rgb.png"), gt_path=str(annotation), fov_path=str(annotation))
            arrays, info = _read_native_arrays("HRF", record, "train")
            self.assertEqual(arrays["g"].shape, (12, 16))
            self.assertTrue(arrays["g"][3, 4] and arrays["g"][4, 5])
            self.assertEqual(info["native_gt_positive_pixels"], 2)
            green = cv2.createCLAHE(clipLimit=5., tileGridSize=(32, 32)).apply(rgb[:, :, 1])
            green = np.uint8(np.clip((green / 255.) ** (1 / 1.2) * 255, 0, 255))
            reference = dict(rgb=torch.tensor(cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR) / 255., dtype=torch.float32).permute(2, 0, 1),
                             x=torch.tensor(cv2.resize(green, (512, 512), interpolation=cv2.INTER_LINEAR) / 255., dtype=torch.float32)[None],
                             g=torch.tensor(arrays["small_g"], dtype=torch.float32)[None],
                             fov=torch.tensor(arrays["small_fov"], dtype=torch.bool)[None])
            old_hashes = {key: value.numpy().tobytes() for key, value in reference.items()}
            hashes = _verify_reference_512(arrays, reference, "synthetic")
            self.assertEqual(set(hashes), {"rgb", "x", "g", "fov"})
            fields = _native_tensors(arrays, torch.full((1, 512, 512), .25, dtype=torch.float32))
            self.assertEqual(fields["n_rgb"].shape, (3, 12, 16))
            self.assertEqual(fields["n_g"].dtype, torch.float32)
            self.assertEqual(fields["n_fov"].dtype, torch.bool)
            self.assertTrue(torch.equal(fields["n_coarse"], torch.full((1, 12, 16), .25)))
            self.assertTrue(all(value.device.type == "cpu" for value in fields.values()))
            self.assertTrue(all(old_hashes[key] == value.numpy().tobytes() for key, value in reference.items()))
            reference["g"] = reference["g"].clone()
            reference["g"][0, 0, 0] = 1 - reference["g"][0, 0, 0]
            with self.assertRaisesRegex(RuntimeError, "does not replay"):
                _verify_reference_512(arrays, reference, "corrupt")

    def test_workspace_rejects_retained_inputs(self):
        from flow_matching_refiner.data import ROOT
        from flow_matching_refiner.native_patch_data import _assert_workspace
        with self.assertRaisesRegex(RuntimeError, "retained"):
            _assert_workspace((ROOT / "revise" / "bad_output").resolve(), {})
        _assert_workspace((ROOT / "outputs" / "new_native_audit").resolve(), {})


if __name__ == "__main__":
    unittest.main()
