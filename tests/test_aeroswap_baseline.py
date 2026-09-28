"""Check pooled, camera, and condition scoring without a PyTorch runtime."""

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from eval.aeroswap_eval import evaluate_brightness_numpy


class BrightnessBaselineTests(unittest.TestCase):
    def test_pooled_and_camera_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "test" / "images"
            masks = root / "test" / "masks"
            images.mkdir(parents=True)
            masks.mkdir(parents=True)

            # Camera 1 is perfect. Camera 2 predicts all sky and gets only one
            # sky pixel right; pooled IoU must sum pixels before division.
            cv2.imwrite(str(images / "1.png"), np.repeat(
                np.array([[255, 255], [0, 0]], dtype=np.uint8)[:, :, None], 3, axis=2))
            cv2.imwrite(str(masks / "1.png"), np.array([[255, 255], [0, 0]], dtype=np.uint8))
            cv2.imwrite(str(images / "2.png"), np.full((2, 2, 3), 255, dtype=np.uint8))
            cv2.imwrite(str(masks / "2.png"), np.array([[255, 0], [0, 0]], dtype=np.uint8))
            with (root / "test.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["camera", "image", "mask", "hour", "night"])
                writer.writerow(["1", "1.png", "1.png", 12, 0])
                writer.writerow(["2", "2.png", "2.png", 22, 1])

            report = evaluate_brightness_numpy(root)
            self.assertEqual(report.pooled["images"], 2)
            self.assertAlmostEqual(report.pooled["sky_iou"], 3 / 6)
            self.assertAlmostEqual(report.pooled["bg_iou"], 2 / 5)
            self.assertAlmostEqual(report.pooled["miou"], 0.45)
            self.assertAlmostEqual(report.per_camera["1"]["miou"], 1.0)
            self.assertAlmostEqual(report.per_camera["2"]["miou"], 0.125)
            self.assertAlmostEqual(report.mean_over_cameras, 0.5625)
            self.assertEqual(report.by_condition["day"]["images"], 1)
            self.assertEqual(report.by_condition["night"]["images"], 1)


if __name__ == "__main__":
    unittest.main()
