"""Check sky compositing and protection of the original photo."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from processing.sky_replace import composite_sky, fill_frame, save_sky_version


class SkyReplaceTests(unittest.TestCase):
    def test_composite_keeps_foreground_when_mask_empty(self):
        fused = np.full((32, 48, 3), (100, 50, 20), dtype=np.uint8)
        sky = np.full((10, 20, 3), (10, 120, 240), dtype=np.uint8)
        empty = np.zeros((512, 512), dtype=np.uint8)
        full = np.full((512, 512), 255, dtype=np.uint8)
        self.assertTrue(np.array_equal(composite_sky(fused, sky, empty), fused))
        self.assertTrue(np.array_equal(composite_sky(fused, sky, full),
                                       fill_frame(sky, 48, 32)))
        self.assertEqual(tuple(fused[0, 0]), (100, 50, 20))

    def test_save_never_overwrites_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fused_path = root / "fused.png"
            sky_path = root / "sky.png"
            output_path = root / "fused_sky.png"
            cv2.imwrite(str(fused_path), np.full((16, 24, 3), 50, dtype=np.uint8))
            cv2.imwrite(str(sky_path), np.full((12, 18, 3), 200, dtype=np.uint8))
            before = fused_path.read_bytes()
            probability = np.full((512, 512), 255, dtype=np.uint8)
            with self.assertRaises(ValueError):
                save_sky_version(fused_path, sky_path, probability, fused_path)
            with self.assertRaises(ValueError):
                save_sky_version(fused_path, sky_path, probability, sky_path)
            save_sky_version(fused_path, sky_path, probability, output_path)
            self.assertEqual(fused_path.read_bytes(), before)
            self.assertTrue(output_path.exists())
            self.assertFalse((root / "fused_sky.tmp.png").exists())
            self.assertEqual(int(cv2.imread(str(output_path))[8, 12, 0]), 200)


if __name__ == "__main__":
    unittest.main()
