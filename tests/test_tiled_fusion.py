"""Check that tiled exposure fusion stays close to whole-image fusion."""

import unittest

import cv2
import numpy as np

from core.scheduler import mertens


class TiledFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(1)

    def test_tiled_result_matches_whole_image_tone(self):
        height, width = 144, 192
        y, x = np.mgrid[:height, :width]
        scene = np.stack([
            35 + 0.6 * x + 0.2 * y,
            45 + 0.4 * x + 0.35 * y,
            60 + 0.3 * x + 0.25 * y,
        ], axis=2).astype(np.float32)
        scene[20:100, 80:160] = [240, 230, 200]  # bright window against a dim wall
        exposures = [np.clip(scene * scale, 0, 255).astype(np.float32)
                     for scale in (0.45, 1.0, 1.8)]

        whole, whole_tiles = mertens(exposures, tile_size=0, overlap=24, guide_factor=4)
        tiled, tile_count = mertens(exposures, tile_size=96, overlap=24, guide_factor=4)

        self.assertEqual(whole_tiles, 1)
        self.assertGreater(tile_count, 1)
        self.assertEqual(tiled.shape, whole.shape)
        self.assertTrue(np.isfinite(tiled).all())
        difference = np.abs(whole - tiled) * 255
        self.assertLess(float(difference.mean()), 8.0)
        self.assertLess(float(np.percentile(difference, 95)), 20.0)

    def test_image_within_one_tile_uses_whole_image_fusion(self):
        image = np.full((32, 48, 3), 80, dtype=np.float32)
        exposures = [image * scale for scale in (0.5, 1.0, 1.5)]
        whole, _ = mertens(exposures, tile_size=0, overlap=8)
        tiled, tile_count = mertens(exposures, tile_size=64, overlap=8)

        self.assertEqual(tile_count, 1)
        np.testing.assert_array_equal(tiled, whole)


if __name__ == "__main__":
    unittest.main()
