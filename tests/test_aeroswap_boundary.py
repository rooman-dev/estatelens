"""Check the boundary metric's image-border and mismatch behavior."""

import unittest

import numpy as np

from eval.aeroswap_eval import boundary_pixel_counts, sky_boundary


class BoundaryIoUTests(unittest.TestCase):
    def test_full_mask_includes_frame_border(self):
        full = np.ones((8, 8), dtype=np.uint8)
        self.assertEqual(int(sky_boundary(full).sum()), 28)
        self.assertEqual(int(sky_boundary(np.zeros((8, 8), dtype=np.uint8)).sum()), 0)

    def test_perfect_and_shifted_masks(self):
        truth = np.zeros((32, 32), dtype=np.uint8)
        truth[:15, :] = 1
        exact = boundary_pixel_counts(truth, sky_boundary(truth))
        self.assertEqual(exact[0], exact[1])
        shifted = np.zeros_like(truth)
        shifted[:18, :] = 1
        mismatch = boundary_pixel_counts(shifted, sky_boundary(truth))
        self.assertLess(mismatch[0], mismatch[1])
        self.assertGreater(mismatch[0], 0)


if __name__ == "__main__":
    unittest.main()
