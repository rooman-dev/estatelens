"""Check the desktop-side preview without importing PyTorch."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from processing.sky_preview import mask_overlay, sky_runner


class SkyPreviewTests(unittest.TestCase):
    def test_overlay_preserves_input_and_shows_selected_sky(self):
        rgb = np.full((64, 64, 3), (80, 60, 40), dtype=np.uint8)
        original = rgb.copy()
        probability = np.full((512, 512), 255, dtype=np.uint8)
        result, coverage = mask_overlay(rgb, probability)
        self.assertEqual(coverage, 1.0)
        self.assertTrue(np.array_equal(rgb, original))
        self.assertFalse(np.array_equal(result[32, 32], rgb[32, 32]))

        empty, coverage = mask_overlay(rgb, np.zeros_like(probability))
        self.assertEqual(coverage, 0.0)
        self.assertTrue(np.array_equal(empty, rgb))

    def test_runtime_config_is_an_argv_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "sky_runtime.json").write_text(json.dumps(["wsl", "--", "python3"]),
                                                   encoding="utf-8")
            self.assertEqual(sky_runner(root), ["wsl", "--", "python3"])
            (data / "sky_runtime.json").write_text('"not a command list"', encoding="utf-8")
            with self.assertRaises(ValueError):
                sky_runner(root)


if __name__ == "__main__":
    unittest.main()
