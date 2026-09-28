"""Regression checks for RAW discovery and mixed-size exposure brackets."""

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from core.prototype import find_frames, group_brackets


class BracketGroupingTests(unittest.TestCase):
    def test_raw_files_take_precedence_over_matching_camera_jpegs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("A.ORF", "A.JPG", "B.SRW", "B.JPG", "C.JPG"):
                (root / name).touch()
            with patch("core.prototype.read_exif", return_value=(None, None, None, None)):
                frames = find_frames(root)
        self.assertEqual([frame["path"].name for frame in frames],
                         ["A.ORF", "B.SRW", "C.JPG"])

    def test_three_and_five_frame_sets_and_long_exposure(self):
        start = datetime(2024, 1, 1)

        def frame(name, seconds, exposure):
            return {"path": Path(name), "timestamp": start + timedelta(seconds=seconds),
                    "exposure": exposure}

        frames = [
            frame("a1", 0, 0.01), frame("a2", 0, 0.02), frame("a3", 1, 0.04),
            frame("b1", 18, 0.01), frame("b2", 18, 0.02), frame("b3", 19, 0.04),
            frame("b4", 19, 0.08), frame("b5", 20, 0.16),
            frame("c1", 80, 0.6), frame("c2", 85, 5), frame("c3", 117, 30),
        ]
        brackets, warnings = group_brackets(frames)
        self.assertEqual([[f["path"].name for f in group] for group in brackets], [
            ["a1", "a2", "a3"],
            ["b1", "b2", "b3", "b4", "b5"],
            ["c1", "c2", "c3"],
        ])
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
