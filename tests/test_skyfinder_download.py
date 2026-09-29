"""Interrupted camera archives must resume without duplicating bytes."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.aeroswap_data import _fetch


class Response(io.BytesIO):
    def __init__(self, data, status, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}


class DownloadTests(unittest.TestCase):
    def test_resumes_partial_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            dest = Path(temporary) / "camera.zip"
            dest.with_name("camera.zip.part").write_bytes(b"abc")
            with patch("urllib.request.urlopen", return_value=Response(b"def", 206, "bytes 3-5/6")) as urlopen:
                _fetch("https://example.invalid/camera.zip", dest, 6)
            self.assertEqual(dest.read_bytes(), b"abcdef")
            self.assertIn("bytes=3-", urlopen.call_args.args[0].headers["Range"])

    def test_restarts_if_server_ignores_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            dest = Path(temporary) / "camera.zip"
            dest.with_name("camera.zip.part").write_bytes(b"abc")
            with patch("urllib.request.urlopen", return_value=Response(b"uvwxyz", 200)):
                _fetch("https://example.invalid/camera.zip", dest, 6)
            self.assertEqual(dest.read_bytes(), b"uvwxyz")


if __name__ == "__main__":
    unittest.main()
