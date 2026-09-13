"""Web-optimised JPEG export.

    export_image(image, source_path, output_path)

Steps: convert to sRGB -> shrink to a max long edge -> 8-bit -> JPEG encode ->
copy EXIF from the source frame with ExifTool.

Encoder, best available first (none is a hard dependency):
    1. MozJPEG `cjpeg` on PATH           (real MozJPEG encoder)
    2. cv2 + mozjpeg_lossless_optimization (cv2 output, losslessly re-optimised)
    3. cv2                                (progressive, optimised Huffman, 4:2:0)

Metadata: ExifTool copies EXIF/XMP/IPTC from the source (RAW files work).
GPS is stripped by default, because property coordinates are the address.
The source ICC profile is not copied: pixels are sRGB after export, and an
untagged JPEG is treated as sRGB by browsers. Orientation is reset because the
pixels are already upright. If ExifTool is missing, the JPEG is still written
without metadata and a warning is logged.

CLI:
    python -m processing.datashield in.tif out.jpg --source IMG_0001.CR2 [--keep-gps]
"""

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MAX_LONG_EDGE = 2048
DEFAULT_QUALITY = 85
MIN_QUALITY = 40

# Embedded previews and maker notes can add megabytes to a web JPEG.
EXIF_EXCLUDE = ["ICC_Profile:all", "PreviewImage", "ThumbnailImage", "JpgFromRaw",
                "OtherImage", "MakerNotes:all"]
GPS_TAGS = ["GPS:all", "XMP-exif:GPS*", "XMP:Location*"]


@dataclass
class ExportResult:
    path: Path
    width: int
    height: int
    quality: int
    bytes: int
    encoder: str
    exif_copied: bool


# ------------------------------------------------------------------- colour

def _srgb_encode(linear):
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(np.maximum(linear, 0.0031308), 1 / 2.4) - 0.055)


def _srgb_decode(encoded):
    return np.where(encoded <= 0.04045, encoded / 12.92,
                    np.power((np.maximum(encoded, 0.04045) + 0.055) / 1.055, 2.4))


def _rgb_to_xyz(primaries, white):
    """RGB->XYZ matrix from xy chromaticities of the primaries and white point."""
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = white
    w = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    return xyz * np.linalg.solve(xyz, w)


_D65 = (0.3127, 0.3290)
_SRGB_TO_XYZ = _rgb_to_xyz([(0.64, 0.33), (0.30, 0.60), (0.15, 0.06)], _D65)
_ADOBE_TO_XYZ = _rgb_to_xyz([(0.64, 0.33), (0.21, 0.71), (0.15, 0.06)], _D65)
_ADOBE_TO_SRGB = np.linalg.solve(_SRGB_TO_XYZ, _ADOBE_TO_XYZ)
_ADOBE_GAMMA = 563 / 256


def to_float(image):
    """RGB uint8 / uint16 / float(0-1) -> float32 0-1."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an RGB image (H, W, 3), got shape {image.shape}")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535
    if np.issubdtype(image.dtype, np.floating):
        return image.astype(np.float32, copy=False)
    raise ValueError(f"unsupported dtype {image.dtype}; use uint8, uint16 or float 0-1")


def to_srgb(image, color_space):
    """Float RGB 0-1 in `color_space` -> float sRGB-encoded 0-1."""
    if color_space == "srgb":
        out = image
    elif color_space == "linear":
        out = _srgb_encode(np.clip(image, 0, 1))
    elif color_space == "adobe_rgb":
        linear = np.power(np.clip(image, 0, 1), _ADOBE_GAMMA)
        # Adobe RGB is wider than sRGB; out-of-gamut colours are clipped.
        out = _srgb_encode(np.clip(linear @ _ADOBE_TO_SRGB.T, 0, 1))
    else:
        raise ValueError(f"unknown color_space {color_space!r}; use srgb, linear or adobe_rgb")
    return np.clip(out, 0, 1).astype(np.float32, copy=False)


def resize_long_edge(image, max_long_edge):
    """Shrink so the long edge is at most max_long_edge. Never enlarges."""
    h, w = image.shape[:2]
    scale = max_long_edge / max(h, w)
    if scale >= 1:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ encoding

def _encode_cv2(rgb8, quality):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality,
              cv2.IMWRITE_JPEG_OPTIMIZE, 1,
              cv2.IMWRITE_JPEG_PROGRESSIVE, 1,
              cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise RuntimeError("cv2 failed to encode JPEG")
    return buf.tobytes()


def _encode_cjpeg(cjpeg, rgb8, quality):
    h, w = rgb8.shape[:2]
    ppm = f"P6\n{w} {h}\n255\n".encode() + np.ascontiguousarray(rgb8).tobytes()
    run = subprocess.run([cjpeg, "-quality", str(quality), "-optimize", "-progressive",
                          "-sample", "2x2"], input=ppm, capture_output=True)
    if run.returncode != 0 or not run.stdout:
        raise RuntimeError(f"cjpeg failed: {run.stderr.decode(errors='replace').strip()}")
    return run.stdout


def _make_encoder():
    """Return (name, encode(rgb8, quality) -> bytes) for the best available encoder."""
    cjpeg = shutil.which("cjpeg")
    if cjpeg:
        return "mozjpeg-cjpeg", lambda img, q: _encode_cjpeg(cjpeg, img, q)
    try:
        import mozjpeg_lossless_optimization as mlo
    except ImportError:
        return "cv2", _encode_cv2
    return "cv2+mozjpeg-lossless", lambda img, q: mlo.optimize(_encode_cv2(img, q))


def encode_jpeg(rgb8, quality=DEFAULT_QUALITY, max_bytes=None):
    """Encode RGB uint8 to JPEG bytes. Returns (bytes, quality used, encoder name).

    With max_bytes, binary-search the highest quality in [MIN_QUALITY, quality]
    that fits; if even MIN_QUALITY is too big, return that and log a warning.
    """
    name, encode = _make_encoder()
    data = encode(rgb8, quality)
    if max_bytes is None or len(data) <= max_bytes:
        return data, quality, name

    lo, hi, best = MIN_QUALITY, quality - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = encode(rgb8, mid)
        if len(candidate) <= max_bytes:
            best, lo = (candidate, mid), mid + 1
        else:
            hi = mid - 1
    if best is None:
        log.warning("cannot fit %d bytes even at quality %d; using it anyway",
                    max_bytes, MIN_QUALITY)
        best = (encode(rgb8, MIN_QUALITY), MIN_QUALITY)
    return best[0], best[1], name


# ------------------------------------------------------------------ metadata

_warned_no_exiftool = False


def copy_metadata(source_path, jpeg_path, width, height, keep_gps=False):
    """Copy metadata from source_path into jpeg_path in place. Returns True on success."""
    global _warned_no_exiftool
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        if not _warned_no_exiftool:
            log.warning("ExifTool not found on PATH; exporting without metadata")
            _warned_no_exiftool = True
        return False
    if not Path(source_path).exists():
        log.warning("source %s not found; exporting without metadata", source_path)
        return False

    exclude = EXIF_EXCLUDE + ([] if keep_gps else GPS_TAGS)
    cmd = [exiftool, "-m", "-q", "-q", "-overwrite_original",
           "-TagsFromFile", str(source_path), "-all:all",
           *(f"--{tag}" for tag in exclude),
           "-Orientation#=1", "-ColorSpace#=1",
           f"-ExifImageWidth={width}", f"-ExifImageHeight={height}",
           str(jpeg_path)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0:
        log.warning("ExifTool failed (%s); exporting without metadata", run.stderr.strip())
        return False
    return True


# -------------------------------------------------------------------- export

def export_image(image, source_path, output_path, max_long_edge=DEFAULT_MAX_LONG_EDGE,
                 quality=DEFAULT_QUALITY, max_bytes=None, color_space="srgb",
                 keep_gps=False):
    """Export an RGB image as a web-ready JPEG with metadata from source_path.

    image: RGB array, uint8 / uint16 / float 0-1, in `color_space`
           ("srgb" for rawpy's default output, "linear", or "adobe_rgb").
    source_path: original frame (RAW or JPEG) to copy metadata from.
    """
    if max_long_edge < 1:
        raise ValueError("max_long_edge must be positive")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be 1-100")
    output_path = Path(output_path)

    rgb = to_srgb(resize_long_edge(to_float(image), max_long_edge), color_space)
    rgb8 = (rgb * 255 + 0.5).astype(np.uint8)
    h, w = rgb8.shape[:2]
    data, used_quality, encoder = encode_jpeg(rgb8, quality, max_bytes)

    # Write to a temp file beside the output and rename into place, so a crash
    # never leaves a half-written JPEG under the real name.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".jpg", dir=output_path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        exif_copied = copy_metadata(source_path, tmp, w, h, keep_gps)
        os.replace(tmp, output_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise

    return ExportResult(output_path, w, h, used_quality, output_path.stat().st_size,
                        encoder, exif_copied)


def main():
    ap = argparse.ArgumentParser(description="Web-optimised JPEG export")
    ap.add_argument("input", type=Path, help="image to export (TIFF, PNG, JPEG)")
    ap.add_argument("output", type=Path)
    ap.add_argument("--source", type=Path, help="frame to copy metadata from (default: input)")
    ap.add_argument("--max-long-edge", type=int, default=DEFAULT_MAX_LONG_EDGE)
    ap.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    ap.add_argument("--max-bytes", type=int)
    ap.add_argument("--color-space", choices=["srgb", "linear", "adobe_rgb"], default="srgb")
    ap.add_argument("--keep-gps", action="store_true",
                    help="keep GPS coordinates (stripped by default: they reveal the address)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    bgr = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        ap.error(f"cannot read {args.input}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)

    r = export_image(rgb, args.source or args.input, args.output, args.max_long_edge,
                     args.quality, args.max_bytes, args.color_space, args.keep_gps)
    print(f"{r.path}: {r.width}x{r.height}, q{r.quality}, {r.bytes / 1024:.0f} KB, "
          f"{r.encoder}, exif {'copied' if r.exif_copied else 'not copied'}")


if __name__ == "__main__":
    main()
