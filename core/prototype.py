"""Throwaway end-to-end prototype: RAW brackets -> Mertens fusion -> JPEGs.

Replaced by processing/chromaraw.py and processing/lumamerge.py once Hadeed
writes them. Do not build on this file.

Usage:
    python -m core.prototype <input_dir> <output_dir> [--half-size] [--no-align] [--dry-run]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import exifread
import numpy as np
import rawpy

RAW_EXTS = {".cr2", ".nef", ".arw", ".dng"}
TIFF_EXTS = {".tif", ".tiff"}  # synthetic test brackets; rawpy cannot read these
BRACKET_SIZE = 3
GAP_SECONDS = 3.0
JPEG_QUALITY = 95


def read_exif(path):
    """Return (timestamp, exposure_time, f_number, iso); missing values are None."""
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    def number(key):
        tag = tags.get(key)
        return float(tag.values[0]) if tag else None

    stamp = tags.get("EXIF DateTimeOriginal")
    try:
        timestamp = datetime.strptime(str(stamp), "%Y:%m:%d %H:%M:%S") if stamp else None
    except ValueError:
        timestamp = None
    return timestamp, number("EXIF ExposureTime"), number("EXIF FNumber"), number("EXIF ISOSpeedRatings")


def describe_exposure(frame):
    if frame["exposure"] is None:
        return "?"
    t = frame["exposure"]
    return f"1/{round(1 / t)}" if t < 1 else f"{t:g}s"


def brightness(frame):
    """Relative exposure: shutter * ISO / aperture^2. None if any value is missing."""
    if None in (frame["exposure"], frame["f_number"], frame["iso"]):
        return None
    return frame["exposure"] * frame["iso"] / frame["f_number"] ** 2


def find_frames(folder):
    frames = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in RAW_EXTS | TIFF_EXTS:
            timestamp, exposure, f_number, iso = read_exif(path)
            frames.append({"path": path, "timestamp": timestamp,
                           "exposure": exposure, "f_number": f_number, "iso": iso})
    return frames


def chunk(cluster, warnings):
    """Split a cluster into full brackets; warn about and drop the leftovers."""
    full = len(cluster) - len(cluster) % BRACKET_SIZE
    leftover = cluster[full:]
    if leftover:
        names = ", ".join(f["path"].name for f in leftover)
        warnings.append(f"incomplete bracket skipped ({len(leftover)} of {BRACKET_SIZE}): {names}")
    return [cluster[i:i + BRACKET_SIZE] for i in range(0, full, BRACKET_SIZE)]


def group_brackets(frames):
    """Group by capture time; fall back to filename order if any timestamp is missing."""
    warnings = []
    if any(f["timestamp"] is None for f in frames):
        warnings.append("some files have no EXIF timestamp; grouping by filename order")
        return chunk(sorted(frames, key=lambda f: f["path"].name), warnings), warnings

    frames = sorted(frames, key=lambda f: (f["timestamp"], f["path"].name))
    clusters = [[frames[0]]]
    for prev, cur in zip(frames, frames[1:]):
        # Timestamps mark the start of an exposure, so measure from the end of the previous one.
        prev_end = prev["timestamp"].timestamp() + (prev["exposure"] or 0)
        if cur["timestamp"].timestamp() - prev_end <= GAP_SECONDS:
            clusters[-1].append(cur)
        else:
            clusters.append([cur])

    brackets = []
    for cluster in clusters:
        brackets.extend(chunk(cluster, warnings))
    return brackets, warnings


def load_image(path, half_size):
    """Return an RGB float32 image scaled to 0-255, the range MergeMertens expects."""
    if path.suffix.lower() in TIFF_EXTS:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("OpenCV could not read the file")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if half_size:
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    else:
        with rawpy.imread(str(path)) as raw:
            # no_auto_bright keeps the exposure differences between frames;
            # camera white balance keeps colour consistent across the bracket.
            img = raw.postprocess(no_auto_bright=True, use_camera_wb=True,
                                  output_bps=16, half_size=half_size)
    scale = 255.0 / np.iinfo(img.dtype).max
    return img.astype(np.float32) * scale


def align(images):
    """Shift every frame onto the middle exposure using median threshold bitmaps."""
    mtb = cv2.createAlignMTB()
    gray = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY) for img in images]
    ref = len(images) // 2
    return [img if i == ref else mtb.shiftMat(img, mtb.calculateShift(gray[ref], gray[i]))
            for i, img in enumerate(images)]


def fuse_bracket(bracket, out_path, half_size, do_align):
    images = [load_image(f["path"], half_size) for f in bracket]
    if len({img.shape for img in images}) != 1:
        raise ValueError("frames have different sizes")
    if do_align:
        images = align(images)
    fused = cv2.createMergeMertens().process(images)
    out = np.clip(fused * 255, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"could not write {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fuse exposure brackets into JPEGs (prototype).")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--half-size", action="store_true", help="decode at half resolution (faster, ~4x less memory)")
    parser.add_argument("--no-align", action="store_true", help="skip alignment (tripod shots)")
    parser.add_argument("--dry-run", action="store_true", help="print the groups without fusing")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"error: {args.input_dir} is not a folder")

    frames = find_frames(args.input_dir)
    if not frames:
        sys.exit(f"error: no RAW or TIFF files in {args.input_dir}")

    brackets, warnings = group_brackets(frames)
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"{len(frames)} files -> {len(brackets)} brackets")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    for i, bracket in enumerate(brackets, 1):
        # Order dark to bright when EXIF allows it; otherwise keep capture/filename order.
        if all(brightness(f) is not None for f in bracket):
            bracket = sorted(bracket, key=brightness)
            if len({brightness(f) for f in bracket}) == 1:
                print(f"WARNING: bracket {i} has identical exposures; may not be a real bracket")

        out_path = args.output_dir / f"{bracket[0]['path'].stem}_fused.jpg"
        names = " ".join(f["path"].name for f in bracket)
        exposures = " ".join(describe_exposure(f) for f in bracket)
        prefix = f"[{i}/{len(brackets)}] {names} | {exposures}"

        if args.dry_run:
            print(prefix)
            continue

        start = time.perf_counter()
        try:
            fuse_bracket(bracket, out_path, args.half_size, not args.no_align)
        except Exception as e:  # prototype: report and keep going
            print(f"{prefix} -> FAILED: {e}")
            failed += 1
            continue
        print(f"{prefix} -> {out_path.name} ({time.perf_counter() - start:.1f}s)")
        done += 1

    if not args.dry_run:
        print(f"done: {done} fused, {failed} failed, output in {args.output_dir}")


if __name__ == "__main__":
    main()
