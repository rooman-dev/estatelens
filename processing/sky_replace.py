"""Create an optional sky-version image after a human reviews the mask.

This module does not decide whether a photo is exterior or whether a mask is
good. The UI gates saving on SceneSense and shows the replacement preview to
the user first. The original fused JPEG is never overwritten.
"""

import os
from pathlib import Path

import cv2
import numpy as np


MASK_SIZE = (512, 512)


def fill_frame(sky_rgb, width, height):
    """Center-crop a sky image to the target aspect ratio, then resize."""
    if sky_rgb.ndim != 3 or sky_rgb.shape[2] != 3 or sky_rgb.dtype != np.uint8:
        raise ValueError("expected a sky RGB uint8 image")
    if width < 1 or height < 1:
        raise ValueError("output dimensions must be positive")
    source_h, source_w = sky_rgb.shape[:2]
    if source_w / source_h > width / height:
        crop_w = max(1, round(source_h * width / height))
        left = (source_w - crop_w) // 2
        cropped = sky_rgb[:, left:left + crop_w]
    else:
        crop_h = max(1, round(source_w * height / width))
        top = (source_h - crop_h) // 2
        cropped = sky_rgb[top:top + crop_h]
    interpolation = cv2.INTER_AREA if cropped.shape[0] > height else cv2.INTER_LINEAR
    return cv2.resize(cropped, (width, height), interpolation=interpolation)


def composite_sky(fused_rgb, sky_rgb, probability, strip_height=512):
    """Composite in strips so large property photos do not need float32 copies."""
    if fused_rgb.ndim != 3 or fused_rgb.shape[2] != 3 or fused_rgb.dtype != np.uint8:
        raise ValueError("expected a fused RGB uint8 image")
    if probability.shape != MASK_SIZE or probability.dtype != np.uint8:
        raise ValueError("expected a 512x512 uint8 sky probability mask")
    if strip_height < 1:
        raise ValueError("strip height must be positive")
    height, width = fused_rgb.shape[:2]
    sky = fill_frame(sky_rgb, width, height)
    resized = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    hard_mask = (resized > 127).astype(np.uint8) * 255
    # A small soft edge avoids a hard one-pixel cut, while keeping the shown
    # mask's 0.5 threshold as the region that will be replaced.
    feather = max(1.0, min(height, width) / 500.0)
    alpha_u8 = cv2.GaussianBlur(hard_mask, (0, 0), sigmaX=feather)
    output = fused_rgb.copy()
    for y in range(0, height, strip_height):
        end = min(y + strip_height, height)
        alpha = alpha_u8[y:end].astype(np.float32)[:, :, None] / 255.0
        foreground = fused_rgb[y:end].astype(np.float32)
        replacement = sky[y:end].astype(np.float32)
        output[y:end] = np.rint(foreground * (1.0 - alpha) + replacement * alpha).astype(np.uint8)
    return output


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_sky_version(fused_path, sky_path, probability, output_path):
    """Write a separate image atomically, refusing the fused photo's path."""
    fused_path, sky_path, output_path = map(Path, (fused_path, sky_path, output_path))
    if output_path.resolve() == fused_path.resolve():
        raise ValueError("sky version cannot overwrite the original fused photo")
    if output_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("sky version must be a JPEG or PNG")
    fused = read_rgb(fused_path)
    sky = read_rgb(sky_path)
    composed = composite_sky(fused, sky, probability)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    options = [cv2.IMWRITE_JPEG_QUALITY, 95] if output_path.suffix.lower() in (".jpg", ".jpeg") else []
    try:
        if not cv2.imwrite(str(temporary), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR), options):
            raise OSError(f"could not write sky version: {temporary}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path
