"""Call AeroSwap out of process and draw a review overlay on a fused photo.

The runner may be native Python or a WSL command. A local, ignored
`data/sky_runtime.json` file can provide an argv list for a machine where
PyTorch cannot load in the desktop app's Python environment.
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SIZE = 512


def sky_runner(root=PROJECT_ROOT):
    config = Path(root) / "data" / "sky_runtime.json"
    if config.exists():
        command = json.loads(config.read_text(encoding="utf-8"))
        if not isinstance(command, list) or not command or not all(
                isinstance(part, str) and part for part in command):
            raise ValueError(f"{config} must contain a nonempty JSON list of command parts")
        return command
    return [sys.executable, "-m", "models.infer_sky"]


def predict_sky_probability(rgb, root=PROJECT_ROOT, timeout=120):
    """Return a 512x512 uint8 sky probability mask from a detached model process."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an RGB uint8 image")
    small = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError("could not encode image for AeroSwap")
    completed = subprocess.run(
        sky_runner(root), input=encoded.tobytes(), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=root, timeout=timeout, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"AeroSwap inference failed: {details[-800:]}")
    mask = cv2.imdecode(np.frombuffer(completed.stdout, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (MODEL_SIZE, MODEL_SIZE):
        raise ValueError("AeroSwap returned an invalid mask image")
    return mask


def mask_overlay(rgb, probability, cutoff=0.5):
    """Highlight predicted sky in blue and draw its boundary in yellow."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an RGB uint8 image")
    if probability.shape != (MODEL_SIZE, MODEL_SIZE) or probability.dtype != np.uint8:
        raise ValueError("expected a 512x512 uint8 probability mask")
    height, width = rgb.shape[:2]
    enlarged = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    selected = enlarged > round(cutoff * 255)
    overlay = rgb.copy()
    overlay[selected] = (0.6 * rgb[selected] + 0.4 * np.array([35, 170, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 230, 0), 2)
    return overlay, float(selected.mean())
