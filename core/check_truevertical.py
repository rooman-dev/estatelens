"""Reproducible check for core/truevertical.py, with numbers for the report.

Renders a synthetic facade, photographs it with a known tilted camera, corrects
it at several focal length guesses, and measures verticality three ways:
  ground truth - the known facade verticals, pushed through both homographies (exact)
  re-detected  - diagnostics["residual_deg"]: lines re-detected on the output
  fit          - diagnostics["fit_residual_deg"]: the lines used to fit, after warping

Usage:
    python -m core.check_truevertical [--save-dir data/truevertical_check]
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from core.truevertical import correct_perspective_with_diagnostics, measure_verticality

PHOTO_W, PHOTO_H = 3000, 2000
TRUE_FOCAL = 0.5 * PHOTO_W             # deliberately different from every guess tested
FOCAL_GUESSES = (0.4, 0.6, 0.9)        # x long side
BENCHMARK_DEG = 0.5


WINDOW_W, WINDOW_H = 400, 650


def make_facade():
    """Frontal facade (as a level camera would see it), its true vertical segments, and window rects."""
    fw, fh = 5200, 4000
    rng = np.random.default_rng(1)
    texture = cv2.GaussianBlur(rng.normal(0, 6, (fh, fw)).astype(np.float32), (0, 0), 3)
    img = np.clip(np.dstack([198 + texture, 204 + texture, 210 + texture]), 0, 255).astype(np.uint8)
    verticals, windows = [], []

    def rect(x0, y0, x1, y1, color, frame=None, thickness=10):
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)
        if frame is not None:
            cv2.rectangle(img, (x0, y0), (x1, y1), frame, thickness)
        verticals.extend([(x0, y0, x0, y1), (x1, y0, x1, y1)])

    for x in (700, 4500):                                  # pilasters / building edges
        rect(x - 60, 300, x + 60, 3700, (150, 150, 158))
    for floor_y in (1300, 2400):                           # floor slabs
        rect(700, floor_y - 25, 4500, floor_y + 25, (160, 165, 170))
    for row, y0 in enumerate((450, 1500, 2600)):           # window grid
        for x0 in range(1000, 4300, 650):
            if row == 2 and 2300 <= x0 <= 2700:
                rect(2350, 2600, 2850, 3700, (70, 50, 40), (30, 30, 30))   # door
                continue
            rect(x0, y0, x0 + WINDOW_W, y0 + WINDOW_H, (120, 110, 95), (40, 40, 45))
            cv2.line(img, (x0 + 200, y0), (x0 + 200, y0 + WINDOW_H), (40, 40, 45), 8)
            verticals.append((x0 + 200, y0, x0 + 200, y0 + WINDOW_H))
            windows.append((x0, y0, x0 + WINDOW_W, y0 + WINDOW_H))
    # Distractors: a leaning ladder (~12 deg from vertical, inside the 20 deg candidate
    # window, so RANSAC must reject it) and a diagonal stair rail.
    cv2.line(img, (3050, 3700), (3280, 2620), (60, 60, 60), 14)
    cv2.line(img, (3200, 3700), (3430, 2620), (60, 60, 60), 14)
    cv2.line(img, (1150, 3700), (2150, 3000), (70, 70, 70), 12)
    return img, np.array(verticals, float), np.array(windows, float)


def rotation(pitch_deg, roll_deg):
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx


def photograph(facade, pitch_deg, roll_deg):
    """Homography facade -> photo for a camera tilted by pitch/roll about the facade centre."""
    fh, fw = facade.shape[:2]
    k = np.diag([TRUE_FOCAL, TRUE_FOCAL, 1.0])
    to_center = np.array([[1, 0, -fw / 2], [0, 1, -fh / 2], [0, 0, 1.0]])
    from_center = np.array([[1, 0, PHOTO_W / 2], [0, 1, PHOTO_H / 2], [0, 0, 1.0]])
    h = from_center @ k @ rotation(pitch_deg, roll_deg) @ np.linalg.inv(k) @ to_center

    corners = np.array([[0, 0, 1], [PHOTO_W, 0, 1], [PHOTO_W, PHOTO_H, 1], [0, PHOTO_H, 1]], float).T
    back = np.linalg.inv(h) @ corners
    back = back[:2] / back[2]
    assert back.min() >= 0 and back[0].max() <= fw and back[1].max() <= fh, "photo sees past the facade"
    return cv2.warpPerspective(facade, h, (PHOTO_W, PHOTO_H), flags=cv2.INTER_CUBIC), h


def map_segments(segs, h):
    ends = np.vstack([np.column_stack([segs[:, :2], np.ones(len(segs))]),
                      np.column_stack([segs[:, 2:], np.ones(len(segs))])]).T
    m = h @ ends
    m = (m[:2] / m[2]).T
    return np.hstack([m[:len(segs)], m[len(segs):]])


def visible(segs, w, h):
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    return (mid[:, 0] >= 0) & (mid[:, 0] < w) & (mid[:, 1] >= 0) & (mid[:, 1] < h)


def angles(segs, from_axis):
    dx, dy = np.abs(segs[:, 2] - segs[:, 0]), np.abs(segs[:, 3] - segs[:, 1])
    return np.degrees(np.arctan2(dx, dy) if from_axis == "vertical" else np.arctan2(dy, dx))


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def window_aspect_error(windows, h, w_out, h_out):
    """Median % error of window height/width after mapping, vs the true 650/400."""
    errors = []
    for x0, y0, x1, y1 in windows:
        c = map_segments(np.array([[x0, y0, x1, y0], [x0, y1, x1, y1]]), h)  # top and bottom edges
        (tl, tr), (bl, br) = c[0].reshape(2, 2), c[1].reshape(2, 2)
        if not all(0 <= p[0] < w_out and 0 <= p[1] < h_out for p in (tl, tr, bl, br)):
            continue
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        errors.append((height / width) / (WINDOW_H / WINDOW_W) - 1)
    return 100 * float(np.median(errors))


def run_tilt_case(facade, verticals, windows, pitch, roll, guesses, save_dir):
    photo, h_photo = photograph(facade, pitch, roll)
    vis = visible(map_segments(verticals, h_photo), PHOTO_W, PHOTO_H)
    before = angles(map_segments(verticals, h_photo)[vis], "vertical")
    print(f"\nTilt: pitch {pitch} deg, roll {roll} deg, true focal {TRUE_FOCAL / PHOTO_W:.1f} x width")
    print(f"  before correction: true verticals deviate median {np.median(before):.2f}, max {before.max():.2f} deg; "
          f"window aspect error {window_aspect_error(windows, h_photo, PHOTO_W, PHOTO_H):+.1f}%")
    print(f"  {'focal':>6} | {'applied':>7} {'correct':>7} {'crop':>5} | {'GT med':>6} {'GT max':>6} | "
          f"{'redet':>5} {'p90':>5} | {'fit':>5} | {'window aspect':>13} | benchmark")
    rows = []
    for frac in guesses:
        out, d = correct_perspective_with_diagnostics(photo, frac * PHOTO_W)
        total = d["homography"] @ h_photo if d["applied"] else h_photo
        oh, ow = out.shape[:2]
        v = map_segments(verticals, total); v = v[visible(v, ow, oh)]
        gt = angles(v, "vertical")
        aspect = window_aspect_error(windows, total, ow, oh)
        ok = d["residual_deg"] is not None and d["residual_deg"] <= BENCHMARK_DEG and gt.max() <= BENCHMARK_DEG
        print(f"  {frac:>6.1f} | {str(d['applied']):>7} {fmt(d['correction_deg'], 1):>7} "
              f"{fmt(d['crop_area_frac'] and d['crop_area_frac'] * 100, 0):>4}% | "
              f"{np.median(gt):>6.3f} {gt.max():>6.3f} | {fmt(d['residual_deg']):>5} {fmt(d['residual_p90_deg']):>5} | "
              f"{fmt(d['fit_residual_deg'], 3):>5} | {aspect:>+12.1f}% | {'PASS' if ok else 'FAIL'}")
        rows.append((frac, d, gt, out))
        if save_dir:
            cv2.imwrite(str(save_dir / f"pitch{pitch}_roll{roll}_f{frac}.jpg"), out)
    if save_dir:
        cv2.imwrite(str(save_dir / f"pitch{pitch}_roll{roll}_input.jpg"), photo)
    return photo, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--save-dir", type=Path, help="write input and corrected images here")
    args = parser.parse_args()
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    facade, verticals, windows = make_facade()
    photo, _ = run_tilt_case(facade, verticals, windows, 12, 3, FOCAL_GUESSES, args.save_dir)
    run_tilt_case(facade, verticals, windows, -8, -2, (0.6,), args.save_dir)

    print("\nMetric resolution: a level view rotated by a known angle; residual_deg should match it")
    level = facade[500:2500, 1100:4100]
    for deg in (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0):
        m = cv2.getRotationMatrix2D((PHOTO_W / 2, PHOTO_H / 2), deg, 1.0)
        img = cv2.warpAffine(level, m, (PHOTO_W, PHOTO_H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        med, p90, n = measure_verticality(img)
        print(f"  true {deg:4.2f} deg -> measured median {fmt(med)}, p90 {fmt(p90)} ({n} lines)")

    print("\nOther inputs:")
    cases = {
        "blank grey": np.full((PHOTO_H, PHOTO_W, 3), 128, np.uint8),
        "random noise": np.random.default_rng(2).integers(0, 256, (PHOTO_H, PHOTO_W, 3), dtype=np.uint8),
        "16-bit facade": photo.astype(np.uint16) * 257,
        "grayscale facade": cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY),
    }
    for name, img in cases.items():
        out, d = correct_perspective_with_diagnostics(img)
        print(f"  {name:<17} applied={d['applied']!s:<5} dtype {img.dtype}->{out.dtype} "
              f"shape {img.shape}->{out.shape}  reason={d['reason']}  residual={fmt(d['residual_deg'])}")


if __name__ == "__main__":
    main()
