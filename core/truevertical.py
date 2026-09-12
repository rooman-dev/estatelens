"""Perspective correction for architectural photos (Rooman's prototype).

A proposal for Hadeed's processing/ module; he may move or rewrite it.

Method: detect near-vertical line segments, find the point they converge to
(the vertical vanishing point), then rotate a virtual camera until that point
is at infinity straight up: H = K R K^-1. Verticals come out exactly parallel
for any focal length guess; the guess only changes how horizontals and
proportions are stretched. Finally crop the largest rectangle with the
original aspect ratio that contains no empty warp border.

    correct_perspective(image) -> image
    correct_perspective_with_diagnostics(image) -> (image, diagnostics)
"""

import math
from itertools import combinations

import cv2
import numpy as np

WORK_SIZE = 1500              # long side (px) for line detection
MIN_LINE_FRAC = 0.05          # shortest line kept, as a fraction of the work image height
NEAR_VERTICAL_DEG = 20.0      # lines within this angle of vertical are candidates
INLIER_DEG = 1.5              # a line "agrees" with a vanishing point within this angle
SNAP_PX = 2.0                 # edge pixels this close to a Hough segment are used to refine it
GAP_PX = 5.0                  # ... as long as they connect to it with gaps no longer than this
REFINE_ITERS = 4
RANSAC_ITERS = 500
MIN_INLIERS = 4
MAX_CORRECTION_DEG = 30.0     # larger corrections are almost always wrong detections
MIN_CORRECTION_DEG = 0.1      # below this, skip the warp to avoid resampling blur
MIN_CROP_FRAC = 0.6           # refuse corrections that keep less of the image than this
CROP_MARGIN_PX = 2            # keep bicubic interpolation away from the empty border
DEFAULT_FOCAL_FRAC = 0.6      # focal length guess, as a fraction of the long side


# --- line detection -------------------------------------------------------------

def _to_gray8(image):
    gray = image
    if gray.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if gray.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(gray, code)  # channel order barely matters for edges
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _refine_segment(seg, edge_pts):
    """Sub-pixel angle: fit a line through all the connected edge pixels along a Hough segment.

    Hough endpoints are whole pixels, and a slightly slanted edge is a pixel
    staircase that Hough splits into short, exactly vertical pieces, so its
    angles are useless below ~0.75 deg. Here each segment grows along its line
    through connected edge pixels (gaps <= GAP_PX) until it spans the whole
    staircase, and a line fitted through those pixels recovers the true angle.
    """
    x1, y1, x2, y2 = seg
    origin = np.array([x1, y1])
    u = np.array([x2 - x1, y2 - y1]) / math.hypot(x2 - x1, y2 - y1)
    t_lo, t_hi = 0.0, math.hypot(x2 - x1, y2 - y1)
    fitted = False
    for _ in range(REFINE_ITERS):
        rel = edge_pts - origin
        across = rel @ np.array([-u[1], u[0]])
        band = np.abs(across) <= SNAP_PX
        pts, t = edge_pts[band], (rel @ u)[band]
        order = np.argsort(t)
        pts, t = pts[order], t[order]
        run = np.concatenate([[0], np.cumsum(np.diff(t) > GAP_PX)])
        keep = np.isin(run, run[(t >= t_lo) & (t <= t_hi)])  # runs overlapping the current extent
        if keep.sum() < 10:
            break
        vx, vy, x0, y0 = cv2.fitLine(pts[keep].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        origin, u = np.array([x0, y0]), np.array([vx, vy])
        t_kept = (pts[keep] - origin) @ u
        t_lo, t_hi = float(t_kept.min()), float(t_kept.max())
        fitted = True
    if not fitted:
        return seg
    return np.concatenate([origin + t_lo * u, origin + t_hi * u])


def detect_vertical_segments(image):
    """Near-vertical line segments as an (N, 4) array of x1, y1, x2, y2, full-resolution pixels."""
    gray = _to_gray8(image)
    scale = min(1.0, WORK_SIZE / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    median = float(np.median(gray))
    edges = cv2.Canny(gray, max(10.0, 0.66 * median), max(30.0, 1.33 * median))
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=40,
                            minLineLength=int(MIN_LINE_FRAC * gray.shape[0]), maxLineGap=5)
    if lines is None:
        return np.empty((0, 4))
    segs = lines.reshape(-1, 4).astype(np.float64)
    segs = segs[_angle_from_vertical(segs) <= NEAR_VERTICAL_DEG]
    ys, xs = np.nonzero(edges)
    edge_pts = np.column_stack([xs, ys]).astype(np.float64)
    segs = np.array([_refine_segment(s, edge_pts) for s in segs]).reshape(-1, 4)
    return segs / scale


def _angle_from_vertical(segs):
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    return np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))


def _lengths(segs):
    return np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    idx = np.searchsorted(cumulative, q * cumulative[-1])
    return float(values[order][min(idx, len(values) - 1)])


# --- vanishing point ----------------------------------------------------------------

def _homogeneous_lines(segs):
    ones = np.ones(len(segs))
    lines = np.cross(np.column_stack([segs[:, 0], segs[:, 1], ones]),
                     np.column_stack([segs[:, 2], segs[:, 3], ones]))
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def _vp_errors(vp, segs):
    """Angle (deg) between each segment and the direction from its midpoint to vp."""
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    seg_dir = segs[:, 2:] - segs[:, :2]
    vp_dir = vp[:2] - mid * vp[2]  # works for a vanishing point at infinity too (vp[2] = 0)
    cos = np.abs(np.sum(seg_dir * vp_dir, axis=1))
    cos /= np.linalg.norm(seg_dir, axis=1) * np.linalg.norm(vp_dir, axis=1) + 1e-12
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def _fit_vp(lines, weights):
    """Least-squares intersection of lines: the vp minimising weighted line-point distance."""
    _, _, vt = np.linalg.svd(lines * np.sqrt(weights)[:, None])
    return vt[-1]


def find_vertical_vp(segs):
    """RANSAC over pairs of lines, then refine on the inliers. Coordinates must be normalised."""
    lines = _homogeneous_lines(segs)
    lengths = _lengths(segs)
    rng = np.random.default_rng(0)  # deterministic: same image, same answer
    best_score, best_inliers = 0.0, None
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(segs), 2, replace=False, p=lengths / lengths.sum())
        vp = np.cross(lines[i], lines[j])
        if np.linalg.norm(vp) < 1e-12:
            continue
        inliers = _vp_errors(vp / np.linalg.norm(vp), segs) < INLIER_DEG
        score = lengths[inliers].sum()
        if score > best_score:
            best_score, best_inliers = score, inliers

    if best_inliers is None or best_inliers.sum() < 2:
        return None, np.zeros(len(segs), bool)
    for _ in range(2):  # refit, re-select inliers, refit
        vp = _fit_vp(lines[best_inliers], lengths[best_inliers])
        best_inliers = _vp_errors(vp, segs) < INLIER_DEG
        if best_inliers.sum() < 2:
            return None, best_inliers
    return _fit_vp(lines[best_inliers], lengths[best_inliers]), best_inliers


def _normalise(segs, w, h):
    """Centre on the principal point (assumed to be the image centre) and scale to ~[-1, 1]
    so the least-squares fit is well conditioned."""
    cx, cy, norm = (w - 1) / 2, (h - 1) / 2, max(h, w) / 2
    return (segs - [cx, cy, cx, cy]) / norm, cx, cy, norm


def measure_verticality(image):
    """Deviation from vertical of the lines in `image` that agree on a common direction.

    Measured by re-detecting lines, independent of how the image was produced, so it
    is an honest benchmark. Only lines agreeing on one vanishing point count, so
    leaning objects (a ladder, a bannister) don't pollute it.
    Returns (length-weighted median deg, 90th percentile deg, line count), or
    (None, None, count) if there are too few lines to judge.
    """
    h, w = image.shape[:2]
    segs = detect_vertical_segments(image)
    if len(segs) < MIN_INLIERS:
        return None, None, len(segs)
    _, inliers = find_vertical_vp(_normalise(segs, w, h)[0])
    if inliers.sum() < MIN_INLIERS:
        return None, None, int(inliers.sum())
    angles, weights = _angle_from_vertical(segs[inliers]), _lengths(segs[inliers])
    return (_weighted_quantile(angles, weights, 0.5),
            _weighted_quantile(angles, weights, 0.9), int(inliers.sum()))


# --- homography and crop ------------------------------------------------------------

def _rotation_to_vertical(vp_centered, focal_px):
    """Smallest camera rotation that points the vanishing direction straight down the y axis."""
    direction = np.array([vp_centered[0] / focal_px, vp_centered[1] / focal_px, vp_centered[2]])
    direction /= np.linalg.norm(direction)
    if direction[1] < 0:  # vp and -vp are the same point; pick the one needing less rotation
        direction = -direction
    target = np.array([0.0, 1.0, 0.0])
    axis = np.cross(direction, target)
    angle = math.atan2(np.linalg.norm(axis), float(direction @ target))
    if np.linalg.norm(axis) < 1e-12:
        return np.eye(3), 0.0
    rotation, _ = cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)
    return rotation, math.degrees(angle)


def _largest_rect(quad, aspect):
    """Largest axis-aligned rectangle of the given aspect (w/h) inside a convex quad.

    Each quad edge is a half-plane n.p <= c. A rectangle with centre (cx, cy) and
    half-width a has all corners inside iff n.(cx, cy) + a(|nx| + |ny|/aspect) <= c,
    which is linear in (cx, cy, a). So this is a 3-variable linear programme whose
    optimum lies where 3 constraints meet: try all 4 triples, keep the best feasible one.
    """
    x, y = quad[:, 0], quad[:, 1]
    sign = 1.0 if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0 else -1.0
    rows, bounds = [], []
    for k in range(4):
        (px, py), (qx, qy) = quad[k], quad[(k + 1) % 4]
        ex, ey = qx - px, qy - py
        nx, ny = sign * ey, -sign * ex
        rows.append([nx, ny, abs(nx) + abs(ny) / aspect])
        bounds.append(sign * (ey * px - ex * py))
    rows, bounds = np.array(rows), np.array(bounds)

    best = None
    for triple in combinations(range(4), 3):
        idx = list(triple)
        try:
            cx, cy, a = np.linalg.solve(rows[idx], bounds[idx])
        except np.linalg.LinAlgError:
            continue
        feasible = np.all(rows @ [cx, cy, a] <= bounds + 1e-6 * (1 + np.abs(bounds)))
        if a > 0 and feasible and (best is None or a > best[2]):
            best = (cx, cy, a)
    return best


def _translation(tx, ty):
    return np.array([[1.0, 0, tx], [0, 1.0, ty], [0, 0, 1.0]])


def estimate_correction(image, focal_px=None):
    """Work out the correction without applying it. See module docstring for the method."""
    h, w = image.shape[:2]
    focal_px = float(focal_px or DEFAULT_FOCAL_FRAC * max(h, w))
    diag = {"applied": False, "reason": None, "focal_px": focal_px, "candidate_lines": 0,
            "inlier_lines": 0, "correction_deg": None, "homography": None, "output_size": None,
            "crop_area_frac": None, "fit_residual_deg": None}

    segs = detect_vertical_segments(image)
    diag["candidate_lines"] = len(segs)
    if len(segs) < MIN_INLIERS:
        diag["reason"] = f"too few near-vertical lines ({len(segs)})"
        return diag

    segs_norm, cx, cy, norm = _normalise(segs, w, h)
    vp, inliers = find_vertical_vp(segs_norm)
    diag["inlier_lines"] = int(inliers.sum())
    if vp is None or inliers.sum() < MIN_INLIERS:
        diag["reason"] = f"too few lines agree on a vanishing point ({int(inliers.sum())})"
        return diag

    vp_centered = np.array([vp[0] * norm, vp[1] * norm, vp[2]])
    rotation, correction_deg = _rotation_to_vertical(vp_centered, focal_px)
    diag["correction_deg"] = correction_deg
    if correction_deg > MAX_CORRECTION_DEG:
        diag["reason"] = f"correction too large ({correction_deg:.1f} deg); likely wrong lines"
        return diag
    if correction_deg < MIN_CORRECTION_DEG:
        diag["reason"] = "already vertical"
        return diag

    k = np.diag([focal_px, focal_px, 1.0])
    h_full = _translation(cx, cy) @ k @ rotation @ np.linalg.inv(k) @ _translation(-cx, -cy)

    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]], float).T
    warped = h_full @ corners
    if np.any(warped[2] <= 0):
        diag["reason"] = "warp sends image corners behind the camera"
        return diag
    quad = (warped[:2] / warped[2]).T

    rect = _largest_rect(quad, w / h)
    if rect is None:
        diag["reason"] = "no valid crop"
        return diag
    rcx, rcy, half_w = rect
    out_w = int(2 * half_w) - 2 * CROP_MARGIN_PX
    out_h = int(out_w * h / w)
    crop_frac = out_w * out_h / (w * h)
    diag["crop_area_frac"] = crop_frac
    if crop_frac < MIN_CROP_FRAC:
        diag["reason"] = f"crop would keep only {crop_frac:.0%} of the image"
        return diag

    homography = _translation(-(rcx - out_w / 2), -(rcy - out_h / 2)) @ h_full
    inlier_segs = segs[inliers]
    ends = np.vstack([np.column_stack([inlier_segs[:, 0:2], np.ones(len(inlier_segs))]),
                      np.column_stack([inlier_segs[:, 2:4], np.ones(len(inlier_segs))])]).T
    mapped = homography @ ends
    mapped = (mapped[:2] / mapped[2]).T
    n = len(inlier_segs)
    mapped_segs = np.hstack([mapped[:n], mapped[n:]])
    diag["fit_residual_deg"] = _weighted_quantile(_angle_from_vertical(mapped_segs), _lengths(mapped_segs), 0.5)

    diag.update(applied=True, homography=homography, output_size=(out_w, out_h))
    return diag


def correct_perspective_with_diagnostics(image, focal_px=None):
    """Correct the image and measure the result.

    Returns (image, diagnostics). If the correction is not trustworthy, the input is
    returned unchanged and diagnostics["reason"] says why. residual_deg is measured by
    re-detecting lines on the returned image (the proposal benchmark: <= 0.5 deg).
    """
    diag = estimate_correction(image, focal_px)
    out = image
    if diag["applied"]:
        out = cv2.warpPerspective(image, diag["homography"], diag["output_size"],
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    diag["residual_deg"], diag["residual_p90_deg"], diag["residual_lines"] = measure_verticality(out)
    return out, diag


def correct_perspective(image, focal_px=None):
    """Make near-vertical architectural lines truly vertical, then crop the empty edges."""
    return correct_perspective_with_diagnostics(image, focal_px)[0]
