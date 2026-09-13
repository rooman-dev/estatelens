"""Scene classification from classical image features (no neural network).

    is_exterior(image)    -> (bool, confidence)   gates sky replacement
    classify_scene(image) -> (label, confidence)  kitchen / bathroom / bedroom-living / exterior / other

`image` is a fused BGR image: uint8, uint16, or float in [0, 1] (display-referred,
as Mertens fusion produces). Colour, edge and brightness statistics are measured
on a copy shrunk to ANALYSIS_LONG_EDGE.

is_exterior never depends on room typing. It weighs two kinds of positive evidence:
  exterior: a sky found by structure, not hue. That means a large smooth region
            spanning down from the top edge, with few fine edges, a smooth
            brightness gradient, and brighter than what lies below it. Colour and
            vegetation are weak support.
  interior: a smooth top region darker than what's below (a ceiling), and
            rectilinear lines.
A confident answer needs one side strong and clearly ahead. "No evidence either
way" (e.g. a sky-less canyon) gives (False, AMBIGUOUS_CONFIDENCE): unsure,
failing safe, since a missed sky replacement is cheaper than a sky pasted onto a
ceiling. A confidence >= 0.5 means a clear decision. Logged label: exterior /
interior / unsure.

classify_scene calls the same exterior scorer first, and types rooms only for
confident interiors. When scores are weak or too close, it returns
("other", OTHER_CONFIDENCE) rather than guessing.

Every threshold below is an untuned first guess. Confidences are heuristic
scores, not calibrated probabilities.

Logging: each public call writes one row to `scene_classifications` in the
SQLite database (features, scores, label, threshold snapshot). Pass db_path=None
to skip. `true_label` is left NULL to be filled in by hand; exterior_score() and
room_scores() take a SceneFeatures, so logged rows can be re-scored with new
thresholds without reloading images.

CLI:
    python -m processing.scenesense fused.tif [--exterior-only] [--verbose] [--db data/estatelens.db | --no-log]
"""

import argparse
import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DB = Path("data/estatelens.db")
ANALYSIS_LONG_EDGE = 512

# HSV bands, OpenCV scale (H 0-179, S and V 0-255).
COLOURED_MIN_S = 40
COLOURED_MIN_V = 40
BLUE_H = (95, 130)
GREEN_H = (35, 85)
WARM_H = (5, 25)
NEUTRAL_MAX_S = 30
NEUTRAL_MIN_V = 60
WHITE_MIN_V = 200

LINE_MIN_LENGTH_FRACTION = 0.06   # of the long edge
AXIS_TOLERANCE_DEG = 10           # meaningful once perspective is corrected

# Sky candidate: structure, not hue
SKY_BLUR_SIGMA = 1.5
SKY_MAX_GRADIENT = 5.0            # per-pixel luminance change (0-255 scale) after blur
SKY_CLOSE_FRACTION = 0.01         # closing kernel on the seed mask, fraction of long edge (5 px at 512). Pinholes
                                  # only: at >= 0.03 closing merged ground/wall patches and flooded every image
# Region growing from the seeds: cloud variation is absorbed, a brightness discontinuity stops growth.
SKY_STEP_TOL = 4                  # max brightness step between neighbouring pixels (0-255, blurred). At 6 growth
                                  # leaked through hazy ridges and down rock walls (waterfall -> confident interior)
SKY_RANGE_TOL = 0.10              # grown pixels must stay within the seed's [p5, p95] luminance +/- this,
                                  # so growth can't chain down a slow gradient into a wall
SKY_MIN_COMPONENT_FRAC = 0.01     # of image area
SKY_FINE_CANNY = (30, 90)         # on the unblurred image, to see texture the blur hid

# Exterior decision: two-sided evidence
EXTERIOR_MIN_EVIDENCE = 0.55      # E >= this to call exterior
INTERIOR_MIN_EVIDENCE = 0.50      # I >= this to call interior
DECISION_MIN_MARGIN = 0.20        # winner must beat the other side by this much
AMBIGUOUS_CONFIDENCE = 0.2

# Room typing
ROOM_MIN_SCORE = 0.55
ROOM_MIN_MARGIN = 0.08
ROOM_SOFTMAX_TEMPERATURE = 0.1
OTHER_CONFIDENCE = 0.2

SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_classifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    source      TEXT,              -- image path, if the caller gave one
    call        TEXT NOT NULL,     -- 'is_exterior' or 'classify_scene'
    label       TEXT NOT NULL,     -- 'exterior' / 'interior' for is_exterior
    confidence  REAL NOT NULL,
    features    TEXT NOT NULL,     -- JSON of SceneFeatures
    scores      TEXT NOT NULL,     -- JSON
    thresholds  TEXT NOT NULL,     -- JSON snapshot of the constants used
    true_label  TEXT               -- filled in by hand for tuning
);
"""


@dataclass
class SceneFeatures:
    # brightness (luminance in [0, 1])
    lum_mean: float
    lum_std: float
    lum_p5: float
    lum_p50: float
    lum_p95: float
    clipped_frac: float
    dark_frac: float
    top_bottom_ratio: float     # mean luminance, top third / bottom third
    # colour
    sat_mean: float
    sat_std: float
    blue_frac: float
    green_frac: float
    warm_frac: float
    neutral_frac: float
    white_frac: float
    green_bottom_frac: float    # green pixels in the bottom half
    # top smooth region (sky or ceiling candidate); see _top_smooth_region
    sky_area_frac: float        # region area / image area
    sky_top_coverage: float     # share of columns where the region starts at the top edge
    sky_edge_density: float     # fine-scale edges inside the region
    sky_residual: float         # luminance std around a fitted quadratic surface (0-1 scale)
    sky_below_ratio: float      # region mean luminance / mean luminance directly below it
    sky_colour_support: float   # share of region pixels with a plausible sky colour (blue, neutral, sunset)
    # edges
    edge_density: float
    line_length_norm: float     # total Hough line length / (width + height)
    axis_line_frac: float       # share of line length within AXIS_TOLERANCE_DEG of horizontal/vertical
    fine_texture: float         # mean |Laplacian| / 255


def _ramp(x, lo, hi):
    """0 at or below lo, 1 at or above hi, linear between."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _prepare(image):
    """Any supported input -> uint8 BGR, long edge at most ANALYSIS_LONG_EDGE."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.uint8:
        img = image
    elif image.dtype == np.uint16:
        img = (image // 257).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.floating):
        img = (np.clip(np.nan_to_num(image), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        raise ValueError(f"unsupported dtype {image.dtype}")

    h, w = img.shape[:2]
    scale = ANALYSIS_LONG_EDGE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _in_band(hue, band):
    return (hue >= band[0]) & (hue <= band[1])


def _grow_region(blurred, seed):
    """Grow `seed` into neighbours whose brightness steps by at most SKY_STEP_TOL.

    floodFill in floating-range mode compares each pixel with the neighbour it was
    reached from, so soft cloud variation is absorbed and a sharp step (roofline,
    horizon) stops growth. Pixels outside the seed's luminance range +/- SKY_RANGE_TOL
    are pre-blocked in the mask to stop slow chaining across gradients.
    """
    h, w = blurred.shape
    if not seed.any():
        return seed
    lo, hi = np.percentile(blurred[seed], [5, 95])
    tol = SKY_RANGE_TOL * 255.0
    blocked = (blurred < lo - tol) | (blurred > hi + tol)
    # floodFill mask is 2 px larger; non-zero cells are never filled. Filled cells become 2.
    mask = np.ones((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = blocked
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8)
    image = blurred.copy()   # floodFill wants a writable image even in mask-only mode
    for x in np.flatnonzero(seed[0]):
        if mask[1, x + 1] == 0:
            cv2.floodFill(image, mask, (int(x), 0), 0, SKY_STEP_TOL, SKY_STEP_TOL, flags)
    return (mask[1:-1, 1:-1] == 2) | seed


def _top_smooth_region(gray):
    """Per-column span of sky candidate running down from the top edge.

    Seeds: low-gradient connected components that touch the top edge and are large
    enough. They are then grown by brightness similarity (_grow_region). Each
    column's span stops at the first non-region pixel. Returns (span mask, bottom
    row per column, -1 where the column has no span).
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (0, 0), SKY_BLUR_SIGMA)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    smooth = (np.hypot(gx, gy) / 8.0 < SKY_MAX_GRADIENT).astype(np.uint8)   # Sobel 3x3 gain is 8
    size = max(3, int(round(SKY_CLOSE_FRACTION * max(h, w))) | 1)   # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(smooth, connectivity=4)
    min_area = SKY_MIN_COMPONENT_FRAC * h * w
    keep = [lab for lab in np.unique(labels[0]) if lab != 0 and stats[lab, cv2.CC_STAT_AREA] >= min_area]
    seed = np.isin(labels, keep) if keep else np.zeros((h, w), bool)
    region = _grow_region(blurred, seed)

    # First non-region row in each column; h if the whole column is region.
    first_gap = np.where(region.all(axis=0), h, np.argmin(region, axis=0))
    bottom = np.where(region[0], first_gap - 1, -1)
    span = np.arange(h)[:, None] <= bottom[None, :]
    return span, bottom


def _sky_features(gray, lum, hue, sat, val):
    h, w = gray.shape
    span, bottom = _top_smooth_region(gray)
    area = int(span.sum())
    if area == 0:
        return dict(sky_area_frac=0.0, sky_top_coverage=0.0, sky_edge_density=0.0,
                    sky_residual=0.0, sky_below_ratio=1.0, sky_colour_support=0.0)

    fine_edges = cv2.Canny(gray, *SKY_FINE_CANNY) > 0

    # Smooth brightness: residual around a quadratic surface fitted to the region.
    ys, xs = np.nonzero(span)
    if len(ys) > 20000:
        pick = np.random.default_rng(0).choice(len(ys), 20000, replace=False)
        ys, xs = ys[pick], xs[pick]
    x, y = xs / w, ys / h
    design = np.column_stack([np.ones_like(x), x, y, x * x, y * y, x * y])
    target = lum[ys, xs]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = float(np.std(target - design @ coef))

    # Brighter than below: compare with the same columns' pixels under the span.
    below = (np.arange(h)[:, None] > bottom[None, :]) & (bottom[None, :] >= 0)
    below_mean = lum[below].mean() if below.any() else lum[span].mean()
    below_ratio = float(lum[span].mean() / (below_mean + 1e-3))

    # Colour, weak support only: blue, neutral, or sunset warm/pink/purple. Green or other saturated hues are not sky.
    plausible = (_in_band(hue, BLUE_H) | (sat < NEUTRAL_MAX_S) | (hue <= WARM_H[1]) | (hue >= 140)) & (val >= 40)
    return dict(
        sky_area_frac=area / (h * w),
        sky_top_coverage=float((bottom >= 0).mean()),
        sky_edge_density=float(fine_edges[span].mean()),
        sky_residual=residual,
        sky_below_ratio=below_ratio,
        sky_colour_support=float(plausible[span].mean()),
    )


def extract_features(image):
    img = _prepare(image)
    h, w = img.shape[:2]
    top = slice(0, max(1, h // 3))
    bottom_third = slice(h - max(1, h // 3), h)
    bottom_half = slice(h // 2, h)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lum = gray.astype(np.float32) / 255.0
    p5, p50, p95 = np.percentile(lum, [5, 50, 95])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    coloured = (sat >= COLOURED_MIN_S) & (val >= COLOURED_MIN_V)
    blue = coloured & _in_band(hue, BLUE_H)
    green = coloured & _in_band(hue, GREEN_H)
    warm = coloured & _in_band(hue, WARM_H)
    neutral = (sat < NEUTRAL_MAX_S) & (val >= NEUTRAL_MIN_V)
    white = neutral & (val >= WHITE_MIN_V)

    # Canny thresholds follow the median, with floors so dark or flat images don't turn noise into edges.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    edges = cv2.Canny(blurred, max(20.0, 0.66 * median), min(255.0, max(60.0, 1.33 * median)))
    edge_mask = edges > 0

    long_edge = max(h, w)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=max(5, int(LINE_MIN_LENGTH_FRACTION * long_edge)), maxLineGap=5)
    total_len = axis_len = 0.0
    if lines is not None:
        x1, y1, x2, y2 = lines.reshape(-1, 4).astype(np.float64).T  # (N,1,4) or (N,4) depending on OpenCV build
        lengths = np.hypot(x2 - x1, y2 - y1)
        angles = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        off_axis = np.minimum.reduce([angles, 180.0 - angles, np.abs(angles - 90.0)])
        total_len = float(lengths.sum())
        axis_len = float(lengths[off_axis <= AXIS_TOLERANCE_DEG].sum())

    laplacian = cv2.Laplacian(gray, cv2.CV_32F)

    return SceneFeatures(
        lum_mean=float(lum.mean()),
        lum_std=float(lum.std()),
        lum_p5=float(p5),
        lum_p50=float(p50),
        lum_p95=float(p95),
        clipped_frac=float((gray >= 250).mean()),
        dark_frac=float((gray <= 10).mean()),
        top_bottom_ratio=float(lum[top].mean() / (lum[bottom_third].mean() + 1e-3)),
        sat_mean=float(sat.mean() / 255.0),
        sat_std=float(sat.std() / 255.0),
        blue_frac=float(blue.mean()),
        green_frac=float(green.mean()),
        warm_frac=float(warm.mean()),
        neutral_frac=float(neutral.mean()),
        white_frac=float(white.mean()),
        green_bottom_frac=float(green[bottom_half].mean()),
        **_sky_features(gray, lum, hue, sat, val),
        edge_density=float(edge_mask.mean()),
        line_length_norm=total_len / (w + h),
        axis_line_frac=axis_len / total_len if total_len > 0 else 0.0,
        fine_texture=float(np.abs(laplacian).mean() / 255.0),
    )


# ---------------------------------------------------------------- scoring

def _top_region_presence(f):
    """How clearly a large smooth region spans the top of the frame, in [0, 1]. Says nothing about sky vs ceiling."""
    return _ramp(f.sky_area_frac, 0.05, 0.25) * _ramp(f.sky_top_coverage, 0.30, 0.80)


def exterior_score(f):
    """Positive evidence for exterior in [0, 1]. Sky structure dominates; colour and vegetation only support."""
    presence = _top_region_presence(f)
    # Brighter-than-below is required, not just weighted: without it a flat grey frame scored as sky.
    brighter = _ramp(f.sky_below_ratio, 1.0, 1.15)
    quality = (0.55 * (1.0 - _ramp(f.sky_edge_density, 0.01, 0.06))
               + 0.45 * (1.0 - _ramp(f.sky_residual, 0.04, 0.15)))
    structure = brighter * (0.4 + 0.6 * quality)
    return (0.80 * presence * structure
            + 0.10 * presence * f.sky_colour_support
            + 0.10 * _ramp(f.green_bottom_frac, 0.05, 0.30))


def interior_score(f):
    """Positive evidence for interior in [0, 1]. The absence of sky is NOT evidence here."""
    # Ceiling: a smooth top region that is darker than, or level with, what's below it. Skies are brighter.
    ceiling = _top_region_presence(f) * (1.0 - _ramp(f.sky_below_ratio, 0.85, 1.05))
    rectilinear = _ramp(f.axis_line_frac, 0.60, 0.90) * _ramp(f.line_length_norm, 1.0, 3.0)
    return 0.70 * ceiling + 0.30 * rectilinear


def exterior_decision(ext, inte):
    """(exterior evidence, interior evidence) -> (is_exterior, confidence, ambiguous).

    A confident answer needs strong evidence for one side that clearly beats the other.
    Weak or conflicting evidence gives (False, AMBIGUOUS_CONFIDENCE): unsure, failing safe.
    """
    if ext >= EXTERIOR_MIN_EVIDENCE and ext - inte >= DECISION_MIN_MARGIN:
        return True, 0.5 + 0.5 * _ramp(ext - inte, DECISION_MIN_MARGIN, 0.70), False
    if inte >= INTERIOR_MIN_EVIDENCE and inte - ext >= DECISION_MIN_MARGIN:
        return False, 0.5 + 0.5 * _ramp(inte - ext, DECISION_MIN_MARGIN, 0.70), False
    return False, AMBIGUOUS_CONFIDENCE, True


def room_scores(f):
    """Evidence for each interior room type, each in [0, 1]."""
    return {
        "bathroom": (0.40 * _ramp(f.white_frac, 0.15, 0.45)
                     + 0.25 * (1.0 - _ramp(f.sat_mean, 0.10, 0.30))
                     + 0.20 * _ramp(f.neutral_frac, 0.30, 0.70)
                     + 0.15 * _ramp(f.fine_texture, 0.02, 0.06)),
        "kitchen": (0.35 * _ramp(f.edge_density, 0.05, 0.14)
                    + 0.30 * _ramp(f.line_length_norm, 0.5, 3.0)
                    + 0.20 * _ramp(f.axis_line_frac, 0.50, 0.85)
                    + 0.15 * _ramp(f.neutral_frac, 0.20, 0.50)),
        "bedroom-living": (0.35 * _ramp(f.warm_frac, 0.10, 0.40)
                           + 0.30 * (1.0 - _ramp(f.edge_density, 0.04, 0.12))
                           + 0.20 * _ramp(f.sat_mean, 0.12, 0.30)
                           + 0.15 * (1.0 - _ramp(f.white_frac, 0.10, 0.35))),
    }


def classify_features(f):
    """SceneFeatures -> (label, confidence, scores). No I/O, so logged rows can be re-scored."""
    ext, inte = exterior_score(f), interior_score(f)
    is_ext, ext_conf, ambiguous = exterior_decision(ext, inte)
    scores = {"exterior": ext, "interior": inte}
    if is_ext:
        return "exterior", ext_conf, scores
    if ambiguous:
        return "other", OTHER_CONFIDENCE, scores   # not sure it is even an interior

    rooms = room_scores(f)
    scores.update(rooms)
    names = list(rooms)
    values = np.array([rooms[n] for n in names])
    probs = np.exp(values / ROOM_SOFTMAX_TEMPERATURE)
    probs /= probs.sum()
    scores["room_probs"] = {n: float(p) for n, p in zip(names, probs)}

    order = np.argsort(values)[::-1]
    top, second = values[order[0]], values[order[1]]
    if top < ROOM_MIN_SCORE or top - second < ROOM_MIN_MARGIN:
        return "other", OTHER_CONFIDENCE, scores
    return names[order[0]], float(probs[order[0]] * ext_conf), scores


# ---------------------------------------------------------------- logging

def _thresholds():
    names = ["ANALYSIS_LONG_EDGE", "COLOURED_MIN_S", "COLOURED_MIN_V", "BLUE_H", "GREEN_H", "WARM_H",
             "NEUTRAL_MAX_S", "NEUTRAL_MIN_V", "WHITE_MIN_V",
             "LINE_MIN_LENGTH_FRACTION", "AXIS_TOLERANCE_DEG", "SKY_BLUR_SIGMA", "SKY_MAX_GRADIENT",
             "SKY_CLOSE_FRACTION", "SKY_STEP_TOL", "SKY_RANGE_TOL", "SKY_MIN_COMPONENT_FRAC", "SKY_FINE_CANNY", "EXTERIOR_MIN_EVIDENCE",
             "INTERIOR_MIN_EVIDENCE", "DECISION_MIN_MARGIN", "AMBIGUOUS_CONFIDENCE", "ROOM_MIN_SCORE", "ROOM_MIN_MARGIN", "ROOM_SOFTMAX_TEMPERATURE",
             "OTHER_CONFIDENCE"]
    return {n: globals()[n] for n in names}


def _log_row(db_path, call, label, confidence, features, scores, source):
    """Write one row. A logging failure is warned about, never raised: classification still returns."""
    if db_path is None:
        return
    try:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path, timeout=30)) as conn:
            conn.executescript(SCHEMA)
            with conn:
                conn.execute(
                    "INSERT INTO scene_classifications "
                    "(timestamp, source, call, label, confidence, features, scores, thresholds) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(sep=" ", timespec="milliseconds"),
                     str(source) if source is not None else None, call, label, float(confidence),
                     json.dumps(asdict(features)), json.dumps(scores), json.dumps(_thresholds())))
    except (sqlite3.Error, OSError) as exc:
        log.warning("scene classification not logged to %s: %s", db_path, exc)


# ---------------------------------------------------------------- public API

def is_exterior(image, source=None, db_path=DEFAULT_DB):
    """-> (is_exterior, confidence). Independent of room typing."""
    f = extract_features(image)
    ext, inte = exterior_score(f), interior_score(f)
    result, conf, ambiguous = exterior_decision(ext, inte)
    label = "exterior" if result else ("unsure" if ambiguous else "interior")
    _log_row(db_path, "is_exterior", label, conf, f, {"exterior": ext, "interior": inte}, source)
    return result, conf


def classify_scene(image, source=None, db_path=DEFAULT_DB):
    """-> (label, confidence), label in kitchen / bathroom / bedroom-living / exterior / other."""
    f = extract_features(image)
    label, conf, scores = classify_features(f)
    _log_row(db_path, "classify_scene", label, conf, f, scores, source)
    return label, conf


def _read_image(path):
    # imdecode instead of imread: imread fails on non-ASCII Windows paths.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return image


def main():
    parser = argparse.ArgumentParser(description="Classify a fused image with classical features.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--exterior-only", action="store_true", help="run is_exterior only")
    parser.add_argument("--verbose", action="store_true", help="print features and scores")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-log", action="store_true", help="don't write to the database")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    image = _read_image(args.image)
    db_path = None if args.no_log else args.db
    if args.exterior_only:
        result, conf = is_exterior(image, source=args.image, db_path=db_path)
        print(f"exterior={result} confidence={conf:.2f}")
    else:
        label, conf = classify_scene(image, source=args.image, db_path=db_path)
        print(f"label={label} confidence={conf:.2f}")

    if args.verbose:
        f = extract_features(image)
        print(json.dumps({"features": asdict(f), "scores": classify_features(f)[2]}, indent=2))


if __name__ == "__main__":
    main()
