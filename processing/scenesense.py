"""Scene classification from classical image features (no neural network).

    is_exterior(image)    -> (bool, confidence)   gates sky replacement
    classify_scene(image) -> (label, confidence)  kitchen / bathroom / bedroom-living / exterior / other

`image` is a fused BGR image: uint8, uint16, or float in [0, 1] (display-referred,
as Mertens fusion produces). Colour, edge and brightness statistics are measured
on a copy shrunk to ANALYSIS_LONG_EDGE.

is_exterior uses only exterior cues (sky in the top third, vegetation at the
bottom, bright smooth top) and never depends on room typing. Ambiguous scores
return (False, AMBIGUOUS_CONFIDENCE): a missed sky replacement is cheaper than a
sky pasted onto a ceiling. A confidence >= 0.5 means a clear decision.

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
SKY_BLUE_MIN_V = 90
SKY_BRIGHT_MIN_V = 215   # overcast sky: near-white and bright

LINE_MIN_LENGTH_FRACTION = 0.06   # of the long edge
AXIS_TOLERANCE_DEG = 10           # meaningful once perspective is corrected

# Exterior decision
EXTERIOR_TRUE_MIN = 0.60    # score >= this -> exterior
EXTERIOR_FALSE_MAX = 0.40   # score <= this -> interior; in between is ambiguous
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
    sky_top_frac: float         # blue or bright-neutral pixels in the top third
    green_bottom_frac: float    # green pixels in the bottom half
    # edges
    edge_density: float
    top_edge_density: float
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
    sky = (blue & (val >= SKY_BLUE_MIN_V)) | (neutral & (val >= SKY_BRIGHT_MIN_V))

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
        x1, y1, x2, y2 = lines[:, 0, :].astype(np.float64).T
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
        sky_top_frac=float(sky[top].mean()),
        green_bottom_frac=float(green[bottom_half].mean()),
        edge_density=float(edge_mask.mean()),
        top_edge_density=float(edge_mask[top].mean()),
        line_length_norm=total_len / (w + h),
        axis_line_frac=axis_len / total_len if total_len > 0 else 0.0,
        fine_texture=float(np.abs(laplacian).mean() / 255.0),
    )


# ---------------------------------------------------------------- scoring

def exterior_score(f):
    """Evidence for an exterior shot in [0, 1]. Uses exterior cues only."""
    # Sky counts only if the top is also smooth: a bright ceiling full of fixtures or a window frame is not sky.
    smooth_top = 1.0 - _ramp(f.top_edge_density, 0.02, 0.10)
    return (0.40 * _ramp(f.sky_top_frac, 0.10, 0.50) * (0.5 + 0.5 * smooth_top)
            + 0.20 * _ramp(f.green_bottom_frac, 0.05, 0.30)
            + 0.15 * _ramp(f.top_bottom_ratio, 1.0, 1.6)
            + 0.15 * smooth_top
            + 0.10 * _ramp(f.blue_frac + f.green_frac, 0.05, 0.30))


def exterior_decision(score):
    """score -> (is_exterior, confidence). Ambiguous scores fail safe to False."""
    if score >= EXTERIOR_TRUE_MIN:
        return True, 0.5 + 0.5 * _ramp(score, EXTERIOR_TRUE_MIN, 0.90)
    if score <= EXTERIOR_FALSE_MAX:
        return False, 0.5 + 0.5 * (1.0 - _ramp(score, 0.10, EXTERIOR_FALSE_MAX))
    return False, AMBIGUOUS_CONFIDENCE


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
    ext = exterior_score(f)
    is_ext, ext_conf = exterior_decision(ext)
    scores = {"exterior": ext}
    if is_ext:
        return "exterior", ext_conf, scores
    if ext_conf <= AMBIGUOUS_CONFIDENCE:
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
             "NEUTRAL_MAX_S", "NEUTRAL_MIN_V", "WHITE_MIN_V", "SKY_BLUE_MIN_V", "SKY_BRIGHT_MIN_V",
             "LINE_MIN_LENGTH_FRACTION", "AXIS_TOLERANCE_DEG", "EXTERIOR_TRUE_MIN", "EXTERIOR_FALSE_MAX",
             "AMBIGUOUS_CONFIDENCE", "ROOM_MIN_SCORE", "ROOM_MIN_MARGIN", "ROOM_SOFTMAX_TEMPERATURE",
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
    score = exterior_score(f)
    result, conf = exterior_decision(score)
    _log_row(db_path, "is_exterior", "exterior" if result else "interior", conf, f,
             {"exterior": score}, source)
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
