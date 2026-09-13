"""Gallery tone consistency: nudge every image of one shoot toward a shared look.

    sync_gallery(images, groups=None) -> SyncResult

`images` is {name: fused BGR image}: uint8, uint16, or float in [0, 1]
(display-referred, as Mertens fusion produces). Corrected images come back in the
dtype they went in with.

Method, per group:
  1. Measure each image in CIELAB on a copy shrunk to ANALYSIS_LONG_EDGE.
     brightness: median L over unclipped pixels (L within VALID_L).
     colour cast: mean a/b over the dominant near-neutral cluster. Start from the
                  median a/b, keep pixels within NEUTRAL_RADIUS of it, re-average,
                  repeat. The cluster follows the cast even when it is strong, and
                  saturated objects (a red sofa) fall outside it.
  2. Target = per-channel median of the group's stats. A median, so one very dark
     or very bright room can't drag the rest.
  3. Needed correction: dL = target_L - L, dab = |target_ab - ab|. An image that
     needs more than MAX_DL or MAX_DAB is skipped and reported, not corrected.
  4. Everything else moves `strength` of the way to the target:
     L:   power curve L' = 100 (L/100)^g, chosen so the median lands on the new
          value. Black stays 0 and white stays 100, so nothing clips.
     a/b: constant shift, tapered to zero in the top and bottom SHIFT_TAPER_L of L
          so blacks and highlights don't pick up a tint.

Groups: pass groups={name: label} to sync interiors and exteriors (or floors,
times of day) against separate targets. The caller decides; nothing is inferred
from content. Default: one group. A group smaller than MIN_GROUP_SIZE has no
meaningful median, so its images are skipped and reported.

Every threshold below is an untuned first guess.

Logging: each group writes one row to `tone_sync_runs` (target, settings) and one
row per image to `tone_sync_images` (stats, needed and applied deltas, status,
reason) in the SQLite database. Pass db_path=None to skip.

CLI:
    python -m processing.tonesync FOLDER [--groups groups.json] [--out DIR] [--strength 0.7] [--db data/estatelens.db | --no-log]
Without --out it is a dry run: report only. groups.json maps file name -> group label.
"""

import argparse
import json
import logging
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DB = Path("data/estatelens.db")
ANALYSIS_LONG_EDGE = 512
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

VALID_L = (3.0, 97.0)             # pixels outside this are clipped or near it; not measured
MIN_VALID_FRACTION = 0.10         # below this the image is too clipped to measure
NEUTRAL_RADIUS = 12.0             # a/b distance from the cluster centre that still counts as neutral
NEUTRAL_ITERATIONS = 3
MIN_NEUTRAL_FRACTION = 0.05       # of valid pixels; below this no cast is measured (colour left alone)

MIN_GROUP_SIZE = 3
DEFAULT_STRENGTH = 0.7            # fraction of the gap to the target that is closed
MAX_DL = 12.0                     # L units (0-100)
MAX_DAB = 8.0                     # a/b distance
L_CLAMP = (1.0, 99.0)             # keeps the power-curve exponent finite
SHIFT_TAPER_L = 10.0

DEFAULT_GROUP = "all"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tone_sync_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL,
    group_name  TEXT NOT NULL,
    n_images    INTEGER NOT NULL,
    target      TEXT,              -- JSON {L, a, b}; NULL if the group was too small
    strength    REAL NOT NULL,
    thresholds  TEXT NOT NULL      -- JSON snapshot of the constants used
);
CREATE TABLE IF NOT EXISTS tone_sync_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES tone_sync_runs(id),
    source      TEXT NOT NULL,     -- name as passed in (file name from the CLI)
    status      TEXT NOT NULL,     -- 'corrected' / 'skipped'
    stats       TEXT,              -- JSON of ImageStats
    needed_dL   REAL,
    needed_dab  REAL,
    applied_dL  REAL,
    applied_da  REAL,
    applied_db  REAL,
    reason      TEXT               -- why it was skipped
);
"""


@dataclass
class ImageStats:
    L: float | None               # None when too clipped to measure
    a: float | None               # a/b None when no neutral cluster was found
    b: float | None
    valid_fraction: float
    neutral_fraction: float


@dataclass
class ImageReport:
    name: str
    group: str
    status: str                   # 'corrected' / 'skipped'
    stats: ImageStats | None = None
    needed_dL: float | None = None
    needed_dab: float | None = None
    applied_dL: float = 0.0
    applied_da: float = 0.0
    applied_db: float = 0.0
    reason: str | None = None


@dataclass
class SyncResult:
    corrected: dict = field(default_factory=dict)   # name -> corrected image
    reports: list = field(default_factory=list)     # ImageReport, input order
    targets: dict = field(default_factory=dict)     # group -> {L, a, b} (None if too small)

    @property
    def skipped(self):
        return [(r.name, r.reason) for r in self.reports if r.status == "skipped"]


# ---------------------------------------------------------------- conversion

def _to_float(image):
    """Any supported input -> float32 BGR in [0, 1]."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.floating):
        return np.clip(np.nan_to_num(image), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"unsupported dtype {image.dtype}")


def _from_float(img, dtype):
    img = np.clip(img, 0.0, 1.0)
    if dtype == np.uint8:
        return (img * 255.0 + 0.5).astype(np.uint8)
    if dtype == np.uint16:
        return (img * 65535.0 + 0.5).astype(np.uint16)
    return img.astype(dtype)


def _shrink(img):
    h, w = img.shape[:2]
    scale = ANALYSIS_LONG_EDGE / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- measurement

def measure(image):
    """-> ImageStats for one image."""
    lab = cv2.cvtColor(_shrink(_to_float(image)), cv2.COLOR_BGR2Lab).reshape(-1, 3)
    L, ab = lab[:, 0], lab[:, 1:]
    valid = (L >= VALID_L[0]) & (L <= VALID_L[1])
    valid_fraction = float(valid.mean())
    if valid_fraction < MIN_VALID_FRACTION:
        return ImageStats(None, None, None, valid_fraction, 0.0)

    ab_valid = ab[valid]
    centre = np.median(ab_valid, axis=0)
    neutral = np.zeros(len(ab_valid), bool)
    for _ in range(NEUTRAL_ITERATIONS):
        neutral = np.hypot(*(ab_valid - centre).T) <= NEUTRAL_RADIUS
        if not neutral.any():
            break
        centre = ab_valid[neutral].mean(axis=0)
    neutral_fraction = float(neutral.mean())

    a = b = None
    if neutral_fraction >= MIN_NEUTRAL_FRACTION:
        a, b = float(centre[0]), float(centre[1])
    return ImageStats(float(np.median(L[valid])), a, b, valid_fraction, neutral_fraction)


def _median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def group_target(stats):
    """Per-channel median of a group's ImageStats -> {L, a, b}."""
    return {"L": _median(s.L for s in stats), "a": _median(s.a for s in stats), "b": _median(s.b for s in stats)}


# ---------------------------------------------------------------- correction

def apply_correction(image, source_L, new_L, da, db):
    """Move median L from source_L to new_L with a power curve and shift a/b. Keeps dtype."""
    img = _to_float(image)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]

    m = min(max(source_L, L_CLAMP[0]), L_CLAMP[1]) / 100.0
    t = min(max(new_L, L_CLAMP[0]), L_CLAMP[1]) / 100.0
    gamma = math.log(t) / math.log(m)
    if abs(gamma - 1.0) > 1e-6:
        L[:] = 100.0 * np.power(np.clip(L, 0.0, 100.0) / 100.0, gamma)

    if da or db:
        weight = np.clip(np.minimum(L, 100.0 - L) / SHIFT_TAPER_L, 0.0, 1.0)
        lab[:, :, 1] += da * weight
        lab[:, :, 2] += db * weight

    out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return _from_float(out, image.dtype)


# ---------------------------------------------------------------- public API

def sync_gallery(images, groups=None, strength=DEFAULT_STRENGTH, max_dL=MAX_DL, max_dab=MAX_DAB,
                 db_path=DEFAULT_DB):
    """Correct a gallery toward per-group median tone. See module docstring."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if groups is None:
        groups = {name: DEFAULT_GROUP for name in images}
    missing = set(images) - set(groups)
    extra = set(groups) - set(images)
    if missing or extra:
        raise ValueError(f"groups must label exactly the images given; "
                         f"missing: {sorted(missing)}, unknown: {sorted(extra)}")

    result = SyncResult()
    reports = {}
    members = {}
    for name in images:
        members.setdefault(groups[name], []).append(name)

    for group, names in members.items():
        stats = {n: measure(images[n]) for n in names}
        if len(names) < MIN_GROUP_SIZE:
            target = None
            for n in names:
                reports[n] = ImageReport(n, group, "skipped", stats[n],
                                         reason=f"group '{group}' has {len(names)} images, needs {MIN_GROUP_SIZE}")
        else:
            target = group_target(stats.values())
            for n in names:
                reports[n] = _correct_one(n, group, images[n], stats[n], target, strength, max_dL, max_dab, result)
        result.targets[group] = target
        _log_group(db_path, group, target, strength, max_dL, max_dab, [reports[n] for n in names])

    result.reports = [reports[n] for n in images]
    return result


def _correct_one(name, group, image, s, target, strength, max_dL, max_dab, result):
    report = ImageReport(name, group, "skipped", s)
    if s.L is None or target["L"] is None:
        report.reason = f"too clipped to measure ({s.valid_fraction:.0%} usable pixels)"
        return report

    report.needed_dL = target["L"] - s.L
    colour_measured = s.a is not None and target["a"] is not None
    if colour_measured:
        report.needed_dab = math.hypot(target["a"] - s.a, target["b"] - s.b)

    over = []
    if abs(report.needed_dL) > max_dL:
        over.append(f"dL={report.needed_dL:+.1f} (cap {max_dL:g})")
    if colour_measured and report.needed_dab > max_dab:
        over.append(f"dab={report.needed_dab:.1f} (cap {max_dab:g})")
    if over:
        report.reason = "needs " + ", ".join(over)
        return report

    report.applied_dL = strength * report.needed_dL
    if colour_measured:
        report.applied_da = strength * (target["a"] - s.a)
        report.applied_db = strength * (target["b"] - s.b)
    result.corrected[name] = apply_correction(image, s.L, s.L + report.applied_dL,
                                              report.applied_da, report.applied_db)
    report.status = "corrected"
    if not colour_measured:
        report.reason = "no neutral pixels found; brightness only"
    return report


# ---------------------------------------------------------------- logging

def _thresholds(max_dL, max_dab):
    names = ["ANALYSIS_LONG_EDGE", "VALID_L", "MIN_VALID_FRACTION", "NEUTRAL_RADIUS", "NEUTRAL_ITERATIONS",
             "MIN_NEUTRAL_FRACTION", "MIN_GROUP_SIZE", "L_CLAMP", "SHIFT_TAPER_L"]
    snapshot = {n: globals()[n] for n in names}
    snapshot.update(MAX_DL=max_dL, MAX_DAB=max_dab)
    return snapshot


def _log_group(db_path, group, target, strength, max_dL, max_dab, reports):
    """Write one run row plus its image rows. A logging failure is warned about, never raised."""
    if db_path is None:
        return
    try:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path, timeout=30)) as conn:
            conn.executescript(SCHEMA)
            with conn:
                cur = conn.execute(
                    "INSERT INTO tone_sync_runs (timestamp, group_name, n_images, target, strength, thresholds) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(sep=" ", timespec="milliseconds"), group, len(reports),
                     json.dumps(target) if target is not None else None, float(strength),
                     json.dumps(_thresholds(max_dL, max_dab))))
                conn.executemany(
                    "INSERT INTO tone_sync_images (run_id, source, status, stats, needed_dL, needed_dab, "
                    "applied_dL, applied_da, applied_db, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(cur.lastrowid, r.name, r.status, json.dumps(asdict(r.stats)) if r.stats else None,
                      r.needed_dL, r.needed_dab, r.applied_dL, r.applied_da, r.applied_db, r.reason)
                     for r in reports])
    except (sqlite3.Error, OSError) as exc:
        log.warning("tone sync not logged to %s: %s", db_path, exc)


# ---------------------------------------------------------------- CLI

def _read_image(path):
    # imdecode instead of imread: imread fails on non-ASCII Windows paths.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"could not read {path}")
    return image


def _write_image(path, image):
    ok, buf = cv2.imencode(path.suffix, image)
    if not ok:
        raise SystemExit(f"could not encode {path}")
    buf.tofile(str(path))


def main():
    parser = argparse.ArgumentParser(description="Make a property gallery's tone consistent.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--groups", type=Path, help="JSON: file name -> group label")
    parser.add_argument("--out", type=Path, help="write corrected images here (default: dry run)")
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--no-log", action="store_true", help="don't write to the database")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    images = {p.name: _read_image(p) for p in paths}
    groups = json.loads(args.groups.read_text()) if args.groups else None
    try:
        result = sync_gallery(images, groups, strength=args.strength, db_path=None if args.no_log else args.db)
    except ValueError as exc:
        raise SystemExit(str(exc))

    for group, t in result.targets.items():
        print(f"[{group}] target: " + ("group too small" if t is None else
              "L={L:.1f} a={a} b={b}".format(L=t["L"], a=_fmt(t["a"]), b=_fmt(t["b"]))))
    for r in result.reports:
        line = f"  {r.status:9} {r.name}"
        if r.needed_dL is not None:
            line += f"  needed dL={r.needed_dL:+.1f} dab={_fmt(r.needed_dab)}"
        if r.status == "corrected":
            line += f"  applied dL={r.applied_dL:+.1f} da={r.applied_da:+.1f} db={r.applied_db:+.1f}"
        if r.reason:
            line += f"  ({r.reason})"
        print(line)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, image in result.corrected.items():
            _write_image(args.out / name, image)
        print(f"wrote {len(result.corrected)} images to {args.out}")


def _fmt(v):
    return "n/a" if v is None else f"{v:.1f}"


if __name__ == "__main__":
    main()
