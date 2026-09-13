"""Learn per-scene tone preferences from the user's slider adjustments.

    log_adjustment(scene_type, param, auto_value, user_value) -> delta
    get_defaults(scene_type) -> {param: value}
    reset_profile(scene_type=None) -> rows deleted

The UI calls log_adjustment when a slider is released: auto_value is what the
pipeline picked, user_value is where the user left it. One row per release, with
delta = user_value - auto_value. A release that leaves the value unchanged is
still logged; it says the automatic value was right and pulls the average to 0.

get_defaults walks a scene's deltas for each param, oldest first, as an
exponential moving average whose alpha shrinks with the sample count:

    alpha_k = max(1 / k, MIN_ALPHA)      avg += alpha_k * (delta_k - avg)

For the first 1/MIN_ALPHA samples this is exactly the plain mean, so a few early
samples each count fully. After that alpha stays at MIN_ALPHA and recent edits
outweigh old ones. The returned value is GLOBAL_DEFAULTS[param] + avg. A param
with fewer than MIN_SAMPLES rows for the scene gets the global default unchanged.

Every constant below is an untuned first guess.
"""

import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DB = Path("data/estatelens.db")

# Same numbers as the current enhance step; copied, not imported, because
# core/prototype.py is going away.
GLOBAL_DEFAULTS = {
    "clahe_clip": 1.5,
    "saturation": 1.1,
}

MIN_SAMPLES = 5        # below this a scene uses GLOBAL_DEFAULTS
MIN_ALPHA = 0.2        # alpha floor once there are many samples
MAX_HISTORY = 200      # most recent rows read per (scene, param)

SCHEMA = """
CREATE TABLE IF NOT EXISTS echolearn_adjustments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_type  TEXT NOT NULL,
    param       TEXT NOT NULL,
    auto_value  REAL NOT NULL,
    user_value  REAL NOT NULL,
    delta       REAL NOT NULL,     -- user_value - auto_value
    timestamp   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS echolearn_scene_param ON echolearn_adjustments (scene_type, param, id);
"""


def _connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.executescript(SCHEMA)
    return conn


def log_adjustment(scene_type, param, auto_value, user_value, db_path=DEFAULT_DB):
    """Record one slider release. Returns the delta. A logging failure is warned about, never raised."""
    delta = float(user_value) - float(auto_value)
    try:
        with closing(_connect(db_path)) as conn, conn:
            conn.execute(
                "INSERT INTO echolearn_adjustments (scene_type, param, auto_value, user_value, delta, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (scene_type, param, float(auto_value), float(user_value), delta,
                 datetime.now().isoformat(sep=" ", timespec="milliseconds")))
    except (sqlite3.Error, OSError) as exc:
        log.warning("adjustment not logged to %s: %s", db_path, exc)
    return delta


def weighted_delta(deltas):
    """Moving average of `deltas` (oldest first), alpha = max(1/k, MIN_ALPHA)."""
    avg = 0.0
    for k, d in enumerate(deltas, 1):
        avg += max(1.0 / k, MIN_ALPHA) * (d - avg)
    return avg


def get_defaults(scene_type, db_path=DEFAULT_DB):
    """{param: value} for every param in GLOBAL_DEFAULTS, learned where there are enough samples."""
    defaults = dict(GLOBAL_DEFAULTS)
    try:
        with closing(_connect(db_path)) as conn:
            for param, base in GLOBAL_DEFAULTS.items():
                rows = conn.execute(
                    "SELECT delta FROM echolearn_adjustments WHERE scene_type = ? AND param = ? "
                    "ORDER BY id DESC LIMIT ?", (scene_type, param, MAX_HISTORY)).fetchall()
                if len(rows) >= MIN_SAMPLES:
                    defaults[param] = base + weighted_delta([r[0] for r in reversed(rows)])
    except (sqlite3.Error, OSError) as exc:
        log.warning("could not read preferences from %s, using global defaults: %s", db_path, exc)
        return dict(GLOBAL_DEFAULTS)
    return defaults


def reset_profile(scene_type=None, db_path=DEFAULT_DB):
    """Forget learned adjustments for one scene type, or all of them. Returns rows deleted."""
    with closing(_connect(db_path)) as conn, conn:
        if scene_type is None:
            cur = conn.execute("DELETE FROM echolearn_adjustments")
        else:
            cur = conn.execute("DELETE FROM echolearn_adjustments WHERE scene_type = ?", (scene_type,))
        return cur.rowcount
