"""SQLite job scheduler for batch bracket fusion.

Each bracket becomes a row in `jobs`. Every status change is committed
immediately and recorded in `runs`, so after a crash the database shows exactly
what happened and when. On launch, jobs left 'running' by a crashed run are
reset to 'pending' and processed along with everything else still pending.

Assumes one scheduler process at a time (no lock file, by design: a stale lock
after a crash would block the recovery this module exists for).

Usage:
    python -m core.scheduler [input_dir output_dir] [--db data/estatelens.db]
                             [--half-size] [--no-align] [--clahe-clip 2.0] [--saturation 1.25]
                             [--retry-failed] [--history]
"""

import argparse
import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil

from core.prototype import CLAHE_CLIP, SATURATION, brightness, find_frames, fuse_bracket, group_brackets, read_exif

DEFAULT_DB = Path("data/estatelens.db")
MEMORY_SAMPLE_SECONDS = 0.02

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bracket_files  TEXT NOT NULL UNIQUE,   -- JSON list of absolute paths, dark to bright
    status         TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'failed')),
    output_path    TEXT NOT NULL,          -- planned at creation; the file exists only once 'complete'
    created_at     TEXT NOT NULL,
    settings       TEXT NOT NULL,          -- JSON: half_size, align, clahe_clip, saturation
    error          TEXT,
    wall_seconds   REAL,
    peak_memory_mb REAL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER NOT NULL REFERENCES jobs (job_id),
    old_status TEXT,                        -- NULL when the job is created
    new_status TEXT NOT NULL,
    timestamp  TEXT NOT NULL
);
"""


def now():
    return datetime.now().isoformat(sep=" ", timespec="milliseconds")


def connect(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def set_status(conn, job_id, new_status, **fields):
    """Change a job's status and log the transition, in one transaction."""
    with conn:  # commits on success, rolls back on error
        old_status = conn.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,)).fetchone()["status"]
        columns = ", ".join(f"{name} = ?" for name in ["status", *fields])
        conn.execute(f"UPDATE jobs SET {columns} WHERE job_id = ?", (new_status, *fields.values(), job_id))
        conn.execute("INSERT INTO runs (job_id, old_status, new_status, timestamp) VALUES (?, ?, ?, ?)",
                     (job_id, old_status, new_status, now()))


def recover_interrupted(conn):
    """A job still 'running' at launch can only mean the last run crashed mid-job."""
    rows = conn.execute("SELECT job_id FROM jobs WHERE status = 'running'").fetchall()
    for row in rows:
        set_status(conn, row["job_id"], "pending")
    if rows:
        print(f"recovered {len(rows)} interrupted job(s): {', '.join(str(r['job_id']) for r in rows)}")


def retry_failed(conn):
    rows = conn.execute("SELECT job_id FROM jobs WHERE status = 'failed'").fetchall()
    for row in rows:
        set_status(conn, row["job_id"], "pending", error=None)
    print(f"requeued {len(rows)} failed job(s)")


def enqueue_folder(conn, input_dir, output_dir, settings):
    frames = find_frames(input_dir)
    if not frames:
        print(f"no RAW or TIFF files in {input_dir}")
        return
    brackets, warnings = group_brackets(frames)
    for w in warnings:
        print(f"WARNING: {w}")

    added = skipped = 0
    for bracket in brackets:
        if all(brightness(f) is not None for f in bracket):
            bracket = sorted(bracket, key=brightness)
        files = json.dumps([str(f["path"].resolve()) for f in bracket])
        output_path = output_dir.resolve() / f"{bracket[0]['path'].stem}_fused.jpg"
        with conn:
            # Check first rather than INSERT OR IGNORE: an ignored insert still
            # consumes an AUTOINCREMENT id, leaving gaps in job_id.
            if conn.execute("SELECT 1 FROM jobs WHERE bracket_files = ?", (files,)).fetchone():
                skipped += 1
                continue
            cur = conn.execute(
                "INSERT INTO jobs (bracket_files, status, output_path, created_at, settings) "
                "VALUES (?, 'pending', ?, ?, ?)",
                (files, str(output_path), now(), json.dumps(settings)))
            conn.execute("INSERT INTO runs (job_id, old_status, new_status, timestamp) VALUES (?, NULL, 'pending', ?)",
                         (cur.lastrowid, now()))
            added += 1
    print(f"queued {added} new job(s), skipped {skipped} already in the database")


class PeakMemory:
    """Track the highest process memory (RSS) while the `with` block runs.

    psutil only reports current memory, and Windows' own peak counter covers the
    whole process lifetime, so a background thread samples every 20 ms instead.
    RSS includes OpenCV's C++ buffers, which Python's tracemalloc cannot see.
    """

    def __enter__(self):
        self._process = psutil.Process()
        self.peak = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(MEMORY_SAMPLE_SECONDS):
            self.peak = max(self.peak, self._process.memory_info().rss)

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)

    @property
    def peak_mb(self):
        return self.peak / 2**20


def frame_from_path(path):
    """Rebuild the frame dict that prototype.find_frames produces."""
    timestamp, exposure, f_number, iso = read_exif(path)
    return {"path": path, "timestamp": timestamp, "exposure": exposure, "f_number": f_number, "iso": iso}


def run_job(conn, job):
    job_id = job["job_id"]
    files = [Path(p) for p in json.loads(job["bracket_files"])]
    settings = json.loads(job["settings"])
    output_path = Path(job["output_path"])
    print(f"[job {job_id}] running: {' '.join(p.name for p in files)}")
    set_status(conn, job_id, "running")

    # KeyboardInterrupt or a hard crash is deliberately not caught: the job stays
    # 'running' and the next launch recovers it.
    start = time.perf_counter()
    memory = PeakMemory()
    try:
        missing = [p.name for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing file(s): {', '.join(missing)}")
        bracket = [frame_from_path(p) for p in files]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with memory:
            fuse_bracket(bracket, output_path, settings["half_size"], settings["align"],
                         settings["clahe_clip"], settings["saturation"])
    except Exception as e:
        wall = time.perf_counter() - start
        peak = memory.peak_mb if hasattr(memory, "peak") else None
        set_status(conn, job_id, "failed", error=str(e), wall_seconds=wall, peak_memory_mb=peak)
        print(f"[job {job_id}] FAILED after {wall:.1f}s: {e}")
        return

    wall = time.perf_counter() - start
    set_status(conn, job_id, "complete", error=None, wall_seconds=wall, peak_memory_mb=memory.peak_mb)
    print(f"[job {job_id}] complete -> {output_path.name} ({wall:.1f}s, peak {memory.peak_mb:.0f} MB)")


def process_pending(conn):
    while True:
        job = conn.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY job_id LIMIT 1").fetchone()
        if job is None:
            break
        run_job(conn, job)


def print_history(conn):
    print("jobs:")
    for r in conn.execute("SELECT job_id, status, wall_seconds, peak_memory_mb, error FROM jobs ORDER BY job_id"):
        stats = f"{r['wall_seconds']:.1f}s, {r['peak_memory_mb']:.0f} MB" if r["wall_seconds"] is not None else ""
        print(f"  {r['job_id']:>4}  {r['status']:<9} {stats}  {r['error'] or ''}".rstrip())
    print("runs:")
    for r in conn.execute("SELECT timestamp, job_id, old_status, new_status FROM runs ORDER BY run_id"):
        print(f"  {r['timestamp']}  job {r['job_id']:>4}  {r['old_status'] or '(new)':<9} -> {r['new_status']}")


def main():
    parser = argparse.ArgumentParser(description="Queue and run bracket-fusion jobs from SQLite.")
    parser.add_argument("input_dir", type=Path, nargs="?", help="folder to queue (omit to only resume)")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--half-size", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP)
    parser.add_argument("--saturation", type=float, default=SATURATION)
    parser.add_argument("--retry-failed", action="store_true", help="requeue failed jobs")
    parser.add_argument("--history", action="store_true", help="print jobs and status transitions, then exit")
    args = parser.parse_args()
    if (args.input_dir is None) != (args.output_dir is None):
        parser.error("give both input_dir and output_dir, or neither")
    if args.input_dir is not None and not args.input_dir.is_dir():
        parser.error(f"{args.input_dir} is not a folder")
    if args.clahe_clip < 0 or args.saturation < 0:
        parser.error("--clahe-clip and --saturation must not be negative")

    conn = connect(args.db)
    if args.history:
        print_history(conn)
        return

    recover_interrupted(conn)
    if args.retry_failed:
        retry_failed(conn)
    if args.input_dir is not None:
        settings = {"half_size": args.half_size, "align": not args.no_align,
                    "clahe_clip": args.clahe_clip, "saturation": args.saturation}
        enqueue_folder(conn, args.input_dir, args.output_dir, settings)
    process_pending(conn)

    counts = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    print("database: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in ("complete", "failed", "pending", "running")))


if __name__ == "__main__":
    main()
