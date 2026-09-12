"""SQLite job scheduler for batch bracket fusion.

Each bracket becomes a row in `jobs`. Every status change is committed
immediately and recorded in `runs`, so after a crash the database shows exactly
what happened and when. On launch, jobs left 'running' by a crashed run are
reset to 'pending' and processed along with everything else still pending.

Assumes one scheduler process at a time (no lock file, by design: a stale lock
after a crash would block the recovery this module exists for).

Tiled fusion (stopgap until processing/lumamerge.py exists): Mertens' weight maps
and pyramids, not the decoded frames, are the memory peak, so large images are
fused in overlapping tiles and feathered back together. Frames are still decoded
whole, so their memory is not reduced; disk-backed frames (np.memmap) would cut
that too and are left as future work. --compare-tiling fuses one bracket both
ways, each in a fresh process, and records peak memory for the two runs.

Usage:
    python -m core.scheduler [input_dir output_dir] [--db data/estatelens.db]
                             [--half-size] [--no-align] [--clahe-clip 2.0] [--saturation 1.25]
                             [--tile-size 2048] [--overlap 256]
                             [--retry-failed] [--history] [--compare-tiling]
"""

import argparse
import json
import multiprocessing
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil

from core.prototype import (CLAHE_CLIP, JPEG_QUALITY, SATURATION, align, brightness, enhance, find_frames,
                            group_brackets, load_image, read_exif)
from core.truevertical import correct_perspective_with_diagnostics

DEFAULT_DB = Path("data/estatelens.db")
MEMORY_SAMPLE_SECONDS = 0.02
# Tile edge in pixels; 0 disables tiling. An image that fits in one tile is fused whole.
TILE_SIZE = 2048
# Large overlap because Mertens' coarse pyramid levels see far beyond a tile's edge,
# so neighbouring tiles differ in low-frequency brightness near their borders.
OVERLAP = 256
# Tiles take their broad tone from a whole-image Mertens run at 1/GUIDE_FACTOR size.
# Fused alone, each tile weights exposures by its own content (all sky vs. mostly
# ground), which left a visible band in the Ladakh Valley sky that feathering
# could not hide. At 1/8 the guide added ~93 MB transiently on a 36 MP bracket.
GUIDE_FACTOR = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    bracket_files  TEXT NOT NULL UNIQUE,   -- JSON list of absolute paths, dark to bright
    status         TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'failed')),
    output_path    TEXT NOT NULL,          -- planned at creation; the file exists only once 'complete'
    created_at     TEXT NOT NULL,
    settings       TEXT NOT NULL,          -- JSON: half_size, align, clahe_clip, saturation, tile_size, overlap
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
CREATE TABLE IF NOT EXISTS memory_benchmarks (
    bench_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bracket_files      TEXT NOT NULL,       -- JSON list, same format as jobs.bracket_files
    mode               TEXT NOT NULL CHECK (mode IN ('whole', 'tiled')),
    tile_size          INTEGER NOT NULL,
    overlap            INTEGER NOT NULL,
    tiles              INTEGER,             -- 1 for whole-image fusion
    width              INTEGER,
    height             INTEGER,
    baseline_memory_mb REAL,                -- RSS after imports, before fusion
    peak_memory_mb     REAL,                -- peak RSS: understates need once Windows starts paging
    peak_commit_mb     REAL,                -- peak private bytes: the real requirement
    wall_seconds       REAL,
    error              TEXT,
    timestamp          TEXT NOT NULL
);
"""


def now():
    return datetime.now().isoformat(sep=" ", timespec="milliseconds")


def connect(db_path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS leaves older databases without later columns.
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(memory_benchmarks)")}
    if "peak_commit_mb" not in columns:
        conn.execute("ALTER TABLE memory_benchmarks ADD COLUMN peak_commit_mb REAL")
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
    """Track the highest process memory while the `with` block runs.

    psutil only reports current memory, and Windows' own peak counter covers the
    whole process lifetime, so a background thread samples every 20 ms instead.
    Both counters include OpenCV's C++ buffers, which Python's tracemalloc cannot see.

    Two counters, because they diverge under memory pressure:
    - RSS (working set) is what sits in RAM. When RAM runs short Windows pages
      memory out to the pagefile, so RSS under-reports what the job needs.
    - Commit (private bytes) is what the process has allocated, in RAM or paged
      out. This is the real memory requirement. Falls back to RSS off Windows.
    """

    def __enter__(self):
        self._process = psutil.Process()
        self.peak = self.peak_commit = 0
        self._update()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _update(self):
        info = self._process.memory_info()
        self.peak = max(self.peak, info.rss)
        self.peak_commit = max(self.peak_commit, getattr(info, "private", info.rss))

    def _sample(self):
        while not self._stop.wait(MEMORY_SAMPLE_SECONDS):
            self._update()

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join()
        self._update()

    @property
    def peak_mb(self):
        return self.peak / 2**20

    @property
    def peak_commit_mb(self):
        return self.peak_commit / 2**20


def frame_from_path(path):
    """Rebuild the frame dict that prototype.find_frames produces."""
    timestamp, exposure, f_number, iso = read_exif(path)
    return {"path": path, "timestamp": timestamp, "exposure": exposure, "f_number": f_number, "iso": iso}


def tile_starts(length, tile_size, overlap):
    """Tile start offsets along one axis. Neighbours overlap by at least `overlap`;
    the last tile is pushed back to end flush with the image, so its overlap can be larger."""
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    return list(range(0, length - tile_size, stride)) + [length - tile_size]


def feather(length, start, stop, overlap):
    """1-D weights for a tile covering [start, stop): raised-cosine ramps on sides
    that have a neighbour, flat 1 on sides at the image edge."""
    weights = np.ones(stop - start, np.float32)
    ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap)).astype(np.float32)
    if start > 0:
        weights[:overlap] = ramp
    if stop < length:
        weights[-overlap:] = np.minimum(weights[-overlap:], ramp[::-1])
    return weights


def guide_crop(guide, x, y, tile_w, tile_h, small_w, small_h, width, height):
    """Resample the region of `guide` under a tile onto the tile's small (small_w x small_h) grid.

    Pixel centres are mapped exactly, so the crop lines up with cv2.resize of the
    tile even though tile edges rarely fall on whole guide pixels.
    """
    gh, gw = guide.shape[:2]
    sx, sy = tile_w / small_w * gw / width, tile_h / small_h * gh / height
    ox = (x + 0.5 * tile_w / small_w) * gw / width - 0.5
    oy = (y + 0.5 * tile_h / small_h) * gh / height - 0.5
    matrix = np.float32([[sx, 0, ox], [0, sy, oy]])
    return cv2.warpAffine(guide, matrix, (small_w, small_h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


def mertens(images, tile_size, overlap, guide_factor=GUIDE_FACTOR):
    """Mertens fusion, tile by tile when the image is larger than one tile.

    Returns (fused float32 image, number of tiles). Each fused tile has its low
    frequencies swapped for those of a downscaled whole-image fusion (the guide),
    so all tiles agree on overall tone. It is then weighted by its feather mask and
    accumulated; dividing by the summed weights makes overlaps a smooth blend
    however the tiles happen to line up.
    """
    height, width = images[0].shape[:2]
    merge = cv2.createMergeMertens()
    if tile_size == 0 or (height <= tile_size and width <= tile_size):
        return merge.process(images), 1

    small = [cv2.resize(img, None, fx=1 / guide_factor, fy=1 / guide_factor, interpolation=cv2.INTER_AREA)
             for img in images]
    guide = merge.process(small)
    del small

    accum = np.zeros((height, width, 3), np.float32)
    weight_sum = np.zeros((height, width), np.float32)
    ys, xs = tile_starts(height, tile_size, overlap), tile_starts(width, tile_size, overlap)
    for y in ys:
        y_end = min(y + tile_size, height)
        wy = feather(height, y, y_end, overlap)
        for x in xs:
            x_end = min(x + tile_size, width)
            weight = np.outer(wy, feather(width, x, x_end, overlap))
            # Contiguous copies: OpenCV rejects or silently copies strided views anyway.
            tiles = [np.ascontiguousarray(img[y:y_end, x:x_end]) for img in images]
            fused = merge.process(tiles)
            del tiles  # only this tile's data lives beyond this point, and not for long
            tile_w, tile_h = x_end - x, y_end - y
            small_w, small_h = -(-tile_w // guide_factor), -(-tile_h // guide_factor)  # ceil
            fused_small = cv2.resize(fused, (small_w, small_h), interpolation=cv2.INTER_AREA)
            correction = guide_crop(guide, x, y, tile_w, tile_h, small_w, small_h, width, height) - fused_small
            # Cubic keeps the 8x-upsampled correction smooth (linear has slope kinks every 8 px).
            fused += cv2.resize(correction, (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
            accum[y:y_end, x:x_end] += fused * weight[:, :, None]
            weight_sum[y:y_end, x:x_end] += weight
            del fused
    accum /= weight_sum[:, :, None]
    return accum, len(ys) * len(xs)


def fuse_bracket_tiled(bracket, out_path, half_size, do_align, clahe_clip, saturation, tile_size, overlap):
    """prototype.fuse_bracket with tiled Mertens; every other step is the same and runs on the whole image.

    Alignment, perspective and CLAHE are global operations, so they must never run
    per tile. Returns (perspective note, (width, height), number of tiles).
    Stopgap: keep in step with prototype.fuse_bracket until lumamerge replaces both.
    """
    images = [load_image(f["path"], half_size) for f in bracket]
    if len({img.shape for img in images}) != 1:
        raise ValueError("frames have different sizes")
    if do_align:
        images = align(images)
    height, width = images[0].shape[:2]
    fused, tiles = mertens(images, tile_size, overlap)
    del images
    out = np.clip(fused * 255, 0, 255).astype(np.uint8)
    del fused
    out, diag = correct_perspective_with_diagnostics(out)
    note = (f"perspective {diag['correction_deg']:.1f} deg"
            if diag["applied"] else f"no perspective ({diag['reason']})")
    out = enhance(out, clahe_clip, saturation)
    if not cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"could not write {out_path}")
    return note, (width, height), tiles


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
            # Jobs queued before tiling existed have no tile settings: fuse them whole, as queued.
            _, _, tiles = fuse_bracket_tiled(bracket, output_path, settings["half_size"], settings["align"],
                                             settings["clahe_clip"], settings["saturation"],
                                             settings.get("tile_size", 0), settings.get("overlap", OVERLAP))
    except Exception as e:
        wall = time.perf_counter() - start
        peak = memory.peak_mb if hasattr(memory, "peak") else None
        set_status(conn, job_id, "failed", error=str(e), wall_seconds=wall, peak_memory_mb=peak)
        print(f"[job {job_id}] FAILED after {wall:.1f}s: {e}")
        return

    wall = time.perf_counter() - start
    set_status(conn, job_id, "complete", error=None, wall_seconds=wall, peak_memory_mb=memory.peak_mb)
    print(f"[job {job_id}] complete -> {output_path.name} "
          f"({wall:.1f}s, peak {memory.peak_mb:.0f} MB, {tiles} tile(s))")


def process_pending(conn):
    while True:
        job = conn.execute("SELECT * FROM jobs WHERE status = 'pending' ORDER BY job_id LIMIT 1").fetchone()
        if job is None:
            break
        run_job(conn, job)


def benchmark_child(files, out_path, settings, tile_size, results):
    """Runs in a fresh process, so the memory peaks belong to this one fusion alone."""
    try:
        bracket = [frame_from_path(Path(p)) for p in files]
        baseline = psutil.Process().memory_info().rss / 2**20
        start = time.perf_counter()
        with PeakMemory() as memory:
            _, (width, height), tiles = fuse_bracket_tiled(
                bracket, out_path, settings["half_size"], settings["align"],
                settings["clahe_clip"], settings["saturation"], tile_size, settings["overlap"])
        results.put({"tiles": tiles, "width": width, "height": height, "baseline_memory_mb": baseline,
                     "peak_memory_mb": memory.peak_mb, "peak_commit_mb": memory.peak_commit_mb,
                     "wall_seconds": time.perf_counter() - start})
    except Exception as e:
        results.put({"error": str(e)})


def compare_tiling(conn, input_dir, output_dir, settings):
    """Fuse the folder's first bracket whole and tiled, and record peak memory for both."""
    frames = find_frames(input_dir)
    brackets, _ = group_brackets(frames) if frames else ([], [])
    if not brackets:
        print(f"no complete bracket in {input_dir}")
        return
    bracket = brackets[0]
    if all(brightness(f) is not None for f in bracket):
        bracket = sorted(bracket, key=brightness)
    files = [str(f["path"].resolve()) for f in bracket]
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = bracket[0]["path"].stem
    print(f"comparing memory on: {' '.join(Path(p).name for p in files)}")

    context = multiprocessing.get_context("spawn")
    outputs = {}
    for mode, tile_size in (("whole", 0), ("tiled", settings["tile_size"])):
        out_path = output_dir / f"{stem}_{mode}.jpg"
        results = context.Queue()
        child = context.Process(target=benchmark_child, args=(files, str(out_path), settings, tile_size, results))
        child.start()
        child.join()
        try:
            result = results.get(timeout=5)
        except queue.Empty:
            # Killed without reporting, e.g. the OS ended it for running out of memory.
            result = {"error": f"process exited with code {child.exitcode} without a result"}
        with conn:
            conn.execute(
                "INSERT INTO memory_benchmarks (bracket_files, mode, tile_size, overlap, tiles, width, height, "
                "baseline_memory_mb, peak_memory_mb, peak_commit_mb, wall_seconds, error, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (json.dumps(files), mode, tile_size, settings["overlap"], result.get("tiles"), result.get("width"),
                 result.get("height"), result.get("baseline_memory_mb"), result.get("peak_memory_mb"),
                 result.get("peak_commit_mb"), result.get("wall_seconds"), result.get("error"), now()))
        if "error" in result:
            print(f"  {mode:<5} FAILED: {result['error']}")
            continue
        outputs[mode] = out_path
        print(f"  {mode:<5} {result['width']}x{result['height']}, {result['tiles']} tile(s): "
              f"peak commit {result['peak_commit_mb']:.0f} MB, peak RSS {result['peak_memory_mb']:.0f} MB, "
              f"{result['wall_seconds']:.1f}s -> {out_path.name}")

    if len(outputs) == 2:
        # Both are 8-bit JPEGs, so compression noise alone gives a difference of about 1.
        whole, tiled = (cv2.imread(str(outputs[m])).astype(np.int16) for m in ("whole", "tiled"))
        diff = np.abs(whole - tiled)
        print(f"  whole vs tiled pixel difference: mean {diff.mean():.2f}, "
              f"99th percentile {np.percentile(diff, 99):.0f}, max {diff.max()} (0-255)")


def print_history(conn):
    print("jobs:")
    for r in conn.execute("SELECT job_id, status, wall_seconds, peak_memory_mb, error FROM jobs ORDER BY job_id"):
        stats = f"{r['wall_seconds']:.1f}s, {r['peak_memory_mb']:.0f} MB" if r["wall_seconds"] is not None else ""
        print(f"  {r['job_id']:>4}  {r['status']:<9} {stats}  {r['error'] or ''}".rstrip())
    print("runs:")
    for r in conn.execute("SELECT timestamp, job_id, old_status, new_status FROM runs ORDER BY run_id"):
        print(f"  {r['timestamp']}  job {r['job_id']:>4}  {r['old_status'] or '(new)':<9} -> {r['new_status']}")
    print("memory benchmarks:")
    for r in conn.execute("SELECT * FROM memory_benchmarks ORDER BY bench_id"):
        name = Path(json.loads(r["bracket_files"])[0]).name
        if r["error"]:
            print(f"  {r['timestamp']}  {name}  {r['mode']:<5} FAILED: {r['error']}")
        else:
            print(f"  {r['timestamp']}  {name}  {r['mode']:<5} {r['width']}x{r['height']}  "
                  f"tile {r['tile_size']}/{r['overlap']}, {r['tiles']} tile(s)  "
                  f"commit {r['peak_commit_mb'] or 0:.0f} MB  RSS {r['peak_memory_mb']:.0f} MB  "
                  f"{r['wall_seconds']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Queue and run bracket-fusion jobs from SQLite.")
    parser.add_argument("input_dir", type=Path, nargs="?", help="folder to queue (omit to only resume)")
    parser.add_argument("output_dir", type=Path, nargs="?")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--half-size", action="store_true")
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP)
    parser.add_argument("--saturation", type=float, default=SATURATION)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE,
                        help=f"tile edge in pixels, 0 to fuse whole (default {TILE_SIZE})")
    parser.add_argument("--overlap", type=int, default=OVERLAP, help=f"tile overlap in pixels (default {OVERLAP})")
    parser.add_argument("--retry-failed", action="store_true", help="requeue failed jobs")
    parser.add_argument("--history", action="store_true", help="print jobs and status transitions, then exit")
    parser.add_argument("--compare-tiling", action="store_true",
                        help="fuse the first bracket in input_dir whole and tiled, record peak memory, then exit")
    args = parser.parse_args()
    if (args.input_dir is None) != (args.output_dir is None):
        parser.error("give both input_dir and output_dir, or neither")
    if args.input_dir is not None and not args.input_dir.is_dir():
        parser.error(f"{args.input_dir} is not a folder")
    if args.clahe_clip < 0 or args.saturation < 0:
        parser.error("--clahe-clip and --saturation must not be negative")
    if args.overlap < 1:
        parser.error("--overlap must be at least 1")
    if args.tile_size != 0 and args.tile_size < 2 * args.overlap:
        # Otherwise a tile's two feather ramps would overlap each other.
        parser.error("--tile-size must be 0 or at least twice --overlap")
    if args.compare_tiling and (args.input_dir is None or args.tile_size == 0):
        parser.error("--compare-tiling needs input_dir, output_dir and a non-zero --tile-size")

    conn = connect(args.db)
    if args.history:
        print_history(conn)
        return

    settings = {"half_size": args.half_size, "align": not args.no_align,
                "clahe_clip": args.clahe_clip, "saturation": args.saturation,
                "tile_size": args.tile_size, "overlap": args.overlap}
    if args.compare_tiling:
        compare_tiling(conn, args.input_dir, args.output_dir, settings)
        return

    recover_interrupted(conn)
    if args.retry_failed:
        retry_failed(conn)
    if args.input_dir is not None:
        enqueue_folder(conn, args.input_dir, args.output_dir, settings)
    process_pending(conn)

    counts = dict(conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall())
    print("database: " + ", ".join(f"{counts.get(s, 0)} {s}" for s in ("complete", "failed", "pending", "running")))


if __name__ == "__main__":
    main()
