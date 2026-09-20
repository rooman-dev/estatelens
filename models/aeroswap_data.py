"""SkyFinder data prep for binary sky segmentation.

SkyFinder (Mihail et al.) is ~90k images from 53 fixed AMOS webcams, with one
hand-labelled sky mask per camera. Because every image from a camera shares
the same mask, the train/val/test split is made over cameras, never images:
a random image split would let a model memorise masks and inflate test scores.

Pipeline:
    select    -> data/skyfinder/subset.json  (cameras picked for varied conditions)
    download  -> data/skyfinder/raw/{<camera>.zip, skyfinder_masks.zip}
    prepare   -> data/skyfinder/processed/<split>/{images,masks}/...
                 data/skyfinder/processed/splits.json
                 data/skyfinder/processed/<split>.csv

Usage:
    python -m models.aeroswap_data --select 15
    python -m models.aeroswap_data --subset --download --prepare
    python -m models.aeroswap_data --download --cameras 858 3888 --prepare

--prepare works on a partial download: missing camera zips are skipped with a
warning. With --subset the split is made over the intended cameras, so it does
not change as more zips arrive.

Night filter: SkyFinder file names carry the capture time (YYYYMMDD_HHMMSS in
local time), so night is decided by the clock, not by brightness. Frames outside
--day-hours are dropped from train and val. Test keeps every frame: some cameras
have a bright hazy night sky under light pollution (camera 9708 is the clear
case), which is a real condition a sky model has to face, so it belongs in the
evaluation rather than being filtered out of it. Every CSV row carries `hour`
and `night`, so eval can report day and night separately.

The older brightness filter (--night-threshold) is off by default. It judged
mean brightness inside the sky region, which misses exactly the light-polluted
cameras: 9708's night frames average 122 there, well above any usable threshold.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ZENODO_RECORD = "https://zenodo.org/api/records/5884485"
METADATA_URL = "https://cs.valdosta.edu/~rpmihail/skyfinder/analysis/complete_table_with_mcr.csv"
MASKS_ZIP = "skyfinder_masks.zip"
METADATA_CSV = "complete_table_with_mcr.csv"
SUBSET_JSON = "subset.json"
SIZE = 512
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# AMOS file names: <camera>/YYYYMMDD_HHMMSS.jpg, local time at the camera.
TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{2})(\d{2})(\d{2})")
DAY_HOURS = (7, 18)          # kept when DAY_HOURS[0] <= hour < DAY_HOURS[1]
FILTERED_SPLITS = ("train", "val")


# ---------------------------------------------------------------- download

def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_remote_files():
    """Return {filename: (url, md5, size_bytes)} for every zip in the Zenodo record."""
    with urllib.request.urlopen(ZENODO_RECORD, timeout=60) as r:
        record = json.load(r)
    files = {}
    for f in record["files"]:
        if f["key"].endswith(".zip"):
            files[f["key"]] = (f["links"]["self"], f["checksum"].removeprefix("md5:"), f["size"])
    return files


def _fetch(url, dest):
    """Download url to dest via a .part file, so a crash never leaves a truncated file."""
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    part.replace(dest)


# ------------------------------------------------------------ subset select

# Per-camera profile used to judge how different two cameras are. Transient
# attributes are 0-1 scores per image (Laffont et al.); weather flags are 0/1
# from the nearest weather station. Each becomes a per-camera mean.
PROFILE_ATTRS = ["night", "dawndusk", "sunny", "clouds", "fog", "storm", "snow"]
PROFILE_FLAGS = ["Fog", "Rain", "Snow"]
MIN_IMAGES = 200


def camera_profiles(metadata_csv):
    """Return {camera_id: {"n": count, "lat": abs latitude, <feature>: mean}}."""
    sums, counts, lat = {}, {}, {}
    keys = PROFILE_ATTRS + PROFILE_FLAGS
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            cam = row["CamId"]
            s = sums.setdefault(cam, dict.fromkeys(keys, 0.0))
            for k in keys:
                try:
                    v = float(row[k])
                except ValueError:
                    continue
                if v >= 0:  # the table uses -999 / -9999 for missing
                    s[k] += v
            counts[cam] = counts.get(cam, 0) + 1
            lat.setdefault(cam, abs(float(row["Latitude"])))
    return {cam: {"n": counts[cam], "lat": lat[cam],
                  **{k: v / counts[cam] for k, v in s.items()}}
            for cam, s in sums.items()}


def select_cameras(profiles, masks, n):
    """Pick n cameras that cover the range of conditions.

    Features are standardised so no single one dominates. Start from the most
    typical camera, then repeatedly add the camera farthest from everything
    already chosen (farthest-point sampling).
    """
    usable = sorted(
        (c for c, p in profiles.items()
         if p["n"] >= MIN_IMAGES and c in masks and masks[c].any() and not masks[c].all()),
        key=int,
    )
    if len(usable) < n:
        sys.exit(f"only {len(usable)} usable cameras, asked for {n}")
    feats = PROFILE_ATTRS + PROFILE_FLAGS + ["lat"]
    x = np.array([[profiles[c][k] for k in feats] for c in usable])
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)

    chosen = [int(np.argmin(np.linalg.norm(x, axis=1)))]
    dist = np.linalg.norm(x - x[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(x - x[nxt], axis=1))
    return sorted((usable[i] for i in chosen), key=int)


def select(root, n):
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    remote = list_remote_files()
    masks_path = raw_dir / MASKS_ZIP
    if not (masks_path.exists() and _md5(masks_path) == remote[MASKS_ZIP][1]):
        print(f"downloading {MASKS_ZIP}")
        _fetch(remote[MASKS_ZIP][0], masks_path)
    meta_path = root / METADATA_CSV
    if not meta_path.exists():
        print(f"downloading {METADATA_CSV} (~92 MB)")
        _fetch(METADATA_URL, meta_path)

    profiles = camera_profiles(meta_path)
    cameras = select_cameras(profiles, load_masks(raw_dir), n)
    (root / SUBSET_JSON).write_text(json.dumps({"cameras": cameras}, indent=2))

    print(f"\n{n} cameras -> {root / SUBSET_JSON}\n")
    print(f"{'cam':>6} {'images':>6} {'MB':>6} {'lat':>5}  "
          + " ".join(f"{k[:6]:>6}" for k in PROFILE_ATTRS + PROFILE_FLAGS))
    total = 0
    for cam in cameras:
        p, size = profiles[cam], remote[f"{cam}.zip"][2]
        total += size
        print(f"{cam:>6} {p['n']:>6} {size / 1e6:>6.0f} {p['lat']:>5.1f}  "
              + " ".join(f"{p[k]:>6.2f}" for k in PROFILE_ATTRS + PROFILE_FLAGS))
    print(f"\ntotal {total / 1e9:.2f} GB. Manual download URLs (save into {raw_dir}):")
    for cam in cameras:
        print(f"  {remote[f'{cam}.zip'][0]}")


def download(raw_dir, cameras=None):
    """Download camera zips (all, or only `cameras`) plus the masks zip.

    Files already present with a matching MD5 are skipped, so this can be
    interrupted and rerun. Downloads go to a .part file first so a crash never
    leaves a truncated zip that looks complete.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    remote = list_remote_files()
    wanted = [MASKS_ZIP] + sorted(
        (k for k in remote if k != MASKS_ZIP
         and (cameras is None or Path(k).stem in cameras)),
        key=lambda k: int(Path(k).stem),
    )
    if cameras:
        missing = set(cameras) - {Path(k).stem for k in wanted}
        if missing:
            sys.exit(f"cameras not in SkyFinder: {sorted(missing)}")

    for i, name in enumerate(wanted, 1):
        url, md5, _ = remote[name]
        dest = raw_dir / name
        if dest.exists() and _md5(dest) == md5:
            print(f"[{i}/{len(wanted)}] {name}: ok, skipping", flush=True)
            continue
        print(f"[{i}/{len(wanted)}] {name}: downloading", flush=True)
        _fetch(url, dest)
        got = _md5(dest)
        if got != md5:
            dest.unlink()
            sys.exit(f"{name}: MD5 mismatch (expected {md5}, got {got})")


# ------------------------------------------------------------------ prepare

def load_masks(raw_dir):
    """Return {camera_id: bool mask, True = sky} at native resolution."""
    masks = {}
    with zipfile.ZipFile(raw_dir / MASKS_ZIP) as z:
        for name in z.namelist():
            p = Path(name)
            if p.suffix.lower() not in IMAGE_EXTS or not p.stem.isdigit():
                continue
            m = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                sys.exit(f"unreadable mask: {name}")
            masks[p.stem] = m > 127
    if not masks:
        sys.exit(f"no masks found in {MASKS_ZIP}")
    return masks


def split_cameras(camera_ids, seed, val_frac, test_frac):
    """Shuffle camera IDs and cut them into train/val/test. No camera spans two splits."""
    ids = sorted(camera_ids, key=int)
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(len(ids) * test_frac))
    n_val = max(1, round(len(ids) * val_frac))
    splits = {
        "test": sorted(ids[:n_test], key=int),
        "val": sorted(ids[n_test:n_test + n_val], key=int),
        "train": sorted(ids[n_test + n_val:], key=int),
    }
    seen = [c for cams in splits.values() for c in cams]
    assert len(seen) == len(set(seen)) == len(ids), "camera leaked across splits"
    return splits


def frame_hour(name):
    """Local hour of capture from an AMOS file name, or None if it has no timestamp."""
    m = TIMESTAMP_RE.search(Path(name).stem)
    return int(m.group(2)) if m else None


def is_night(hour, day_hours):
    """True when `hour` falls outside the daylight window. Unknown hour is never night."""
    if hour is None or not day_hours:
        return False
    return not (day_hours[0] <= hour < day_hours[1])


def prepare(raw_dir, out_dir, cameras=None, seed=0, val_frac=0.15, test_frac=0.15,
            night_threshold=0.0, max_per_camera=0, day_hours=DAY_HOURS):
    """Build the processed dataset.

    `cameras` is the intended camera list (e.g. the subset). The split is made
    over that list, so it stays fixed while zips are still arriving; cameras
    whose zip is missing just contribute no images yet. Without it, the split
    is over whatever zips are present.
    """
    masks = load_masks(raw_dir)
    zips = {p.stem: p for p in raw_dir.glob("*.zip") if p.stem.isdigit()}
    if cameras is None:
        cameras = sorted(zips, key=int)
    no_mask = [c for c in cameras if c not in masks]
    for cam in no_mask:
        print(f"warning: camera {cam} has no mask, skipped")
    cameras = [c for c in cameras if c in masks]
    if len(cameras) < 3:
        sys.exit(f"need at least 3 cameras with masks to split, found {len(cameras)}")
    missing = [c for c in cameras if c not in zips]
    if missing:
        print(f"warning: {len(missing)}/{len(cameras)} camera zips not downloaded yet: "
              f"{' '.join(missing)}")

    splits = split_cameras(cameras, seed, val_frac, test_frac)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "splits.json").write_text(json.dumps(
        {"seed": seed, "size": SIZE, "night_threshold": night_threshold,
         "day_hours": list(day_hours) if day_hours else None,
         "night_filtered_splits": list(FILTERED_SPLITS),
         "max_per_camera": max_per_camera, "missing": missing, **splits}, indent=2))

    for split, cams in splits.items():
        img_dir = out_dir / split / "images"
        mask_dir = out_dir / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        # Test keeps night frames: a lit, hazy night sky is a condition the model
        # must be measured on, not one the dataset hides.
        split_day_hours = day_hours if split in FILTERED_SPLITS else None
        rows = []
        for cam in cams:
            if cam not in zips:
                continue
            rows += _prepare_camera(cam, zips[cam], masks[cam], img_dir, mask_dir,
                                    night_threshold, max_per_camera, seed, split_day_hours)
        with open(out_dir / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["camera", "image", "mask", "hour", "night"])
            w.writerows(rows)
        n_night = sum(1 for r in rows if r[4] == 1)
        kept_night = "night kept" if split_day_hours is None else "night dropped"
        print(f"{split}: {len(cams)} cameras, {len(rows)} images "
              f"({n_night} night, {kept_night})")


def _prepare_camera(cam, zip_path, mask, img_dir, mask_dir,
                    night_threshold, max_per_camera, seed, day_hours):
    h, w = mask.shape
    if mask.all() or not mask.any():
        print(f"warning: camera {cam} mask is all one class")
    # Nearest-neighbour keeps the mask strictly 0/255 after resizing.
    mask_small = cv2.resize(mask.astype(np.uint8) * 255, (SIZE, SIZE),
                            interpolation=cv2.INTER_NEAREST)
    mask_name = f"{cam}.png"
    cv2.imwrite(str(mask_dir / mask_name), mask_small)

    kept = dropped_bad = dropped_night = dropped_dark = no_timestamp = kept_night = 0
    rows = []
    try:
        z = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        print(f"warning: camera {cam}: {zip_path.name} is corrupt or incomplete, skipped")
        return rows
    with z:
        names = sorted(n for n in z.namelist() if Path(n).suffix.lower() in IMAGE_EXTS)
        if max_per_camera and len(names) > max_per_camera:
            names = sorted(random.Random(f"{seed}-{cam}").sample(names, max_per_camera))
        for name in names:
            hour = frame_hour(name)
            if hour is None:
                no_timestamp += 1
            night = is_night(hour, day_hours) if day_hours else is_night(hour, DAY_HOURS)
            if day_hours and night:
                dropped_night += 1
                continue
            img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (h, w):
                dropped_bad += 1
                continue
            # Night filter: judge brightness inside the sky region only, since
            # lit buildings can make a night frame look bright overall.
            if night_threshold > 0:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if gray[mask].mean() < night_threshold:
                    dropped_dark += 1
                    continue
            # Squash to 512x512 (aspect ratio not preserved). INTER_AREA is the
            # right filter for downscaling.
            small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            out_name = f"{cam}_{Path(name).stem}.jpg"
            cv2.imwrite(str(img_dir / out_name), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append([cam, out_name, mask_name, "" if hour is None else hour, int(night)])
            kept += 1
            kept_night += night
    extra = f", no timestamp {no_timestamp}" if no_timestamp else ""
    if day_hours:
        print(f"  camera {cam}: kept {kept}, unreadable/wrong size {dropped_bad}, "
              f"night dropped {dropped_night}, too dark {dropped_dark}{extra}")
    else:
        print(f"  camera {cam}: kept {kept} ({kept_night} night, kept on purpose), "
              f"unreadable/wrong size {dropped_bad}, too dark {dropped_dark}{extra}")
    return rows


# ------------------------------------------------------------------ dataset

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # data prep itself does not need torch
    torch = None
    Dataset = object


class SkyFinderDataset(Dataset):
    """Prepared SkyFinder split.

    Returns (image, mask):
        image: float32 tensor [3, 512, 512], RGB, range 0-1
        mask:  float32 tensor [1, 512, 512], 1 = sky, 0 = not sky

    `transform`, if given, is called as transform(image, mask) on the tensors
    and must return the pair; use it for augmentation later.
    """

    def __init__(self, root, split, transform=None):
        if torch is None:
            raise ImportError("SkyFinderDataset requires torch")
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split: {split}")
        self.dir = Path(root) / split
        with open(Path(root) / f"{split}.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform
        self._masks = {}  # one mask per camera; cache instead of rereading

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = cv2.imread(str(self.dir / "images" / row["image"]), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"unreadable image: {row['image']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img).permute(2, 0, 1).float().div_(255)

        if row["mask"] not in self._masks:
            m = cv2.imread(str(self.dir / "masks" / row["mask"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise OSError(f"unreadable mask: {row['mask']}")
            self._masks[row["mask"]] = torch.from_numpy(m > 127).float().unsqueeze(0)
        mask = self._masks[row["mask"]].clone()

        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask


# ---------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("data/skyfinder"))
    ap.add_argument("--select", type=int, metavar="N",
                    help="pick N cameras with varied conditions, save subset.json, print URLs")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--subset", action="store_true",
                    help="use the cameras in subset.json for --download and --prepare")
    ap.add_argument("--cameras", nargs="+", help="use these camera IDs for --download and --prepare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--night-threshold", type=float, default=0.0,
                    help="also drop images whose mean sky brightness (0-255) is below this; "
                         "0 disables. Off by default: the clock filter replaced it")
    ap.add_argument("--day-hours", type=int, nargs=2, metavar=("START", "END"), default=list(DAY_HOURS),
                    help=f"keep frames captured at START <= hour < END (local time from the file "
                         f"name); applies to {'/'.join(FILTERED_SPLITS)} only. Pass 0 0 to disable")
    ap.add_argument("--max-per-camera", type=int, default=0,
                    help="randomly keep at most N images per camera; 0 keeps all")
    args = ap.parse_args()
    if not (args.select or args.download or args.prepare):
        ap.error("pass --select, --download and/or --prepare")
    if args.subset and args.cameras:
        ap.error("--subset and --cameras are mutually exclusive")

    raw, processed = args.root / "raw", args.root / "processed"
    if args.select:
        select(args.root, args.select)

    cameras = args.cameras
    if args.subset:
        subset_path = args.root / SUBSET_JSON
        if not subset_path.exists():
            ap.error(f"{subset_path} not found; run --select first")
        cameras = json.loads(subset_path.read_text())["cameras"]
    if args.download:
        download(raw, set(cameras) if cameras else None)
    if args.prepare:
        day_hours = tuple(args.day_hours) if args.day_hours[1] > args.day_hours[0] else None
        prepare(raw, processed, cameras, args.seed, args.val_frac, args.test_frac,
                args.night_threshold, args.max_per_camera, day_hours)


if __name__ == "__main__":
    main()
