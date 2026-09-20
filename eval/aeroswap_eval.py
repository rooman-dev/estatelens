"""Sky segmentation scoring: mIoU per camera, per condition, and pooled.

    evaluate(predict, root, split) -> Report

`predict` takes a float32 image batch [N, 3, H, W] in 0-1 (RGB) and returns
either probabilities or a 0/1 mask, shaped [N, 1, H, W] or [N, H, W]. Anything
above `threshold` counts as sky. Passing a plain function instead of a model
keeps this usable before a trained checkpoint exists (see --baseline).

Why the breakdown, not one number:

  per camera   Every image from an AMOS camera shares one hand-labelled mask and
               one viewpoint, so images within a camera are far from independent.
               The cameras also differ in size by about 4x, so a pooled number is
               really a weighted average that the biggest cameras dominate. Both
               pooled and the mean over cameras are reported; a gap between them
               means the per-camera scores are uneven.

  day / night  The test split keeps its night frames on purpose. Some cameras
               (9708 above all) have a bright hazy night sky under light
               pollution, which is a real condition and the one a sky model is
               most likely to collapse on. Pooled over everything, a collapse on
               a quarter of the frames hides inside a decent-looking average.

IoU is accumulated as intersection and union pixel counts, then divided once at
the end. That is the standard dataset-level IoU: averaging per-image IoUs would
let frames with almost no sky swing the result. mIoU is the mean of the sky and
non-sky IoUs, so predicting all-sky or all-background cannot score well.

CLI:
    python -m eval.aeroswap_eval --checkpoint sky.pt [--split test] [--out report.json]
    python -m eval.aeroswap_eval --baseline            # brightness baseline, no model
`--checkpoint` takes a TorchScript file (torch.jit.save), so this does not
depend on a model class.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.aeroswap_data import SkyFinderDataset

DEFAULT_ROOT = Path("data/skyfinder/processed")
THRESHOLD = 0.5
BATCH_SIZE = 8


@dataclass
class Counts:
    """Pixel counts for one group; IoU is computed from these, never averaged."""
    inter_sky: int = 0
    union_sky: int = 0
    inter_bg: int = 0
    union_bg: int = 0
    images: int = 0

    def add(self, pred, true):
        p, t = pred.bool(), true.bool()
        self.inter_sky += int((p & t).sum())
        self.union_sky += int((p | t).sum())
        self.inter_bg += int((~p & ~t).sum())
        self.union_bg += int((~p | ~t).sum())
        self.images += pred.shape[0]

    def metrics(self):
        sky = self.inter_sky / self.union_sky if self.union_sky else float("nan")
        bg = self.inter_bg / self.union_bg if self.union_bg else float("nan")
        return {"images": self.images, "sky_iou": sky, "bg_iou": bg,
                "miou": (sky + bg) / 2 if self.union_sky and self.union_bg else float("nan")}


@dataclass
class Report:
    split: str
    pooled: dict = field(default_factory=dict)
    mean_over_cameras: float = float("nan")
    per_camera: dict = field(default_factory=dict)      # camera -> metrics, plus day/night
    by_condition: dict = field(default_factory=dict)    # 'day'/'night' -> metrics

    def to_json(self):
        return {"split": self.split, "pooled": self.pooled,
                "mean_over_cameras_miou": self.mean_over_cameras,
                "per_camera": self.per_camera, "by_condition": self.by_condition}


def evaluate(predict, root=DEFAULT_ROOT, split="test", threshold=THRESHOLD,
             batch_size=BATCH_SIZE, device="cpu"):
    """Score `predict` on one prepared split. Returns a Report."""
    root = Path(root)
    dataset = SkyFinderDataset(root, split)
    if not len(dataset):
        raise SystemExit(f"{split} split is empty: run models.aeroswap_data --prepare first")
    # DataLoader hands back tensors only, so read the per-row metadata alongside it.
    rows = dataset.rows
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    pooled = Counts()
    per_camera = defaultdict(Counts)
    per_camera_condition = defaultdict(Counts)
    per_condition = defaultdict(Counts)

    seen = 0
    for images, masks in loader:
        images = images.to(device)
        with torch.no_grad():
            out = predict(images)
        if out.dim() == 3:
            out = out.unsqueeze(1)
        pred = (out.detach().cpu() > threshold)
        true = masks > 0.5
        for i in range(pred.shape[0]):
            row = rows[seen + i]
            cam = row["camera"]
            cond = "night" if str(row.get("night", "0")) == "1" else "day"
            p, t = pred[i:i + 1], true[i:i + 1]
            pooled.add(p, t)
            per_camera[cam].add(p, t)
            per_camera_condition[(cam, cond)].add(p, t)
            per_condition[cond].add(p, t)
        seen += pred.shape[0]

    report = Report(split=split, pooled=pooled.metrics())
    for cam, counts in sorted(per_camera.items(), key=lambda kv: int(kv[0])):
        entry = counts.metrics()
        for cond in ("day", "night"):
            c = per_camera_condition.get((cam, cond))
            if c is not None and c.images:
                entry[cond] = c.metrics()
        report.per_camera[cam] = entry
    mious = [m["miou"] for m in report.per_camera.values() if not np.isnan(m["miou"])]
    report.mean_over_cameras = float(np.mean(mious)) if mious else float("nan")
    report.by_condition = {c: per_condition[c].metrics() for c in ("day", "night") if per_condition[c].images}
    return report


def brightness_baseline(threshold=0.55):
    """Trivial 'bright pixels are sky' predictor. Not a model: a floor to compare against."""
    def predict(images):
        return images.mean(dim=1, keepdim=True) > threshold
    return predict


def load_checkpoint(path, device="cpu"):
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def print_report(report):
    p = report.pooled
    print(f"\n{report.split}: {p['images']} images, "
          f"{len(report.per_camera)} cameras")
    print(f"  pooled            mIoU {p['miou']:.4f}   sky {p['sky_iou']:.4f}  bg {p['bg_iou']:.4f}")
    print(f"  mean over cameras mIoU {report.mean_over_cameras:.4f}")
    if report.by_condition:
        for cond, m in report.by_condition.items():
            print(f"  {cond:<5} (pooled)    mIoU {m['miou']:.4f}   sky {m['sky_iou']:.4f}  "
                  f"bg {m['bg_iou']:.4f}  n={m['images']}")
    print(f"\n  {'camera':>8} {'n':>6} {'mIoU':>8} {'sky':>8} {'bg':>8}  {'day mIoU':>9} {'night mIoU':>10}")
    for cam, m in report.per_camera.items():
        day = f"{m['day']['miou']:.4f}" if "day" in m else "-"
        night = f"{m['night']['miou']:.4f}" if "night" in m else "-"
        print(f"  {cam:>8} {m['images']:>6} {m['miou']:>8.4f} {m['sky_iou']:>8.4f} "
              f"{m['bg_iou']:>8.4f}  {day:>9} {night:>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--checkpoint", type=Path, help="TorchScript model file")
    ap.add_argument("--baseline", action="store_true",
                    help="score the brightness baseline instead of a model")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, help="write the report as JSON")
    args = ap.parse_args()
    if bool(args.checkpoint) == bool(args.baseline):
        ap.error("pass either --checkpoint or --baseline")

    predict = brightness_baseline() if args.baseline else load_checkpoint(args.checkpoint, args.device)
    report = evaluate(predict, args.root, args.split, args.threshold, args.batch_size, args.device)
    print_report(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_json(), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
