"""Render deterministic held-out SkyFinder examples for AeroSwap error review.

One chronological midpoint is chosen from each test camera's day and night
frames. Selection never uses predictions or IoU, so examples are not chosen to
make the model look especially good or bad. The resulting image is for visual
diagnosis; the full-split metrics are produced by eval.aeroswap_eval.

Run in the PyTorch environment:
    python -m eval.aeroswap_examples --checkpoint models/checkpoints/aeroswap_skyfinder_v1.ts
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from models.aeroswap_data import SkyFinderDataset


def choose_examples(rows):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        condition = "night" if row["night"] == "1" else "day"
        groups[(row["camera"], condition)].append(index)
    return [(camera, condition, sorted(indices, key=lambda i: rows[i]["image"])[len(indices) // 2])
            for (camera, condition), indices in sorted(
                groups.items(), key=lambda item: (int(item[0][0]), item[0][1]))]


def image_miou(pred, true):
    sky_union = np.logical_or(pred, true).sum()
    bg_union = np.logical_or(~pred, ~true).sum()
    sky_iou = np.logical_and(pred, true).sum() / sky_union if sky_union else float("nan")
    bg_iou = np.logical_and(~pred, ~true).sum() / bg_union if bg_union else float("nan")
    return float((sky_iou + bg_iou) / 2)


def panel(image, mask=None, error=None, width=320):
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        image = image.copy()
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    if error is not None:
        false_sky, missed_sky = error
        overlay = image.copy()
        overlay[false_sky] = (0, 0, 255)  # red: replaced non-sky pixels
        overlay[missed_sky] = (255, 0, 0)  # blue: missed sky pixels
        image = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    return cv2.resize(image, (width, width), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/skyfinder/processed"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/aeroswap/examples.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    if args.width < 64:
        parser.error("--width must be at least 64")

    cv2.setNumThreads(1)
    dataset = SkyFinderDataset(args.root, "test")
    model = torch.jit.load(str(args.checkpoint), map_location=args.device).eval()
    rows = []
    metadata = []
    for camera, condition, index in choose_examples(dataset.rows):
        image, target = dataset[index]
        with torch.inference_mode():
            probability = model(image[None].to(args.device))[0, 0].cpu().numpy()
        predicted = probability > 0.5
        true = target[0].numpy() > 0.5
        bgr = cv2.cvtColor((image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        pictures = [panel(bgr, width=args.width),
                    panel(bgr, mask=true, width=args.width),
                    panel(bgr, mask=predicted, width=args.width),
                    panel(bgr, error=(predicted & ~true, ~predicted & true), width=args.width)]
        score = image_miou(predicted, true)
        metadata.append({"camera": camera, "condition": condition,
                         "image": dataset.rows[index]["image"], "miou": score})
        caption = f"Camera {camera} | {condition} | image mIoU {score:.3f}"
        strip = np.full((38, args.width * 4, 3), 32, dtype=np.uint8)
        cv2.putText(strip, caption, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.vstack((strip, np.hstack(pictures))))

    heading = np.full((52, args.width * 4, 3), 18, dtype=np.uint8)
    for col, label in enumerate(("Original", "Reference edge", "Predicted edge",
                                  "Errors: red=FP, blue=FN")):
        cv2.putText(heading, label, (col * args.width + 8, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    output = np.vstack((heading, *rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), output):
        raise OSError(f"could not write {args.out}")
    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} and {metadata_path}; {len(metadata)} examples")


if __name__ == "__main__":
    main()
