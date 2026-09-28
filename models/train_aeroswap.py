"""Train AeroSwap on camera-separated SkyFinder data and export TorchScript.

The test split is never used for training or checkpoint selection. A best model
and a resumable last checkpoint are written after each epoch.

Usage:
    python -m models.train_aeroswap --epochs 10 --batch-size 2
    python -m models.train_aeroswap --resume data/aeroswap/last.pt
"""

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from models.aeroswap import AeroSwapNet
from models.aeroswap_data import SkyFinderDataset


class TrainTransform:
    def __call__(self, image, mask):
        if torch.rand(()) < 0.5:
            image = image.flip(-1)
            mask = mask.flip(-1)
        # SkyFinder spans seasons and times of day. Small exposure changes make
        # the model less dependent on sky brightness without changing the mask.
        if torch.rand(()) < 0.5:
            gain = 0.8 + 0.4 * torch.rand(())
            bias = -0.05 + 0.1 * torch.rand(())
            image = (image * gain + bias).clamp(0, 1)
        return image, mask


def sampled_indices(rows, per_camera, seed):
    if per_camera <= 0:
        return list(range(len(rows)))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["camera"]].append(index)
    rng = random.Random(seed)
    chosen = []
    for camera in sorted(groups, key=int):
        indices = groups[camera]
        chosen.extend(rng.sample(indices, min(per_camera, len(indices))))
    return sorted(chosen)


def segmentation_loss(probability, target):
    bce = F.binary_cross_entropy(probability, target)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    total = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1) / (total + 1)
    return bce + (1 - dice).mean()


def pooled_miou(model, loader, device):
    model.eval()
    inter_sky = union_sky = inter_bg = union_bg = 0
    loss_sum = samples = 0
    with torch.inference_mode():
        for image, target in loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            probability = model(image)
            loss_sum += segmentation_loss(probability, target).item() * image.shape[0]
            samples += image.shape[0]
            pred = probability > 0.5
            true = target > 0.5
            inter_sky += int((pred & true).sum())
            union_sky += int((pred | true).sum())
            inter_bg += int((~pred & ~true).sum())
            union_bg += int((~pred | ~true).sum())
    sky = inter_sky / union_sky if union_sky else 0.0
    bg = inter_bg / union_bg if union_bg else 0.0
    return {"loss": loss_sum / samples, "miou": (sky + bg) / 2,
            "sky_iou": sky, "bg_iou": bg, "images": samples}


def save_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def export_script(model, path):
    model.eval().cpu()
    script = torch.jit.script(model)
    temporary = path.with_name(path.name + ".tmp")
    script.save(str(temporary))
    temporary.replace(path)


def train(args):
    if args.batch_size < 1 or args.accumulate < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("batch size, accumulation, epochs, and patience must be positive")
    if args.max_steps < 0 or args.train_per_camera < 0 or args.val_per_camera < 0:
        raise ValueError("max steps and per-camera limits cannot be negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(1)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu")
    use_amp = device.type == "cuda"
    print(f"device {device}; PyTorch {torch.__version__}; AMP {use_amp}", flush=True)

    train_data = SkyFinderDataset(args.root, "train", transform=TrainTransform())
    val_data = SkyFinderDataset(args.root, "val")
    train_ids = sampled_indices(train_data.rows, args.train_per_camera, args.seed)
    val_ids = sampled_indices(val_data.rows, args.val_per_camera, args.seed)
    if not train_ids or not val_ids:
        raise SystemExit("train and val must both contain images")
    train_loader = DataLoader(Subset(train_data, train_ids), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=use_amp)
    val_loader = DataLoader(Subset(val_data, val_ids), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=use_amp)
    print(f"train {len(train_ids)} images; val {len(val_ids)} images; "
          f"test untouched", flush=True)

    model = AeroSwapNet(pretrained=args.resume is None and not args.no_pretrained).to(device)
    encoder = list(model.enc2.parameters()) + list(model.enc4.parameters()) + list(model.enc8.parameters()) \
              + list(model.enc16.parameters()) + list(model.enc32.parameters())
    decoder = list(model.up16.parameters()) + list(model.up8.parameters()) \
              + list(model.up4.parameters()) + list(model.up2.parameters()) \
              + list(model.final.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": args.lr * 0.1},
        {"params": decoder, "lr": args.lr},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_miou, stale = 1, -1.0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_miou = state["best_miou"]
        stale = state["stale"]
        print(f"resumed after epoch {state['epoch']}; best val mIoU {best_miou:.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "training.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = seen = 0
        optimizer.zero_grad(set_to_none=True)
        steps_this_epoch = min(len(train_loader), args.max_steps) if args.max_steps else len(train_loader)
        for step, (image, target) in enumerate(train_loader, 1):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = segmentation_loss(model(image), target)
            window_start = ((step - 1) // args.accumulate) * args.accumulate
            window_size = min(args.accumulate, steps_this_epoch - window_start)
            scaler.scale(loss / window_size).backward()
            if step % args.accumulate == 0 or step == steps_this_epoch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * image.shape[0]
            seen += image.shape[0]
            if args.max_steps and step >= args.max_steps:
                break
        val = pooled_miou(model, val_loader, device)
        improved = val["miou"] > best_miou
        best_miou = max(best_miou, val["miou"])
        stale = 0 if improved else stale + 1
        entry = {"epoch": epoch, "train_loss": loss_sum / seen, "val": val,
                 "seconds": time.perf_counter() - start, "best_miou": best_miou,
                 "train_images": seen}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")
        save_atomic({"epoch": epoch, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "best_miou": best_miou, "stale": stale,
                     "train_images": len(train_ids), "val_images": len(val_ids)},
                    args.out / "last.pt")
        if improved:
            save_atomic({"epoch": epoch, "model": model.state_dict(),
                         "val": val, "train_images": len(train_ids),
                         "val_images": len(val_ids)}, args.out / "best.pt")
            export_script(model, args.out / "aeroswap.ts")
            model.to(device)
        print(f"epoch {epoch}: train loss {entry['train_loss']:.4f}; "
              f"val mIoU {val['miou']:.4f} (sky {val['sky_iou']:.4f}, "
              f"bg {val['bg_iou']:.4f}); {entry['seconds']:.0f}s" +
              (" BEST" if improved else ""), flush=True)
        if stale >= args.patience:
            print(f"early stop: no improvement for {stale} epochs", flush=True)
            break
    print(f"best val mIoU {best_miou:.4f}; TorchScript {args.out / 'aeroswap.ts'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/skyfinder/processed"))
    parser.add_argument("--out", type=Path, default=Path("data/aeroswap"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-camera", type=int, default=250,
                        help="maximum training frames per camera; 0 uses all")
    parser.add_argument("--val-per-camera", type=int, default=250,
                        help="maximum validation frames per camera; 0 uses all")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="limit train batches per epoch for a smoke test")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="initialize encoder randomly (for offline smoke tests)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
