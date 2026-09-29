"""Read a PNG from stdin and write AeroSwap's probability PNG to stdout.

This binary pipe lets the Windows GUI call a model in WSL without translating
photo paths. Keep stdout binary-only; errors and warnings belong on stderr.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "aeroswap_skyfinder_v1.ts"
SIZE = 512


def infer(image, model, device):
    image = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
    with torch.inference_mode():
        probability = model(tensor)[0, 0].cpu().numpy()
    if probability.shape != (SIZE, SIZE) or not np.isfinite(probability).all():
        raise ValueError("model returned an invalid sky probability mask")
    return np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    encoded = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        parser.error("stdin must contain one encoded image")
    model = torch.jit.load(str(args.checkpoint), map_location=args.device).eval()
    mask = infer(image, model, args.device)
    ok, result = cv2.imencode(".png", mask)
    if not ok:
        raise OSError("could not encode sky mask")
    sys.stdout.buffer.write(result.tobytes())


if __name__ == "__main__":
    main()
