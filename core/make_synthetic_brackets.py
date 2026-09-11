"""Make a fake 3-frame bracket from one RAW file, for testing core/prototype.py.

Decodes the RAW three times with different rawpy `bright` values (-2, 0, +2 EV)
and saves 16-bit TIFFs. They carry no EXIF, so the prototype groups them by
filename order.

Usage:
    python -m core.make_synthetic_brackets [raw_file] [output_dir]
"""

import sys
from pathlib import Path

import cv2
import rawpy

BRIGHTS = {"-2ev": 0.25, "0ev": 1.0, "+2ev": 4.0}


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/test.cr2")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    with rawpy.imread(str(src)) as raw:
        for n, (label, bright) in enumerate(BRIGHTS.items(), 1):
            # half_size keeps the test files small (~75 MB each instead of ~300 MB)
            rgb = raw.postprocess(bright=bright, no_auto_bright=True, use_camera_wb=True,
                                  output_bps=16, half_size=True)
            path = out_dir / f"synth_{n:03d}_{label}.tif"
            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            print(f"wrote {path} (bright={bright})")


if __name__ == "__main__":
    main()
