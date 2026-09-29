"""Verify the prepared SkyFinder reference split from its written files.

Usage:
    python -m eval.verify_skyfinder --out eval/skyfinder_reference/prepared_summary.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REFERENCE = Path(__file__).resolve().parent / "skyfinder_reference" / "splits.json"
SPLITS = ("train", "val", "test")


def verify(root, reference=REFERENCE):
    root = Path(root)
    expected = json.loads(Path(reference).read_text())
    prepared = json.loads((root / "splits.json").read_text())
    for key in ("seed", "size", "night_threshold", "day_hours",
                "night_filtered_splits", "max_per_camera", *SPLITS):
        if prepared[key] != expected[key]:
            raise ValueError(f"prepared {key} differs from the reference")
    if prepared["missing"]:
        raise ValueError(f"camera archives are missing: {prepared['missing']}")

    summary = {"source": "SkyFinder Zenodo record 5884485", "image_size": prepared["size"],
               "splits": {}}
    seen = set()
    for split in SPLITS:
        with (root / f"{split}.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        cameras = Counter(row["camera"] for row in rows)
        intended = set(expected[split])
        if set(cameras) != intended:
            raise ValueError(f"{split} has cameras {sorted(cameras)}; expected {sorted(intended)}")
        if seen & set(cameras):
            raise ValueError(f"camera leaked into {split}: {sorted(seen & set(cameras))}")
        seen.update(cameras)
        night = 0
        for row in rows:
            if row["mask"] != f"{row['camera']}.png":
                raise ValueError(f"wrong mask for camera {row['camera']}: {row['mask']}")
            if not (root / split / "images" / row["image"]).is_file():
                raise ValueError(f"missing image in {split}: {row['image']}")
            if not (root / split / "masks" / row["mask"]).is_file():
                raise ValueError(f"missing mask in {split}: {row['mask']}")
            night += row["night"] == "1"
        if split != "test" and night:
            raise ValueError(f"{split} contains {night} night frames despite the reference filter")
        summary["splits"][split] = {"cameras": len(cameras), "images": len(rows),
                                     "night_images": night,
                                     "images_per_camera": dict(sorted(cameras.items(),
                                                                       key=lambda pair: int(pair[0])))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, default=Path("data/skyfinder/processed"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = verify(args.root)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
