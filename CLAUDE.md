# EstateLens

Final year project at Air University. A Windows desktop app that batch-processes
real estate photos: reads camera RAW files, merges exposure brackets into HDR,
fixes perspective, replaces skies, and exports web-ready JPEGs.

## How to work with me

Explain the plan before writing code. The user is learning, not just shipping.

End every response with this block. Keep it short and factual:

```
---STATUS---
DID: <one line, what you actually changed>
FILES: <files created or modified>
RESULT: <works / broken / untested>
NEXT: <what should happen next>
ERRORS: <any errors, or "none">
---END---
```

## Stack

- Python 3.14. The pins in requirements.txt are tested on it, and other
  versions may not install them.
- PySide6 (UI), OpenCV, rawpy, numpy, PyTorch + torchvision (sky model)
- SQLite via the standard library `sqlite3` module (not in requirements.txt)

Setup: `py -3.14 -m venv venv`, activate it, then `pip install -r requirements.txt`.

## Folder structure

All nine modules are built by one person; folders organise code, not ownership.

| Folder        | Contents                                   |
|---------------|--------------------------------------------|
| `core/`       | Pipeline runner, job scheduling            |
| `ui/`         | PySide6 interface                          |
| `processing/` | RAW loading, HDR fusion, perspective       |
| `models/`     | Sky segmentation model                     |
| `eval/`       | Benchmarks and metrics                     |
| `data/`       | Test images and datasets. Never committed  |

`data/` is gitignored, so it does not exist after cloning. Create it locally
and put test images and downloaded datasets in it.

## Prototype vs. real implementation

`core/prototype.py` is a throwaway end-to-end prototype (RAW loading and
bracket fusion). It exists only to prove the pipeline works. The real
versions go in `processing/`:

- `processing/chromaraw.py`: RAW loading
- `processing/lumamerge.py`: HDR fusion

Once those exist, the prototype is swapped out and deleted. Do not build on it.

## Git

- Image files (`.jpg .jpeg .png .tif .tiff .cr2 .cr3 .nef .arw .dng`, any case)
  are ignored, except under `ui/assets/` and `eval/figures/`.
