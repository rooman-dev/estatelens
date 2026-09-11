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

- Python 3.14. Everyone on the team uses 3.14; the pins in requirements.txt
  are tested on it, and other versions may not install them.
- PySide6 (UI), OpenCV, rawpy, numpy
- SQLite via the standard library `sqlite3` module (not in requirements.txt)

Setup: `py -3.14 -m venv venv`, activate it, then `pip install -r requirements.txt`.

## Folder ownership

Ownership is strict. Do not create or edit files in a folder the current user
does not own. If a change is needed there, write it up as a proposal for the owner.

| Folder        | Owner  | Contents                                   |
|---------------|--------|--------------------------------------------|
| `core/`       | Rooman | Pipeline runner, job scheduling            |
| `ui/`         | Rooman | PySide6 interface                          |
| `processing/` | Hadeed | RAW loading, HDR fusion, perspective       |
| `models/`     | Arham  | Sky segmentation model                     |
| `eval/`       | Arham  | Benchmarks and metrics                     |
| `data/`       | -      | Test images. Never committed               |

`data/` is gitignored, so it does not exist after cloning. Each person creates
it locally and puts their own test images in it.

## Prototype vs. real implementation

`core/prototype.py` is Rooman's throwaway end-to-end prototype (RAW loading and
bracket fusion). It exists only to prove the pipeline works. Hadeed writes the
real versions in `processing/`:

- `processing/chromaraw.py`: RAW loading
- `processing/lumamerge.py`: HDR fusion

Once those exist, the prototype is swapped out and deleted. Do not build on it.

## Git

- Image files (`.jpg .jpeg .png .tif .tiff .cr2 .cr3 .nef .arw .dng`, any case)
  are ignored, except under `ui/assets/` and `eval/figures/`.
