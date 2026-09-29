# EstateLens: Hadeed's progress record

**As of:** 30 September 2026
**Project:** EstateLens, Air University BSCS FYP
**Current assignment before the presentation:** visual review of 15 real bracket outputs

This is a record of work and evidence, not a claim that three image-processing
modules are complete. Rooman's 30 September direction supersedes the earlier
20-day sprint plan. Before the presentation, Hadeed is reviewing the 15 outputs
and reporting visual findings. Rooman owns `core/`, `ui/`, integration, tuning,
the presentation and paper. Hadeed takes `processing/` after the refactor,
`tests/`, AeroSwap in FYP-III and the exterior dataset. The earlier sprint
role split with Arham is no longer current.

## Contributions recorded under Hadeed's Git identity

The code changes listed here were made with Codex assistance and committed as
`Hadeed Zahid`. The earlier implementations of the core modules were committed
by `Rumman Ahmed` (Rooman in the sprint plan). Commit authorship alone should
not be used to claim independent implementation.

| Work | Evidence | Current result |
| --- | --- | --- |
| Support mixed RAW/JPEG bracket datasets, including `.ORF` and `.SRW` RAW files and matching camera JPEGs | `ea7917d` | The supplied dataset can be scanned without counting a RAW+JPEG pair as two exposures. Two grouping regression tests cover this path. |
| Run real exposure brackets through the batch pipeline | Local `data/reddit_bkt/batch.db` and `data/reddit_bkt/batch/` | 15 of 15 jobs marked complete; all 15 output JPEGs exist; no recorded job errors. Only four show furnished homes; the rest include damaged/industrial and outdoor scenes. Input and output photos are ignored by Git. This verifies processing completed, not that every image looks good. |
| Add a tiled-versus-whole fusion regression check | `badc6e6`, `tests/test_tiled_fusion.py` | Two tests exercise the single-tile path and compare multi-tile output with whole-image fusion on a synthetic interior-like scene. |

The full automated suite currently reports **13 passing tests**. Most of these
test the experimental sky work; they do not constitute full coverage of the
three assigned image-processing modules.

The RAW+JPEG fix addresses a silent grouping defect: a camera saving both
formats for each shutter press could count one exposure twice, breaking bracket
boundaries. The same commit added `.ORF` and `.SRW` support for the supplied RAW
set. This is a finding for the FYP verification report, with `ea7917d` and
`tests/test_bracket_grouping.py` as evidence.

## Existing work inherited from Rooman

- `core/prototype.py` already reads RAW/JPEG exposures, groups brackets,
  aligns them, and fuses them with OpenCV Mertens.
- `core/truevertical.py` already implements perspective correction.
- `core/scheduler.py` already implements tiled fusion and the batch queue.
- `ui/main_window.py` already provides the desktop processing flow.

These were working starting points when Hadeed's commits began. They should
be credited to Rooman in contribution reports unless a specific later change
is identified by its own commit.

## Experimental work outside the current sprint scope

Commits from `e65e826` through `90e19bd` added AeroSwap training, evaluation,
and an optional sky preview/save flow with Codex assistance. This was started
earlier than the team's intended schedule. The held-out SkyFinder report records
pooled mIoU **0.6627** versus a brightness baseline of **0.5377** across 4,880
test images, but visual checks show false sky on buildings and roofs. It has
not been validated on exterior property photos. Commit `c0be1b6` hides the
experimental controls from the normal app. AeroSwap is **parked** for the later
project phase and is not presented as a finished FYP-I module.

## Current work and later work

**Before the presentation:** the visual review of all 15 outputs is recorded in
[`interior_visual_review_2026-09-30.md`](interior_visual_review_2026-09-30.md).
The local `data/reddit_bkt/review_share.zip` holds before/after screenshots for
all 15. The review found two damaging perspective corrections, weak room/window
balance in several examples, and a dataset gap for the main real-estate claim.
The report and evidence are ready for Hadeed to share with Rooman for tuning
decisions. No retuning or refactoring was done.

**After the presentation:** move ChromaRaw, LumaMerge and TrueVertical to
standalone `processing/` modules and re-measure the published results; add
de-ghosting; then expand regression tests. ChromaRaw's SHA-256 deduplication and
explicit 16-bit interface remain unimplemented. These are planned work, not
completed contributions. Do not edit Rooman's live `core/` or `ui/` areas
without coordinating with him.
