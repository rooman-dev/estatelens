# EstateLens: visual review of 15 real brackets

**Reviewer:** Hadeed Zahid, with Codex assistance
**Date:** 30 September 2026
**Scope:** visual inspection only, before the FYP presentation

## Result for the presentation

All 15 batch jobs completed and produced JPEGs. The review does **not** establish
that the current pipeline produces listing-ready interiors with both a bright
room and a clear window view. Only four brackets show a furnished home; their
windows have curtains or blinds, and the room or foreground remains dark in the
fused images. The other eleven show a damaged building or derelict industrial
site, including close-ups and outdoor scenes. The dataset is useful for finding
defects, but it is a weak sample for the central real-estate claim. The
description "15 real interior brackets" in earlier notes needs this qualifier.

The most urgent visual defects are two damaging perspective corrections
(`SAM_9433` and `P6076653`) and a window scene with a darker interior than the
camera exposure (`P6076520`). `SAM_9292` also has a conspicuous blue cast and
blue fringe around a pole. These are observations for Rooman to assess; no
processing settings or code were changed for this review.

## How the review was done

- Checked every existing fused JPEG at full image size and compared it with a
  camera exposure near the middle of its bracket. For one bracket (`9220966`),
  a RAW embedded preview was used because a matching camera JPEG was absent.
  These comparisons show appearance, not a controlled quality score.
- Reviewed windows, plain walls and ceilings, framing, colour, halos and
  possible ghosting. A damaged wall or ceiling was compared with the source
  before calling it a processing defect.
- Re-ran the **existing** perspective estimator on the source brackets without
  overwriting any batch output. The batch database stores the final output and
  dimensions, but not the original per-job TrueVertical decision. The counts
  below are therefore a reproduction, not a recovered execution log. Small
  rerun differences occurred on jobs 4 and 10.
- The side-by-side evidence and diagnostic JSON are local under
  `data/reddit_bkt/review/`. The photos are excluded from Git; the report can
  be shared with the local `data/reddit_bkt/review_share.zip` evidence bundle.

## Findings across the set

| Check | Finding |
| --- | --- |
| Window and room balance | No furnished-home example clearly proves both a well-lit room and a clear exterior view. Lace curtains or blinds obscure three views; `9220972` reveals more outdoors but leaves the keyboard and room dark. `P6076520` visibly darkens an already readable interior. |
| Plain walls and ceilings | No definite regular CLAHE tile pattern was found. The ceiling in `SAM_9312` is already cracked and peeling in the camera photo. The painted purple wall in `9220966` has some uneven shading, but not a clear before/after processing defect. There are too few clean, evenly lit painted walls to test the predicted blotching reliably. |
| TrueVertical | Reproduced estimator decisions: **14 applied, 1 declined**. `SAM_9430` declined a proposed 62.8° correction. The accepted 27.4° and 27.0° corrections in `SAM_9433` and `P6076653` visibly damage composition. Output dimensions alone miss the loss in `P6076653`. |
| Other artifacts | No unambiguous motion ghost was identified in these largely static scenes. `SAM_9292` has a strong blue cast and blue fringe around a pole. `P6076626` shows a cool/cyan edge appearance around foliage and the window, while its foreground stays very dark. These observations do not establish which processing stage caused them. |

## Per-output review

`TV` below is the reproduced TrueVertical decision; percentages are the
estimator's **predicted retained area**, not a measured quality score. For
`P6076653`, the estimator reports 100% retained area even though the image is
visibly shifted and loses important foreground content.

| # | Output | Window / exposure and wall check | TV and frame | Visible issue or value |
| --- | --- | --- | --- | --- |
| 1 | `SAM_9289` | Broken wired-glass window, not a normal room/window test; outdoors remains pale. No clean painted surface. | Applied 6.8°, 86.5% area. | Glass shifts noticeably toward blue; dark left edge. |
| 2 | `SAM_9292` | Close-up of broken glass, no room. Wall check inapplicable. | Applied 6.1°, 88.1% area. | Strong blue cast and blue fringe around the pole; **issue**. |
| 3 | `SAM_9312` | Damaged room brightens; foliage through lower window becomes visible, while upper panes stay nearly white. Ceiling was already peeling in the source. | Applied 2.4°, 93.2% area; frame remains usable. | Partial exposure recovery, not balanced across the full window. |
| 4 | `SAM_9415` | Derelict corridor; inside and trees beyond the windows remain readable. Surfaces are aged/textured. | Applied 1.9°, about 93% area; sensible frame. | Comparatively strong exposure example, with limited relevance to a listing photo. |
| 5 | `SAM_9424` | Broken opening looking into trees, not a room/window balance test. No suitable flat wall. | Applied 2.8°, 92.4% area; acceptable frame. | No clear visual defect, but little evidence for the central claim. |
| 6 | `SAM_9430` | Roofless/outdoor ruin; no indoor window or flat wall test. | **Declined** 62.8° candidate; original dimensions retained. | Good safety abstention in this case. |
| 7 | `SAM_9433` | Outdoor ruin; window and wall checks inapplicable. | Applied 27.4°, 79.9% area. | **Major issue:** scene is canted and cropped compared with the camera frame. |
| 8 | `P6076515` | Derelict room; foreground and exterior are readable, but camera exposure was already similar. Paint is heavily damaged, so CLAHE blotches cannot be isolated. | Applied 0.2°, 99.0% area. | Little visible gain, no definite new artifact. |
| 9 | `P6076520` | Open view to buildings; fused exterior has detail, but the interior brick and wall are darker than in the camera exposure. No clean flat wall. | Applied 2.7°, 94.3% area; frame remains usable. | **Major issue:** window balance worsens the room. |
| 10 | `P6076626` | More outside detail appears through a broken window, but the foreground floor remains very dark. No suitable flat wall. | Applied about 0.1°, about 98.5% area. | Cool/cyan edge appearance around leaves/window; inspect at full size. |
| 11 | `P6076653` | Stairwell, no meaningful window/flat wall test; walls are damaged. | Applied 27.0°; estimator reports 100% area at the same output dimensions. | **Major issue:** image tilts/zooms and loses much of the foreground floor and staircase framing. |
| 12 | `9220963` | Furnished home; curtain gains texture, but cabinet and room remain dark. Curtain blocks a clear outside view. No broad plain wall. | Applied 5.5°, 88.5% area; crop is noticeable. | Window claim remains unverified; room exposure is weak. |
| 13 | `9220966` | Furnished home; plant, curtain and painted purple wall brighten. Curtain still blocks a clear exterior view. No definite CLAHE grid on the wall. | Applied 1.8°, 94.8% area; frame acceptable. | Comparatively strong partial home result, but not a clear indoor/outdoor success. |
| 14 | `9220969` | Furnished home; some exterior foliage improves through blinds, while the room stays dark. No reliable plain-wall sample. | Applied 11.5°, 84.3% area; tighter crop. | Window balance still weak; crop merits review. |
| 15 | `9220972` | Furnished home; foliage outside and keyboard details improve somewhat, but the keyboard/room remain dark. Curtains/blinds limit the view. | Applied 1.7°, 96.4% area; frame acceptable. | Comparatively strong partial recovery, not a balanced listing image. |

## Screenshots to review with Rooman

The local evidence bundle includes **all 15** labeled before/after comparisons,
covering each failure and the comparatively strongest examples. Start with:

- Better partial results: `04_SAM_9415_fused_comparison.jpg`,
  `13_9220966_fused_comparison.jpg`, `15_9220972_fused_comparison.jpg`.
- Major failures: `07_SAM_9433_fused_comparison.jpg`,
  `09_P6076520_fused_comparison.jpg`, `11_P6076653_fused_comparison.jpg`.
- Additional review points: `02_SAM_9292_fused_comparison.jpg` (blue cast),
  `03_SAM_9312_fused_comparison.jpg` (still pale upper panes),
  `10_P6076626_fused_comparison.jpg` (dark foreground/cool edges), and the
  four furnished-home comparisons numbered 12–15.

These are selected as diagnostic examples, **not** three finished real-estate
marketing photos. Rooman can decide whether the presentation should show them,
how to describe them, and whether later tuning is warranted. A subsequent test
needs normal furnished interiors with unobstructed windows and smooth painted
walls, photographed with controlled brackets. This is a data gap to report,
not a request to retune the pipeline before the presentation.
