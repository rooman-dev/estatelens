# SkyFinder FYP reference split

These manifests are the original `subset.json` and `splits.json` supplied with
the project handover. They select 15 fixed cameras: 9 for training, 3 for
validation, and 3 for testing. No camera appears in more than one split.

To download and prepare this exact split:

```powershell
python -m models.aeroswap_data --reference --download --prepare
```

The reference settings keep frames captured from 07:00 to 17:59 in training
and validation, while retaining day and night frames in testing. The prepared
images and camera archives live under `data/skyfinder/` and are ignored by Git.
The brightness baseline was reproduced on all 4,880 test images from the three
test cameras. Pooled mIoU was **0.537748** (0.5377 to four decimals), matching
the handover. Camera 9708 scored 0.397863 and camera 10870 scored 0.626280.
The full per-camera and day/night report is in `baseline.json`.

The test-only preparation used the three test camera archives and the committed
reference settings. It did not need train or validation camera archives. The
full preparation has now been verified from its written CSV and image files:
7,845 training images across 9 cameras, 1,944 validation images across 3 cameras,
and 4,880 test images across 3 cameras. No camera occurs in two splits. The
full preparation's test baseline report is identical to `baseline.json`.

To verify a prepared dataset again:

```powershell
python -m eval.verify_skyfinder
```

The verification counts and per-camera image totals are in
`prepared_summary.json`.

## AeroSwap v1 result

The trained model is in `models/checkpoints/aeroswap_skyfinder_v1.ts`; its
training recipe and artifact hash are in that directory's README. The model
was selected by validation performance only. The full 4,880-image test split
was then evaluated in the WSL training environment with the same pooled
pixel-count mIoU code used for the brightness baseline:

```shell
python -m eval.aeroswap_eval --checkpoint models/checkpoints/aeroswap_skyfinder_v1.ts --device cuda --batch-size 2
```

| Test group | Images | Brightness baseline | AeroSwap v1 | Change |
| --- | ---: | ---: | ---: | ---: |
| All images, pooled | 4,880 | 0.5377 | 0.6627 | +0.1250 |
| Day, pooled | 2,208 | 0.6023 | 0.7204 | +0.1181 |
| Night, pooled | 2,672 | 0.4832 | 0.6180 | +0.1347 |
| Mean over cameras | 3 cameras | 0.5103 | 0.6125 | +0.1022 |
| Camera 204 | 2,494 | 0.5067 | 0.7183 | +0.2116 |
| Camera 9708 | 750 | 0.3979 | 0.5217 | +0.1239 |
| Camera 10870 | 1,636 | 0.6263 | 0.5974 | -0.0289 |

The model clears the pooled baseline and improves both pooled day and night
scores. It remains weaker than the baseline on camera 10870, especially by
day (0.7272 versus 0.7835). This is only three held-out camera viewpoints;
the pooled score is heavily influenced by camera 204's 2,494 images. We
should inspect actual mask overlays and measure boundary quality before
claiming the model makes replacement look convincing on real properties.

Full precision and sky/background IoU for each camera and condition are in
`aeroswap_test.json`. `aeroswap_training.jsonl` records all four training
epochs, including validation scores. Model weights and test scores are
versioned separately so the training milestone and evaluation are visible in
Git history.
