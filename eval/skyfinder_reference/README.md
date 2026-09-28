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
training and validation data must still be prepared before model training.
