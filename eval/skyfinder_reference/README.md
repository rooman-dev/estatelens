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
The handover reports a 0.5377 brightness baseline on this split; that score
needs to be reproduced from the prepared files before comparing a trained model.
