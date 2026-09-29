# AeroSwap SkyFinder v1

`aeroswap_skyfinder_v1.ts` is the selected TorchScript sky segmentation model.
It accepts RGB float32 tensors shaped `[N, 3, 512, 512]` in the range `[0, 1]`
and returns sky probabilities shaped `[N, 1, 512, 512]`. Use `> 0.5` for a
binary mask. ImageNet normalization is built into the model.

The model was trained on the committed SkyFinder camera split using the
MobileNetV3-Small encoder in `models/aeroswap.py`. Training used the pinned
`requirements-train.txt` environment on WSL Ubuntu with an RTX 3050 Laptop
GPU. The exact command was:

```shell
python -m models.train_aeroswap --epochs 10 --batch-size 2 --accumulate 2 \
    --train-per-camera 250 --val-per-camera 250 --patience 3 --out data/aeroswap
```

This sampled 2,250 training images from 9 cameras and 750 validation images
from 3 separate cameras. Early stopping ended training after epoch 4; epoch 1
was selected with validation pooled mIoU 0.79511177. The three test cameras
were not used for training or model selection. The full test report is kept
in `eval/skyfinder_reference/aeroswap_test.json`.

SHA-256: `5e2015d5d9e5b5a60be9ac966713f020cba609a8aa52cd913ac486bce93f1069`

TorchScript worked in the pinned PyTorch 2.14.0 environment, although PyTorch
warns that its JIT API is not supported on Python 3.14 and could break in a
future release. Verify loading and inference after changing the runtime.

For desktop sky-mask previews, `models.infer_sky` reads an encoded image from
stdin and returns a 512x512 probability PNG on stdout. The desktop-side helper
in `processing/sky_preview.py` invokes it in a separate process. By default it
uses the desktop Python environment. On a machine where PyTorch runs in WSL,
create an ignored `data/sky_runtime.json` containing a command argument list,
for example:

```json
["wsl", "--", "/home/your-user/venv/bin/python", "-m", "models.infer_sky"]
```

The current development PC has this local configuration. Paths to the photo
are never sent to WSL; the image and mask travel as PNG bytes through the
process pipe.

In the desktop app, process a bracket, select its row, and click **Preview sky
mask**. Blue shows pixels AeroSwap would call sky; the yellow line shows the
predicted boundary. SceneSense warns when the photo is not confirmed as an
exterior; **Choose sky** remains disabled for those photos. On a confirmed
exterior, **Choose sky** accepts a JPEG or PNG and shows the full replacement
preview. **Save sky version** asks for a new filename and writes a separate
image only after that preview is visible. **Show photo** returns to the normal
after view. Neither preview nor save overwrites the fused JPEG or the sky
source image.
SceneSense may decline some genuine exteriors; those require a later reviewed
override rather than silently bypassing the gate.

The save flow has been checked on a SkyFinder exterior with a generated test
sky. It is a functional check, not validation on real estate exteriors. In
that test the model wrongly replaces parts of a roof, so the preview must be
reviewed carefully. The current dataset of 15 real property brackets contains
interiors only; an exterior property example is still needed to evaluate the
feature for its intended use.
