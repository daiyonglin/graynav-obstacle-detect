# GrayNav Fast-SCNN

This directory implements the single-channel road-surface branch used by
GrayNav.  The network follows Fast-SCNN and the Apache-2.0 PaddleSeg reference
implementation, with only deployment-oriented changes:

- a true `1x1x256x256` grayscale input;
- fixed `(1, 2, 4, 8)` pyramid pooling on the `8x8` feature map;
- nearest-neighbour feature resizing;
- a raw `1x4x32x32` logits output with no Softmax or ArgMax in ONNX.

References:

- https://arxiv.org/abs/1902.04502
- https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.10/paddleseg/models/fast_scnn.py

The four deployed classes are `ground_candidate`, `blocked_surface`,
`step_or_drop`, and `pothole`.  Value 255 is ignored during training.

Typical workflow:

```bash
python -m pip install -r segmentation/requirements_surface.txt
# Install the Paddle/Paddle-GPU wheel matching the cloud CUDA runtime only when
# import_paddleseg_fast_scnn.py is used.

# 0. Optional: import the official PaddleSeg RGB checkpoint.  This folds only
#    the first Conv2D and fails if any remaining ordered tensor is incompatible.
# Official release/2.10 model-zoo URL:
# https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams
python scripts/import_paddleseg_fast_scnn.py \
  --paddle-checkpoint /weights/fast_scnn_cityscapes.pdparams \
  --output runs/graynav_fast_scnn/paddleseg_gray1_init.pt

# 1. Prepare official train/validation splits.  config_v1.2.json is mandatory
#    and its critical ids are checked before any masks are written.
python scripts/prepare_graynav_surface_dataset.py \
  --mapillary-root /data/mapillary-vistas \
  --stair-root /data/stairnetv3 \
  --output /data/graynav_surface

python scripts/train_graynav_fast_scnn.py \
  --data /data/graynav_surface \
  --output runs/graynav_fast_scnn \
  --pretrained runs/graynav_fast_scnn/paddleseg_gray1_init.pt \
  --amp

# Resume an interrupted paid-cloud run from the same output directory.
python scripts/train_graynav_fast_scnn.py \
  --data /data/graynav_surface \
  --output runs/graynav_fast_scnn \
  --resume runs/graynav_fast_scnn/last.pt \
  --amp

# 2. Export static raw logits and enforce the A1 operator contract.
python scripts/export_graynav_fast_scnn.py \
  --checkpoint runs/graynav_fast_scnn/best.pt \
  --onnx runs/graynav_fast_scnn/graynav_fast_scnn_gray1_4cls_256.onnx

python scripts/audit_surface_onnx.py \
  --onnx runs/graynav_fast_scnn/graynav_fast_scnn_gray1_4cls_256.onnx

# 3. Verify PyTorch/ONNX grid agreement and prepare a deterministic 200-image
#    single-channel INT8 calibration set from validation data only.
python scripts/validate_surface_onnx.py \
  --checkpoint runs/graynav_fast_scnn/best.pt \
  --onnx runs/graynav_fast_scnn/graynav_fast_scnn_gray1_4cls_256.onnx \
  --images /data/graynav_surface/images/val \
  --report runs/graynav_fast_scnn/onnx_consistency.json

python scripts/build_surface_calibration_set.py \
  --data /data/graynav_surface \
  --output runs/graynav_fast_scnn/int8_calibration \
  --count 200
```

The prepared public RGB images are converted to one grayscale channel before
training.  Neither training nor export replicates grayscale into RGB.

The following describes the legacy four-class segmentation-only route. The A1
vendor converter is intentionally a separate, fail-closed step because
it is not distributed in this repository.  Feed it the audited ONNX and the
generated Y8 calibration images, require an INT8 `1x4x32x32` output, then name
the artifact `graynav_fast_scnn_gray1_int8.m1model` and place it in:

```text
board/obstacle_detect/app_assets/models/
```

Do not substitute an RGB calibration tensor or add a final 256x256 upsample.
After conversion, rebuild the SDK and run the board smoke test from
`/app_demo`:

```sh
A1_DUAL_SMOKE_SECONDS=1800 ./scripts/run_dual_model_smoke.sh
```

The active route is now `graynav_surface_depth.py`, whose two outputs are
`1x3x64x64` surface logits and `1x16x64x64` depth logits. Its final deployment
artifact is `graynav_surface_depth_gray1_int8.m1model`; see
`docs/GRAYNAV_SURFACE_DEPTH_IMPLEMENTATION_2026-08-04.md`.

The board executable loads YOLO with static allocation and Fast-SCNN with
dynamic allocation.  If that fails it reinitializes SSNE and retries both with
dynamic allocation.  If residency still fails, repeat import/train/export with
`--width-mult 0.75`; this is a predefined channel-sliced initialization and
ordinary fine-tuning path, not knowledge distillation.  Three consecutive segmentation inference failures cause
a detector-only fallback and one non-repeating degradation announcement.

The checked-in code can be built without the final surface model; that build is
only a detector-only integration build.  Offline F1/IoU, INT8 agreement, dual
model residency, latency and scene acceptance are not considered passed until
the trained `.m1model`, converter and physical A1 board are available.

For the persistent-disk layout, one-epoch smoke test, resumable 4090 command,
artifact return and firmware rollback gates, see
`docs/GRAYNAV_4090_CLOUD_TRAINING_RUNBOOK_2026-08-02.md`.  The convenience entry
point is `scripts/run_graynav_surface_cloud.sh`; it intentionally does not
download license-gated datasets or store cloud credentials.
