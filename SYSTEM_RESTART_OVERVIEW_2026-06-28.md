# System Restart Overview

Date: 2026-06-28

## Current Objective

Rebuild the project context and restart optimization with the board available.
The immediate goal is not to change everything at once. We need a controlled
loop:

```text
code/model change -> Docker build -> flash board -> Aurora/serial observation -> log result -> next change
```

## Board System

Main board project:

```text
E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk\smart_software\src\app_demo\obstacle_detect\ssne_ai_demo
```

Runtime pipeline:

```text
SC132GS mono camera, 720x1280
  -> IMAGEPROCESSOR online pipeline, Y8 frame
  -> YOLOV8GRAY preprocessing, 384x384 letterbox, SSNE_BGR input
  -> yolov8n_head6.m1model on A1 NPU
  -> CPU YOLOv8 head6 decode: HWC output read, sigmoid, DFL, reverse letterbox
  -> semantic class filtering, distance estimate, MultiTargetNMS
  -> ObstacleTracker, stable boxes, risk and action decision
  -> Aurora OSD
  -> optional UART voice notifier
```

Key source files:

| File | Role |
| --- | --- |
| `demo_obstacle.cpp` | main loop, frame capture, detect, tracker, OSD, voice, serial summaries |
| `src/yolov8_gray.cpp` | YOLOv8 head6 preprocess/inference/postprocess/distance |
| `src/tracker.cpp` | temporal stability, track IDs, decision generation |
| `src/semantic_config.cpp` | COCO raw class -> navigation semantic class, thresholds, risk weights |
| `src/voice_notifier.cpp` | UART output arbitration |
| `src/osd-device.cpp` / `src/utils.cpp` | OSD drawing and NMS utilities |

Current model:

```text
app_assets/models/yolov8n_head6.m1model
```

It is still the official YOLOv8n-derived head6 model. The board code expects:

- 6 raw output heads;
- 3 classification heads with 80 channels;
- 3 regression heads with 64 channels;
- strides 8, 16, 32;
- postprocessing on CPU.

## New Board Observability

`demo_obstacle.cpp` now supports runtime output controls:

```sh
A1_OUTPUT_HUMAN=1
A1_OUTPUT_JSON=0
A1_OUTPUT_SERIAL_DIAG=1
A1_OUTPUT_INTERVAL_FRAMES=5
```

Postprocess diagnostics:

```sh
A1_DEBUG_POSTPROCESS=1
A1_DEBUG_POSTPROCESS_INTERVAL=30
A1_DUMP_PREPROCESS_ONCE=1
A1_PREPROCESS_DUMP_PATH=/tmp/yolov8_input.bin
```

This is the first restart-stage code change. It makes board tests measurable
without recompiling for every logging mode.

During board testing, record:

- FPS and average FPS;
- light state: dark, bright, low contrast, normal;
- raw candidate count;
- post-NMS count;
- stable tracker count;
- nearest class, raw class, confidence, distance, risk;
- current action: STOP, SLOW, LEFT, RIGHT, CLEAR;
- whether OSD boxes flicker or drift.

## Model Optimization State

Model optimization workspace:

```text
E:\jichuang\model_optimization
```

Previous adapter experiment:

- `GGG baseline`: gray copied to 3 channels.
- `LUT adapter`: learnable 3x256 per-channel lookup table.
- `Conv adapter`: small conv adapter.

Truth evaluation on COCO val2017 subset showed:

```text
baseline all AP:      0.30670
LUT all AP:           0.30848
Conv all AP:          0.29272

baseline nav AP:      0.26850
LUT nav AP:           0.26854
Conv nav AP:          0.25209
```

Conclusion:

- LUT was only marginally positive.
- Conv degraded detection.
- Pure distillation is not enough because it mainly imitates the gray-copy
  baseline.

## New Model Direction

The adapter should not claim to recover real color. A single-channel grayscale
image has already lost real RGB chromatic information.

The correct target is:

```text
single-channel gray -> pseudo-RGB feature encoding
```

The pseudo-RGB channels should be less redundant than gray-copy and should
better stimulate RGB-pretrained YOLOv8n first-layer filters.

New candidate:

```text
SpatialGrayAdapter:
  channel 0: tone-adjusted intensity
  channel 1: local contrast / illumination-normalized structure
  channel 2: Sobel edge and texture emphasis
```

Scripts added:

| Script | Role |
| --- | --- |
| `scripts/optimize_spatial_adapter_metric.py` | searches stronger spatial adapter parameters using COCO truth TP/FP/FN objective |
| `scripts/diagnose_adapter_distribution.py` | checks channel correlation and YOLO first-conv activation distribution |
| `scripts/evaluate_gray_adapters_coco.py` | truth eval for baseline/LUT/Conv/Spatial |
| `scripts/summarize_adapter_truth_eval.py` | generates markdown report and plots |

Cloud update package:

```text
E:\jichuang\graynav_model_optimization_spatial_adapter_v4.tar.gz
```

## Restart Plan

### Phase 1: Board Baseline Measurement

Build and flash the current code with observability enabled.

Run with:

```sh
A1_OUTPUT_HUMAN=1 \
A1_OUTPUT_SERIAL_DIAG=1 \
A1_OUTPUT_INTERVAL_FRAMES=5 \
A1_DEBUG_POSTPROCESS=1 \
A1_DEBUG_POSTPROCESS_INTERVAL=30 \
./scripts/run.sh
```

Test scenes:

- empty scene;
- single person;
- single chair;
- black chair;
- table/desk;
- bag/suitcase;
- multiple people;
- multiple obstacles;
- low light;
- strong light;
- fast camera shake.

Acceptance for this phase:

- Aurora shows stable boxes with no OSD residue;
- first-frame output heads are correct;
- raw/post-NMS/stable counts are plausible;
- serial logs make failures diagnosable.

### Phase 2: Board Postprocess Stabilization

Only tune postprocess/tracker after Phase 1 logs exist.

Likely knobs:

- semantic thresholds in `semantic_config.cpp`;
- NMS and `keep_top_k`;
- tracker start/hit/miss stability;
- risk threshold and OSD text stability;
- distance parameter config.

### Phase 3: Model Adapter Experiment

On cloud, run `SpatialGrayAdapter` search and truth evaluation.

Do not integrate adapter into board until PC truth eval shows a meaningful gain:

- navigation AP50 improves by at least 1 percentage point; or
- navigation Recall/AR100 improves by at least 1 percentage point without AP
  dropping; or
- true-positive/FN improvements are visually confirmed on target scenes.

### Phase 4: Model Conversion and Board A/B Test

If the model side improves:

- first test PC gray input;
- then convert/export;
- then compare board baseline vs adapter/model under the same scenes.

## Immediate Next Work

1. Flash and run baseline board test with the new diagnostics.
2. Collect serial logs and Aurora observations.
3. In parallel, run spatial adapter search on cloud.
4. Decide next code change based on board logs, not guesswork.

## Build Record

### 2026-06-28 Restart Build

Docker container:

```text
A1_Builder
```

Build command:

```sh
cd /home/smartsens_flying_chip_a1_sdk/A1_SDK_SC132GS/smartsens_sdk
bash ./scripts/a1_sc132gs_build.sh
```

Result:

```text
success, exit code 0
elapsed about 12.6 minutes
```

Generated host-visible images:

```text
E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk\output\images\rootfs.cpio
E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk\output\images\rootfs.cpio.gz
E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk\output\images\zImage.smartsens-m1-evb
```

The board application was installed into the rootfs at:

```text
output/target/app_demo/ssne_ai_demo
```

Aurora flashing uses:

```text
zImage.smartsens-m1-evb
```

Official notes in `files/烧录流程.txt` say Aurora can browse Docker output via:

```text
http://127.0.0.1:8080/
```

If Aurora does not show the file through Docker HTTP, use the host path above
and select `zImage.smartsens-m1-evb` directly in the burn dialog.
