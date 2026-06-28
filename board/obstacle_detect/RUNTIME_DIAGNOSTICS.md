# Runtime Diagnostics and Consistency Workflow

This project now treats PC/ONNX/head6/board consistency as the first model
optimization gate. Do not judge a new model only by PC mAP before this workflow
passes.

## Board Debug Environment Variables

Use these only during diagnosis. Disable them for normal Aurora demo runs.

```sh
A1_OUTPUT_HUMAN=1
A1_OUTPUT_JSON=0
A1_OUTPUT_SERIAL_DIAG=1
A1_OUTPUT_INTERVAL_FRAMES=5
A1_DEBUG_POSTPROCESS=1
A1_DEBUG_POSTPROCESS_INTERVAL=30
A1_DUMP_PREPROCESS_ONCE=1
A1_PREPROCESS_DUMP_PATH=/tmp/yolov8_input.bin
```

Meaning:

- `A1_OUTPUT_HUMAN`: prints compact readable navigation summaries.
- `A1_OUTPUT_JSON`: prints JSON packets for offline log parsing.
- `A1_OUTPUT_SERIAL_DIAG`: appends raw/post-NMS/stable diagnostics to human
  summaries.
- `A1_OUTPUT_INTERVAL_FRAMES`: controls summary output interval, default `5`.
- `A1_DEBUG_POSTPROCESS`: prints paired head indexes, strides, candidate counts,
  NMS counts and kept result counts.
- `A1_DEBUG_POSTPROCESS_INTERVAL`: frame interval for repeated debug summaries.
- `A1_DUMP_PREPROCESS_ONCE`: saves the first model input tensor after A1
  preprocess.
- `A1_PREPROCESS_DUMP_PATH`: output path for the dumped preprocessed tensor.

Camera and distance parameters can be tuned without recompiling:

```sh
A1_CAM_FOV_H_DEG=49.7
A1_CAM_FOV_V_DEG=78.9
A1_CAM_HEIGHT_M=0.85
A1_CAM_PITCH_DOWN_DEG=15.0
A1_DIST_MIN_M=0.2
A1_DIST_MAX_M=8.0
```

## PC Golden Case Generation

In the cloud/model workspace, generate fixed gray-input golden cases:

```sh
python scripts/generate_yolov8_golden_cases.py \
  --source /data/coco/val2017 \
  --weights yolov8n.pt \
  --out-dir artifacts/golden_yolov8n_gray \
  --imgsz 384 \
  --max-images 100
```

The golden file records preprocessed gray3 inputs, predicted boxes, scores,
classes and letterbox metadata.

## Offline Head6 Validation

Before flashing a converted model, run the SDK offline validator:

```sh
python tools/offline_yolov8_head6_test.py \
  --model E:/jichuang/gray-test/yolov8n_head6.onnx \
  --inputs E:/jichuang/pc_gray_yolov8_test/inputs_gray_3ch_for_yolo \
  --output-dir E:/jichuang/gray-test/runs/offline_obstacle_detect \
  --limit 32 \
  --conf 0.20 \
  --iou 0.60 \
  --top-k 800 \
  --keep-top-k 80
```

Acceptance target:

- mean box coordinate error below 2 px against the selected golden reference;
- score error below 1e-3 when comparing the same raw head output path;
- class id must match;
- head grouping must be exactly 3 classification heads and 3 regression heads.

## Board First-Frame Checks

On the board, the first run must print 6 output heads with valid metadata:

```text
inferred_c=80 or 64
stride=8/16/32
decode = HWC + sigmoid + DFL + anchor decode + reverse letterbox
```

If the grouping fails or any stride/channel is unexpected, do not tune NMS or
tracker thresholds. Fix model conversion, tensor layout or head extraction first.

## UART Voice Reminder

Voice output receives only low-rate navigation decisions, not every detection.
Use:

```sh
A1_OUTPUT_MODE=both A1_VOICE_UART=/dev/ttyS1 ./scripts/run_voice_both.sh
```

P4 wiring remains:

- pin 4 `A1_D1_UART1TX` -> module RX
- pin 6 `A1_D3_UART1RX` -> module TX, optional
- pin 33/34/47/48 GND -> module GND
