# GrayNav Obstacle Detection

Blind-navigation obstacle detection project for the Flyingchip A1 Vision Pi
and SC132GS mono camera. This repository keeps only the project-specific code:
board application sources, OSD/runtime assets, UART voice integration code, and
model optimization scripts.

The full A1 SDK, Docker buildroot output, datasets, training results, and large
model-development artifacts are intentionally not tracked here.

## Repository Layout

```text
board/obstacle_detect/
  demo_obstacle.cpp          A1 runtime entry: camera, NPU, postprocess, OSD, UART
  include/                   Board application headers
  src/                       YOLOv8 head6 decode, tracker, semantics, OSD, voice
  app_assets/                Required OSD assets and current deployed .m1model
  scripts/                   Board-side helper scripts
  cmake_config/, tools/      SDK build integration files

model_optimization/
  scripts/                   Gray adapter training, export, evaluation, diagnostics
  configs/                   Dataset/model configuration files
  requirements.txt           Cloud training Python dependencies
```

## Board Development

Canonical SDK app path on the local machine:

```text
E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk\smart_software\src\app_demo\obstacle_detect\ssne_ai_demo
```

After editing code in this repository, sync the changed board files back to the
SDK app path before Docker compilation.

Build inside the `A1_Builder` container:

```sh
cd /home/smartsens_flying_chip_a1_sdk/A1_SDK_SC132GS/smartsens_sdk
bash ./scripts/a1_sc132gs_build.sh
```

The flashable image is generated at:

```text
output/images/zImage.smartsens-m1-evb
```

On the Windows host this is normally visible as:

```text
E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk\output\images\zImage.smartsens-m1-evb
```

## Runtime Diagnostics

Useful board runtime switches:

```sh
A1_OUTPUT_HUMAN=1
A1_OUTPUT_JSON=0
A1_OUTPUT_SERIAL_DIAG=1
A1_OUTPUT_INTERVAL_FRAMES=5
A1_DEBUG_POSTPROCESS=1
A1_DEBUG_POSTPROCESS_INTERVAL=30
```

Use these during board tests to compare raw candidates, NMS results, tracker
stability, action decisions, and OSD behavior.

## Model Optimization

Current model direction:

```text
M1: gray image -> [G,G,G] -> YOLOv8n GrayNav fine-tuning -> head6
M2: M1 first-conv folded/tied as GrayStem-BC -> fine-tuning -> head6
```

Earlier pseudo-RGB adapter routes are kept for reproducibility, but the active
optimization line has moved to gray-domain detector fine-tuning. GrayStem-BC
uses the identity that `[G,G,G]` makes the first RGB convolution equivalent to a
single gray convolution, then keeps the exported model compatible with the
existing three-channel A1 input pipeline.

Main scripts:

```text
model_optimization/run_graystem_experiment.sh
model_optimization/scripts/prepare_graystem_dataset.py
model_optimization/scripts/train_yolov8n_gray_obstacle8.py
model_optimization/scripts/graystem_yolov8.py
model_optimization/scripts/evaluate_graystem_obstacle8.py
model_optimization/scripts/export_yolov8_head6.py
```

Cloud upload packages should be generated only when training is about to run on
the cloud machine.

## Development Rules

- Do not commit full SDK trees, build outputs, datasets, cloud results, or
  temporary archives.
- Add or update stage-summary documents only after a phase is complete.
- Keep comments meaningful: each new class/function should state its role, and
  complex logic blocks should have a short orienting comment.
- Each optimization iteration should be committed with a focused message after
  it builds or after its experiment result is recorded.
