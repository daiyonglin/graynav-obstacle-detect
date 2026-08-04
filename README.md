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

Current target model direction:

```text
M1: SC132GS Y8 -> true-mono COCO80 YOLOv8n -> head6
M2: SC132GS Y8 -> shared Fast-SCNN -> 3-class surface + 16-bin depth heads
```

Earlier ROD25 and pseudo-RGB routes remain for reproducibility and rollback.
Both target models have a strict one-channel deployment contract. RGB public
weights use the exact repeated-gray identity `W_gray = W_R + W_G + W_B` for the
first convolution; public training data is converted to grayscale before it
enters the loader.

Main scripts for the new line:

```text
model_optimization/build_coco80_gray1_detector.sh
model_optimization/prepare_public_datasets.sh
model_optimization/run_surface_depth_cloud.sh
model_optimization/scripts/prepare_graynav_surface_depth_dataset.py
model_optimization/scripts/train_graynav_surface_depth.py
model_optimization/scripts/export_graynav_surface_depth.py
model_optimization/scripts/audit_surface_depth_onnx.py
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
