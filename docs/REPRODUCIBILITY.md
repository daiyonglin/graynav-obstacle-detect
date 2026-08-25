# Reproducibility Guide

## 1. Environment and storage

Windows 复现默认将虚拟环境、数据、缓存、日志和 checkpoint 放在 `E:\GrayNavWorkspace`。仓库只保存源码、配置和发布权重。

```powershell
powershell -ExecutionPolicy Bypass -File training/setup_windows.ps1 `
  -WorkRoot E:\GrayNavWorkspace `
  -BasePython E:\Anaconda3\python.exe
```

依赖基线为 Python 3.10、PyTorch 2.5.1+cu118、torchvision 0.20.1、Ultralytics 8.3.0、ONNX/ONNX Runtime、OpenCV、TensorBoard 和 tqdm。

## 2. Data

### Detection

`training/prepare_detection_data.ps1` 下载 VOC2007 trainval（460,032,000 bytes），筛选 Indoor8 映射并保留少量负样本。`training/scripts/download_coco_indoor8_subset.py` 和 `prepare_coco_indoor8.py` 可按类别下载紧凑 COCO 子集；默认流程不需要完整 COCO。

### Scene and depth

在 `E:\GrayNavWorkspace\data\raw` 放置：

- `ADEChallengeData2016.zip`；
- `RGB-D stair dataset.zip`；
- `nyu_depth_v2_labeled.mat`；
- `splits.mat`。

运行 `training/prepare_scene_data.ps1` 后会生成单通道图像、4 类标签、深度监督、train/val manifest 和审计报告。NYUv2 使用官方 795/654 划分；数据脚本检查跨划分 source-id 重叠。

## 3. Initialization assets

仓库 `weights/` 提供：

- `yolov8n.pt`：检测初始化；
- `graynav_surface_depth_e3_epoch49.pt`：场景/深度初始化；
- `graynav_unified_best_safety_epoch29.pt`：最终选择 checkpoint；
- `graynav_unified_indoor8_scene21.onnx`：静态导出图。

所有文件均在 `weights/README.md` 记录字节数和 SHA256。

## 4. Training

```powershell
powershell -ExecutionPolicy Bypass -File training/train_local.ps1 `
  -WorkRoot E:\GrayNavWorkspace `
  -BatchSize 16 `
  -AccumulationSteps 2 `
  -Workers 4 `
  -Epochs 35
```

固定参数：seed 42、AdamW、lr `3e-4`、weight decay `0.01`、AMP、5 epoch 场景预热。有效 batch 为 32。tqdm 输出 batch 进度，TensorBoard 路径为 `runs\unified_indoor8_v1\tensorboard`。

## 5. Evaluation and visualization

每个工具均提供 `--help`，并以 manifest 而非手工挑图确定样本：

```powershell
$py = "E:\GrayNavWorkspace\env\Scripts\python.exe"
& $py training/scripts/evaluate_unified.py --help
& $py training/scripts/visualize_unified.py --help
```

评估分别报告检测类别、局部人体、场景类别、台阶误报、深度排序和梯度指标。可视化为每个样本独立输出 mono、GT、seg、depth 和 edge，避免在一个画布中压缩信息。

## 6. ONNX and A1 conversion

```powershell
& $py training/scripts/export_unified.py --help
& $py training/scripts/validate_unified_onnx.py --help
& $py training/scripts/audit_unified_onnx.py --help
& $py training/scripts/build_a1_calibration.py --help
& $py training/scripts/package_a1_submission.py --help
```

导出必须满足静态 `1×1×384×384` 输入和 7 输出契约。校准/评估 `.npy` 与模型输入保持单通道 NCHW。A1 二进制转换由官方编译器执行；仓库脚本负责生成提交包、独立审计官方余弦相似度并核对 m1model SHA256。

## 7. Board build

官方 A1 SDK 不随本仓库分发。将仓库板端文件同步到已安装 SDK：

```powershell
powershell -ExecutionPolicy Bypass -File board/sync_to_sdk.ps1 `
  -SdkRoot E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk
```

随后在 `A1_Builder` 容器执行完整 SDK Buildroot 构建。构建后必须核对：

1. rootfs 仅含一个 `.m1model`；
2. 模型 SHA256 为 `33eec832...d66da8`；
3. CMake cache 显示 8 类、1 通道、voice ON；
4. `zImage < 15 MiB`；
5. OSD 固定资源清单完整；
6. host/ARM 测试和运行脚本契约通过。
