# GrayNav COCO80 + SurfaceDepth 实施交接（2026-08-04）

## 1. 本轮已经实现的边界

本轮实现的是可训练、可转换、可编译的完整软件链路，不伪装成已经完成的模型或板测结果：

- 检测：官方 COCO80 YOLOv8n，RGB 首层按 `W_R + W_G + W_B` 折叠为真单通道，保留 6 个 raw head。
- 道路：真单通道 A1-safe Fast-SCNN，共享编码器，输出 `1x3x64x64` 分割 logits 和 `1x16x64x64` 深度 logits。
- 板端：`D -> D -> D -> SD` 分时推理、三走廊后处理、台阶时序锁存、深度/几何保守融合、降级回退。
- 演示：Aurora 不改源码，不依赖颜色；使用 PATH 空心边界、WALL 双框/X、STEP 双横线/向下箭头、UNKNOWN 虚线和 FAIL。
- 语音：危险类型首次长描述；普通动作或深度等级没有变化时不重复排队；STOP 和系统故障仍可抢占和重复保护。

尚未完成且必须由外部资源继续的步骤：4090 训练、官方 A1 转换、最终 `.m1model`、双模型实板运行和场景验收。

## 2. 不可变模型契约

检测模型：

```text
input: images, float32, 1x1x384x384
output: 3 x 80-channel class heads + 3 x 64-channel DFL heads
REG_MAX: 16
CPU: sigmoid, DFL, box decode, NMS, tracker
```

SurfaceDepth 模型：

```text
input:        images       1x1x256x256
output[0]:    seg_logits   1x3x64x64
output[1]:    depth_logits 1x16x64x64
classes:      ground_candidate, blocked_surface, step_or_drop
depth bins:   16 log-spaced centers over 0.3 to 8.0 m
CPU:          argmax, softmax/expectation, medians, temporal vote, scale fusion
```

所有训练图像、ONNX 输入、A1 校准 `.npy` 和板端 tensor 都保持单通道。禁止把 Y8 复制为 3 通道。

## 3. 云端从零执行

建议镜像：Miniconda / Python 3.10 / Ubuntu 22.04 / CUDA 11.8。所有环境、缓存、数据和结果必须位于持久盘 `/root/autodl-tmp`。

### 3.1 上传并解压代码

本地只打包已经提交的 Git 内容：

```powershell
Set-Location E:\jichuang\graynav-obstacle-detect
$Commit = (git rev-parse --short=12 HEAD).Trim()
$Package = "E:\jichuang\cloud_packages\graynav-$Commit.tar.gz"
New-Item -ItemType Directory -Force -Path (Split-Path $Package) | Out-Null
git archive --format=tar.gz --output $Package HEAD
Get-FileHash -Algorithm SHA256 $Package
```

上传到 `/root/autodl-tmp/graynav/uploads/` 后：

```bash
export GRAYNAV_WORK_ROOT=/root/autodl-tmp/graynav
mkdir -p "$GRAYNAV_WORK_ROOT"/{uploads,src,public,weights,runs,logs,artifacts}
cd "$GRAYNAV_WORK_ROOT/uploads"
sha256sum graynav-*.tar.gz
mkdir -p "$GRAYNAV_WORK_ROOT/src/graynav-obstacle-detect"
tar -xzf graynav-*.tar.gz -C "$GRAYNAV_WORK_ROOT/src/graynav-obstacle-detect"
cd "$GRAYNAV_WORK_ROOT/src/graynav-obstacle-detect"
```

### 3.2 环境

```bash
nvidia-smi
df -h / /root/autodl-tmp
free -h
chmod +x model_optimization/*.sh
bash model_optimization/setup_cloud_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/graynav/env
```

环境脚本会在持久盘创建 conda prefix，固定安装 PyTorch 2.5.1/cu118、torchvision 0.20.1、Ultralytics 8.3.0、Paddle CPU 2.6.2、ONNX/ONNX Runtime、OpenCV、h5py 和 scipy；可用空间低于 45 GiB 会停止。

### 3.3 先构建 COCO80 真单通道检测 ONNX

Ultralytics 会在首次使用时取得官方 `yolov8n.pt`；也可先把权重放到指定路径：

```bash
cd "$GRAYNAV_WORK_ROOT/src/graynav-obstacle-detect"
export YOLOV8N_WEIGHTS=yolov8n.pt
bash model_optimization/build_coco80_gray1_detector.sh
cat /root/autodl-tmp/graynav_coco80_gray1/a1_ops.json
```

只有 `a1_precheck_passed=true`、输入通道为 1、输出恰好为 3 个 80 通道和 3 个 64 通道 head 才能提交官方 A1 转换。

### 3.4 SurfaceDepth 随机图转换闸门

训练前必须先执行：

```bash
export GRAYNAV_PREFLIGHT_ONLY=1
bash model_optimization/run_surface_depth_cloud.sh
cat /root/autodl-tmp/graynav_surface_depth_run/preflight/a1_ops.json
```

把随机权重 ONNX 提交官方 A1 转换。确认双输出模型转换成功前，不要开始计费训练。转换成功后：

```bash
unset GRAYNAV_PREFLIGHT_ONLY
export GRAYNAV_A1_PREFLIGHT_CONFIRMED=1
```

### 3.5 公开数据

需要自行接受各数据集条款并放置为：

```text
/root/autodl-tmp/graynav_public/ADEChallengeData2016/
/root/autodl-tmp/graynav_public/nyu_depth_v2_labeled.mat
/root/autodl-tmp/graynav_public/splits.mat
/root/autodl-tmp/graynav_public/StairNetV3/{train,val}/...
```

下载后记录原压缩包 SHA256，确认空间，再运行自动转换：

```bash
export GRAYNAV_PUBLIC_ROOT=/root/autodl-tmp/graynav_public
export GRAYNAV_PREPARED=/root/autodl-tmp/graynav_surface_depth_prepared
bash model_optimization/prepare_public_datasets.sh
cat "$GRAYNAV_PREPARED/dataset_summary.json"
```

审计必须满足：3 个 source 均非空、train/val 无 `source_id` 重叠、`input_channels=1`、`rgb_input_used=false`。ADE20K 提供道路/地面/墙面/楼梯语义；NYUv2 提供 NYU40 地面/阻挡语义和深度；StairNetV3 提供台阶掩膜，并在原数据确有 depth 时自动加入深度监督。缺失任务用 loss mask，不生成伪标签。

### 3.6 官方 Fast-SCNN 初始化与完整训练

```bash
mkdir -p /root/autodl-tmp/graynav/weights
cd /root/autodl-tmp/graynav/weights
wget -c -O fast_scnn_cityscapes.pdparams \
  https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams
sha256sum fast_scnn_cityscapes.pdparams

cd "$GRAYNAV_WORK_ROOT/src/graynav-obstacle-detect"
export GRAYNAV_PADDLE_CHECKPOINT=/root/autodl-tmp/graynav/weights/fast_scnn_cityscapes.pdparams
export GRAYNAV_A1_PREFLIGHT_CONFIRMED=1
export GRAYNAV_WIDTH_MULT=1.0
tmux new -s graynav
bash model_optimization/run_surface_depth_cloud.sh 2>&1 | tee /root/autodl-tmp/graynav/logs/surface_depth.log
```

训练循环使用 tqdm 显示 epoch、batch、总损失、分割损失、深度损失和学习率。另开一个终端启动 TensorBoard：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/graynav/env
tensorboard \
  --logdir /root/autodl-tmp/graynav_surface_depth_run/tensorboard \
  --host 0.0.0.0 --port 6006
```

TensorBoard 记录逐 batch 总损失与学习率，并按 epoch 记录分割/深度损失、三类 IoU/F1、危险类别平均 F1、AbsRel、delta1、近远顺序正确率和模型选择分数。

导入器必须打印：

```text
input_shape=1x1x256x256
first_conv_shape=(..., 1, 3, 3)
one_channel_first_conv_initialized=True
rgb_input_used=False
```

训练完成后检查：

```bash
cat /root/autodl-tmp/graynav_surface_depth_run/train/history.json
cat /root/autodl-tmp/graynav_surface_depth_run/deploy/a1_ops.json
cat /root/autodl-tmp/graynav_surface_depth_run/deploy/onnx_consistency.json
cat /root/autodl-tmp/graynav_surface_depth_run/deploy/datasets_contract.json
sha256sum -c /root/autodl-tmp/graynav_surface_depth_run/deploy/SHA256SUMS.txt
```

若 1.0 模型无法在 A1 与检测模型常驻，使用 `GRAYNAV_WIDTH_MULT=0.75` 从官方初始化重新训练；不能裁切 1.0 checkpoint，也不进行蒸馏。

## 4. A1 转换与回传

分别提交：

```text
graynav_yolov8n80_gray1_head6.onnx
graynav_surface_depth_gray1.onnx
datasets.zip
datasets_contract.json
a1_ops.json
```

A1 转换成功后，最终文件固定命名：

```text
graynav_yolov8n80_gray1_head6.m1model
graynav_surface_depth_gray1_int8.m1model
```

检测与 SurfaceDepth 的校准输入尺寸不同，不能混用同一个压缩包。数据准备完成后再次运行检测构建脚本，它会额外生成 `384x384` 的检测 `deploy/datasets.zip`；SurfaceDepth 目录中的压缩包固定为 `256x256`。两个包内都必须是 `float32 NCHW`、`1x1xHxW`、范围 `[0,1]`。转换时分别使用：

```text
model_optimization/configs/a1_coco80_gray1.toml
model_optimization/configs/a1_surface_depth_gray1.toml
```

复制到 Git 与 SDK 的 `app_assets/models/`，再完整编译。不得通过 `scp` 临时替换板端模型。最终验收还必须比较 FP32/INT8 的网格类别和深度等级，并确认两个 SSNE `model_id` 不同。

## 5. 板端构建与演示

Buildroot 配置默认值已切换为 COCO80 + SurfaceDepth。模型未就绪时，编译验证可显式覆盖为旧模型；这不构成新版本镜像。最终候选必须使用默认新契约：

```powershell
docker exec A1_Builder sh -lc `
'cd /home/smartsens_flying_chip_a1_sdk/A1_SDK_SC132GS/smartsens_sdk && ./scripts/a1_sc132gs_build.sh'
```

烧录前检查：

```text
CMakeCache: classes=80, input_channels=1, surface=ON
models: 两个最终文件名与 SHA256
zImage: < 15 MiB
runtime: D/D/D/SD，surface-depth >= 2 Hz
stability: 30 分钟无崩溃、无持续内存增长
fallback: SurfaceDepth 三次失败后检测框/串口/语音继续
```

Aurora 只显示板端生成的灰度图层，不修改软件：

- PATH：细空心走廊；
- WALL：双边界与大 X；
- STEP/DROP：平行横线与向下箭头；
- UNKNOWN：虚线与 `UN`；
- SurfaceDepth 故障：清除道路图形，显示 `FAIL`，保留检测框；
- FAR/MID/NEAR：一格/两格/三格风险条，不显示米数。

## 6. 镜像保护和候选归档

受保护回退镜像：

```text
bytes  = 8214488
SHA256 = A7976710ECB456CB312D18F0195DCAE496ED652EFC582AB698EBC3EB7B055530
```

当前可靠副本至少位于：

```text
E:\jichuang\files\zImage.smartsens-m1-evb
E:\jichuang\firmware_archive\GrayNav_B3_1ch_DCE_25class_A7976710\zImage.smartsens-m1-evb
```

新候选不得覆盖以上路径。完成双模型板测后使用：

```powershell
& E:\jichuang\graynav-obstacle-detect\board\obstacle_detect\scripts\archive_candidate.ps1 `
  -CandidateName GrayNav_COCO80_SurfaceDepth_<日期> `
  -ZImage <新zImage> `
  -DetectorModel <检测m1model> `
  -SurfaceDepthModel <双头m1model> `
  -OnnxAudit <a1_ops.json> `
  -CalibrationContract <datasets_contract.json> `
  -BoardLog <30分钟板端日志>
```

脚本拒绝覆盖已有候选目录，也拒绝把受保护回退镜像误当作新候选；归档会记录 Git commit、工作区状态、全部文件大小与 SHA256。

## 7. 验收解释

离线指标和板端场景必须分别报告。单目深度不输出厘米级承诺；无几何锚点时只用于相对 FAR/MID/NEAR。只有最近至少 3 个可靠 person/bench/chair/couch/table 几何锚点建立稳定尺度后，学习深度才允许写入内部规划距离。几何与学习深度相差超过 40% 时采用更保守风险并降低置信度，不做平均。

任何随机权重 ONNX、只通过交叉编译的 zImage、未完成 A1 转换的模型，都不能标记为“已部署”或“已验收”。
