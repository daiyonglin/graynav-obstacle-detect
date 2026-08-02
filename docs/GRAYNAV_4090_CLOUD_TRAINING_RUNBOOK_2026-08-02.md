# GrayNav 4090 单卡云训练与演示交付 Runbook

本文面向 AutoDL RTX 4090 24 GB、Ubuntu 22.04、Python 3.10、CUDA 11.8 镜像。所有大文件、环境、缓存、数据与训练结果必须放在持久盘 `/root/autodl-tmp`，不得放在 30 GB 系统盘。

## 0. 开机前检查

1. 数据盘至少准备 120 GB，推荐 150 GB。当前截图中的 50 GB 不足以同时保存原始压缩包、解压后的 Mapillary、灰度转换集、环境和训练输出。
2. 选择 `Miniconda / conda3 / Python 3.10 / Ubuntu 22.04 / CUDA 11.8`。
3. 在精确的优化前 zImage 找到或完成可靠回退前，不覆盖当前开发板。
4. Mapillary Vistas 必须先由使用者登录并接受官方条款；不要把临时下载链接或账号令牌提交到 Git。

## 1. 云实例预检

```bash
nvidia-smi
df -h / /root/autodl-tmp
free -h
python --version
conda --version
```

预期：能识别 RTX 4090；`/root/autodl-tmp` 有至少 100 GB 可用空间。空间不足时立即停下，不开始下载和计费训练。

所有长任务在 tmux 中运行：

```bash
tmux new -s graynav
# 离开但不中止：Ctrl-b d
# 重新进入：tmux attach -t graynav
```

## 2. 持久目录与 Python 环境

```bash
export GRAYNAV_WORK_ROOT=/root/autodl-tmp/graynav
mkdir -p "$GRAYNAV_WORK_ROOT"/{src,uploads,downloads,datasets/raw,weights,runs,logs,cache/pip,cache/torch,artifacts}
export PIP_CACHE_DIR="$GRAYNAV_WORK_ROOT/cache/pip"
export TORCH_HOME="$GRAYNAV_WORK_ROOT/cache/torch"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$GRAYNAV_WORK_ROOT/env" python=3.10
conda activate "$GRAYNAV_WORK_ROOT/env"
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$GRAYNAV_WORK_ROOT/src/graynav-obstacle-detect/model_optimization/segmentation/requirements_surface.txt"
# 只用于读取一次官方 .pdparams，CPU 版可避免与 PyTorch CUDA 依赖相互覆盖。
python -m pip install paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

环境验收：

```bash
python - <<'PY'
import cv2, onnx, onnxruntime, paddle, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("paddle", paddle.__version__)
print("opencv", cv2.__version__)
print("onnx", onnx.__version__, "onnxruntime", onnxruntime.__version__)
assert torch.cuda.is_available()
PY
```

## 3. 本地代码打包和上传

先在 Windows 本地完成有选择的 Git 提交，确保数据、日志、权重和 SDK 输出未进入提交。不要把本机 GitHub 私钥复制到租用实例。

在本地 PowerShell 中，从已提交版本生成可复现包：

```powershell
Set-Location E:\jichuang\graynav-obstacle-detect
git status --short
$Commit = git rev-parse --short=12 HEAD
$Package = "E:\jichuang\cloud_packages\graynav-surface-$Commit.tar.gz"
New-Item -ItemType Directory -Force -Path (Split-Path $Package) | Out-Null
git archive --format=tar.gz --output $Package HEAD
Get-FileHash -Algorithm SHA256 $Package
```

使用 AutoDL 文件界面把包上传到 `/root/autodl-tmp/graynav/uploads/`，或在取得实例 SSH 主机和端口后使用：

```powershell
scp -P <SSH端口> $Package root@<SSH主机>:/root/autodl-tmp/graynav/uploads/
```

云端校验并解压：

```bash
cd /root/autodl-tmp/graynav/uploads
sha256sum graynav-surface-*.tar.gz
mkdir -p /root/autodl-tmp/graynav/src/graynav-obstacle-detect
tar -xzf graynav-surface-*.tar.gz -C /root/autodl-tmp/graynav/src/graynav-obstacle-detect
```

每次上传都记录 Git commit 和压缩包 SHA256。首轮采用打包上传，避免把私有仓库凭据留在临时云主机；后续如需云端拉取，可单独配置只读 deploy key。

## 4. 数据集放置和审计

数据根目录约定：

```text
/root/autodl-tmp/graynav/datasets/raw/mapillary-vistas/
  config_v1.2.json
  training/images/
  training/v1.2/labels/
  validation/images/
  validation/v1.2/labels/

/root/autodl-tmp/graynav/datasets/raw/stairnetv3/
  train/images/
  train/segmentations/
  val/images/
  val/segmentations/
```

Mapillary 使用 v1.2 语义标签，不得混用 v2.0 taxonomy。楼梯数据只读取 RGB `images` 和现成 `segmentations`，不读取 depth。

若数据提供方给出登录后的临时直链，在云端续传：

```bash
cd /root/autodl-tmp/graynav/downloads
aria2c -c -x 8 -s 8 -o <archive-name> '<signed-download-url>'
sha256sum <archive-name>
```

若平台没有 `aria2c`，使用 `wget -c`。不要把含 token 的完整命令写入项目日志。下载后先观察实际结构再解压：

```bash
find /root/autodl-tmp/graynav/datasets/raw -maxdepth 4 -type d | sort | head -n 100
du -sh /root/autodl-tmp/graynav/datasets/raw/*
df -h /root/autodl-tmp
```

运行自动映射：

```bash
cd /root/autodl-tmp/graynav/src/graynav-obstacle-detect/model_optimization
python scripts/prepare_graynav_surface_dataset.py \
  --mapillary-root /root/autodl-tmp/graynav/datasets/raw/mapillary-vistas \
  --stair-root /root/autodl-tmp/graynav/datasets/raw/stairnetv3 \
  --output /root/autodl-tmp/graynav/datasets/graynav_surface
```

审计输出：

```bash
cat /root/autodl-tmp/graynav/datasets/graynav_surface/dataset_summary.json
wc -l /root/autodl-tmp/graynav/datasets/graynav_surface/manifest_{train,val}.jsonl
du -sh /root/autodl-tmp/graynav/datasets/graynav_surface
```

必须确认 train/val 非空、`source_id_overlap=0`、`input_channels=1`、`rgb_input_used=false`，且训练清单同时含 `mapillary` 和 `stair`。

## 5. 官方权重折叠

把 PaddleSeg 官方 Fast-SCNN Cityscapes `.pdparams` 放到：

```text
/root/autodl-tmp/graynav/weights/fast_scnn_cityscapes.pdparams
```

导入并检查真正单通道首层：

```bash
cd /root/autodl-tmp/graynav/src/graynav-obstacle-detect/model_optimization
python scripts/import_paddleseg_fast_scnn.py \
  --paddle-checkpoint /root/autodl-tmp/graynav/weights/fast_scnn_cityscapes.pdparams \
  --output /root/autodl-tmp/graynav/runs/graynav_fast_scnn_w1.0/paddleseg_gray1_init.pt
```

只有出现以下契约才继续：

```text
input_shape=1x1x256x256
first_conv_shape=(..., 1, 3, 3)
one_channel_first_conv_initialized=True
rgb_input_used=False
```

## 6. 训练：先 1 epoch，再 80 epoch

统一入口会自动准备数据、折叠权重、断点续训、导出 ONNX、审计算子、验证一致性并生成 200 张校准集：

```bash
cd /root/autodl-tmp/graynav/src/graynav-obstacle-detect
chmod +x model_optimization/scripts/run_graynav_surface_cloud.sh

export GRAYNAV_WORK_ROOT=/root/autodl-tmp/graynav
export GRAYNAV_MAPILLARY_ROOT=$GRAYNAV_WORK_ROOT/datasets/raw/mapillary-vistas
export GRAYNAV_STAIR_ROOT=$GRAYNAV_WORK_ROOT/datasets/raw/stairnetv3
export GRAYNAV_PADDLE_CHECKPOINT=$GRAYNAV_WORK_ROOT/weights/fast_scnn_cityscapes.pdparams
export GRAYNAV_BATCH_SIZE=16
export GRAYNAV_WORKERS=8
export GRAYNAV_WIDTH_MULT=1.0
export GRAYNAV_EPOCHS=1

bash model_optimization/scripts/run_graynav_surface_cloud.sh
```

1 epoch 冒烟必须确认：CUDA 被使用、显存无 OOM、loss 有限、四类混淆矩阵可生成、`last.pt/best.pt/ONNX` 存在。记录耗时后再跑完整训练：

```bash
export GRAYNAV_EPOCHS=80
bash model_optimization/scripts/run_graynav_surface_cloud.sh
```

脚本检测到 `last.pt` 后会从 epoch 2 继续。实例中断后重新激活同一环境、重新设置上述环境变量并执行同一命令即可。

监控：

```bash
watch -n 2 nvidia-smi
tail -f /root/autodl-tmp/graynav/logs/train_w1.0.log
df -h /root/autodl-tmp
```

如 batch 16 OOM，先改为 12，再改为 8；不要先改图像尺寸、类别或网络结构。如 1.0 模型不能在 A1 双模型常驻，再把 `GRAYNAV_WIDTH_MULT=0.75` 作为独立实验从头训练，不能复用 1.0 checkpoint。

## 7. 训练产物回传

仅打包可复现实验所需的小文件，不打包公开数据和完整缓存：

```bash
cd /root/autodl-tmp/graynav
tar -czf artifacts/graynav-fastscnn-w1.0.tar.gz \
  runs/graynav_fast_scnn_w1.0/best.pt \
  runs/graynav_fast_scnn_w1.0/last.pt \
  runs/graynav_fast_scnn_w1.0/history.json \
  runs/graynav_fast_scnn_w1.0/onnx_consistency.json \
  runs/graynav_fast_scnn_w1.0/graynav_fast_scnn_gray1_4cls_256.onnx \
  runs/graynav_fast_scnn_w1.0/error_samples \
  runs/graynav_fast_scnn_w1.0/int8_calibration \
  datasets/graynav_surface/dataset_summary.json \
  logs/train_w1.0.log
sha256sum artifacts/graynav-fastscnn-w1.0.tar.gz
```

下载到本地后再次校验 SHA256。`best.pt`、公开数据和训练缓存不提交 Git；最终通过 A1 转换并板测的约 1 MB `.m1model` 才作为部署资产跟随板端代码版本管理，同时记录 ONNX、校准集和 `.m1model` 哈希。

## 8. A1 与演示门禁

训练完成不等于可以烧录。顺序必须是：

1. 离线指标和 ONNX 审计通过。
2. A1 转换得到 INT8 `1x4x32x32` logits。
3. 双模型 30 分钟常驻与交替推理通过。
4. 新镜像单独命名、计算 SHA256，不覆盖回退目录。
5. 先在备用板或可立即回退条件下烧录，再跑安全静态场景。

Aurora 演示采用黑白画面、检测框、左/中/右走廊状态框和顶部最终动作；串口主行只显示 `action/hazard/sector/proximity/source/degraded`，比例细节放调试模式。语音第一次出现新危险时播报“危险类型 + 方向/动作”，危险类型不变时只播短动作；`STOP/system_fault` 可抢占，`clear` 只在检测与分割都连续稳定后播报一次，不持续刷“直行”。
