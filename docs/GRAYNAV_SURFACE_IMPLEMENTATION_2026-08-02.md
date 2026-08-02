# GrayNav 单通道道路感知实施交接（2026-08-02）

## 1. 本轮已经实现的范围

本轮已经把“检测 + 道路分割 + 统一决策”的软件链路实现到可编译状态，但没有伪造训练、A1 转换或板测结果。当前仓库缺少以下外部资源，因此最终分割权重和双模型实机验收仍需继续执行：

- Mapillary Vistas v1.2 与 StairNetV3 原始数据；
- PaddleSeg 官方 Fast-SCNN 预训练参数；
- A1-AI-Tool 转换服务或入口；
- 可运行最终 `.m1model` 的 A1 开发板。

没有最终分割模型时，镜像会明确记录警告并按 detector-only 模式启动，不会把集成构建表述为完整功能已经验收。

## 2. 模型和数据工具

目录：`model_optimization/segmentation` 与 `model_optimization/scripts`。

已实现：

1. `GrayNavFastSCNN`：严格 `1x1x256x256` 输入，输出 `1x4x32x32` raw logits。
2. 固定 8x8 最深特征图、`8/4/2/1` AveragePool、静态 nearest Resize。
3. 1.0 与 0.75 两个宽度版本；0.75 用于双模型无法常驻时的后备重训。
4. Paddle 参数有序导入与首层 `W_R + W_G + W_B` 折叠；0.75 版本按目标通道裁切初始化，不使用蒸馏。
5. Mapillary v1.2 配置文件强制检查、四类自动映射、OpenCV BT.601 兼容灰度转换、官方 split 重叠审计。
6. 50% 普通 Mapillary、25% curb/pothole 中心裁剪、25% 楼梯样本的分组采样。
7. AdamW、5 epoch warm-up、cosine、0.7 CE + 0.3 Dice、固定类别权重与灰度增强。
8. 按危险三类 macro F1 主排序、危险误判为地面比例次排序保存最佳权重。
9. 逐类 IoU/F1、混淆矩阵、危险误判率、最终错误样例输出。
10. ONNX 静态导出、输入输出审计、A1 算子/卷积核限制审计、PyTorch/ONNX 网格一致性工具。
11. 200 张 validation 灰度校准图自动选择与固定 256x256 输出。

### 云端执行顺序

```bash
cd model_optimization
python -m pip install -r segmentation/requirements_surface.txt

python scripts/import_paddleseg_fast_scnn.py \
  --paddle-checkpoint /weights/fast_scnn_cityscapes.pdparams \
  --output runs/graynav_fast_scnn/paddleseg_gray1_init.pt

python scripts/prepare_graynav_surface_dataset.py \
  --mapillary-root /data/mapillary-vistas \
  --stair-root /data/stairnetv3 \
  --output /data/graynav_surface

python scripts/train_graynav_fast_scnn.py \
  --data /data/graynav_surface \
  --output runs/graynav_fast_scnn \
  --pretrained runs/graynav_fast_scnn/paddleseg_gray1_init.pt \
  --epochs 80 --batch-size 16 --width-mult 1.0

python scripts/export_graynav_fast_scnn.py \
  --checkpoint runs/graynav_fast_scnn/best.pt \
  --onnx runs/graynav_fast_scnn/graynav_fast_scnn_gray1_4cls_256.onnx

python scripts/audit_surface_onnx.py \
  --onnx runs/graynav_fast_scnn/graynav_fast_scnn_gray1_4cls_256.onnx \
  --report runs/graynav_fast_scnn/onnx_audit.json

python scripts/validate_surface_onnx.py \
  --checkpoint runs/graynav_fast_scnn/best.pt \
  --onnx runs/graynav_fast_scnn/graynav_fast_scnn_gray1_4cls_256.onnx \
  --images /data/graynav_surface/images/val \
  --report runs/graynav_fast_scnn/onnx_consistency.json

python scripts/build_surface_calibration_set.py \
  --data /data/graynav_surface \
  --output runs/graynav_fast_scnn/int8_calibration --count 200
```

Paddle 只用于导入官方 `.pdparams`，应安装与云端 CUDA 对应的 Paddle/Paddle-GPU 版本。训练主流程使用 PyTorch。导入器会在任意层形状或顺序不匹配时停止，不能用 `strict=False` 静默跳过大量权重。

## 3. A1 转换交接

把已经通过 audit 的 ONNX 与 `int8_calibration` 送入 A1-AI-Tool：

- 输入必须保持一个通道；
- 校准图必须作为 Y8 灰度处理，禁止复制成 BGR；
- batch 固定为 1；
- 输出为四类 32x32 logits；
- 输出量化必须允许跨四类直接 ArgMax（共用量化尺度，或由 SSNE 返回反量化 float）；
- 不添加 Softmax、ArgMax、Transpose 或 256x256 上采样；
- 结果命名为 `graynav_fast_scnn_gray1_int8.m1model`。

转换完成后，同时复制到 Git 管理目录和 SDK 编译目录中的：

```text
app_assets/models/graynav_fast_scnn_gray1_int8.m1model
```

然后重新执行完整 SDK 构建。Buildroot 白名单已经保留 YOLO 和这个分割模型，其他历史 `.m1model` 仍会被删除。

## 4. 板端实现

主要模块：

- `surface_segmentation.*`：动态加载、Y8 ROI 预处理、raw logits、3x3 多数滤波、连通域、三走廊统计、2/3 触发与连续 4 次清除；
- `surface_fusion.*`：按系统故障、检测 STOP、道路落差/坑洼、阻挡面、不确定路况的顺序融合；
- `ObstacleTracker::PredictOnly`：分割帧保留目标轨迹，不把它当成空检测；
- `demo_obstacle.cpp`：D/D/D/S、1.5 秒过期、连续三次失败降级、双模型动态分配重试；
- `VISUALIZER`：左中右绿/黄/红走廊框；
- `VoiceNotifier`：危险类型首次长描述、后续短动作、一次性降级提示；
- JSON/串口：只追加 hazard/source/confidence/degraded 与 surface 字段。

分割结果不写 `distance_m`。`near/mid/unknown` 只来自危险区域在 720x720 下方 ROI 中的纵向位置。

## 5. 构建与板测

完整构建：

```powershell
docker exec A1_Builder sh -lc `
'cd /home/smartsens_flying_chip_a1_sdk/A1_SDK_SC132GS/smartsens_sdk && ./scripts/a1_sc132gs_build.sh'
```

烧录后先运行 30 分钟资源烟雾测试：

```sh
cd /app_demo
A1_DUAL_SMOKE_SECONDS=1800 ./scripts/run_dual_model_smoke.sh
```

日志必须同时出现不同的 YOLO/Surface model_id、`npu_slot=detection` 与 `npu_slot=surface`，且不能出现 `[SURFACE][DEGRADED]`。烟雾脚本记录前后 `MemAvailable`、两类槽位数量和分割错误数。

之后再按平路、墙面、上行台阶、下行边缘、坑洼、阴影/水渍/井盖、多风险叠加、暗光/遮挡、分割故障逐类重复测试。没有安全员保护时不得由测试人员闭眼依赖系统行走。

## 6. 本地已完成的验证

- PyTorch 1.0/0.75 输入输出与首层折叠单元测试通过；
- Mapillary v1.2 关键类别映射单元测试通过；
- 随机权重 1.0 ONNX 约 4.28 MiB，0.75 ONNX 约 2.45 MiB；
- 两个 ONNX 均为 45 Conv、4 AveragePool、4 Resize，无 Shape/Slice/Transpose/Div/Sub/Softmax/ArgMax；
- 板端主程序和 surface logic test 均成功完成 ARM 交叉编译链接；
- detector-only 集成镜像构建成功，zImage 为 8,230,256 bytes；
- CMakeCache 已确认单通道 YOLO、语音和 Surface 分割全部启用；
- Git 板端源码与 SDK 编译源码逐文件 hash 一致。

注意：以上不等于离线精度、INT8 一致性、双模型常驻或实机场景验收已经完成。
