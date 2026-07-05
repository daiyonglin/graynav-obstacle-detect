# GrayNav YOLOv8n 灰度微调与 GrayStem-BC 实验阶段报告

日期：2026-07-05

本阶段目标是重新验证“模型侧灰度适配”是否能带来稳定收益。实验不再使用前置 LUT、Conv adapter 或 GMFE 伪彩色模块，而是围绕 YOLOv8n 的灰度域微调和第一层结构改造进行严谨对照。

## 1. 实验背景

当前板端输入来自 SC132GS 单通道灰度相机。此前 PC 端测试显示，官方 `yolov8n.pt` 在“灰度图复制三通道 `[G,G,G]`”输入下已经有较好的检测效果；但模型切分、格式转换并部署到 A1 开发板后出现检测框不稳定、漏检和误检等问题。

因此本阶段需要回答两个问题：

1. 当前训练和评估是否真的使用了单通道灰度图像，而不是误用 RGB 原图。
2. 在保持 YOLOv8n 预训练能力的基础上，灰度域微调或 GrayStem 第一层结构改造是否能超过原始 YOLOv8n 灰度复制输入。

## 2. 理论依据与实验路线

### 2.1 灰度复制基线

基线输入为：

```text
Y = RGB2GRAY(RGB)
X = [Y, Y, Y]
```

也就是把单通道灰度图复制成三通道后输入官方 YOLOv8n。这个方法与当前板端链路一致，作为 `M0`。

### 2.2 保留 80 类检测头的灰度域微调

前一阶段尝试过把 YOLOv8n 改成 8 类导航检测头，但效果明显下降。主要原因是 8 类头会破坏 COCO 80 类预训练检测头，使分类头重新学习，训练数据规模又不足以恢复原模型能力。

因此本阶段采用更保守的做法：

- 保留 YOLOv8n 原始 80 类检测头；
- 训练数据仍使用 COCO 80 类标签；
- 输入统一为灰度复制三通道；
- 评估阶段再把 80 类预测映射到 GrayNav 导航语义类别。

该模型记为 `M1`。

### 2.3 GrayStem-BC 第一层灰度结构约束

对于灰度复制输入 `[G,G,G]`，YOLOv8n 第一层卷积满足：

```text
Conv([G,G,G]) = W_R * G + W_G * G + W_B * G
              = (W_R + W_G + W_B) * G
```

因此可以把第一层约束到灰度等价子空间：

```text
W_R = W_G = W_B = (W_R + W_G + W_B) / 3
```

这样在约束施加瞬间，对 `[G,G,G]` 输入的第一层输出保持数学等价，同时模型结构上表达“灰度输入不再依赖 RGB 三个独立色彩通道”。该模型记为 `M2 GrayStem-BC`。

本轮训练脚本在以下回调中持续把第一层投影回 GrayStem-BC 子空间：

- `on_train_start`
- `on_train_epoch_start`
- `on_train_batch_end`
- `on_fit_epoch_end`
- `on_train_end`

训练日志中记录的初始等价误差为：

```text
initial_graycopy_equivalence_error = 5.9604645e-07
```

说明 GrayStem-BC 投影在训练开始时对灰度复制输入基本保持数值等价。

## 3. 代码与实验入口

本阶段核心代码：

- `model_optimization/scripts/prepare_yolov8n80_gray_dataset.py`
  - 从 COCO train2017 zip 流式抽样；
  - 生成灰度复制三通道图像；
  - 保留 COCO 80 类 YOLO 标签；
  - 额外生成 GrayNav 8 类评估标注。

- `model_optimization/scripts/audit_gray_dataset.py`
  - 审计训练集、验证集是否满足 `R=G=B`；
  - 输出最大通道差、失败样本数和审计报告。

- `model_optimization/scripts/train_yolov8n_gray_obstacle8.py`
  - 用 Ultralytics 官方训练流程训练 M1；
  - 保留 80 类检测头；
  - 支持 warmup/adapt/stabilize 三阶段。

- `model_optimization/scripts/train_yolov8n_graystem_bc.py`
  - 训练 M2 GrayStem-BC；
  - 持续约束第一层卷积权重满足 GrayStem-BC。

- `model_optimization/scripts/evaluate_graystem_obstacle8.py`
  - 在 GrayNav 8 类语义上统一评估 M0/M1/M2；
  - 支持多种灰度扰动；
  - 统计 COCO AP/AR 和空场景误检。

- `model_optimization/run_graystem_bc_next.sh`
  - 一键执行数据准备、灰度审计、M1 训练、M2 训练、导出和评估。

## 4. 数据集构建

### 4.1 数据来源

本阶段使用 COCO2017：

- 训练来源：`train2017.zip`
- 验证来源：`val2017`
- 标注来源：`instances_train2017.json`、`instances_val2017.json`

训练时不解压全量 train2017，而是从 zip 中流式抽样，避免 50GB 云端数据盘不足。

### 4.2 灰度转换方式

图像转换逻辑：

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
```

这意味着训练和验证图片都是三通道文件，但三个通道完全相同，本质上仍是单通道灰度输入。

### 4.3 数据集规模

本轮数据集参数：

```text
DATA_ROOT      = /root/autodl-tmp/datasets/gray_coco80_graystem_bc
TARGET_IMAGES  = 30000
CONTEXT_IMAGES = 8000
NEGATIVE_IMAGES = 0
```

实际生成：

| Split | Images | Total COCO80 instances |
|---|---:|---:|
| train | 30000 | 237499 |
| val | 5000 | 35821 |

训练集主要实例数：

| Class | Instances |
|---|---:|
| person | 69895 |
| chair | 16007 |
| car | 9865 |
| book | 7044 |
| cup | 6321 |
| dining table | 6312 |
| bottle | 5987 |
| handbag | 4678 |
| bench | 3599 |
| backpack | 3186 |

验证集主要实例数：

| Class | Instances |
|---|---:|
| person | 10633 |
| car | 1907 |
| chair | 1758 |
| book | 1047 |
| bottle | 1007 |
| cup | 893 |
| dining table | 684 |
| traffic light | 601 |
| handbag | 537 |
| truck | 414 |

## 5. 灰度数据审计

审计脚本随机抽查训练集和验证集各 1200 张图像。

| Root | Total images | Sampled | Failed | Max channel delta | Max bad pixel ratio |
|---|---:|---:|---:|---:|---:|
| train | 30000 | 1200 | 0 | 0 | 0.0 |
| val | 5000 | 1200 | 0 | 0 | 0.0 |

结论：

- 训练集和验证集都确认为灰度复制三通道；
- 不存在 RGB 原图泄漏；
- 本轮实验的输入域是可信的。

## 6. 实验模型

### 6.1 M0: 原始 YOLOv8n 灰度复制输入

```text
weights = /root/autodl-tmp/yolov8n.pt
input   = [G,G,G]
train   = no
head    = COCO80
```

基础权重信息：

```text
name   = yolov8n.pt
bytes  = 6549796
sha256 = f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
```

### 6.2 M1: 保守灰度域微调

```text
base weights = yolov8n.pt
input        = [G,G,G]
head         = COCO80
schedule     = warmup + adapt + stabilize
```

训练配置：

| Stage | Epochs | Freeze | lr0 | Mosaic |
|---|---:|---:|---:|---:|
| warmup | 3 | 10 | 0.00012 | 0.00 |
| adapt | 30 | 0 | 0.00008 | 0.08 |
| stabilize | 8 | 0 | 0.00002 | 0.00 |

Ultralytics 内部验证结果：

```text
all images = 5000
instances  = 35821
P          = 0.572
R          = 0.375
mAP50      = 0.401
mAP50-95   = 0.263
```

最终权重：

```text
/root/autodl-tmp/graynav-graystem-bc/model_optimization/runs/detect/runs/detect/M1_yolov8n80_gray_conservative_stabilize/weights/best.pt
```

### 6.3 M2: GrayStem-BC 约束模型

```text
base weights = M1 best.pt
input        = [G,G,G]
head         = COCO80
first layer  = GrayStem-BC constrained first conv
schedule     = warmup + adapt + stabilize
```

训练配置：

| Stage | Epochs | Freeze | lr0 | Mosaic |
|---|---:|---:|---:|---:|
| warmup | 2 | 10 | 0.00008 | 0.00 |
| adapt | 18 | 0 | 0.00005 | 0.05 |
| stabilize | 6 | 0 | 0.000015 | 0.00 |

Ultralytics 内部验证结果：

```text
all images = 5000
instances  = 35821
P          = 0.527
R          = 0.374
mAP50      = 0.386
mAP50-95   = 0.256
```

最终权重：

```text
/root/autodl-tmp/graynav-graystem-bc/model_optimization/runs/detect/runs/detect/M2_graystem_bc_yolov8n80_stabilize/weights/best.pt
```

## 7. 评估设置

评估不是直接看 COCO80 指标，而是统一映射到 GrayNav 导航语义类别后进行 COCOeval。

模型输出仍为 COCO80 类，评估阶段映射到 GrayNav 8 类：

```text
person
chair/seat
table/desk
sofa/bed
bag/suitcase
small_object
vehicle/bicycle
generic_obstacle
```

评估场景：

| Scene | Description |
|---|---|
| normal | 原始灰度验证集 |
| low_light | 低照度 |
| high_exposure | 过曝 |
| low_contrast | 低对比度 |
| motion_blur | 水平运动模糊 |
| noise | 高斯噪声 |
| shadow | 阴影/渐暗 |

指标：

- `AP`
- `AP50`
- `AP75`
- `AR100`
- `empty_fp_per_image`
- `prediction_count`
- `fps`

## 8. 完整评估结果

### 8.1 AP50 对比

| Scene | M0 AP50 | M1 AP50 | M2 AP50 | M1-M0 | M2-M1 | M2-M0 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0.409077 | 0.405512 | 0.397080 | -0.003565 | -0.008432 | -0.011997 |
| low_light | 0.390610 | 0.389255 | 0.378320 | -0.001355 | -0.010935 | -0.012290 |
| high_exposure | 0.372228 | 0.357758 | 0.349863 | -0.014470 | -0.007895 | -0.022365 |
| low_contrast | 0.363738 | 0.371164 | 0.353578 | +0.007426 | -0.017586 | -0.010161 |
| motion_blur | 0.341398 | 0.336599 | 0.317672 | -0.004799 | -0.018927 | -0.023726 |
| noise | 0.359021 | 0.347114 | 0.323119 | -0.011907 | -0.023995 | -0.035902 |
| shadow | 0.406562 | 0.404590 | 0.395452 | -0.001972 | -0.009138 | -0.011109 |

### 8.2 AP 对比

| Scene | M0 AP | M1 AP | M2 AP |
|---|---:|---:|---:|
| normal | 0.271754 | 0.258661 | 0.255116 |
| low_light | 0.258262 | 0.247702 | 0.240424 |
| high_exposure | 0.240773 | 0.222963 | 0.218308 |
| low_contrast | 0.240350 | 0.235001 | 0.226844 |
| motion_blur | 0.224101 | 0.214477 | 0.202684 |
| noise | 0.232738 | 0.215238 | 0.200339 |
| shadow | 0.269579 | 0.257313 | 0.253564 |

### 8.3 AR100 对比

| Scene | M0 AR100 | M1 AR100 | M2 AR100 |
|---|---:|---:|---:|
| normal | 0.449847 | 0.437361 | 0.439564 |
| low_light | 0.434043 | 0.427419 | 0.423756 |
| high_exposure | 0.416731 | 0.402367 | 0.402440 |
| low_contrast | 0.415141 | 0.412185 | 0.410772 |
| motion_blur | 0.394506 | 0.385349 | 0.377485 |
| noise | 0.413251 | 0.399590 | 0.389391 |
| shadow | 0.447128 | 0.436048 | 0.438197 |

### 8.4 空场景误检

| Scene | M0 empty FP/img | M1 empty FP/img | M2 empty FP/img |
|---|---:|---:|---:|
| normal | 0.144 | 0.144 | 0.156 |
| low_light | 0.117 | 0.142 | 0.151 |
| high_exposure | 0.148 | 0.124 | 0.153 |
| low_contrast | 0.119 | 0.117 | 0.140 |
| motion_blur | 0.191 | 0.222 | 0.234 |
| noise | 0.136 | 0.146 | 0.152 |
| shadow | 0.134 | 0.133 | 0.140 |

### 8.5 预测数量

| Scene | M0 predictions | M1 predictions | M2 predictions |
|---|---:|---:|---:|
| normal | 256566 | 279956 | 267997 |
| low_light | 249338 | 276822 | 262844 |
| high_exposure | 260138 | 275518 | 266228 |
| low_contrast | 248444 | 254152 | 237775 |
| motion_blur | 264497 | 281318 | 258749 |
| noise | 261694 | 274452 | 262231 |
| shadow | 252105 | 275598 | 262499 |

## 9. 平均结果

7 种场景平均结果：

| Model | AP50 | AP | AR100 | empty FP/img |
|---|---:|---:|---:|---:|
| M0 raw YOLOv8n graycopy | 0.377519 | 0.248223 | 0.424378 | 0.141286 |
| M1 conservative gray fine-tune | 0.373142 | 0.235908 | 0.414331 | 0.146857 |
| M2 GrayStem-BC | 0.359298 | 0.228183 | 0.411658 | 0.160857 |

相对差值：

| Comparison | AP50 | AP | AR100 | empty FP/img |
|---|---:|---:|---:|---:|
| M1 - M0 | -0.004377 | -0.012314 | -0.010047 | +0.005571 |
| M2 - M1 | -0.013844 | -0.007725 | -0.002673 | +0.014000 |
| M2 - M0 | -0.018222 | -0.020040 | -0.012720 | +0.019571 |

## 10. 导出结果

M2 成功导出 ONNX 和 head6 ONNX。

导出日志显示：

```text
input shape = (1, 3, 384, 384)
full output = (1, 84, 3024)
```

head6 输出：

```text
cls head 1: [1, 80, 48, 48]
cls head 2: [1, 80, 24, 24]
cls head 3: [1, 80, 12, 12]
box head 1: [1, 64, 48, 48]
box head 2: [1, 64, 24, 24]
box head 3: [1, 64, 12, 12]
```

这说明导出链路本身没有失败，M2 的问题主要是精度表现下降，不是导出失败。

## 11. 结论

### 11.1 关键结论

1. 本轮训练和评估确实使用了灰度图像。
   - 训练集和验证集审计均通过；
   - 不存在 RGB 原图泄漏。

2. 保留 80 类检测头的灰度微调比此前 8 类重训更合理，但仍未超过原始 YOLOv8n。
   - `M1` 平均 AP50 比 `M0` 低 `0.004377`；
   - `M1` 平均 AP 比 `M0` 低 `0.012314`；
   - 空场景误检略有上升。

3. GrayStem-BC 结构改造没有带来收益。
   - `M2` 在所有测试场景中 AP50 均低于 `M1`；
   - `M2` 相比 `M0` 平均 AP50 下降 `0.018222`；
   - `M2` 空场景误检最高。

4. 原始 YOLOv8n 对灰度复制输入已经非常强。
   - 单纯使用 COCO 灰度数据继续微调，提升空间很小；
   - 继续在该方向堆训练轮数或轻量结构约束，不太可能显著超过 `M0`。

### 11.2 对 GrayStem-BC 的判断

GrayStem-BC 的数学初始化是成立的，初始等价误差约为 `5.96e-07`。但训练过程中持续约束第一层会降低模型自由度，尤其是在噪声、运动模糊和低对比度场景中下降明显。

因此，GrayStem-BC 可以作为“尝试过且有理论解释的结构改造”写入研究过程，但不建议作为下一版上板模型。

### 11.3 下一步建议

本阶段结果支持以下判断：

- 当前 PC 端模型能力不是主要短板；
- 板端效果下降更可能来自模型切分、量化/转换、输入预处理、head6 解码、DFL、letterbox 反变换、NMS 或 tracker/OSD 稳定性。

下一阶段建议重心转回板端一致性：

1. 固定 `M0 yolov8n.pt graycopy` 作为 PC 参考。
2. 固定一批 golden gray images。
3. 逐级比较：
   - PyTorch full model；
   - ONNX full model；
   - ONNX head6 raw output；
   - Python head6 decoder；
   - C++ board decoder；
   - A1 `.m1model` 输出。
4. 重点检查：
   - BGR/RGB/GRAY 输入顺序；
   - letterbox padding；
   - head stride 对应关系；
   - DFL softmax 维度；
   - cls/box head 顺序；
   - NMS 阈值和坐标反变换；
   - INT8 量化校准集是否覆盖真实灰度场景。

## 12. 本阶段产物

云端最小结果包：

```text
graystem_bc_next_min_results.tar.gz
```

本地分析结果：

```text
E:\jichuang\analysis\graystem_bc_next_min
E:\jichuang\analysis\graystem_bc_next_min\graystem_bc_result_table.csv
```

关键结果文件：

```text
artifacts/graystem_bc_eval/graystem_eval_summary.json
artifacts/graystem_bc_state/gray_dataset_audit.json
artifacts/graystem_bc_state/M1_final_weights.txt
artifacts/graystem_bc_state/M2_final_weights.txt
root/autodl-tmp/datasets/gray_coco80_graystem_bc/dataset_manifest.json
```
