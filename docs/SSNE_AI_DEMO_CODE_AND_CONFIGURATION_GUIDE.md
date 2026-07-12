# GrayNav `ssne_ai_demo` 代码与配置详解

## 1. 文档范围

本文对应实际部署目录：

```text
smart_software/src/app_demo/obstacle_detect/ssne_ai_demo
```

当前编译配置为：

```text
model = graynav_rod25_gray1_dce_b3_head6.m1model
input_channels = 1
classes = 25
voice = ON
```

工程实现从完整灰度画面到语音指令的闭环：

```text
capture -> dual ROI -> grayscale preprocess -> NPU head6
-> DFL decode/NMS -> multi-object tracker -> ranging
-> corridor planner -> OSD/serial/voice -> health recovery
```

## 2. 目录与文件职责

### 2.1 构建和入口

| 文件 | 职责 |
|---|---|
| `CMakeLists.txt` | 源文件、A1 库、模型资产、编译宏和安装规则 |
| `cmake_config/Paths.cmake` | SDK 头文件与库路径 |
| `demo_obstacle.cpp` | 主程序、模块初始化、帧循环、健康管理、输出 |
| `scripts/run.sh` | 通用启动配置 |
| `scripts/run_voice_both.sh` | OSD+语音的正式硬件启动配置 |

### 2.2 算法接口

| 头文件 | 核心类型 |
|---|---|
| `include/common.hpp` | `DetectionItem`、`DetectionResult`、`ZoneStatus`、`AvoidanceDecision` |
| `include/semantic_config.hpp` | 原始类别、导航语义类别和风险权重接口 |
| `include/tracker.hpp` | 轨迹状态、类别证据、深度状态 |
| `include/ranging.hpp` | 单目几何和尺寸先验测距接口 |
| `include/avoidance_planner.hpp` | 三通行走廊与动作状态机 |
| `include/osd-device.hpp` | Aurora OSD 初始化和绘制 |
| `include/voice_notifier.hpp` | UART 语音工作线程和事务状态 |
| `include/utils.hpp` | IoU、NMS、绘制和通用数学函数 |

### 2.3 实现文件

| 文件 | 核心工作 |
|---|---|
| `src/pipeline_image.cpp` | 从在线 pipeline 获取传感器帧 |
| `src/yolov8_gray.cpp` | 双 ROI、灰度增强、NPU 推理、head6 解码和 NMS |
| `src/semantic_config.cpp` | ROD25 标签映射和类别风险配置 |
| `src/tracker.cpp` | 多目标关联、框平滑、类别滞回、距离时序滤波 |
| `src/ranging.cpp` | 地面投影、尺寸先验、近场上界、不确定度融合 |
| `src/avoidance_planner.cpp` | 三走廊净空评估和行动决策 |
| `src/osd-device.cpp` | 检测框、动作和风险文本叠加 |
| `src/voice_notifier.cpp` | SYN6288 固定帧连续播报 |
| `src/utils.cpp` | NMS、绘制、坐标与格式辅助逻辑 |

## 3. `demo_obstacle.cpp` 主流程

### 3.1 初始化顺序

1. 初始化 SSNE runtime；
2. 创建 `PipelineImageProcessor` 并连接在线摄像头 pipeline；
3. 从 `/app_demo/app_assets/models` 加载编译时指定模型；
4. 初始化 `ObstacleTracker`，内部继续初始化测距和规划器；
5. 初始化 OSD；
6. 根据 `A1_OUTPUT_MODE` 初始化语音；
7. 启动非阻塞退出监听线程；
8. 进入逐帧循环。

### 3.2 每帧执行顺序

```text
GetImageData
  -> analyze_light_stats
  -> detector.Predict
  -> system_health.UpdateResource/RefreshState
  -> tracker.Update
  -> health decision override
  -> OSD
  -> voice_notifier.Update
  -> serial/performance output
```

异常状态优先级高于正常规划。`FaultActive()` 时清除显示目标，使用 `SafeDecision()` 将动作强制设为 `system_fault`。

### 3.3 `SystemHealth`

检测三类异常：

- 数据/摄像头：连续取帧失败、暗/亮遮挡、平坦帧、冻结帧；
- 推理：连续 NPU 推理失败；
- 资源：低 FPS、高 P95、低内存、候选爆炸。

关键阈值：

| 项目 | 阈值 |
|---|---:|
| capture failure | 3 帧 |
| inference failure | 2 帧 |
| bad image evidence | 8 |
| frozen image | 15 帧 |
| low FPS/resource | 20 帧 |
| low memory | 3 次 |
| candidate burst | 5 帧 |
| 故障恢复 | 异常归零后连续 30 个健康帧 |

图像异常统计使用采样均值、标准差、暗/亮像素比例和 FNV 风格采样哈希。坏图证据采用 `+1/-1` 衰减，避免亮边导致遮挡状态瞬时清零。

## 4. 图像输入和双 ROI

传感器完整图像为 `720x1280` 单通道。固定单 ROI 会裁掉画面顶部或底部目标，因此 `yolov8_gray.cpp` 交替处理两个重叠视图：

| 视图 | ROI |
|---|---|
| UPPER | `(0, 0, 720, 720)` |
| LOWER | `(0, 560, 720, 720)` |

两个 ROI 重叠 160 px，每帧只推理一个 ROI，并统一映射回完整坐标。跟踪器知道当前 ROI，只在目标应当可见的视图中增加 missed，避免交替视图误删轨迹。

运行参数：

```bash
A1_FULL_FRAME_WIDTH=720
A1_FULL_FRAME_HEIGHT=1280
A1_DUAL_ROI=1
A1_ROI_UPPER_Y=0
A1_ROI_LOWER_Y=560
```

## 5. 单通道预处理和自适应灰度

模型输入为 `SSNE_Y_8`，尺寸 `384x384`。ROI 经过保持比例缩放和 letterbox，不进行灰度复制三通道。

自适应灰度只在输入统计明显偏暗或偏亮时工作，以有限比例混合局部归一化结果，避免正常光照被过增强：

```bash
A1_ADAPTIVE_GRAY=1
A1_ADAPTIVE_GRAY_DARK_MEAN=75
A1_ADAPTIVE_GRAY_BRIGHT_MEAN=195
A1_ADAPTIVE_GRAY_BLEND=60
```

`A1_DUMP_PREPROCESS_ONCE=1` 可把两个实际 384 输入保存到 `/tmp/yolov8_input*`，用于确认模型真正看到的区域和灰度分布。

## 6. 模型和 head6 后处理

### 6.1 模型接口

当前 B3 模型有六个 HWC 输出：

- 分类头：25 通道，空间尺度 48、24、12；
- DFL 回归头：64 通道，空间尺度 48、24、12；
- stride：8、16、32；
- `reg_max=16`，四条边各 16 个离散 bin。

初始化阶段校验 head 数、尺寸、通道、dtype、layout 和 stride。配置不匹配时不得继续解码。

### 6.2 解码

对每个 anchor：

1. 扫描 25 类获得 top-1；
2. 分类 logit 未达到预计算阈值则跳过；
3. 仅对通过分类筛选的 anchor 计算四组 DFL softmax；
4. 计算期望距离并结合 anchor/stride 得到模型输入坐标框；
5. 逆 letterbox；
6. 加 ROI 原点，映射回 `720x1280`。

候选默认限制：

```bash
A1_NMS_TOP_K=300
A1_NMS_KEEP_TOP_K=40
```

### 6.3 NMS 和大框抑制

同类框约使用 0.60 IoU NMS；中心、面积和高度一致的跨类框执行重复去除。宽度超过约 90% 或低质量 coarse 框不能压制内部更可靠的小框。道路类被屏蔽，粗糙宽框仅作为不确定性证据，不直接产生全域 STOP。

调试入口：

```bash
A1_DEBUG_POSTPROCESS=1
A1_DUMP_HEADS_ONCE=1
A1_HEAD_DUMP_PREFIX=/tmp/yolov8_head
```

## 7. `semantic_config.cpp` 类别层

模型输出保持 ROD25 的 25 个原始类别。语义层把原始标签映射为导航类别并分配风险权重，不改变模型输出张量。

当前原始类别包括：bike、building、car、person、stairs、traffic_sign、electrical_pole、motorcycle、dustbin、dog、manhole、tree、guard_rail、pedestrian_crosswalk、truck、bus、bench、traffic_cone、fire_hydrant、traffic_barrel、plant_pot、electrical_box、chair、bicycle_rack 等。`road` 不作为障碍动作来源。

建筑、交通标志、电气箱等场景结构类只有具备可信近场接地证据时才参与避障，减少背景误检触发紧急动作。

## 8. 多目标跟踪

`ObstacleTracker` 维护：框、类别证据、命中/丢失次数、ROI 可见性、距离、径向速度、方差和 TTC。

关联分数：

```text
0.52*IoU + 0.28*center_similarity
+ 0.12*size_similarity + 0.08*class_compatibility
```

突变门限会拒绝低 IoU 且中心跳跃、尺寸突变或 coarse 框拖拽稳定轨迹的匹配。person 与非 person 的跨类匹配受到更强惩罚。

新轨迹门限：普通目标 0.22，高置信直接 0.45，person 可降至 0.16；coarse 和异常宽框不能建立轨迹。低置信目标至少命中两次才显示。

框平滑根据运动自适应：静止框旧权重约 0.65，快速移动降至 0.15；高置信检测进一步降低旧权重，兼顾稳定和低延迟。当前 ROI 中丢失超过一次或总未见时间超过 700 ms 删除；非当前 ROI 最多短暂保留 250 ms。

类别证据每次衰减 0.92，候选新类别证据超过当前类别 1.20 倍才切换，避免类别闪烁。

## 9. 测距

### 9.1 相机参数

```bash
A1_CAM_FOV_H_DEG=49.7
A1_CAM_FOV_V_DEG=78.9
A1_CAM_HEIGHT_M=0.85
A1_CAM_PITCH_DOWN_DEG=15.0
A1_DIST_MIN_M=0.20
A1_DIST_MAX_M=8.0
```

内参近似：

```text
fx = W / (2*tan(FOV_h/2))
fy = H / (2*tan(FOV_v/2))
```

### 9.2 距离来源

- ground：框底中心反投影射线与地面相交；
- size：person 1.70 m、bench 0.80 m、plant pot 0.35 m、chair 0.85 m 尺寸先验；
- partial person：头部宽度 0.18 m 先验；
- nearfield bound：靠近底部且面积较大时给 0.45/0.70/1.00 m 上界；
- fused：ground 与 size 归一化残差不超过 2.5 时逆方差融合。

输出保守距离：

```text
safe_distance = estimated_distance - sigma
```

风险阈值：`<0.80 urgent`、`<1.05 near`、`<2.00 warning`，否则 far。coarse 或分数低于 0.16 的框不输出伪精确米数。

### 9.3 时序距离和 TTC

跟踪器以 alpha-beta 形式预测距离和径向速度，权重随测量置信度变化。至少三次可靠距离、接近速度大于 0.08 m/s 后计算：

```text
TTC = safe_distance / approach_speed
```

TTC 小于 1.5 s 直接视为 urgent。

## 10. 避障规划

地面横向坐标按约 `x<-0.35`、`-0.35<=x<=0.35`、`x>0.35` 分入左、中、右走廊。每条走廊维护最近保守距离和最小 TTC。

决策：

- `STOP`：wide 可信近障、中央 TTC<1.5 s、中央近障且两侧未验证安全；
- `LEFT`：中央/右侧阻塞，左侧净空>1.35 m 且比右侧多 0.25 m；
- `RIGHT`：镜像条件；
- `SLOW`：侧边近障、任一 warning、目标质量不确定；
- `CLEAR`：无可靠风险。

双 ROI 均需在 500 ms 内有观测，侧方走廊才标记为 verified。系统不会在看不清侧方时冒险提示转向。

## 11. OSD 和串口输出

OSD 最多显示高风险、高质量的目标，顶部显示 `STOP/SLOW/LEFT/RIGHT/CLEAR`，辅助位图显示方向和风险，如 `C_NEAR`、`L_WARN`、`WIDE_NEAR`。默认每 2 帧更新，框坐标均为完整画面坐标。

人类可读串口摘要包含帧号、动作、方向、类别、距离和风险。诊断模式额外输出 capture、preprocess、inference、decode/NMS、tracking/ranging/planning、总延迟、平均 FPS、P95 和抖动。

## 12. 语音模块

语音模块详见 `A1_SYN6288_VOICE_SYSTEM_TECHNICAL_GUIDE.md`。关键原则是：推理线程不等待 UART；最新动作覆盖旧动作；兼容模式使用 2 ms 字节间隔和短语定时完成；遮挡期间 `system_fault` 独占语音。

## 13. 编译期配置

`CMakeLists.txt` 的关键缓存变量：

| 变量 | 当前值 | 影响 |
|---|---|---|
| `A1_ENABLE_VOICE` | ON | 编译语音模块并链接 UART/GPIO |
| `A1_YOLO_NUM_CLASSES` | 25 | 分类 head 校验、类别数组和 tracker 证据长度 |
| `A1_YOLO_INPUT_CHANNELS` | 1 | 输入格式为 `SSNE_Y_8` |
| `A1_MODEL_FILENAME` | B3 head6 | 安装到 rootfs 的唯一主模型 |

修改这些变量后必须完整重新配置和编译，不能只替换 `.m1model`。

## 14. 运行时配置索引

### 14.1 输入/推理

| 参数 | 默认值 |
|---|---:|
| `A1_FULL_FRAME_WIDTH/HEIGHT` | 720/1280 |
| `A1_DUAL_ROI` | 1 |
| `A1_ROI_UPPER_Y/LOWER_Y` | 0/560 |
| `A1_NMS_TOP_K/KEEP_TOP_K` | 300/40 |
| `A1_ADAPTIVE_GRAY` | 1 |
| `A1_ADAPTIVE_GRAY_DARK_MEAN` | 75 |
| `A1_ADAPTIVE_GRAY_BRIGHT_MEAN` | 195 |
| `A1_ADAPTIVE_GRAY_BLEND` | 60 |

### 14.2 显示/诊断

| 参数 | 默认值 |
|---|---:|
| `A1_OSD_INTERVAL_FRAMES` | 2 |
| `A1_OUTPUT_INTERVAL_FRAMES` | 5 |
| `A1_PERF_INTERVAL_FRAMES` | 60 |
| `A1_OUTPUT_SERIAL_DIAG` | 0 |
| `A1_DEBUG_POSTPROCESS` | 0 |
| `A1_CAPTURE_AUTO_RESTART` | 1 |

### 14.3 语音

| 参数 | 默认值 |
|---|---:|
| `A1_OUTPUT_MODE` | both |
| `A1_VOICE_BAUD` | 9600 |
| `A1_VOICE_INTERVAL_FRAMES` | 2 |
| `A1_VOICE_CLEAR_REPEAT_MS` | 1200 |
| `A1_VOICE_STOP_REPEAT_MS` | 1600 |
| `A1_VOICE_FAULT_REPEAT_MS` | 1800 |
| `A1_VOICE_FAULT_HOLD_MS` | 2500 |
| `A1_VOICE_BYTE_GAP_US` | 2000 |
| `A1_VOICE_POST_TX_DELAY_MS` | 30 |
| `A1_VOICE_RX_POLL_MS` | 3 |

## 15. 工具和资产

- `tools/offline_yolov8_head6_test.py`：离线 head6 解码和一致性检查；
- `tools/generate_osd_ssbmp.py`：生成 A1 OSD 位图；
- `tools/build_gray_obstacle_dataset.py`：构建灰度检测数据；
- `tools/voice_gateway_pc.py`：PC 串口语音网关实验；
- `app_assets/osd/*.ssbmp`：动作和风险 OSD 位图；
- `app_assets/models/*.m1model`：源码目录可能保留历史模型，但安装规则会删除非当前模型，控制最终镜像体积。

`src/scrfd_gray.cpp.bak` 是历史备份，不参与 CMake 构建。`__pycache__` 也不属于板端功能代码。

## 16. 修改和验证规范

1. 先修改 Git 管理目录 `board/obstacle_detect`；
2. 同步到实际 SDK 的 `ssne_ai_demo`；
3. 运行应用级交叉编译；
4. 运行 `a1_sc132gs_build.sh` 生成完整镜像；
5. 检查模型名、类别数、输入通道、语音参数和镜像大小；
6. 上板进行 60 秒以上检测、测距、动作、语音和异常回归；
7. 通过后提交并打标签。

当前已验证连续语音回档点：`voice-continuous-stable-20260712`。当前“停下+异常锁存”实现提交为 `9fbec87`。
