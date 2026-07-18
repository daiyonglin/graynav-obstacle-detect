# GrayNav 测距算法技术说明

本文档说明 GrayNav 板端系统中目标距离估计、距离置信度、风险等级和避障决策之间的算法关系。当前系统没有引入额外深度估计网络，而是在目标检测框基础上使用单目几何、类别尺寸先验、近场规则和 track 级时序滤波完成实时测距。该设计的核心目标是：在 A1 单核 Cortex-A7、0.8TOPS INT8 NPU 和低内存约束下，以极低 CPU 开销输出足够稳定的近/中/远风险判断。

## 1. 问题定义

检测模型输出每个障碍物的二维框：

```text
B_i = (x1, y1, x2, y2), score_i, raw_class_i
```

测距模块需要估计：

```text
z_i = distance_m
c_i = distance_confidence
r_i = risk_level
```

其中：

- `distance_m` 是相机到目标地面接触点或目标主体的近似水平距离；
- `distance_confidence` 表示该距离估计的可信度；
- `risk_level` 是避障策略真正使用的离散风险等级；
- 当测距不可靠时，`distance_m = -1`，系统只保留目标存在性和风险兜底，不强行输出米级距离。

系统设计原则是：盲人避障优先要求近场风险判断稳定，而不是追求所有类别的精确米级深度。因此算法在近场采用保守估计，在低质量框上避免过度解释。

## 2. 坐标与预处理链路

### 2.1 图像坐标

Aurora 全画面坐标定义为：

```text
u: x axis, left -> right
v: y axis, top -> bottom
W: full image width
H: full image height
```

当前典型配置为：

```text
full image: 720 x 1280
crop image: 720 x 540
model input: 384 x 384
```

检测框先在 384x384 letterbox 空间中解码，再通过 reverse letterbox 映射回全画面坐标。测距、OSD、避障分区全部使用全画面坐标，避免不同模块坐标系不一致。

### 2.2 相机内参近似

系统没有完整标定矩阵时，用水平/垂直 FOV 近似焦距：

```text
fx = W / (2 * tan(fov_h / 2))
fy = H / (2 * tan(fov_v / 2))
cx = W / 2
cy = H / 2
```

运行参数包括：

```text
camera_height_m
camera_pitch_down_deg
fov_h_deg
fov_v_deg
```

其中 `camera_pitch_down_deg` 表示相机光轴相对水平面向下的俯仰角。安装高度和俯仰角是单目几何测距最敏感的参数，必须结合实物距离点标定。

## 3. Ground Projection：地面交点测距

### 3.1 几何假设

对行人、椅子、路锥、楼梯、垃圾桶等落地障碍，检测框底边 `yb = y2` 近似对应目标与地面的接触区域。若地面近似平坦，相机高度为 `h_cam`，则可以通过底边像素对应的俯角求水平距离。

### 3.2 像素到视线角

设检测框底边像素为：

```text
yb = y2
```

垂直方向视线相对光轴的角度近似为：

```text
alpha = atan((yb - cy) / fy)
```

考虑相机向下俯仰：

```text
theta = pitch_down + alpha
```

其中 `theta` 是该像素射线相对水平面的向下夹角。

### 3.3 距离公式

地面交点水平距离为：

```text
z_ground = h_cam / tan(theta)
```

有效性约束：

```text
theta > theta_min
z_min <= z_ground <= z_max
```

若 `theta` 太小，射线接近平行地面，距离会发散；若框底边来自背景结构而非落地点，结果会严重偏差。因此算法会结合框位置、框质量、类别和近场规则判断 ground 距离是否可信。

### 3.4 适用类别

优先使用 ground projection 的类别：

- `person`
- `stairs`
- `traffic_cone`
- `bench`
- `chair`
- `dustbin`
- `plant_pot`
- `generic_obstacle` 中底边可信的障碍

不优先使用 ground projection 的情况：

- 宽大横框；
- 低置信 coarse 框；
- 框底边不靠近真实地面；
- 目标位于画面上部且尺寸较小；
- 屏幕、墙面、栏杆等非落地平面结构。

## 4. Size Prior：类别尺寸先验测距

### 4.1 针孔模型

对于物理尺寸相对稳定的目标，可根据目标真实尺寸与图像尺寸估计距离：

```text
z_size_h = H_real * fy / h_box
z_size_w = W_real * fx / w_box
```

其中：

```text
h_box = y2 - y1
w_box = x2 - x1
```

若类别高度先验更可靠，则使用 `z_size_h`；若宽度先验更稳定，则使用 `z_size_w`。

### 4.2 类别尺寸先验

尺寸先验不是精确物体建模，而是用于稳定风险等级的弱先验。典型先验包括：

| 类别 | 使用维度 | 目的 |
|---|---:|---|
| person | 高度 | 行人主体距离估计 |
| bench/chair | 高度/宽度 | 座椅类近场障碍估计 |
| dustbin/traffic barrel | 高度 | 箱体、容器类落地障碍辅助估计，使用较大方差 |
| traffic_cone | 高度 | 路锥近场识别 |
| plant_pot | 高度 | 小型落地障碍兜底 |
| vehicle/bicycle | 宽度/风险等级 | 室外大目标风险估计，米级距离不强依赖 |

### 4.3 误差来源

尺寸先验的主要误差来自：

- 检测框只框住目标局部；
- 目标姿态变化；
- 目标实际尺寸差异；
- 遮挡导致框高/框宽偏小；
- 近距离鱼眼/广角畸变。

因此 `size prior` 不能单独作为最终距离，必须和 ground projection、近场规则、历史 track 状态共同融合。

### 4.4 部分人体框处理

人体只检测到头部、上半身或腿部时，检测框底边不等于脚底。系统先用候选地面距离反推完整人体期望像素高度：

```text
h_expected = fy * 1.70 / z_ground
visible_fraction = h_box / h_expected
```

当 `visible_fraction < 0.52` 时，拒绝当前 ground estimate，避免把胸口或头部框底误当作地面接触点。随后根据框宽高比选择头宽、肩宽或窄身体宽度先验，并把相对标准差提高到约 32%~45%。这类结果可以稳定风险等级，但不作为厘米级距离。

## 5. Nearfield Fallback：近场保守兜底

盲人避障系统最关键的是近场安全。当前系统对检测框做近场条件判断：

```text
bottom_ratio = y2 / H
height_ratio = h_box / H
area_ratio = box_area / (W * H)
center_overlap = overlap(box, center_zone)
```

当满足以下条件时，即使几何测距不稳定，也进入近场估计：

```text
bottom_ratio high
height_ratio or area_ratio large
center_overlap significant
score above minimum threshold
```

近场兜底输出离散距离：

```text
z_near = 0.45m / 0.70m / 1.00m
```

该距离不是精确测量值，而是风险代理值。它用于触发 `NEAR` 或 `URGENT`，避免近距离大目标因为几何公式异常而被误判为安全。

对低置信、宽大、非贴地框，系统不使用 nearfield 距离，而是输出：

```text
distance_m = -1
distance_source = existence
distance_confidence = low
```

这样可以防止屏幕、墙面、桌沿、床栏等背景结构被错误解释为近距离全域障碍。

## 6. 距离置信度建模

每个距离估计都会附带置信度：

```text
c_dist in [0, 1]
```

置信度由以下因素共同决定：

```text
c_dist = f(score, box_quality, distance_source, bottom_ratio, class_prior)
```

其中：

- `score`：检测置信度；
- `box_quality`：`good / low / coarse`；
- `distance_source`：`ground / size / fused / nearfield / existence / unknown`；
- `bottom_ratio`：落地目标底边是否可信；
- `class_prior`：类别是否适合该测距方式。

典型置信度排序：

```text
fused > ground ~= size > nearfield > existence > unknown
```

但若检测框质量差，`ground` 或 `size` 也会被降权。置信度不仅用于当前帧风险判断，还用于 track 级 Kalman 滤波的测量噪声估计。

## 7. 多源距离融合

### 7.1 融合输入

每个目标最多产生三类候选距离：

```text
z_ground
z_size
z_near
```

系统按类别和置信度选择或融合。

### 7.2 一致性判断

当 ground 与 size 同时有效时，先判断相对差异：

```text
delta = abs(z_ground - z_size) / max(z_ground, z_size)
```

若：

```text
delta < delta_consistent
```

则认为两种估计一致，输出加权融合：

```text
z_fused = (w_g * z_ground + w_s * z_size) / (w_g + w_s)
```

权重来自距离源置信度：

```text
w_g = c_ground
w_s = c_size
```

### 7.3 不一致时的选择

若两种估计明显不一致：

- 对落地强类别，优先 ground；
- 对框底部不可靠但尺寸稳定的类别，优先 size；
- 对贴近画面底部的大目标，优先 nearfield；
- 对低质量宽框，不输出精确距离，只给存在性。

这是一种保守选择策略，目标是减少危险漏报，而不是最小化所有场景的平均距离误差。

## 8. Track 级距离滤波

### 8.1 状态定义

每个稳定 track 维护一维深度状态：

```text
x_k = [z_k, v_k]^T
```

其中：

- `z_k`：当前距离；
- `v_k`：距离变化速度，负值表示目标接近；
- `P_k`：状态协方差。

代码中对应：

```text
depth_m
depth_velocity_mps
depth_cov
```

### 8.2 预测模型

假设短时间内距离变化近似匀速：

```text
z_k^- = z_{k-1} + v_{k-1} * dt
v_k^- = v_{k-1}
P_k^- = P_{k-1} + Q
```

`dt` 由帧间隔估计。过程噪声 `Q` 用于允许目标运动、检测框抖动和测距误差。

### 8.3 测量更新

当前帧测得距离：

```text
z_meas
```

测量噪声随置信度变化：

```text
R = R_min + R_scale * (1 - c_dist)
```

Kalman 增益：

```text
K = P^- / (P^- + R)
```

更新：

```text
z_k = z_k^- + K * (z_meas - z_k^-)
P_k = (1 - K) * P_k^-
```

速度使用平滑的一阶差分估计：

```text
v_meas = (z_k - z_{k-1}) / dt
v_k = beta * v_{k-1} + (1 - beta) * v_meas
```

该滤波可以抑制单帧检测框抖动导致的距离跳变，同时仍能响应持续接近的障碍。

### 8.4 非对称跳变确认

导盲场景中，误把近障判断为更远比误把远障判断为更近更危险。因此距离跳变采用非对称处理：

- 若单帧测量比预测值突然增大超过 `max(0.55m, 35%)`，先保持预测值；
- 只有连续两次远距离测量在 `max(0.35m, 20%)` 范围内一致，才接纳新的远距离；
- 若测量突然变近，则立即接纳，不等待第二帧；
- 等待确认期间增大协方差，并将来源标记为 `temporal_hold_far_outlier`。

该策略就是板端的“跳帧确认”：它不简单丢弃固定帧，而是只延迟安全方向上的突变，危险方向仍保持实时响应。

## 9. TTC：碰撞时间估计

若目标正在接近：

```text
approach_mps = max(0, -v_k)
```

则碰撞时间：

```text
TTC = z_k / approach_mps
```

当：

```text
TTC < 1.5s
```

即使 `z_k` 尚未低于近场距离阈值，也提升风险等级。这可以覆盖“目标快速靠近”或“行人向障碍移动”的动态场景。

## 10. 风险等级定义

测距模块最终输出离散风险：

```text
URGENT / NEAR / WARNING / FAR / UNKNOWN
```

建议判定边界：

| 风险等级 | 条件 |
|---|---|
| URGENT | 中心或 wide 近障 `< 0.8m`，或 `TTC < 1.5s` |
| NEAR | 任一区域可靠目标 `< 1.0m` |
| WARNING | 中心目标 `< 2.0m`，侧边目标 `< 1.5m` |
| FAR | 远距离可靠目标 |
| UNKNOWN | 有目标但距离不可信 |

对盲人导航而言，`UNKNOWN` 不等于安全。若目标处于中心区域且框质量不差，UNKNOWN 会作为弱风险参与 `SLOW` 判断。

## 11. 空间区域建模

画面被划分为三类通行区域：

```text
left / center / right
```

检测框根据横向重叠比例分配区域：

- 主要落在左侧：`left`
- 主要落在中间：`center`
- 主要落在右侧：`right`
- 横跨中心和侧边：`left_center / center_right`
- 横跨三域：`wide`

`wide` 框只有在质量好、贴地可信、近场可信时才会同时影响三域。低质量 wide/coarse 框只作为背景风险证据，不允许单独触发全域阻塞。

## 12. 从测距到避障动作

避障策略不是直接根据单个最近目标决策，而是根据三个区域的 clearance 和风险综合判断：

```text
clearance(zone) = nearest reliable distance in zone
```

动作规则：

| 动作 | 触发条件 |
|---|---|
| STOP | 中心近距离阻塞且左右也近，或 wide 可靠近障，或 TTC 极短 |
| LEFT | 右侧近障，或中心阻塞但左侧 clearance 更大 |
| RIGHT | 左侧近障，或中心阻塞但右侧 clearance 更大 |
| SLOW | 中心 warning、侧边 warning、多目标不稳定但前方存在风险 |
| CLEAR | 连续稳定无可靠近障 |

策略目标是让输出动作稳定且短促，方便 OSD 和语音播报。

## 13. 误差来源与补偿策略

### 13.1 误差来源

1. 相机高度/俯仰角标定误差；
2. 广角畸变导致边缘几何误差；
3. 检测框底边不等于真实落地点；
4. 遮挡或截断导致框尺寸偏小；
5. 多目标合并成宽框；
6. 低光、强光、运动模糊导致检测框抖动；
7. 类别误判导致尺寸先验选择错误。

### 13.2 补偿策略

系统使用以下方法降低误差影响：

- 用 `quality` 标记 `good / low / coarse`；
- 对 wide/coarse 框降低距离置信度；
- 用 ground 与 size 做一致性验证；
- 用 nearfield fallback 保守处理近距离大目标；
- 用 track 级 Kalman 平滑距离；
- 用 TTC 捕捉快速接近风险；
- 用区域策略代替单目标最近距离策略。

## 14. 参数标定建议

建议在板端用以下固定距离点做标定：

```text
0.5m, 1.0m, 1.5m, 2.0m
```

每个距离点测试：

- person
- chair/bench
- traffic_cone 或等效小障碍
- stairs/台阶边缘

标定顺序：

1. 固定相机高度；
2. 调整 `camera_pitch_down_deg`，使 ground projection 在 1m 和 1.5m 处误差最小；
3. 检查 0.5m 是否稳定触发 NEAR/URGENT；
4. 检查 2m 是否不误触发 STOP；
5. 若宽框频繁误报近距离，提高 coarse/wide 过滤强度；
6. 若近障漏报，提高 nearfield fallback 灵敏度。

## 15. 异常保护策略

测距模块依赖图像和检测输出，因此异常处理必须优先于普通避障策略。

### 15.1 摄像头/数据异常

检测条件：

- 连续采集失败；
- 图像全黑、全白；
- 长时间低对比；
- 帧长时间不更新。

保护策略：

```text
clear OSD boxes
action = STOP
voice = 停下
try restart capture pipeline
```

### 15.2 推理异常

检测条件：

- `ssne_inference` 返回失败；
- 输出 head 数量、shape、dtype、layout 不匹配；
- 候选长期为 0 或候选数量异常爆炸。

保护策略：

```text
discard current frame
action = STOP
print AI ERR / tensor metadata
```

### 15.3 资源异常

检测条件：

- FPS 长期过低；
- 后处理候选数量过多；
- UART 发送/回传异常；
- OSD 绘制失败。

保护策略：

```text
reduce output complexity
limit display boxes
increase candidate threshold if needed
action = STOP when uncertainty is high
```

## 16. 算法特点总结

GrayNav 测距不是单一公式，而是一个轻量级多源估计系统：

```text
detection box
  -> ground projection
  -> size prior
  -> nearfield fallback
  -> confidence estimation
  -> source fusion
  -> track-level Kalman filter
  -> TTC
  -> zone risk
  -> avoidance action
```

这种设计牺牲了一部分精确深度建模能力，但满足板端实时性、可解释性和安全优先要求。对于盲人避障任务，最终验收重点应是近/中/远风险等级和动作决策正确率，而不是所有类别的绝对米级误差。
