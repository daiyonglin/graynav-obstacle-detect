# GrayNav Embedded Navigation System

## 1. Data path

SC132GS 输出 720×1280 Y8。系统不执行 Y→BGR，直接创建 `SSNE_Y_8` 输入。每次推理从原图截取 720×720 ROI 并缩放到 384×384；调度周期为 `LOWER → LOWER → UPPER`。所有 ROI 由同一个模型处理，只有 LOWER 推理推进场景和台阶的时序状态。

七个输出按名称、完整 shape、元素数、dtype 和量化 scale 绑定。任一契约不满足时整次结果失效，系统进入故障保护，不复用过期的路况或深度结果。

## 2. Detection and tracking

CPU 对三个尺度执行 sigmoid、DFL、anchor decode、reverse letterbox 和 NMS。每个 anchor 只保留 top-1 类别；同类嵌套框依据 IoS、中心距离和面积关系抑制。

Tracker 以 IoU、中心距离和类别证据关联目标。人体连续 2 次命中后发布，家具和包类连续 3 次后发布。框中心与宽高分别 EMA：高重叠目标权重 0.25，中等移动 0.40，明显移动 0.65；跨 ROI 更新权重不超过 0.25，单次宽高变化限制为 ±20%。Indoor8 类别证据以 0.85 衰减并允许高置信新类别纠正历史误类。

## 3. Monocular ranging

### 3.1 Camera model

默认安装参数：

| Parameter | Value |
|---|---:|
| Full image | 720×1280 |
| Horizontal / vertical FOV | 49.7° / 78.9° |
| Camera height | 0.71 m |
| Downward pitch | 15.0° |
| Ground geometry scale | 1.60 |
| Size-prior scale | 1.00 |
| Planning safety scale | 1.00 |

焦距由视场角计算：

\[
f_x=\frac{W/2}{\tan(\mathrm{FOV}_x/2)},\qquad
f_y=\frac{H/2}{\tan(\mathrm{FOV}_y/2)}.
\]

### 3.2 Ground-contact estimate

对可信完整目标，以框底中心近似落地点。令 \(v\) 为校正后的底边像素、\(c_y=H/2\)、相机俯角为 \(\phi\)，则：

\[
\theta=\arctan\frac{v-c_y}{f_y}+\phi,\qquad
z_g=s_g\frac{h}{\tan\theta}.
\]

底边定位误差通过 \(\partial z/\partial v\) 传播为距离方差；远场小角度会自然获得更大的不确定度。被边界截断或不符合完整人体纵横比的 person 框拒绝地面接触估计。

### 3.3 Size-prior estimate

对完整人体、椅子、桌子、沙发和长椅使用：

\[
z_s=s_s\frac{f_yH_{prior}}{h_{px}}.
\]

局部人体禁止把框高解释为 1.70 m，而使用肩部/躯干/腿部宽度先验：

\[
z_p=s_s\frac{f_xW_{prior}}{w_{px}},
\]

并赋予 38%–45% 的相对不确定度。包等尺寸变化过大的类别不使用固定尺寸先验。

### 3.4 Uncertainty fusion and temporal filtering

当地面与尺寸估计的归一化残差

\[
r=\frac{|z_g-z_s|}{\sqrt{\sigma_g^2+\sigma_s^2}}
\]

不超过 2.5 时，按逆方差融合：

\[
\hat z=\frac{z_g/\sigma_g^2+z_s/\sigma_s^2}
{1/\sigma_g^2+1/\sigma_s^2}.
\]

冲突时保留更可靠证据并扩大方差。公开距离是连续期望值；规划距离使用 `min(display_mean−sigma, planning_mean−sigma)`，并可被近场占屏上界收紧。Tracker 对逆深度保存 5 点历史，使用中位数和非对称 alpha-beta 更新；接近方向的观测更快进入，远离方向要求更多连续证据，防止框漂移把近障碍瞬时推远。

学习深度只用于 `NEAR/MID/FAR`、接近趋势和场景相对顺序，不独立宣称绝对米制精度。

## 4. Scene and stair reasoning

64×64 等效场景网格经 3×3 多数滤波后投影到左、中、右走廊。台阶候选综合：

1. 中央走廊 step 语义比例与连通域；
2. 水平边缘峰值和横向跨度；
3. 边缘上下至少两个深度等级的跳变；
4. LOWER ROI 多帧投票；
5. 与稳定人体/家具框的遮挡关系。

单一床沿或椅背只能形成疑似证据，不能直接触发 confirmed stair。确认台阶在近中距离触发 STOP，远距或疑似台阶触发 SLOW。

## 5. Three-zone navigation

水平分区边界为 `0.35 / 0.65`，中央约占 30%。每个稳定目标按中心进入唯一主分区；只有跨度超过 0.88 的真实宽框才同时封堵多个区域。动作遵循可解释的距离序列：

| Condition | Action |
|---|---|
| Central/wide obstacle `<0.80 m` | STOP |
| Central obstacle `0.80–1.50 m` | SLOW |
| Obstacle `≥1.50 m` and no higher-priority hazard | CLEAR |
| Left-side obstacle `<1.50 m` | RIGHT |
| Right-side obstacle `<1.50 m` | LEFT |
| Confirmed stair or system fault | priority override |

TTC 仅诊断，不参与动作。墙面语义默认不直接改变动作，以避免灰度地面纹理误判导致持续减速；墙面信息仍可出现在诊断与场景融合中。

## 6. Stability, output and fault protection

风险升级快速生效，STOP 解除、左右反转和 CLEAR 恢复采用非对称滞回。最终 `StableGuidance` 是 OSD、串口和语音的唯一输入。

- OSD：Layer 1 显示动作，Layer 2 显示距离档位与方位，Layer 4 最多两个稳定框；Layer 3 仅允许已验证台阶几何。
- Serial：每帧可输出 `[Fxxxxxx] ACTION dir=... cls=... dist=... risk=... zones=...`；诊断字段由开关控制。
- Voice：UART0、9600 8N1、固定 GBK 短语；播报 `直行/减速/停下/左转/右转/异常`，异步 latest-action mailbox 不阻塞推理。
- Camera cover：仅当均值≤45、中心均值≤50、暗像素比例≥80% 且连续 3 帧成立时触发；连续 18 帧恢复。白墙和过曝不按黑色遮挡处理。
- Runtime fault：捕获、模型输出、资源或 UART 异常被隔离；视频、推理和语音线程按各自保护策略降级。
