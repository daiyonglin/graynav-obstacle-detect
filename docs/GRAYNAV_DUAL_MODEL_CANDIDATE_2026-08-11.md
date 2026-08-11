# GrayNav COCO80 + SurfaceDepth E3 构建记录

日期：2026-08-11

## 状态

本记录对应一个“已构建、已烧录、实板验证失败”的实验镜像。当前可靠 ROD25
回退版本未被覆盖；该双模型镜像不得提升为部署候选。

## 实板失败记录

2026-08-11 Aurora 和串口实测得到以下事实：

- COCO80 人体检测可以输出检测框；
- 串口持续出现 `perception=DETECTION_DEGRADED_SURFACE_DEPTH`、
  `degraded=1`、`hazard=UNKNOWN`；
- 未获得地面、墙面或台阶的有效道路感知与可视化；
- Aurora 出现动态点阵文字/风险条重叠形成的黑点和杂乱状态；
- 高频多行串口混合导航、风险和语音诊断信息，不适合作为演示界面；
- 双模型分时与降级链路没有达到实时、稳定、可解释的目标。

这组现象证明当前运行的是 detector-only 降级路径。失败原因仍需由完整启动日志区分
模型加载、预处理、首次推理、输出数量/次序、dtype 或 layout 绑定问题；在原因查明前，
不能把“没有道路提示”解释为单纯的分割精度不足。

## 模型契约

| 支路 | 模型 | 输入 | 输出/用途 |
|---|---|---|---|
| 物体检测 | `yolov8n80_graycopy_head6.m1model` | Y8 复制为 `1x3x384x384 [G,G,G]` | COCO80 六个 raw head，CPU DFL/NMS/跟踪 |
| 道路与深度 | `graynav_surface_depth_e3_gray1.m1model` | 真单通道 `1x1x256x256` | 4 类道路分割 + 16 级相对深度 |

SurfaceDepth 文件是用户提供的正式 E3 转换结果，仅规范重命名，字节内容未改变：

```text
bytes  = 1,459,634
SHA256 = D40B6F6C6392D062A5C39625B3F39C69E579255583498E2C218BB8C2593106F1
```

## 本轮实现

- E3 四类输出与 `unknown_other` 保守语义；
- INT8 输出尺度处理和 HWC/CHW 契约检查；
- 16 级深度按 NEAR/MID/FAR 聚合，margin 小于 `0.20` 时输出 UNKNOWN；
- 台阶持续危险覆盖错误 FAR，UNKNOWN 不允许 clear；
- 三走廊 PATH/WALL/STEP/UNKNOWN 灰度形状编码；
- OSD 增加主要物体文字、道路状态和风险条；
- 串口增加四类比例、深度概率/margin/歧义字段；
- 语音首次播报危险类型，稳定状态下抑制重复播报；
- SurfaceDepth 连续失败后保留 COCO80 检测、OSD、串口和语音链路。

## 验证证据

主机纯 CPU 后处理测试实际运行通过；A1 完整交叉编译和链接通过。Buildroot
目标根文件系统及 initramfs 中恰好包含上述两个模型。

```text
CMakeCache:
  A1_YOLO_NUM_CLASSES=80
  A1_YOLO_INPUT_CHANNELS=3
  A1_ENABLE_SURFACE_SEG=ON
  A1_MODEL_FILENAME=yolov8n80_graycopy_head6.m1model
  A1_SEG_MODEL_FILENAME=graynav_surface_depth_e3_gray1.m1model

zImage:
  bytes  = 9,219,520
  SHA256 = F0E34C79E84B3B2DEEA84990C3D4F59FE9442B10B4585D66D21B22F8639226FB
```

构建成功只证明代码、配置和模型能够进入镜像，不能证明它们能在 A1 上稳定运行。
本次烧录已经否定该候选。后续保留本记录用于复盘输出契约与降级原因，产品路线改为
单一共享骨干的检测、道路分割和相对深度多任务模型。
