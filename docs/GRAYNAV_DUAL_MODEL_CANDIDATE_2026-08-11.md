# GrayNav COCO80 + SurfaceDepth E3 构建记录

日期：2026-08-11

## 状态

本记录对应一个“已构建、未烧录、未实板验收”的候选。当前板上的可靠 ROD25
回退版本未被覆盖。

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

构建成功只证明代码、配置和模型能够进入镜像，不能证明双模型可以在 A1 上同时
常驻，也不能证明真实场景识别效果。烧录前必须先归档；烧录后按计划检查两个
`model_id`、输出 dtype/layout、30 分钟稳定性以及平地、墙壁、台阶、人体和椅子场景。
