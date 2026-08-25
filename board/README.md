# Flyingchip A1 Runtime

`board/` 是 GrayNav 的可部署 C++ 路径。`obstacle_detect/` 对应 SDK 应用，`sdk_overlay/` 和 `rootfs_overlay/` 分别提供 Buildroot package 与启动配置。

## Runtime modules

| Module | Responsibility |
|---|---|
| `yolov8_gray.cpp` | SSNE 单模型加载、Y8 预处理、7 输出绑定、DFL/NMS、Scene21 解码 |
| `tracker.cpp` | 多 ROI 关联、类别证据、框与距离时序滤波 |
| `ranging.cpp` | 地面交点、尺寸先验、不确定度融合与近场上界 |
| `surface_segmentation.cpp` | 场景网格、深度等级、台阶多证据确认 |
| `surface_fusion.cpp` | 目标与场景风险融合 |
| `avoidance_planner.cpp` | 左/中/右三区动作规划 |
| `guidance_stabilizer.cpp` | 距离、方位和动作非对称稳定 |
| `utils.cpp` | OSD、串口格式和共享导航输出 |
| `voice_notifier.cpp` | SYN6288 UART 异步短指令播报 |

## Model contract

编译固定：

```text
A1_YOLO_INPUT_CHANNELS=1
A1_YOLO_NUM_CLASSES=8
A1_MODEL_FILENAME=graynav_unified_indoor8_scene21.m1model
A1_ENABLE_VOICE=ON
```

模型清单见 `obstacle_detect/app_assets/models/MODEL_MANIFEST.json`。CMake 只安装该模型及固定 OSD 资产。

## Sync and build

```powershell
powershell -ExecutionPolicy Bypass -File board/sync_to_sdk.ps1 `
  -SdkRoot E:\jichuang\docker\docker_test\data\A1_SDK_SC132GS\smartsens_sdk
```

同步脚本会核对模型 SHA256。完整镜像必须在 `A1_Builder` 容器内按官方 SDK Buildroot 流程生成。构建后检查：

```text
rootfs model count = 1
zImage < 15 MiB
model SHA256 = 33eec832...d66da8
voice = ON
input channels = 1
classes = 8
```

## Host contract tests

```powershell
python -m unittest board/obstacle_detect/tests/test_demo_contract.py -v
python board/obstacle_detect/tools/audit_osd_assets.py `
  --root board/obstacle_detect/app_assets/osd
```

`test_surface_logic.cpp` 可用 SDK 交叉编译器构建，覆盖跟踪、测距、场景、三区规划、稳定导航和故障逻辑。

## Runtime configuration

生产默认参数集中在 `obstacle_detect/scripts/run.sh`。关键参数包括：

- `A1_RANGE_NEAR_M=0.80`；
- `A1_RANGE_WARNING_M=1.50`；
- `A1_SECTOR_LEFT_BOUND=0.35`；
- `A1_SECTOR_RIGHT_BOUND=0.65`；
- `A1_ENABLE_WALL_GUIDANCE=0`；
- `A1_COVER_BLACK_MEAN_MAX=45`；
- `A1_COVER_BLACK_RATIO_PCT=80`；
- `A1_VOICE_BAUD=9600`；
- `A1_OUTPUT_SERIAL_DIAG=0`；
- `A1_VOICE_DIAG=0`。

诊断开关只增加日志，不改变模型契约和动作阈值。
