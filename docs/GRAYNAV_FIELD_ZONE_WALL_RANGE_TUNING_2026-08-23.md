# GrayNav 现场分区、墙面与测距修正记录（2026-08-23）

## 1. 本轮问题与结论

本轮保持统一 Indoor8 + scene21 模型及其量化文件不变，只修改 CPU 后处理与
板端运行参数。现场现象对应三条独立链路：

1. 旧分区边界为 `0.42/0.58`，中央仅占画面宽度的 16%，目标稍偏离中心就会
   触发侧向绕行。
2. `blocked_surface` 与地面灰度纹理存在域偏差，时序锁存后会把无命名目标的
   正常地面升级成墙面减速。
3. 当前单目几何测距依赖近似 FOV、相机高度、俯角和检测框接地点，现场
   4～5 m 目标曾显示为 2～3 m，说明整体存在系统性低估。

## 2. 画面区域改为 35% / 30% / 35%

归一化目标框中心为

```text
u = (x1 + x2) / (2W)
```

新区域规则为：

```text
LEFT    u < 0.35
CENTER  0.35 <= u <= 0.65
RIGHT   u > 0.65
```

因此外侧各占 35%，中央占 30%。只有目标中心真正进入外侧区域时，左侧障碍才
建议右转、右侧障碍才建议左转；中央目标继续根据风险输出直行、减速或停下。
宽度超过 `A1_WIDE_BOX_RATIO=0.88` 的可信宽框仍按封堵证据处理。

运行参数：

```text
A1_SECTOR_LEFT_BOUND=0.35
A1_SECTOR_RIGHT_BOUND=0.65
```

## 3. 默认关闭墙面动作，保留台阶能力

新增：

```text
A1_ENABLE_WALL_GUIDANCE=0
```

默认行为：

- `blocked_surface` 仍被模型计算、后处理和时序统计，可在诊断模式检查；
- `blocked_surface` 不再覆盖目标检测规划器的动作，也不再生成墙面 OSD 原因；
- 地面被误分为 blocked 时，不会单独造成持续 `SLOW`；
- `step_or_drop`、台阶边缘和深度跳变链路完全保留；
- 模型故障、摄像头遮挡和输出契约异常保护完全保留；
- 真正无法判断的 `unknown_other` 仍按保守策略减速。

如果后续取得稳定的本机墙面负样本并完成针对性微调，可将该开关设为 `1`
重新验证，不需要改模型接口。

## 4. 测距公式与现场比例修正

地面交点法使用目标框底部接地点 `v`：

```text
fy = H / (2 tan(FOVv / 2))
theta = atan((v - H/2) / fy) + pitch
z_ground_raw = camera_height / tan(theta)
```

尺寸先验法使用类别高度 `S` 与像素高度 `h_px`：

```text
z_size_raw = fy * S / h_px
```

两者在归一化残差不超过阈值时按逆方差融合；规划器使用
`safe_distance = mean - sigma`，串口显示的是经过 tracker 时序滤波的期望距离。

本轮不篡改物理安装参数，保留：

```text
FOVh=49.7 deg
FOVv=78.9 deg
camera_height=0.71 m
pitch_down=15.0 deg
ground_contact_offset_ratio=0.012
```

新增明确的几何比例参数：

```text
A1_RANGE_GEOMETRY_SCALE=1.60
```

它同时缩放地面法、尺寸法的均值及几何不确定度，再进入融合、tracker 和学习
深度尺度锚定。`1.60` 是根据“4～5 m 被低估为 2～3 m”选择的偏保守首轮值：
它能显著修正远场低估，同时避免直接采用中点比值 1.8 所带来的过度放远风险。

该比例不是厘米级标定。新镜像上板后，应固定安装姿态，在 2、3、4、5 m
放置完整人体或椅子，各静止采集不少于 10 个稳定读数，并记录：

```text
真实距离 / dist / distance_source / confidence / safe_distance
```

只有多距离点误差方向一致时才继续调整比例；若误差随距离非线性变化，应复核
FOV、俯角和框底接地点，不能继续用一个比例掩盖几何模型问题。

## 5. 回归检查

自动测试覆盖：

- 默认分区必须为 `0.35/0.65`；
- 默认墙面动作关闭，blocked 诊断结果不能把 clear 改成 slow；
- unknown 与台阶保守策略仍有效；
- 同一检测框在 scale=1.00 与 scale=1.60 下，几何输出按预期增大；
- 近场占比仍只限制 `safe_distance_m`，不伪造串口精确距离；
- 模型文件保持唯一且 SHA256 不变。

统一模型：

```text
graynav_unified_indoor8_scene21.m1model
bytes  = 4,150,950
SHA256 = 33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
```

## 6. 实板验收顺序

1. 空旷正常地面连续观察 2 分钟，应稳定出现 `CLEAR/直行`，不得因 WALL 持续减速。
2. 目标框中心依次置于 20%、34%、40%、50%、60%、66%、80% 位置，确认只有
   35% 外侧区域触发反向绕行。
3. 在中央 2、3、4、5 m 做静止测距表；随后缓慢接近，确认距离单调下降。
4. 用真实上/下台阶复核台阶提示；关闭墙面动作不得影响台阶边缘显示和警告。
5. 手掌完全遮挡摄像头，确认异常 OSD、串口与语音保护仍然生效。

## 7. Docker 构建结果

完整构建在 `A1_Builder` 中执行：

```text
docker exec A1_Builder sh -lc \
  'cd /home/smartsens_flying_chip_a1_sdk/A1_SDK_SC132GS/smartsens_sdk && \
   ./scripts/a1_sc132gs_build.sh'
```

构建与镜像审计结果：

```text
CMake classes/input/voice = 8 / 1 / ON
rootfs m1model count       = 1
ARM surface_logic_test     = linked successfully
A1_BUILD_SURFACE_TESTS     = OFF after audit
zImage bytes               = 8,133,992
zImage SHA256              = DCFFA53AD6A411321FC145B0D72D851E1F45353DB7809695E822DA53B5C889C7
```

可烧录镜像已独立归档：

```text
E:\jichuang\firmware_archive\GrayNav_FieldZone_WallOff_RangeScale_20260823_DCFFA53A\zImage.smartsens-m1-evb
```
