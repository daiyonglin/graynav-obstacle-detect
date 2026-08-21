# GrayNav 每帧导航、区域去重与语音恢复记录

日期：2026-08-21  
输入证据：实板 Aurora 截图与完整串口日志  
模型：`graynav_unified_indoor8_scene21.m1model`（未修改）

## 1. 日志复核

日志包含 79 条稳定导航记录，全部为 `SLOW`：

```text
unknown   1
person   54
multiple 15
blocked   9
```

PERSON 区域分布：

```text
仅 RIGHT       53
仅 CENTER       1
CENTER + RIGHT 13
LEFT + RIGHT    2
```

因此，上一版将单个检测框按中心点投入唯一分区的修改已经生效；右侧独占是主要状态。剩余 `CENTER+RIGHT` 和 `LEFT+RIGHT` 来自交替 UPPER/LOWER ROI 暂时保留的两个 PERSON 轨迹，而不是同一个框被复制到三区。

串口启动信息明确显示：

```text
output interval = 90 frames
NAV heartbeat   = 2000 ms
```

这解释了用户看到的十几帧乃至几十帧才输出一条日志。模型推理没有因此停止，慢的是输出策略。

## 2. 区域与重复人体修复

普通目标继续严格按框中心进入一个区域：

```text
x_center < 0.42W            -> LEFT
0.42W <= x_center <= 0.58W  -> CENTER
x_center > 0.58W            -> RIGHT
```

同一人体的面部/肩部框与躯干框现在允许存在小范围纵向间隙，只要：

- 横向中心距离小于 `0.16W`；
- 横向交集不低于较窄框宽度的 55%；
- 纵向间隔不超过 `0.14H`。

该规则在普通 IoU 的“无相交立即返回”之前执行，因此分离的面部和躯干片段能够真正合并。横向明显分离的两个人仍会保留为两个目标。

## 3. 转向与 OSD 更新

旧链路存在两层动作稳定：规划器确认转向后，`GuidanceStabilizer` 又等待两帧。交替 ROI 下原因和目标组合变化，使最终 OSD 长期停在 `SLOW`。

现在：

- 单独右侧障碍形成 `turn_left`；单独左侧障碍形成 `turn_right`；
- `CENTER+RIGHT` 受阻且 LEFT 没有目标时形成 `turn_left`；
- `LEFT+CENTER` 受阻且 RIGHT 没有目标时形成 `turn_right`；
- 规划器已把 SLOW 转换为明确侧向逃生动作时，稳定输出立即采用该转向；
- STOP、左右反向切换和风险解除仍保留原有保守滞回。

因此 Aurora 顶部动作将由最终稳定动作驱动，不应再长期固定为 SLOW。

## 4. 每帧串口输出

运行参数改为：

```text
A1_OUTPUT_INTERVAL_FRAMES=1
```

当该值为 1 时，每次有效主循环均输出一条 NAV：

```text
[Fxxxxxx] LEFT dir=left cls=person dist=1.95m risk=WARNING zones=L:clear,C:clear,R:person@1.95
```

`heartbeat` 和 `min_change` 仍保留为低频模式的兼容参数，但在每帧模式下不再限制 NAV 输出。

## 5. 语音恢复

上一版使用 600 ms 无 ACK 固定短语事务。实板只听到第一次“减速”，推断后续帧在模块仍忙时过早到达并被连续拒绝；详细 TX/DONE 日志此前关闭，串口无法验证实际事务。

本版设置：

```text
A1_VOICE_ACTION_PROMPT_MS=1000
A1_VOICE_COOLDOWN_MS=0
A1_VOICE_STOP_REPEAT_MS=500
A1_VOICE_DIAG=1
```

语音线程每两帧刷新一次最新动作邮箱；工作线程完成一个约 1 秒事务后继续播报当前动作。STOP 在事务结束后增加 500 ms 间隔。验证镜像临时开启语音诊断，串口应连续出现：

```text
[VOICE] seq=1 TX ... action=slow
[VOICE] seq=1 DONE source=timer ...
[VOICE] seq=2 TX ... action=slow
[VOICE] seq=2 DONE source=timer ...
```

若 TX/DONE 连续增长但扬声器不发声，问题位于 SYN6288 电路、忙状态或功放链路；若序号停止增长，则继续依据最后一个状态和错误行修复软件状态机。

## 6. 自动测试与镜像

已通过：

- Python 演示契约测试 9 项；
- C++ 纯 CPU 后处理测试；
- 互斥区域、C+R 左转、纵向人体片段去重和立即发布转向测试；
- A1 SDK Docker 全量构建；
- rootfs 单一模型、CMake 与运行参数审计。

```text
Git commit:   98876df
zImage bytes: 8,134,064
zImage SHA256: 4C9355FA563492A66B36E82629FA772B09196CEA38C1D6B07FE19AC7FEE5D718
model count: 1
model SHA256: 33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
```

该镜像是语音与实时输出验证候选。`A1_VOICE_DIAG=1` 在语音稳定确认后可恢复为 0，以减少最终演示日志。
