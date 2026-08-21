# GrayNav 侧向避障、连续语音与墙面显示调试记录（2026-08-21）

## 1. 实板日志结论

本轮输入日志包含 91 条稳定导航记录：

```text
STOP             43
SLOW             48
LEFT/RIGHT        0
cls=multiple     68
L/C/R 同为 person 51
cls=blocked       9
```

因此持续播报“停下”的主要原因不是墙面分割，而是一个较大的右侧人体框同时覆盖
左、中、右走廊。旧规划器把三个走廊的占用视为三个独立障碍，生成
`three_corridors_near -> STOP`；稳定决策和语音随后正确地执行了这个错误规划结果。

## 2. 规划修正

目标框现在同时保留两种互不混淆的信息：

- **footprint occupancy**：框覆盖哪些走廊，用于估算候选路径的安全余量；
- **primary sector**：目标中心及轨迹的主方位，用于判断障碍实际位于左、中或右。

只有宽度达到 `WideBoxRatio=0.88` 的真正宽框才丢失侧向身份。普通局部人体框即使
覆盖画面宽度的 55% 到 75%，只要中心位于右侧，就优先产生 `turn_left`；左侧同理
产生 `turn_right`。一个侧向轨迹不会再因为覆盖三个走廊而被解释成三个障碍。

发布稳定轨迹前再次执行包含率去重，抑制上下 ROI 对同一人体形成的嵌套框。真正的
宽近场目标、中央紧急 TTC 或多个具有不同主方位的近场目标仍可触发 STOP。

## 3. 语音节拍

语音内容仍严格限定为：

```text
直行 / 减速 / 停下 / 左转 / 右转 / 异常
```

运行参数调整为：

```text
A1_VOICE_COOLDOWN_MS=0
A1_VOICE_STOP_REPEAT_MS=1000
A1_VOICE_TX_GAP_MS=0
A1_VOICE_STOP_FOLLOWUP_HOLD_MS=0
A1_VOICE_TURN_FOLLOWUP_HOLD_MS=0
```

固定短词的无 ACK 事务仍保留约 900 ms 的播放保护时间。普通动作在该事务完成后立刻
按邮箱中的最新稳定动作继续播报；STOP 在完成后额外等待约 1 s。STOP 降级为
SLOW/LEFT/RIGHT 时不再保留旧动作 2.5 s，从而避免 Aurora 已变化而语音仍旧停留。

## 4. 墙面识别与可视化

墙面不是目标检测框类别，而来自统一模型 `scene_logits` 的
`blocked_surface` 通道：

1. 对 48×48 场景网格逐格 ArgMax；
2. 执行 3×3 多数滤波；
3. 投影到左、中、右走廊；
4. 单走廊 `blocked_ratio >= 0.40` 且最大连通区不少于 12 格，形成当前帧证据；
5. 最近 4 次 LOWER 推理中至少 3 次成立才锁存；连续 4 次消失后解除。

有稳定人物或家具动作时，目标规划优先，目标背后的 blocked mask 不覆盖为墙面。
只有没有命名目标主导、且 blocked_surface 已通过时序确认时才输出墙面原因。

Aurora 不恢复旧的墙面 X、走廊和密集图形。墙面状态使用单一静态位图：

```text
WALL MID FRONT
```

普通目标仍显示 `MID RIGHT` 等距离/方位并配合目标框。因此两者视觉语义清楚：
**有框的是目标障碍；无目标框且带 WALL 的是持续阻挡面。**

## 5. 构建证据与边界

```text
host C++ surface/planner/tracker test = PASS
Python demo contract tests           = PASS (8 tests)
OSD asset audit                      = PASS (40 files, 364570 bytes)
ARM compile/link                     = PASS
full Docker/Buildroot build          = PASS
rootfs .m1model count                = 1
rootfs NAV asset count               = 40
model SHA256                         = 33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
zImage bytes                         = 8134344
zImage SHA256                        = A29912BFB60BA8580E1D13240F098DDF1044813ED6B6E9C9563BC6A8FBAF9878
```

该镜像是待烧录候选，不代表实板方向与语音已完成验收。烧录后应先验证右侧单人体
稳定输出 `LEFT dir=left`，再接 SYN6288 检查 STOP 切换到 LEFT/SLOW 时是否在当前短词
结束后立即跟随最新动作。墙面测试必须在没有人物/家具框主导的情况下进行。

