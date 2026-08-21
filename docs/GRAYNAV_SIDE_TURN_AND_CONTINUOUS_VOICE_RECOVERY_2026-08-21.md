# GrayNav 侧向转弯与持续语音恢复记录

日期：2026-08-21

## 1. 实板日志结论

本次日志共包含 163 条导航记录：

```text
SLOW  138
STOP   24
RIGHT   1
LEFT    0
```

其中 52 条已经是唯一右侧占用 `L:clear,C:clear,R:person`，但最终动作仍为
`SLOW`。这说明中心点唯一分区已在部分帧生效，转向是在分区之后被场景融合
或动作释放滞回降级。

日志启动契约仍为旧版本：

```text
output interval = 90 frames
voice runtime TX/DONE lines = 0
build contract line = missing
```

因此运行环境没有获得上一版依赖启动脚本设置的逐帧输出与语音诊断参数。

## 2. 规划与融合修复

- 单目标继续依据检测框中心唯一进入 LEFT、CENTER 或 RIGHT。
- 右侧目标生成 `turn_left`，左侧目标生成 `turn_right`。
- 从 `SLOW` 或 `STOP` 进入已经由规划器确认的侧向逃生动作时，不再叠加
  第二层四帧 STOP 释放等待。
- 人物后方的普通 `blocked_surface` 不再覆盖明确的侧方目标转向。
- 所选方向存在持续的 `step_or_drop` 证据时仍可否决转向，保留安全边界。
- 墙面本身仍由没有命名目标时的场景阻挡分支处理。

## 3. 程序内置实时输出契约

为了防止直接启动二进制或旧脚本使参数退回默认值，以下行为已写入程序：

```text
NAV output       = every completed processing loop
Aurora OSD       = every completed processing loop
voice mailbox    = every frame
voice ACK gate   = disabled by default
voice idle query = disabled by default
voice diagnostic = enabled by default for validation
```

启动时必须出现：

```text
[BUILD] contract=per_frame_nav_continuous_voice_side_turn_v2 \
nav_every_loop=1 osd_every_loop=1 voice_continuous=1
```

若缺少该行，说明烧录或启动的不是本记录对应镜像。

## 4. 持续语音状态机

语音仍只播报最终动作：直行、减速、停下、左转、右转、异常。当前动作保存在
latest-action mailbox；每个约 1 秒的固定短词事务由计时器完成，随后立即读取
并播报最新动作，不依赖不稳定的 SYN6288 ACK/IDLE 回传。

验证镜像应持续输出：

```text
[VOICE] seq=N TX frame=... action=turn_left bytes=...
[VOICE] seq=N DONE source=timer duration_ms=...
```

序号持续增长而无声音表示外部语音硬件链路问题；序号停止则表示软件事务问题。

## 5. 构建证据

- Git commit：`262ba6a`
- Python 契约测试：9 项通过。
- A1 主程序完整交叉编译通过。
- 可选 `surface_logic_test` 交叉链接通过。
- rootfs 统一模型数量：1。
- 模型 SHA256：`33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8`。
- zImage 大小：`8,134,552 bytes`。
- zImage SHA256：`3E4B0F81B58A37EAC6BFA08B79505145ED977E23417A70604F41674F4E314FDD`。

## 6. 下一次实板检查

1. 启动必须出现新的 BUILD contract 行，且 `output interval = 1 frames`。
2. 单人在右侧应持续得到 `R:person` 与 `LEFT dir=left`。
3. 单人在左侧应持续得到 `L:person` 与 `RIGHT dir=right`。
4. OSD 动作应与同帧 NAV 动作一致。
5. `[VOICE] TX/DONE` 应连续增长，播报内容应随最终动作切换。
