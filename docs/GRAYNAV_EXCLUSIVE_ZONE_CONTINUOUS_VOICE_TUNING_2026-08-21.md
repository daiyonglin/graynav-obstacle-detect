# GrayNav 互斥障碍分区与连续语音修复记录

日期：2026-08-21  
目标平台：Flyingchip A1 + SC132GS 单通道灰度相机  
模型：`graynav_unified_indoor8_scene21.m1model`

## 1. 本次实板日志结论

对用户上传的完整串口日志解析得到 153 条稳定导航记录：

- `SLOW` 123 条，`STOP` 30 条，`LEFT/RIGHT` 0 条；
- 同一 PERSON 同时写入左、中、右三区 49 次；
- 右侧独占 PERSON 20 次，中右同时占用 54 次；
- 语音日志只有启动信息，因为正常模式关闭了事务级诊断日志，不能用日志行数代表实际播报次数。

截图中的人体主要位于画面右侧，但其检测框较高、较宽。旧实现按目标框与走廊的覆盖关系，把同一目标同时投入多个走廊，导致：

```text
zones=L:person,C:person,R:person
```

规划器因此误以为三侧共同受阻，持续给出 `STOP/hold` 或 `SLOW/forward`，无法形成预期的右侧障碍向左绕行。

## 2. 障碍分区修复

普通命名目标现在只进入一个主分区，按目标框中心横坐标划分：

```text
x_center < 0.42W          -> LEFT
0.42W <= x_center <= 0.58W -> CENTER
x_center > 0.58W          -> RIGHT
```

只有宽度达到画面 `0.88W` 的真正超宽目标才允许同时封堵三区。这样既消除了同一个人的三区复制，又保留正面大面积遮挡物或近距离宽障碍触发停止的能力。

预期串口结果：

```text
# 人体位于右侧
[Fxxxxxx] LEFT dir=left cls=person dist=1.50m risk=WARNING zones=L:clear,C:clear,R:person@1.50

# 人体位于左侧
[Fxxxxxx] RIGHT dir=right cls=person dist=1.50m risk=WARNING zones=L:person@1.50,C:clear,R:clear

# 人体位于中央且达到近场阈值
[Fxxxxxx] STOP dir=hold cls=person dist=0.90m risk=URGENT zones=L:clear,C:person@0.90,R:clear
```

## 3. 转向稳定器修复

旧稳定器按 `(action, cause)` 联合计数。即使动作一直是 `LEFT`，内部原因在 `person/multiple/blocked` 之间变化也会重置计数，稳定输出便长期停留在先前的 `SLOW` 或 `STOP`。

现在改为只按动作累计稳定证据：

- `LEFT/RIGHT/SLOW/STOP/CLEAR` 决定是否完成状态切换；
- `cause` 仅作串口解释和语音上下文，不再阻断动作稳定；
- STOP 的保守退出策略保持不变。

## 4. 同一人体重复框抑制

上下 ROI 轮换时可能分别形成面部/上身框和躯干框。新增同列人体抑制条件：

- 横向中心距离小于 `0.12W`；
- 横向交集占较窄框宽度不低于 0.65；
- 纵向相交，或纵向间隔不超过 `0.06H`。

满足条件时将其视为同一人体，优先保留合理的较完整身体框，避免一个人生成多个轨迹并再次污染多个分区。

## 5. 连续语音策略

语音内容继续只播报动作，不播报障碍物名称：

```text
直行 / 减速 / 停下 / 左转 / 右转 / 异常
```

运行参数：

```text
A1_VOICE_COOLDOWN_MS=0
A1_VOICE_CLEAR_REPEAT_MS=0
A1_VOICE_STOP_REPEAT_MS=1000
A1_VOICE_FAULT_REPEAT_MS=0
A1_VOICE_TX_GAP_MS=0
A1_VOICE_STOP_FOLLOWUP_HOLD_MS=0
A1_VOICE_TURN_FOLLOWUP_HOLD_MS=0
A1_VOICE_ACTION_PROMPT_MS=600
```

此前无 ACK 固定语音帧仍沿用通用 900 ms 事务计时，短动作播报完成后产生额外静默。现在固定动作帧按 600 ms 完成一次事务，worker 保留当前稳定动作并立即进入下一次播报：

- `CLEAR/SLOW/LEFT/RIGHT/FAULT`：固定帧事务结束后连续重复；
- `STOP`：每次播报后保留约 1 秒额外停顿，避免高优先级停止指令过密；
- UART 初始化或发送失败仍不会阻塞推理与视频链路。

600 ms 是软件事务时长，不等同于语音芯片的实际发音时长。上板必须听测是否有截断；若硬件语速较慢，可仅调大 `A1_VOICE_ACTION_PROMPT_MS`，无需修改决策代码。

## 6. 自动测试与构建证据

已通过：

- Python 演示契约测试：8 项；
- C++ 后处理测试，包括互斥分区、右侧障碍左转、类别稳定、人体框去重、原因抖动不阻断转向；
- A1 SDK 完整 Docker 构建；
- rootfs 中仅包含一个统一 `.m1model`。

构建产物：

```text
zImage bytes  = 8,133,392
zImage SHA256 = 5C1B9AD4B5C06EE741C382D199CAA9EFC1241BD6E482C209CA47BE0095D7B40C

model count   = 1
model SHA256  = 33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
```

关键构建契约：

```text
A1_ENABLE_VOICE=ON
A1_MODEL_FILENAME=graynav_unified_indoor8_scene21.m1model
A1_REQUIRE_MODEL=ON
A1_YOLO_INPUT_CHANNELS=1
A1_YOLO_NUM_CLASSES=8
```

## 7. 下一次实板验收

1. 单人稳定处于右侧：串口只能在 `R` 出现 PERSON，动作应为 `LEFT dir=left`。
2. 单人稳定处于左侧：串口只能在 `L` 出现 PERSON，动作应为 `RIGHT dir=right`。
3. 单人处于中央：只能占用 `C`；达到近场风险时才允许 STOP。
4. 超宽近场目标或真实多目标封堵：应保留 `multiple/blocked` 和 STOP。
5. 从系统启动完成后持续听测六种动作语音；确认非 STOP 动作基本无额外静默，STOP 保留约 1 秒间隔。
6. 如发生语句截断，记录具体动作并提高 `A1_VOICE_ACTION_PROMPT_MS`，不要放宽避障阈值。

本文件记录的是已完成的软件测试与构建证据。最终行为仍须以新镜像的实板测试为准。
