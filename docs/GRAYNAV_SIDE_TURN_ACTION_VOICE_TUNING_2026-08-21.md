# GrayNav 侧方避障转向与纯动作语音调优证据

日期：2026-08-21

实现提交：`a05bbf1`

适用模型：`graynav_unified_indoor8_scene21.m1model`

## 1. 本轮问题

实板行走测试发现两个影响连续导航的问题：

1. 位于画面左侧或右侧的较大目标会同时覆盖侧区和狭窄中央走廊，旧规划逻辑容易把它
   当成“两走廊无可靠出口”，直接输出 `STOP`；
2. 道路分割没有把候选转向侧稳定判定为 `PATH` 时，旧融合逻辑会把“未知”当成“明确
   不可通行”，再次把转向升级成 `STOP`；
3. 语音会把目标类别或场景描述映射成“前方有……，请……”长句。类别、深度或场景状态
   变化时还可能重复触发，降低实时性和可懂度。

## 2. 侧方避障逻辑

规划器继续使用目标框对左、中、右走廊的覆盖率统计多障碍，但另外记录每个目标框中心的
**主要横向来源**：

```text
box_center < 0.42  -> primary LEFT
0.42..0.58        -> primary CENTER
box_center > 0.58  -> primary RIGHT
box_width >= 0.55 -> 按 CENTER/WIDE 阻挡处理
```

修改后的关键规则为：

| 稳定风险 | 输出动作 |
|---|---|
| 仅主要左侧目标达到 NEAR/WARNING，右侧没有独立风险 | `turn_right` |
| 仅主要右侧目标达到 NEAR/WARNING，左侧没有独立风险 | `turn_left` |
| 主要中央目标 TTC `<1.40 s` | `stop` |
| 三走廊同时近场或可靠宽目标近场 | `stop` |
| 中央近场且只有左侧已验证安全 | `turn_left` |
| 中央近场且只有右侧已验证安全 | `turn_right` |
| 中央近场且无可验证出口 | `stop` |

因此，一个左侧椅子即使边框有一部分进入中央走廊，仍优先提示右转；只有目标真正以中央
为主要位置、形成宽近场阻挡或封住三条走廊时才停止。

道路融合层同时区分：

- `unknown/not-safe`：没有足够证据证明可通行，但也没有检测到持续危险；保留目标规划器
  已给出的转向；
- `persistent_hazard/blocked_persistent`：目标转向侧存在经过时序确认的台阶或阻挡面；拒绝
  该转向，另一侧可靠时改向，否则根据风险程度减速或停止。

## 3. 语音改为纯动作合同

语音线程不再读取 `cause/object_label/hazard_sector/depth_level` 生成描述句，只消费最终稳定
动作：

```text
clear        -> 直行
slow         -> 减速
stop         -> 停下
turn_left    -> 左转
turn_right   -> 右转
system_fault -> 异常
```

已删除 person、obstacle、stair、surface 等长文本 GBK 负载。语音邮箱键现在只等于动作，
目标类别、方位或深度改变但动作未改变时不会重新触发描述。固定短词帧、UART 常开、异步
worker、冷却、STOP/异常优先级和故障恢复机制保持不变。

另外修复了一个时序细节：普通动作只有在满足 `frame_interval`、真正进入发送邮箱后，才被
记录为“已请求”。这样在非采样帧出现的 LEFT/RIGHT/SLOW 变化不会被提前吞掉。

## 4. 自动验证

新增测试覆盖：

- 左侧大框同时覆盖左区与部分中央区，连续确认后必须输出 `turn_right`；
- 右侧大框同时覆盖右区与部分中央区，连续确认后必须输出 `turn_left`；
- 中央近场且 TTC 紧急时仍输出 `stop`；
- 目标转向侧仅为未知路面时保留转向；
- 语音源码中不存在 person/obstacle/stair 长句动作，邮箱只使用最终动作。

验证结果：

```text
Python contract tests       6/6 passed
C++ surface/planner tests   passed
A1 cross compile/link       passed
Docker full zImage build    passed
```

## 5. 新烧录候选

```text
zImage bytes  = 8,128,848
zImage SHA256 = FBF72C05FED3CB6B092953DA1F49DCCF6B053F3AA66BA1E03ED65B19B735963C

model bytes   = 4,150,950
model SHA256  = 33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
rootfs models = 1
```

CMakeCache 继续满足：

```text
A1_ENABLE_VOICE=ON
A1_YOLO_NUM_CLASSES=8
A1_YOLO_INPUT_CHANNELS=1
A1_REQUIRE_MODEL=ON
A1_MODEL_FILENAME=graynav_unified_indoor8_scene21.m1model
```

## 6. 实板复测重点

每项建议连续保持目标 5 秒以上：

1. 左侧放置椅子或人体：状态稳定后应显示/播报“右转”，不应持续停下；
2. 右侧放置椅子或人体：应显示/播报“左转”；
3. 中央近场放置宽椅子或人体：仍应停下；
4. 左侧和右侧同时阻挡、中央可通行：应减速直行，不左右反复；
5. 三个区域同时阻挡：应停下；
6. 在同一动作下切换人体、椅子和桌子：语音不得播报物体名称，也不得因类别变化重复长句；
7. 依次制造 `clear/slow/stop/turn_left/turn_right/system_fault`，确认只播报“直行、减速、
   停下、左转、右转、异常”。

本轮没有修改模型权重、检测阈值、测距公式、OSD 图层预算或回退镜像。
