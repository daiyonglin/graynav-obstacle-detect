# A1-SYN6288 导盲语音播报系统技术说明

## 1. 系统目标与边界

语音子系统把板端目标检测、测距和避障规划产生的高频状态，转换为盲人能够及时执行的低带宽行动指令。系统只播报 `直行、减速、停下、左转、右转、异常`，不在语音中加入类别和距离，避免长句阻塞后续紧急指令。

完整链路如下：

```text
SC132GS 灰度图像
  -> A1 NPU 目标检测
  -> CPU 跟踪、测距、通行走廊规划
  -> AvoidanceDecision.action
  -> VoiceNotifier 最新动作邮箱
  -> UART0 固定协议帧
  -> 电平转换 1.8 V <-> 3.3 V
  -> SYN6288
  -> 功放/喇叭
```

当前已验证稳定版本使用 `syn6288_compat_state_machine_v2` 和 `persistent_paced_duplex`。已验证的连续播报基线 Git 标签为 `voice-continuous-stable-20260712`。

## 2. 硬件连接

### 2.1 引脚定义

当前使用 A1 UART0：

| A1 P4 引脚 | A1 信号 | 方向 | 电平转换后连接 |
|---|---|---|---|
| P4-15 | `A1_D0_UART0TX` | A1 输出 | SYN6288 `RXD` |
| P4-16 | `A1_D2_UART0RX` | A1 输入 | SYN6288 `TXD` |
| P4-33/34/47/48 之一 | GND | 公共参考 | SYN6288、电平转换模块 GND |

串口配置固定为 `9600 baud, 8 data bits, no parity, 1 stop bit`，即 9600 8N1。

### 2.2 电平与供电

A1 外设侧为 1.8 V 逻辑，语音模块侧为 3.3 V UART，因此 TX、RX 均经过双向电平转换。电平转换两侧必须共地。语音模块供电按照模块载板要求单独提供，不能把 P4 的 1.8 V 逻辑电源当作语音功放电源。

BUSY GPIO 当前未使用。忙闲管理由软件节奏和 UART RX 状态字节共同完成。

## 3. 从避障决策到语音动作

`AvoidancePlanner` 输出 `AvoidanceDecision.action`：

| 内部动作 | 中文语音 | 触发含义 |
|---|---|---|
| `clear` | 直行 | 中央通路连续确认安全 |
| `slow` | 减速 | 警告距离、侧方近障或测距不确定 |
| `stop` | 停下 | 中央紧急阻塞、宽近障或极短 TTC |
| `turn_left` | 左转 | 中央/右侧阻塞且左侧通路已验证 |
| `turn_right` | 右转 | 中央/左侧阻塞且右侧通路已验证 |
| `system_fault` | 异常 | 摄像头、图像、推理或资源异常 |

避障规划不是按单个框直接播报，而是把目标映射到左、中、右三个地面通行走廊。转向提示只有在候选侧连续观测安全、净空大于 1.35 m 且优于另一侧至少 0.25 m 时产生。无法确认侧方安全时退化为减速或停下。

动作时序具有滞回：风险升级立即进入；普通动作至少两次确认；左右反向需稳定 300 ms；`clear` 需稳定 700 ms；`stop` 解除需稳定 500 ms。

## 4. SYN6288 数据帧

### 4.1 帧格式

文本合成命令采用：

```text
FD | LEN_H | LEN_L | CMD | PARAM | GBK_TEXT... | XOR
```

- `FD`：帧头。
- `LEN`：从命令字到校验字节的数据区长度。
- `CMD=01`：语音合成命令。
- `PARAM=01`：GBK 文本、无背景音乐。
- `XOR`：此前所有字节逐字节异或。

为消除运行时编码和动态内存差异，正式运行使用预计算固定帧：

| 动作 | GBK 文本 | 完整十六进制帧 |
|---|---|---|
| 直行 | `D6 B1 D0 D0` | `FD 00 07 01 01 D6 B1 D0 D0 9D` |
| 减速 | `BC F5 CB D9` | `FD 00 07 01 01 BC F5 CB D9 A1` |
| 停下 | `CD A3 CF C2` | `FD 00 07 01 01 CD A3 CF C2 99` |
| 左转 | `D7 F3 D7 AA` | `FD 00 07 01 01 D7 F3 D7 AA A3` |
| 右转 | `D3 D2 D7 AA` | `FD 00 07 01 01 D3 D2 D7 AA 86` |
| 异常 | `D2 EC B3 A3` | `FD 00 07 01 01 D2 EC B3 A3 D4` |

### 4.2 回传状态

软件持续读取 UART RX，识别 SYN6288 状态码：

| 状态码 | 含义 | 当前处理 |
|---|---|---|
| `0x41` | 接收成功 | 统计和诊断 |
| `0x45` | 接收失败 | 统计；兼容模式不锁死播报 |
| `0x4A` | 初始化完成 | 更新模块状态 |
| `0x4E` | 正在播放 | 更新模块状态 |
| `0x4F` | 播放完成/空闲 | 强 ACK 模式使用；兼容模式忽略迟到完成码 |

实测载板在整帧 API 调用和状态查询下会持续返回 `0x45`，因此正式配置不把 ACK 或查询结果作为允许播报的硬门槛。

## 5. 稳定连续播报机制

### 5.1 持久 UART 与字节节拍

UART 在程序初始化时打开，在退出时关闭，不对每条语音重复 open/close。A1 API 每次发送一个字节，字节间隔默认 2000 us，整帧后等待 30 ms。该节拍是当前硬件组合上实测可连续工作的兼容路径。

### 5.2 推理线程与语音线程解耦

推理主线程只调用 `VoiceNotifier::Update()` 更新最新动作。独立工作线程负责：

1. 每 3 ms 唤醒并消费 RX FIFO；
2. 读取最新动作邮箱；
3. 检查当前语音事务是否完成；
4. 到达重复周期后发送下一帧。

UART 延迟不会阻塞相机采集、NPU 推理和 OSD。

### 5.3 最新动作邮箱

系统不维护 FIFO 语音队列。高频决策只覆盖 `pending_action_`，避免几秒后仍播报已经过期的行动。

兼容模式不会在模块忙时直接抢发。当前短语最长按 900 ms 定时完成，随后立即发送邮箱中的最新动作。这避免忙时收到 `0x45` 丢弃“停下/异常”，也避免上一条语音迟到的 `0x4F` 截断新短语。

### 5.4 重复播报

| 动作类型 | 默认重复周期 |
|---|---:|
| 直行 | 1200 ms |
| 减速/左转/右转 | 1200 ms |
| 停下 | 1600 ms |
| 异常 | 1800 ms |

重复周期从上一条事务完成后计算。当前动作不变也会周期播报，从而提供持续导航反馈。

### 5.5 异常独占与恢复

摄像头遮挡不是一帧二值判断，而是采用故障证据累计和恢复滞回：

- 暗遮挡：均值 `<38`、暗像素比例 `>0.82`、标准差 `<10`；
- 强光遮挡：均值 `>230`、亮像素比例 `>0.88`；
- 平坦帧：标准差在 `(0, 2.5)`；
- 异常证据按坏帧 `+1`、好帧 `-1` 衰减，达到 8 触发；
- 故障锁存后，所有异常计数必须归零，再连续 30 个健康处理帧才能解除；
- 语音层在最后一次异常后额外保持 `system_fault` 2500 ms。

因此遮挡期间普通的 `clear` 结果不能覆盖“异常”；解除遮挡后也不会因短暂亮边立刻播报“直行”。

## 6. 关键代码

| 文件 | 作用 |
|---|---|
| `include/voice_notifier.hpp` | 语音状态、线程、邮箱、UART 和统计字段定义 |
| `src/voice_notifier.cpp` | 固定帧、UART 初始化、节拍发送、RX 消费、事务调度 |
| `src/avoidance_planner.cpp` | 从三走廊风险生成行动指令 |
| `demo_obstacle.cpp` | SystemHealth、异常锁存、将安全决策送入语音模块 |
| `scripts/run_voice_both.sh` | 已验证的运行参数和硬件后端配置 |
| `CMakeLists.txt` | `A1_ENABLE_VOICE` 编译开关和 UART/GPIO 库链接 |

## 7. 正式运行参数

```bash
A1_OUTPUT_MODE=both
A1_VOICE_BACKEND=a1_uart
A1_VOICE_BAUD=9600
A1_VOICE_FIXED_FRAME=1
A1_VOICE_REOPEN_EACH_TX=0
A1_VOICE_BYTE_GAP_US=2000
A1_VOICE_POST_TX_DELAY_MS=30
A1_VOICE_PASSIVE_RX=1
A1_VOICE_ACK=0
A1_VOICE_REQUIRE_ACK=0
A1_VOICE_QUERY_IDLE=0
A1_VOICE_RX_POLL_MS=3
A1_VOICE_FAULT_HOLD_MS=2500
```

不得在未经回归测试时同时调整字节间隔、重复周期和短语长度；三者共同决定模块是否忙时接收下一帧。

## 8. 日志与验收

启动应打印：

```text
protocol=syn6288_compat_state_machine_v2
transport=persistent_paced_duplex
```

典型事务：

```text
[VOICE] seq=12 TX frame=... action=clear bytes=10
[VOICE] seq=12 DONE source=timer duration_ms=...
```

健康摘要中的 `tx` 应持续增加，`completed` 应随之增加，`recoveries` 在兼容模式下应保持为 0。验收至少覆盖：持续直行、减速、完整“停下”、左右转、遮挡期间持续“异常”、解除遮挡后恢复导航，以及 60 秒以上连续运行。

## 9. 回档

当前连续播报基线：

```bash
git switch --detach voice-continuous-stable-20260712
```

该标签对应 `c43385b`。回档后需要把受管仓库的 `board/obstacle_detect` 同步到实际 SDK，并重新执行完整 Docker 编译，不能只替换主机侧源码。
