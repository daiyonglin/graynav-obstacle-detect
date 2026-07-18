# A1 持续控制 SYN6288 语音播报实现方案

本文档梳理 A1 开发板通过 UART 持续控制 SYN6288 语音合成模块播报避障动作的完整链路。目标是让板端检测/测距/避障决策产生的低频动作指令稳定播报为：`直行`、`减速`、`停下`、`左转`、`右转`。

## 1. 系统链路

整体数据流为：

```mermaid
flowchart LR
    A["SC132GS 灰度相机"] --> B["A1 NPU 目标检测"]
    B --> C["CPU 后处理 / tracker / 测距"]
    C --> D["避障决策: STOP/SLOW/CLEAR/LEFT/RIGHT"]
    D --> E["Aurora OSD 显示"]
    D --> F["VoiceNotifier 语音仲裁"]
    F --> G["A1 UART_TX0/RX0"]
    G --> H["电平转换 1.8V <-> 3.3V"]
    H --> I["SYN6288 语音模块"]
    I --> J["功放 / 喇叭"]
```

语音模块只播报“动作”，不播报类别、距离和长句。这样可以避免一条语音过长导致行动提示滞后。

## 2. 硬件连接

当前接线方案：

| A1 开发板 | 中间模块 | SYN6288 模块 | 说明 |
| --- | --- | --- | --- |
| `A1_D0_UART0TX` | 1.8V -> 3.3V | `RXD` | A1 向语音模块发送命令帧 |
| `A1_D2_UART0RX` | 3.3V -> 1.8V | `TXD` | SYN6288 向 A1 回传状态字节 |
| `GND` | 共地 | `GND` | 必须共地 |
| 可选 GPIO 输入 | 3.3V -> 1.8V | `Ready/Busy STATUS` | 推荐接入，用于稳定判断忙闲 |

A1 UART 引脚是 1.8V 电平，SYN6288 模块常见 TTL 侧为 3.3V，因此中间必须使用双向或方向正确的电平转换。TX/RX 必须交叉连接。

## 3. A1 UART 驱动要求

依据本地 `UART驱动说明`：

- A1 使用官方 `uart_api.h`。
- 使用前加载 `uart_kmod.ko`，GPIO 使用前加载 `gpio_kmod.ko`。
- A1 支持 `UART_TX0` 和 `UART_RX0`。
- `GPIO_PIN_0` 配置为 `UART_TX0` 输出复用。
- `GPIO_PIN_2` 配置为 `UART_RX0` 输入复用。
- `uart_init()` 后配置 TX0/RX0 波特率。
- TX0 和 RX0 波特率必须一致，并且与外设一致。
- SYN6288 默认稳定波特率是 `9600`，所以当前使用 `9600 8N1`。
- UART FIFO 为 32 字节，发送超过 32 字节必须分片。当前短词帧均小于 32 字节，但代码仍保留分片发送。

当前代码入口：

- `include/voice_notifier.hpp`
- `src/voice_notifier.cpp`

核心初始化流程：

```cpp
gpio_init();
gpio_set_alternate(gpio, GPIO_PIN_0, GPIO_AF_INPUT_NONE, GPIO_AF_OUTPUT_UART_TX0);
gpio_set_alternate(gpio, GPIO_PIN_2, GPIO_AF_INPUT_UART_RX0, GPIO_AF_OUTPUT_NONE);
uart_init();
uart_set_baudrate(uart, UART_TX0, 9600);
uart_set_baudrate(uart, UART_RX0, 9600);
uart_set_parity(uart, UART_TX0, UART_PARITY_NONE);
uart_set_parity(uart, UART_RX0, UART_PARITY_NONE);
```

## 4. SYN6288 协议帧

SYN6288 合成播放命令帧格式：

| 字段 | 字节 |
| --- | --- |
| 帧头 | `0xFD` |
| 数据区长度高字节 | `LEN_H` |
| 数据区长度低字节 | `LEN_L` |
| 命令字 | `0x01`，合成播放 |
| 命令参数 | `0x01`，GBK/GB2312 文本，无背景音 |
| 文本 payload | GBK 编码中文短词 |
| 校验 | 从帧头到文本最后一字节逐字节 XOR |

数据区长度为：`命令字 1 + 参数 1 + 文本长度 + 校验 1`，即 `payload_len + 3`。

当前五条短词帧：

| 动作 | 中文 | GBK payload | 完整帧 |
| --- | --- | --- | --- |
| `clear` | 直行 | `D6 B1 D0 D0` | `FD 00 07 01 01 D6 B1 D0 D0 9D` |
| `slow` | 减速 | `BC F5 CB D9` | `FD 00 07 01 01 BC F5 CB D9 A1` |
| `stop` | 停下 | `CD A3 CF C2` | `FD 00 07 01 01 CD A3 CF C2 99` |
| `turn_left` | 左转 | `D7 F3 D7 AA` | `FD 00 07 01 01 D7 F3 D7 AA A3` |
| `turn_right` | 右转 | `D3 D2 D7 AA` | `FD 00 07 01 01 D3 D2 D7 AA 86` |

控制命令：

| 功能 | 完整帧 |
| --- | --- |
| 停止合成 | `FD 00 02 02 FD` |
| 状态查询 | `FD 00 02 21 DE` |

## 5. SYN6288 回传与 BUSY 机制

数据手册中关键返回码：

| 返回码 | 含义 |
| --- | --- |
| `0x4A` | 芯片初始化成功 |
| `0x41` | 收到正确命令帧，接收成功 |
| `0x45` | 命令帧不能识别，接收失败 |
| `0x4E` | 状态查询返回：正在播音 |
| `0x4F` | 状态查询返回：空闲；或一帧合成结束后自动回传 |

`Ready/Busy-STATUS` 引脚含义：

| 电平 | 含义 |
| --- | --- |
| 低电平 | 芯片空闲，可以接收命令和数据 |
| 高电平 | 芯片忙，正在语音合成或播音 |

手册建议连续播报时，在收到上一帧播报完毕的 `0x4F` 后，延迟约 1ms 再发送下一帧。硬件上也可以直接读取 `Ready/Busy-STATUS`，低电平后再发。

## 6. 当前现象判断

这次串口日志显示：

```text
[VOICE][INFO] enabled ... pre_stop=0 ack=0 query_idle=0 ...
[VOICE][TX] frame=10 action=slow status=ok detail=pre_stop=ok,speak:ack_disabled
[VOICE][TX] frame=60 action=slow status=ok detail=pre_stop=ok,speak:ack_disabled
[VOICE][TX] frame=195 action=clear status=ok detail=speak:ack_disabled
[VOICE][TX] frame=310 action=clear status=ok detail=speak:ack_disabled
```

这说明：

1. A1 主程序不是只发送一次，后续帧仍在调用语音发送。
2. 当前 `ack=0`，所以 `status=ok` 只代表 `uart_send_data()` 返回成功，不代表 SYN6288 已接收或已经播报。
3. 如果模块只播第一句，问题集中在 SYN6288 是否接收了后续帧、是否处于 busy 状态、是否需要等待 `0x4F` 或 BUSY 低电平。
4. 之前启用 `pre_stop=1/ack=1/retry=1` 后出现全部失败，说明每条语音前盲目发送停止命令不是可靠默认策略。

## 7. 正确的持续播报状态机

语音系统不应该把每帧检测结果都发给模块，而应维护一个低频状态机。

### 7.1 输入

来自避障决策：

```text
action in {clear, slow, stop, turn_left, turn_right}
risk in {CLEAR, FAR, WARNING, NEAR, URGENT}
frame_id
health_state
```

异常处理也要能进入语音：

- 摄像头异常、图像全黑/全白/卡帧：播 `停下`
- 推理异常、head 不匹配、候选爆炸：播 `停下`
- 资源异常、FPS 过低、UART 持续失败：优先串口告警，必要时播 `停下`

### 7.2 仲裁规则

1. 风险升级立即更新目标动作：`clear -> slow/stop`、`slow -> stop` 不等待长冷却。
2. 风险降低需要稳定确认：`stop -> slow/clear`、`slow -> clear` 必须连续稳定若干帧。
3. 同一动作不高频重复：
   - `stop` 可短冷却重复，建议 3-5 秒。
   - `slow/left/right` 建议 4-6 秒。
   - `clear` 建议 8-12 秒，避免一直播直行。
4. 不排队旧动作。语音模块忙时只保留最新动作，丢弃过期动作。

### 7.3 发送策略

推荐最终实现为：

```text
new_decision -> action arbiter -> pending_action

if module_idle:
    send latest pending_action
    pending_action = none
else:
    keep only latest high-priority action
```

其中 `module_idle` 的判断优先级：

1. 最可靠：读取 SYN6288 `Ready/Busy-STATUS` GPIO，低电平为空闲。
2. 次可靠：读取串口 `0x4F` 或发送状态查询 `FD 00 02 21 DE` 后等待 `0x4F`。
3. 调试保底：按固定时间间隔盲发，但这只能证明 A1 TX 正常，不能证明模块接受正常。

## 8. 推荐实现方案

### 8.1 当前保守版本

当前先使用最小干扰策略：

```sh
A1_VOICE_BAUD=9600
A1_VOICE_PRE_STOP=0
A1_VOICE_ACK=0
A1_VOICE_REQUIRE_ACK=0
A1_VOICE_QUERY_IDLE=0
A1_VOICE_USE_PREFIX=0
A1_VOICE_RETRY=0
A1_VOICE_TX_GAP_MS=3000
```

这版用于确认：A1 是否持续向 SYN6288 RXD 发送正确短词帧。

若这版仍只播第一句，需要进入硬件/协议闭环排查，不能再仅靠 `uart_send_data()` 的返回值判断。

### 8.2 标准可靠版本

接入 BUSY 后：

1. SYN6288 `Ready/Busy-STATUS` 经过 3.3V -> 1.8V 电平转换接到 A1 可读 GPIO。
2. A1 将该 GPIO 设置为输入模式。
3. 每次发送前读取 BUSY：
   - 低电平：立即发送最新动作帧。
   - 高电平：不发送，保留最新 pending action。
4. 发送后可选读取 `0x41`，确认接收成功。
5. 若收到 `0x45`：
   - 记录错误计数。
   - 等待 BUSY 低或 `0x4F` 后重发一次。
   - 连续失败时进入语音降级，但 OSD 和串口继续工作。

GPIO API 支持 `gpio_read_pin()`，官方说明中可读 GPIO 范围包含赛题 1/2 可用的 GPIO。需要结合当前相机/传感器占用确认最终使用哪一个未占用 GPIO。

### 8.3 不推荐默认启用的策略

不建议默认每次发送前都发停止命令：

```text
FD 00 02 02 FD
```

原因：

- 停止命令适合抢占高优先级 `stop`，不适合作为每条短词的前置动作。
- 实测启用 `pre_stop=1` 后出现完全不播或 `0x45` 拒收。
- 对于短词播报，等待空闲比强行停止更稳定。

## 9. 当前代码职责

| 函数 | 职责 |
| --- | --- |
| `VoiceNotifier::InitializeFromEnv()` | 读取环境变量，初始化 GPIO/UART，打印配置 |
| `VoiceNotifier::OpenA1UartApi()` | 配置 PIN0/PIN2 复用，打开官方 UART API |
| `VoiceNotifier::Update()` | 每隔若干帧读取最新避障动作，判断是否需要播报 |
| `VoiceNotifier::ShouldSend()` | 动作稳定性、冷却时间、重复播报控制 |
| `VoiceNotifier::BuildPromptPayload()` | 将动作映射为 GBK 中文短词 |
| `VoiceNotifier::BuildSyn6288Frame()` | 封装 SYN6288 合成播放帧和 XOR 校验 |
| `VoiceNotifier::SendBytes()` | 通过 `uart_send_data(UART_TX0)` 分片发送 |
| `VoiceNotifier::ReadResponseByte()` | 通过 `uart_receive_data(UART_RX0)` 读取模块回传 |
| `VoiceNotifier::QueryBusyState()` | 发送状态查询帧并读取 `0x4E/0x4F` |
| `VoiceNotifier::RunStartupSelfTest()` | 启动时播五条短词验证模块是否能连续接收 |

## 10. 现场调试流程

### 10.1 先确认 A1 配置

启动日志必须出现：

```text
UART: Device initialized successfully (TX0/RX0 only)
[VOICE][INFO] backend=a1_uart pins=UART_TX0/UART_RX0
[VOICE][INFO] enabled ... baud=9600 ... pre_stop=0 ack=0 query_idle=0 prefix=0 ...
```

如果看到：

```text
pre_stop=1
ack=1
retry=1
```

说明运行脚本仍处于高风险调试模式，需要改回保守模式。

### 10.2 语音自检

短时启用：

```sh
A1_VOICE_SELFTEST=1
```

预期连续播报：

```text
直行 -> 减速 -> 停下 -> 左转 -> 右转
```

若自检也只播第一句，说明问题在语音模块接收/忙闲/硬件链路，不在避障决策。

### 10.3 打开 ACK 验证 RX 回传

短时启用：

```sh
A1_VOICE_ACK=1
A1_VOICE_REQUIRE_ACK=0
A1_VOICE_PRE_STOP=0
A1_VOICE_RETRY=0
```

观察：

- `ack=0x41`：模块接受帧。
- `ack=0x45`：模块拒收帧。
- `ack_timeout`：A1 没收到模块回传，需要检查 SYN6288 TXD -> A1 RX0 链路。

如果第一条 `0x41`、后续 `0x45`，最可能原因是没有等待模块空闲，建议接 BUSY 或等 `0x4F`。

### 10.4 脱离 A1 做模块对照

用 USB-TTL 串口工具直连 SYN6288：

1. USB-TTL TX -> SYN6288 RXD
2. USB-TTL RX <- SYN6288 TXD
3. GND 共地
4. 串口设置 9600 8N1
5. 发送完整十六进制帧：

```text
FD 00 07 01 01 D6 B1 D0 D0 9D
FD 00 07 01 01 BC F5 CB D9 A1
FD 00 07 01 01 CD A3 CF C2 99
```

如果 USB-TTL 也只能播第一句，问题在模块忙闲/供电/功放/协议使用方式。若 USB-TTL 能持续播，问题在 A1 UART 时序或电平转换。

## 11. 与商家沟通的关键信息

可以直接向商家确认：

1. SYN6288 模块是否允许在上一句未播完时直接发送下一条合成命令？
2. 若不允许，是否必须等待 `0x4F` 或 `Ready/Busy` 低电平？
3. `Ready/Busy-STATUS` 引脚在该模块板上是否已引出，电平是否为 3.3V TTL？
4. 模块收到 `FD 00 07 01 01 D6 B1 D0 D0 9D` 后能否返回 `0x41` 并播报“直行”？
5. 连续发送短词时，推荐间隔是多少？是否必须先发停止命令？
6. 收到 `0x45` 后，推荐恢复方式是重发、等待空闲、发停止命令，还是硬件复位？
7. 模块供电电压、电流余量、功放和喇叭负载是否会导致第一句后模块复位或异常？

## 12. 下一步结论

当前 A1 软件已经能多次触发 UART 发送，但还没有可靠证明 SYN6288 接收并播放了后续帧。要从“能偶尔播一句”升级到“持续可靠播报”，必须增加至少一种模块侧闭环：

1. 推荐：接入 `Ready/Busy-STATUS` 到 A1 可读 GPIO，低电平后发送。
2. 备选：使用 UART RX 读取 `0x4F` 或状态查询 `0x4E/0x4F`，确认空闲后发送。
3. 保底：固定间隔盲发只能作为临时演示，不适合作为最终系统。

在 BUSY/ACK 闭环稳定之前，语音播报问题不能仅通过调整冷却时间解决。
