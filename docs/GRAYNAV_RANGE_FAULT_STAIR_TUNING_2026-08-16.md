# GrayNav 测距、异常保护与台阶提示调优证据（2026-08-16）

## 1. 本次实板日志结论

输入日志为 2026-08-16 用户实板测试的完整 UART 记录。正常导航摘要共 294 条：

- 动作：`STOP=152`、`SLOW=109`、`SYSTEM_FAULT=33`；
- 主障碍：`blocked=98`、`multiple=97`、`person=92`、`chair=4`、`stair=2`、`unknown=1`；
- 风险：`URGENT=152`、`WARNING=108`、`FAULT=33`、`UNKNOWN=1`；
- 有效数值距离 259 条，中位数约 `1.88m`，范围 `0.45m～3.99m`。

人体和椅子已经能够形成稳定且可纠正的 Indoor8 轨迹，未再出现上一版“椅子长期锁死为
PERSON”的问题。椅子片段在连续帧中保持为 `chair`，因此分类与 tracker 修复有效。

日志仍暴露出两个距离问题：

1. 近场规则会将连续几何估计硬截成 `0.45/0.70/1.00m`；
2. 面向 UART 的五次中值历史没有绑定 `track_id`，切换目标后会混入上一个目标距离。

例如日志存在 `primary=0.45m`、对应走廊却约为 `1.9～3.2m` 的记录；53/259 条主距离与
最近走廊距离相差超过 `0.5m`。这说明问题来自软件汇总与近场上界，而不只是单帧框抖动。

台阶只形成两条疑似记录，说明模型确实给出过台阶证据，但旧的 `3/5` 疑似和 `4/6`
确认时序对当前 INT8 edge head 偏严，难以在 Aurora 上持续看到边缘图形。

## 2. 本轮修复

### 2.1 目标绑定的连续测距

- `distance_m` 保留地面射线、尺寸先验和学习深度融合后的连续期望值；
- `0.45/0.70/1.00m` 近场规则只约束规划器使用的 `safe_distance_m`，不再作为 UART 米数；
- tracker 继续保留该安全上界，但不会用它污染目标的连续距离状态；
- 导航距离历史按 `track_id + class` 或 `scene + sector` 分组，目标切换时立即清空；
- 每个目标使用 3 次中值和非对称 EMA：接近响应快，突然跳远响应慢。

这些修改提高的是一致性与鲁棒性，不等于完成物理相机标定。串口两位小数仍是单目估计，
不能在答辩或 README 中表述为厘米级真实距离。若要继续提高绝对误差，需要用固定安装高度、
俯角和若干已知距离点修正 `A1_CAM_HEIGHT_M`、`A1_CAM_PITCH_DOWN_DEG` 与 FOV 参数。

### 2.2 遮挡摄像头异常

现有健康监控已能识别 `LENS_BLOCKED_OR_INVALID_IMAGE`，日志中也出现过 ACTIVE 与
RECOVERED。此次补齐异常输出隔离：

- 进入保护状态立即输出 `SYSTEM_FAULT / cls=abnormal / dist=-- / risk=FAULT`；
- 清空旧 person、chair、blocked 和三区摘要，避免异常行继续携带过期障碍；
- 导航固定为 `STOP/hold`，Aurora 进入 `AI_FAIL`；
- SYN6288 使用 GBK 固定词“异常”，语音异步发送；
- 手移开后需经过健康恢复时序才释放保护，避免亮边单帧误恢复。

### 2.3 疑似台阶边缘

- 边缘连通跨度统计阈值从 `0.50` 降到 `0.42`；
- 疑似门控使用较弱但联合的三证据：语义区域、水平 edge、上下深度跳变；
- edge 必须再有语义或深度支持，单独的床沿/桌沿仍不能触发台阶；
- 疑似由最近 4 次中的 2 次进入，显示两条短平行边缘带并只触发 `SLOW`；
- 确认仍要求语义、强边缘、至少 2 个深度等级跳变三项同时成立，最近 6 次中 3 次进入；
- 确认状态绘制双线危险框并按 NEAR/MID 触发停止。

## 3. 自动验证与镜像

Git 实现提交：

```text
29e7d54  fix(board): improve range fault and stair guidance
```

验证结果：

```text
Python demo contract tests     5/5 passed
Indoor8 CPU surface test       passed
A1 ARM cross compilation       passed
A1_ENABLE_VOICE                ON
A1_YOLO_NUM_CLASSES            8
A1_YOLO_INPUT_CHANNELS         1
rootfs .m1model count          1
model SHA256                   33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
zImage bytes                   8,131,568
zImage SHA256                  9E870915ED3C9851E7B02B7972B2E62AE8D9C416BFF761AC50F223EBAFBBE3D5
```

受保护回退镜像保持只读且哈希不变：

```text
bytes   8,214,488
SHA256  A7976710ECB456CB312D18F0195DCAE496ED652EFC582AB698EBC3EB7B055530
```

## 4. 下一次实板检查

1. 在 `0.8m / 1.5m / 2.5m` 三个已知位置各静止 10 秒，分别测试 person 和 chair；
2. 检查同一 track 的数值距离是否连续，换目标后是否立即切换而不保留旧值；
3. 用手完整遮挡镜头，确认一次“异常”语音、`STOP` 保护及 `cls=abnormal dist=--`；
4. 对真实上行和下行边缘测试，先看到疑似双平行线，再在三证据持续时进入确认框；
5. 床沿、桌沿连续测试 2 分钟，不得进入确认台阶或因台阶触发 STOP；
6. 若启用 `A1_OUTPUT_SERIAL_DIAG=1`，记录 `stair=state/peak/span/jump`，用于下一轮只调门控而
   不盲目放宽阈值。
