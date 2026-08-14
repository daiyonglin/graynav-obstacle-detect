# GrayNav 统一模型板端稳定化证据（2026-08-14）

## 1. 本轮边界

本轮不重新训练、不重新量化，也不修改 Aurora 客户端。继续使用已通过官方 A1 转换的真单通道统一模型：

```text
model  = graynav_unified_indoor8_scene21.m1model
bytes  = 4,150,950
SHA256 = 33EEC832710706B1153F468F219C08389A52BA3D21CBDFFCDE32CA5E25D66DA8
input  = 1 x 1 x 384 x 384
output = 6 个检测张量 + 1 个 packed scene_logits
```

失败测试日志中的主要异常位于 CPU 后处理和演示层：274 条导航状态中，93.4% 的相邻记录发生变化；距离档位跳变率为 58.0%，`PERSON/NONE` 跳变率为 40.1%；75% 的记录被判为台阶，16 次语音中有 13 次播报台阶。模型加载、七输出契约和单次 NPU 推理本身均正常。

## 2. 已实现修改

### 2.1 检测与跟踪

- Indoor8 每个 anchor 只保留 top-1 类别，ROD25 的弱 person 附加候选仅在 25 类契约下启用。
- 同类别框在交集/小框面积不低于 0.75、中心横向距离小于画面宽度 10% 时视为同一实体。
- 人体嵌套框优先保留较大的非 coarse 身体框；只有大框分数低于小框 60% 时才保留小框。
- person 连续命中 2 次发布，其他室内类连续命中 3 次发布；类别切换需连续 3 次且新证据达到旧类 1.5 倍。
- 中心和宽高分别进行 ROI 感知 EMA；单次宽高变化限制为上一框的正负 20%。当前 ROI 短暂漏检时最多保持 500 ms，900 ms 无有效观测后删除。
- Aurora 最多显示两个已经确认、互不嵌套的目标框。

### 2.2 台阶与统一决策

`SurfaceResult` 增加 `STAIR_NONE / STAIR_SUSPECTED / STAIR_CONFIRMED` 以及边缘峰值、跨度、深度跳变和物体遮挡证据。

- 语义证据：中央走廊 step 比例 4% 到 35%，连通区域至少 12 格。
- 边缘证据：水平峰值至少 0.55，跨度至少为走廊宽度的 45%。
- 深度证据：边缘上下至少相差 2 个等级。
- 任意两项成立并在最近 5 次 LOWER 结果中出现 3 次，进入 suspected。
- 三项同时成立并在最近 6 次 LOWER 结果中出现 4 次，进入 confirmed。
- 边缘落在稳定 person/chair/table/couch 框内时，不允许直接 confirmed。
- suspected 仅产生 `SLOW + STEP?`；confirmed 且 NEAR/MID 才产生 `STOP + STAIR`。

新增 `StableGuidance`，作为 OSD、串口和语音的唯一数据源。STOP 连续两次进入、连续四次风险下降后退出；距离使用五次投票并对 NEAR 采用非对称进退；方位切换连续确认三次；目标名称保持 600 ms。稳定命名目标优先于其背景的 blocked_surface。

### 2.3 Aurora、串口与语音

- Layer 0 和 Layer 3 始终清空，不再绘制走廊、墙面 X、台阶箭头、横线、十字或点阵文字。
- Layer 1 仅有一张动作位图，Layer 2 仅有一张两行组合位图，Layer 4 最多两个检测框。
- 组合 HUD 覆盖 12 个主标签、4 个距离档位和 3 个方位，共 144 个 `.ssbmp`；构建前执行完整性审计。
- 正常串口格式固定为 `[NAV] STOP | PERSON | NEAR | FRONT | AI_OK`；变化触发，稳定状态 2 秒心跳，关键变化最短间隔 500 ms。
- 正常模式关闭 `[VOICE] TX/DONE/HEALTH` 事务日志，只保留初始化和真实 UART 错误；`A1_VOICE_DIAG=1` 可恢复诊断。
- suspected 和 confirmed 使用不同语音：前者提示“疑似台阶，请慢行”，后者才提示台阶停止。

## 3. 自动测试与构建审计

```text
host surface/depth/guidance tests = PASS
Python OSD scripts py_compile    = PASS
OSD asset audit                  = PASS (144 files, 4,084,074 bytes)
git diff --check                 = PASS
ARM application compile/link     = PASS
full Docker/Buildroot build      = PASS
rootfs .m1model count            = 1
rootfs INFO HUD count            = 144
model SHA256                     = 33EEC832...5D66DA8 (unchanged)
```

最终镜像：

```text
bytes  = 8,145,744
SHA256 = 9C5D03B54A4376480F12DC75679785873CBB8AD50E97EFA883E2CCACE442D02A
limit  = 15 MiB
status = 已构建、待烧录；尚未完成实板验收
```

## 4. 烧录后的验收顺序

1. 启动日志确认只有一个 `model_id`、七个输出契约全部通过、无 OSD add/flush failure。
2. 对准同一人体的脸、上半身和全身，确认最多一个稳定框；静止框中心抖动不超过画面宽度 5%。
3. 椅子重复 10 次，至少 7 次稳定显示 CHAIR，且不得持续显示 PERSON。
4. 床沿和椅背连续观察 2 分钟，不得进入 confirmed STAIR 或因台阶触发 STOP。
5. 平坦地面、墙面和真实台阶依次测试；真实台阶应先 STEP?，持续证据成立后再 STAIR。
6. 核对 Aurora 只剩动作、两行信息和最多两个框；稳定串口每 2 秒最多一行。
7. 接入 SYN6288，分别试听人物、阻挡、疑似台阶、确认台阶、转向和系统异常短语。
8. 连续运行 30 分钟，确认无崩溃和持续内存增长。

测试人员不得闭眼依赖系统行走。本镜像在完成以上实板测试前只能称为“烧录候选”，不能标记为验收版本；受保护的 A797 回退镜像不得覆盖。
