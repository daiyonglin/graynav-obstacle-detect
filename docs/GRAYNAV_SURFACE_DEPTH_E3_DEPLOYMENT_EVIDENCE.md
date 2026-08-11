# GrayNav SurfaceDepth E3 部署证据

日期：2026-08-11

状态：训练、ONNX 导出和官方 A1 INT8 转换已完成；尚未完成 SDK 集成、烧录和实板验收。

## 1. 最终选择

| 项目 | 值 |
|---|---|
| 实验 | E3 |
| 架构 | Fast-SCNN `detail64` 共享编码器，4 类分割头 + 16 级深度头 |
| checkpoint | epoch 49 `best_seg.pt` |
| checkpoint bytes | 14,160,726 |
| checkpoint SHA256 | `31a305b4bb50ba6eec7aa590e5b5c09271cdf7b3e9fdfad08527f624ad5cdc2a` |
| E2 回退 checkpoint SHA256 | `be5c9ba4d55604b31a97ee8de29011ab1b869808c194814b8dcc28f8d6217388` |

模型契约：

```text
images        float32  1 x 1 x 256 x 256  [0, 1]
seg_logits             1 x 4 x 64 x 64
depth_logits           1 x 16 x 64 x 64

surface classes:
0 ground_candidate
1 blocked_surface
2 step_or_drop
3 unknown_other
```

## 2. 训练与验证边界

E3 epoch 49 的公开验证集结果：

| 指标 | 值 |
|---|---:|
| ground IoU | 0.6209358 |
| blocked IoU | 0.6440821 |
| step precision | 0.7777237 |
| step recall | 0.9317864 |
| step F1 | 0.8478128 |
| unknown IoU | 0.6144420 |
| hazard-to-ground | 0.0337182 |
| NYUv2 AbsRel | 0.2415585 |
| NYUv2 delta1 | 0.6479838 |
| near/far order accuracy | 0.8777919 |

本模型没有通过最初设定的全部研究门槛：ground/blocked IoU 未达到 0.65/0.70，固定验证集中仍存在楼梯区域过填充，ADE 小台阶能力有限。选择 E3 的原因是它在现有公开数据、算子预算和训练次数内取得了较平衡的分割、深度层次与边缘细节，适合进入保守的板端功能验证，而不是因为它已经达到独立导航安全标准。

## 3. ONNX 证据

| 产物 | bytes | SHA256 |
|---|---:|---|
| `graynav_surface_depth_e3_gray1.onnx` | 4,593,443 | `8e304b72fb025159ab2c58f51c044094b53243d1ba3939b315f965d7a136b73b` |
| `datasets.zip` | 12,747,195 | `8ca9c129fefb5fe6efa54e2723e868f6ec5ca788df6239a01a0e9630da8ed79b` |
| `config.toml` | 119 | `ca3223d7577ead25efcb45d5bfc18aa3a3dd132b3650c17ed60a287a1fabffff` |

本地 ONNX 审计：

```text
input       images       1 x 1 x 256 x 256
output      seg_logits   1 x 4 x 64 x 64
output      depth_logits 1 x 16 x 64 x 64
opset       12
operators   Add 8, AveragePool 4, Concat 1, Constant 10,
            Conv 50, Relu 38, Resize 5
unsupported 0
forbidden   0
```

PyTorch/ONNX 一致性（200 个样本）：

```text
seg_grid_agreement     = 1.0
depth_level_agreement  = 0.99999755859375
max_abs_logit_error    = 9.918212890625e-05
```

校准集共 160 个样本：ADE20K 64、NYUv2 56、StairNetV3 40。量化评估集共 40 个样本：ADE20K 16、NYUv2 14、StairNetV3 10。两者没有样本重叠。

## 4. 官方 A1 转换结果

| 项目 | 值 |
|---|---|
| 输出 `.m1model` bytes | 1,459,634 |
| 输出 `.m1model` SHA256 | `d40b6f6c6392d062a5c39625b3f39c69e579255583498e2c218bb8c2593106f1` |
| 输入 scale | `images: 0.003921568859` |
| 输出 scale | `seg_logits: 0.0690593868494` |
| 输出 scale | `depth_logits: 0.367095053196` |
| 官方评估样本数 | 10 |
| 官方余弦相似度 | `seg_logits: 0.9859874547` |
| 官方余弦相似度 | `depth_logits: 0.9958205223` |

离线补充比较（不替代官方报告）：

```text
seg ArgMax grid agreement          = 94.46%
depth exact 16-bin agreement       = 77.97%
depth NEAR/MID/FAR cell agreement  = 95.61%
expected-depth mean relative diff  = 6.02%
```

精确 16-bin 一致率低于分组远近一致率，说明板端不应直接把单个最大深度 bin 当作稳定距离输出。最终后处理必须基于分组概率、空间中位数与时序信息。

## 5. 强制板端保护

1. 分割输出严格按 4 类解析，`unknown_other` 不得计入可通行地面。
2. 深度只产生 `NEAR / MID / FAR / UNKNOWN`；禁止在 Aurora 或语音中播报学习深度米数。
3. NEAR/MID/FAR 分组最高概率与次高概率的差小于 `0.20` 时，输出 `UNKNOWN`，规划器至少进入 `slow`。
4. 持续 `step_or_drop` 必须覆盖深度头给出的 `FAR`。
5. 分割危险使用走廊比例、连通区域与多帧投票；单帧高分不能直接触发稳定危险。
6. SurfaceDepth 推理连续失败时进入降级状态，保留检测、串口、OSD 和异步语音链路。
7. 在没有 SC132GS 本域深度标定的条件下，学习深度只作为相对证据，与检测框几何距离冲突时采用更保守的风险等级。

## 6. 本地归档核验

证据归档：

```text
graynav_surface_depth_e3_a1_evidence.tar.gz
bytes  = 5,081
SHA256 = 6f97b0972fe326c21e6b1595cd6949713a72122af6c1e5a4818276e7a3508707
```

证据归档包含：模型契约、A1 算子审计、ONNX 一致性、校准数据契约、最终选择记录和转换清单。官方转换器输出压缩包及 `.m1model` 作为本地受控二进制产物保存；它们只有在 C++ 契约升级和主机测试通过后才进入板端资产提交。

## 7. 后续验收状态

以下项目当前均为未完成，不得在报告中描述为“已部署”或“已验收”：

- 4 类分割 C++ 后处理和 `UNKNOWN_OTHER` 决策语义；
- INT8 深度分组概率与 `0.20` 歧义保护；
- 正式 `.m1model` 进入 Git 受控板端资产；
- Git 管理副本与 A1 SDK 编译副本同步；
- 双模型常驻、输出次序、量化 scale 和 30 分钟交替推理；
- 候选 `zImage` 构建、归档、烧录和真实场景测试。
