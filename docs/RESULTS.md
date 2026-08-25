# Results and Reliability Evidence

## 1. Offline model results

| Task | Metric | Value |
|---|---|---:|
| Detection | Person AP50 | 0.7712 |
| Detection | Person recall | 0.9199 |
| Detection | Partial-person recall | 0.9598 |
| Detection | Chair AP50 | 0.5831 |
| Detection | Dining-table AP50 | 0.6112 |
| Detection | Couch AP50 | 0.6701 |
| Scene | Ground IoU | 0.5731 |
| Scene | Blocked-surface IoU | 0.5735 |
| Stair | Step F1 | 0.7055 |
| Stair | Stair-edge F1 | 0.0805 |
| Stair | No-stair false-positive rate | 0.1244 |
| Depth | AbsRel | 0.3668 |
| Depth | δ1 | 0.4835 |
| Depth | Near/far ordering accuracy | 0.8348 |

Macro mAP50 为 0.3765，受紧凑训练集中 handbag/suitcase/bench 等稀有或缺失类别显著影响，因此仓库同时报告与室内演示和安全决策直接相关的逐类指标。

## 2. PyTorch-to-A1 consistency

正式转换的 10 个官方评估样本结果：

| Output | Mean cosine | Minimum sample cosine |
|---|---:|---:|
| cls_p3 | 0.994585 | 0.991113 |
| reg_p3 | 0.963706 | 0.949598 |
| cls_p4 | 0.991130 | 0.986160 |
| reg_p4 | 0.941258 | 0.916017 |
| cls_p5 | 0.990634 | 0.986307 |
| reg_p5 | 0.968374 | 0.935184 |
| scene_logits | 0.969735 | 0.950890 |

所有输出高于 0.90 转换门槛。官方报告与独立重算一致；完整逐样本记录保存在 `results/a1_conversion.json`。

## 3. Reliability mechanisms

- 模型加载后校验 input/output 数量、shape、元素数和量化属性；
- 单模型架构避免多 model-id 常驻和跨模型时序错配；
- 检测发布、类别切换、框尺寸和距离均有显式时序门控；
- 台阶必须经过语义、边缘、深度和时序多证据确认；
- OSD 有固定图层和图元预算，资源由脚本生成并审计；
- 相机黑色遮挡、捕获超时、推理失败、输出契约错误和 UART 错误分别隔离；
- CMake/rootfs 使用模型与资产白名单，发布镜像执行 15 MiB 门控；
- 串口、OSD 和语音共享同一稳定导航状态，避免不同输出互相矛盾。

## 4. Artifact integrity

| Artifact | Bytes | SHA256 |
|---|---:|---|
| FP32 checkpoint | 19,117,704 | `f28e5732ce4ed2432523d2ad508a21699c2494672f44295f2de8a1d572645f23` |
| ONNX | 12,328,061 | `2902d0aede72dd21ecd3d543142ff7c125ad5e00b19b5974d090cfce9837ae54` |
| A1 m1model | 4,150,950 | `33eec832710706b1153f468f219c08389a52ba3d21cbdffcde32ca5e25d66da8` |
| Final zImage | 8,134,208 | `de1c5b2fc4a311b4be9b1a402d572e15313e76323c6cb9577b132eb591b7afea` |

## 5. Interpretation boundary

单目深度和几何距离受相机安装、检测框完整性和场景域变化影响；米制结果是带不确定度的决策估计，不是精密测量。stair-edge 单分支 F1 较低，因此任何仅凭一条水平线直接停车的策略都不符合本系统证据门控。该系统用于研究和辅助提示，不构成独立出行安全认证。
