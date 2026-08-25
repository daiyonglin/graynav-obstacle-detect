# Released Weights

| File | Bytes | SHA256 |
|---|---:|---|
| `yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `graynav_surface_depth_e3_epoch49.pt` | 14,160,726 | `31a305b4bb50ba6eec7aa590e5b5c09271cdf7b3e9fdfad08527f624ad5cdc2a` |
| `graynav_unified_best_safety_epoch29.pt` | 19,117,704 | `f28e5732ce4ed2432523d2ad508a21699c2494672f44295f2de8a1d572645f23` |
| `graynav_unified_indoor8_scene21.onnx` | 12,328,061 | `2902d0aede72dd21ecd3d543142ff7c125ad5e00b19b5974d090cfce9837ae54` |

前两个 checkpoint 用于非随机初始化，epoch29 checkpoint 是统一模型选择结果，ONNX 是与 A1 INT8 模型对应的静态部署图。A1 二进制位于 `board/obstacle_detect/app_assets/models/`。
