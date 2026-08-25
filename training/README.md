# Training Pipeline

`training/` 包含从公开数据准备到 A1 转换提交包的完整模型侧代码。

## Modules

- `models/graynav_unified_perception.py`：统一 Indoor8 + Scene21 模型；
- `models/graynav_surface_depth.py`：场景/深度迁移初始化网络；
- `models/graynav_fast_scnn.py`：A1-safe 轻量编码器组件；
- `scripts/prepare_*`：公开数据自动转换；
- `scripts/train_*`：SurfaceDepth 与统一模型训练；
- `scripts/evaluate_*`、`visualize_*`：定量和独立样本可视化；
- `scripts/export_*`、`validate_*`、`audit_*`：ONNX 契约与一致性；
- `scripts/build_a1_calibration.py`：平衡校准/量化评估集；
- `tests/`：数据映射、损失、模型、采样和转换审计测试。

## Quick start

```powershell
powershell -ExecutionPolicy Bypass -File training/setup_windows.ps1
powershell -ExecutionPolicy Bypass -File training/prepare_detection_data.ps1
powershell -ExecutionPolicy Bypass -File training/prepare_scene_data.ps1
powershell -ExecutionPolicy Bypass -File training/train_local.ps1
```

默认工作区为 `E:\GrayNavWorkspace`。运行输出与数据不会写入 Git 仓库。

## Tests

```powershell
E:\GrayNavWorkspace\env\Scripts\python.exe -m unittest discover `
  -s training/tests -p "test_*.py" -v
```

训练 checkpoint 必须记录输入通道、首层 shape、类别顺序、scene channel 契约和随机种子。正式导出必须先通过 ONNX checker、算子审计以及 PyTorch/ONNX 网格一致性。
