#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn
from ultralytics import YOLO

from gray_adapter import load_adapter_bundle


FORBIDDEN_OPS = {"Sqrt", "Sub", "Div", "Softmax", "Clip", "NonMaxSuppression"}
EXPECTED_SAFE_OPS = {"Conv", "AveragePool", "BatchNormalization", "Add", "Mul", "Concat", "Relu", "Constant", "Identity"}


class BoardCompatibleAdaptedYolo(nn.Module):
    """Adapter+YOLO wrapper with board-compatible 3-channel input."""

    def __init__(self, adapter: nn.Module, yolo_model: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.yolo_model = yolo_model

    def forward(self, images: torch.Tensor) -> Any:
        return self.yolo_model(self.adapter(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BC-GMFE-DCA + YOLOv8n ONNX and check deployment ops.")
    parser.add_argument("--adapter", type=Path, required=True, help="bc_gmfe_dca_best.pt")
    parser.add_argument("--weights", type=Path, required=True, help="YOLOv8n .pt weights")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/bc_gmfe_dca"))
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def inspect_onnx_ops(path: Path) -> dict[str, Any]:
    """Summarize ONNX operator usage and flag high-risk deployment ops."""
    model = onnx.load(str(path))
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    forbidden = sorted(op for op in counts if op in FORBIDDEN_OPS)
    unknown = sorted(op for op in counts if op not in EXPECTED_SAFE_OPS)
    return {
        "op_counts": dict(sorted(counts.items())),
        "forbidden_ops": forbidden,
        "non_whitelist_ops": unknown,
        "has_forbidden_ops": bool(forbidden),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    adapter = load_adapter_bundle(args.adapter, map_location=device).to(device).eval()
    yolo_model = YOLO(str(args.weights)).model.to(device).eval()
    for param in yolo_model.parameters():
        param.requires_grad_(False)
    model = BoardCompatibleAdaptedYolo(adapter, yolo_model).to(device).eval()

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    onnx_path = args.out_dir / "bc_gmfe_dca_yolov8_full.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["predictions"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    op_report = inspect_onnx_ops(onnx_path)
    meta = {
        "full_onnx": str(onnx_path),
        "adapter": str(args.adapter),
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "input": "1x3xHxW board-compatible gray-replicated BGR/RGB float normalized to 0..1",
        "op_report": op_report,
        "note": "Use this full ONNX for operator feasibility inspection first. Head6 extraction should be done after confirming output node names for this graph.",
    }
    meta_path = args.out_dir / "bc_gmfe_dca_export_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported: {onnx_path}")
    print(f"meta: {meta_path}")
    if op_report["forbidden_ops"]:
        raise SystemExit(f"forbidden ONNX ops found: {op_report['forbidden_ops']}")


if __name__ == "__main__":
    main()
