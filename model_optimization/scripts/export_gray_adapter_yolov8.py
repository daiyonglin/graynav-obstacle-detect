#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from gray_adapter import AdaptedYolo, load_adapter_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="gray_adapter.pt")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/gray_adapter"))
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    adapter = load_adapter_bundle(args.adapter, map_location=device).to(device).eval()
    yolo_model = YOLO(args.weights).model.to(device).eval()
    for p in yolo_model.parameters():
        p.requires_grad_(False)
    model = AdaptedYolo(adapter, yolo_model).to(device).eval()

    dummy = torch.zeros(1, 1, args.imgsz, args.imgsz, device=device)
    onnx_path = args.out_dir / "gray_adapter_yolov8_full.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["gray"],
        output_names=["predictions"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    meta = {
        "full_onnx": str(onnx_path),
        "adapter": str(args.adapter),
        "weights": args.weights,
        "imgsz": args.imgsz,
        "input": "1xHxW gray float normalized to 0..1",
        "note": "Full adapter+YOLO ONNX is for conversion feasibility tests. For A1 head6 deployment, use the LUT metadata with the original YOLOv8 head6 path unless the toolchain accepts this graph.",
    }
    (args.out_dir / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported: {onnx_path}")


if __name__ == "__main__":
    main()
