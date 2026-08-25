#!/usr/bin/env python3
"""Compare PyTorch and ONNX segmentation plus ordinal-depth outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from models.graynav_surface_depth import GrayNavSurfaceDepth  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    contract = payload.get("contract", {})
    width = float(contract.get("width_mult", 1.0))
    model = GrayNavSurfaceDepth(
        width_mult=width, detail64=bool(contract.get("detail64", False))
    ).eval()
    model.load_state_dict(payload["model"], strict=True)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    paths = sorted(
        path for path in args.images.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )[: args.limit]
    if not paths:
        raise RuntimeError(f"no images under {args.images}")
    seg_match = depth_match = cells = 0
    max_abs = 0.0
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = gray.astype(np.float32)[None, None] / 255.0
        with torch.no_grad():
            torch_seg, torch_depth = (item.numpy() for item in model(torch.from_numpy(tensor)))
        onnx_seg, onnx_depth = session.run(["seg_logits", "depth_logits"], {"images": tensor})
        max_abs = max(
            max_abs,
            float(np.max(np.abs(torch_seg - onnx_seg))),
            float(np.max(np.abs(torch_depth - onnx_depth))),
        )
        seg_match += int((torch_seg.argmax(1) == onnx_seg.argmax(1)).sum())
        depth_match += int((torch_depth.argmax(1) == onnx_depth.argmax(1)).sum())
        cells += int(torch_seg.shape[2] * torch_seg.shape[3])
    report = {
        "samples": len(paths),
        "seg_grid_agreement": seg_match / cells,
        "depth_level_agreement": depth_match / cells,
        "max_abs_logit_error": max_abs,
    }
    report["passed"] = report["seg_grid_agreement"] >= 0.99 and report["depth_level_agreement"] >= 0.99
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
