#!/usr/bin/env python3
"""Compare PyTorch and ONNX raw logits/grid decisions on grayscale samples."""

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

from segmentation.graynav_fast_scnn import GrayNavFastSCNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    contract = payload.get("contract", {}) if isinstance(payload, dict) else {}
    model = GrayNavFastSCNN(width_mult=float(contract.get("width_mult", 1.0))).eval()
    model.load_state_dict(payload.get("model", payload), strict=True)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    files = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})[: args.limit]
    if not files:
        raise RuntimeError(f"no images found under {args.images}")

    matched = 0
    total = 0
    max_abs = 0.0
    for path in files:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = gray.astype(np.float32)[None, None] / 255.0
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(tensor)).numpy()
        onnx_logits = session.run(["seg_logits"], {"images": tensor})[0]
        max_abs = max(max_abs, float(np.max(np.abs(torch_logits - onnx_logits))))
        a = torch_logits.argmax(axis=1)
        b = onnx_logits.argmax(axis=1)
        matched += int((a == b).sum())
        total += int(a.size)
    report = {
        "samples": len(files),
        "grid_cells": total,
        "grid_class_agreement": matched / total,
        "max_abs_logit_error": max_abs,
        "passed": matched / total >= 0.99,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
