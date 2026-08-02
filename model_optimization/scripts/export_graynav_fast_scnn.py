#!/usr/bin/env python3
"""Export the static one-channel GrayNav Fast-SCNN deployment graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from segmentation.graynav_fast_scnn import GrayNavFastSCNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--width-mult", type=float, choices=(1.0, 0.75))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    contract = payload.get("contract", {}) if isinstance(payload, dict) else {}
    width_mult = args.width_mult if args.width_mult is not None else float(contract.get("width_mult", 1.0))
    model = GrayNavFastSCNN(width_mult=width_mult)
    model.load_state_dict(state, strict=True)
    model.eval()
    if tuple(model.first_conv.weight.shape)[1] != 1:
        raise RuntimeError("refusing to export a non-grayscale stem")
    dummy = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    with torch.no_grad():
        output = model(dummy)
    if tuple(output.shape) != (1, 4, 32, 32):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        input_names=["images"],
        output_names=["seg_logits"],
        opset_version=12,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    contract = {
        "model": "graynav_fast_scnn_gray1_4cls_256",
        "input_name": "images",
        "input_shape": [1, 1, 256, 256],
        "input_semantics": "single-channel grayscale derived from SC132GS Y8",
        "output_name": "seg_logits",
        "output_shape": [1, 4, 32, 32],
        "classes": ["ground_candidate", "blocked_surface", "step_or_drop", "pothole"],
        "width_mult": width_mult,
        "postprocess": "CPU ArgMax, 3x3 majority, corridor ratios, temporal voting",
    }
    args.onnx.with_suffix(".contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
