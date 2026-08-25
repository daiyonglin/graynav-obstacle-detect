#!/usr/bin/env python3
"""Evaluate E0/E1/E2/E3 fairly on prepared-v2 with the repaired metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from models.graynav_surface_depth import GrayNavSurfaceDepth  # noqa: E402
from train_surface_depth import SurfaceDepthDataset, evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default="checkpoint")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    contract = payload.get("contract", {})
    seg_classes = int(contract.get("seg_shape", [1, 3])[1])
    if seg_classes not in (3, 4):
        raise RuntimeError(f"unsupported segmentation head: {seg_classes}")
    model = GrayNavSurfaceDepth(
        width_mult=float(contract.get("width_mult", 1.0)),
        detail64=bool(contract.get("detail64", False)),
        num_surface_classes=seg_classes,
    )
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dataset = SurfaceDepthDataset(args.data, "val", str(contract.get("experiment", "e1")))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics = evaluate(model, loader, device)
    report = {
        "name": args.name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "contract": contract,
        "prepared_data": str(args.data),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "name": args.name,
        "seg_classes": seg_classes,
        "depth_absrel": metrics["depth"]["absrel"],
        "depth_gradient_mae": metrics["depth"]["gradient_mae"],
        "whole_frame_step_prediction_count": metrics["safety"]["whole_frame_step_prediction_count"],
        "false_whole_frame_step_prediction_count": metrics["safety"][
            "false_whole_frame_step_prediction_count"
        ],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    print("GRAYNAV_CHECKPOINT_EVALUATION_OK")


if __name__ == "__main__":
    main()
