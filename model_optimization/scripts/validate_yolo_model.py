#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from graynav_dce import register_ultralytics_dce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ultralytics validation and save compact metrics JSON.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="validate_model")
    return parser.parse_args()


def as_float(value: Any) -> float:
    """Convert tensor-like metric values to JSON-safe floats."""
    try:
        return float(value)
    except Exception:
        return 0.0


def main() -> None:
    args = parse_args()
    register_ultralytics_dce()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        verbose=True,
    )
    names = getattr(metrics, "names", None) or getattr(model, "names", {})
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    box = metrics.box
    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "metrics": {
            "mp": as_float(box.mp),
            "mr": as_float(box.mr),
            "map50": as_float(box.map50),
            "map": as_float(box.map),
            "map75": as_float(box.map75),
        },
        "class_names": list(names),
        "per_class_map50_95": [as_float(x) for x in getattr(box, "maps", [])],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"validation_summary={args.out}")


if __name__ == "__main__":
    main()
