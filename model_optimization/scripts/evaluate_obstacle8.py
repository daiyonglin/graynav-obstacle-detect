#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument("--name", default="obstacle8_eval")
    parser.add_argument("--predict-source", type=Path)
    parser.add_argument("--conf", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        conf=args.conf,
        iou=0.60,
        plots=True,
        save_json=True,
    )
    print(metrics)
    if args.predict_source:
        model.predict(
            source=str(args.predict_source),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.60,
            device=args.device,
            project=args.project,
            name=f"{args.name}_pred",
            save=True,
            save_txt=True,
            save_conf=True,
        )


if __name__ == "__main__":
    main()

