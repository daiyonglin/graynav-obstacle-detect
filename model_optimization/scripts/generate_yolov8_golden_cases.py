#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="input image directory")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/golden_yolov8n_gray"))
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def letterbox_gray3(img: np.ndarray, size: int) -> tuple[np.ndarray, Dict[str, float]]:
    h, w = img.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    gray3 = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return gray3, {
        "src_w": int(w),
        "src_h": int(h),
        "dst_w": int(size),
        "dst_h": int(size),
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
    }


def result_to_items(result) -> List[Dict]:
    items = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return items
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    names = result.names
    for box, score, class_id in zip(xyxy, conf, cls):
        items.append({
            "box": [round(float(x), 4) for x in box],
            "score": round(float(score), 6),
            "class_id": int(class_id),
            "label": names.get(int(class_id), str(class_id)),
        })
    return items


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pre_dir = args.out_dir / "preprocessed_gray3"
    vis_dir = args.out_dir / "predictions"
    pre_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in VALID_EXTS)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"no images found under {args.source}")

    model = YOLO(args.weights)
    cases = []
    for src in tqdm(images, desc="golden"):
        raw = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        gray3, lb = letterbox_gray3(raw, args.imgsz)
        pre_path = pre_dir / f"{src.stem}_gray3_{args.imgsz}.png"
        cv2.imwrite(str(pre_path), gray3)
        result = model.predict(str(pre_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)[0]
        vis_path = vis_dir / f"{src.stem}_pred.jpg"
        cv2.imwrite(str(vis_path), result.plot())
        npy_path = pre_dir / f"{src.stem}_gray3_{args.imgsz}.npy"
        tensor = gray3[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        np.save(npy_path, tensor)
        cases.append({
            "source": str(src),
            "preprocessed_image": str(pre_path),
            "preprocessed_npy_chw_rgb01": str(npy_path),
            "prediction_image": str(vis_path),
            "letterbox": lb,
            "detections": result_to_items(result),
        })

    summary = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "cases": cases,
        "acceptance": {
            "box_mean_abs_error_px": 2.0,
            "score_abs_error": 0.001,
            "class_id_must_match": True,
        },
    }
    (args.out_dir / "golden_predictions.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"golden cases: {args.out_dir / 'golden_predictions.json'}")


if __name__ == "__main__":
    main()
