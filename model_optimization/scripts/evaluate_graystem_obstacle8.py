#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO

from graynav_obstacle8 import SEMANTIC_NAMES, yolo_class_to_semantic


CORRUPTIONS = ["normal", "low_light", "high_exposure", "low_contrast", "motion_blur", "noise", "shadow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair COCOeval comparison for M0/M1/M2 on GrayNav-Obstacle8.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--m0-weights", type=Path)
    parser.add_argument("--m1-weights", type=Path, required=True)
    parser.add_argument("--m2-weights", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--corruptions", default="normal,low_light,high_exposure,low_contrast,motion_blur,noise,shadow")
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/tensorboard/graystem"))
    return parser.parse_args()


def apply_corruption(gray: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Apply deterministic mono-sensor-like degradation for robust validation."""
    g = gray.astype(np.float32) / 255.0
    if mode == "normal":
        out = g
    elif mode == "low_light":
        out = np.power(np.clip(g * 0.55, 0.0, 1.0), 1.25)
    elif mode == "high_exposure":
        out = np.clip(g * 1.45 + 0.08, 0.0, 1.0)
        out[out > 0.88] = 1.0
    elif mode == "low_contrast":
        mean = float(g.mean())
        out = np.clip(mean + 0.45 * (g - mean), 0.0, 1.0)
    elif mode == "motion_blur":
        kernel = 7
        padded = np.pad(g, ((0, 0), (kernel // 2, kernel // 2)), mode="edge")
        out = sum(padded[:, i : i + g.shape[1]] for i in range(kernel)) / float(kernel)
    elif mode == "noise":
        rng = np.random.default_rng(seed)
        out = np.clip(g + rng.normal(0, 0.04, size=g.shape), 0.0, 1.0)
    elif mode == "shadow":
        h, w = g.shape
        yy = np.linspace(0.55, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(0.85, 1.0, w, dtype=np.float32)[None, :]
        out = np.clip(g * np.minimum(yy, xx), 0.0, 1.0)
    else:
        raise ValueError(f"unknown corruption: {mode}")
    return np.round(out * 255.0).astype(np.uint8)


def prepare_eval_images(dataset_root: Path, out_dir: Path, corruption: str) -> Path:
    """Create exact GGG validation images for a corruption mode."""
    if corruption == "normal":
        return dataset_root / "images" / "val"
    dst = out_dir / "eval_inputs" / corruption
    if dst.exists() and any(dst.glob("*.jpg")):
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    src_dir = dataset_root / "images" / "val"
    for src in tqdm(sorted(src_dir.glob("*.jpg")), desc=f"prepare {corruption}"):
        image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"failed to read image: {src}")
        seed = sum(bytearray(src.name.encode("utf-8"))) & 0xFFFF
        gray = apply_corruption(image, corruption, seed)
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dst / src.name), gray3, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return dst


def image_id_by_name(coco: COCO) -> dict[str, int]:
    """Map validation file names to COCO image ids."""
    return {Path(img["file_name"]).name: int(img_id) for img_id, img in coco.imgs.items()}


def predict_to_coco_json(
    weights: Path,
    images: Path,
    annotations: Path,
    out_json: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run YOLO prediction and remap model classes to GrayNav-Obstacle8 categories."""
    coco = COCO(str(annotations))
    id_map = image_id_by_name(coco)
    model = YOLO(str(weights))
    model_to_sem = yolo_class_to_semantic(model.names)
    predictions: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    results = model.predict(
        source=str(images),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        stream=True,
        verbose=False,
    )
    count = 0
    for result in tqdm(results, desc=f"predict {weights.name}"):
        count += 1
        image_id = id_map.get(Path(result.path).name)
        if image_id is None or result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, conf, cls):
            sem = model_to_sem.get(int(cls_id))
            if sem is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": sem + 1,
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(predictions), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    return predictions, {"images": float(count), "wall_seconds": elapsed, "fps": float(count / elapsed) if elapsed > 0 else 0.0}


def coco_eval(annotations: Path, predictions: list[dict[str, Any]]) -> dict[str, float]:
    """Run COCO bbox evaluation on GrayNav-Obstacle8 categories."""
    coco_gt = COCO(str(annotations))
    if not predictions:
        return {k: 0.0 for k in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100"]}
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.catIds = [idx + 1 for idx in range(len(SEMANTIC_NAMES))]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {name: float(value) for name, value in zip(names, ev.stats)}


def empty_false_positive_rate(annotations: Path, predictions: list[dict[str, Any]], conf: float = 0.25) -> dict[str, float]:
    """Compute false positives per empty validation image."""
    coco = COCO(str(annotations))
    non_empty = {int(ann["image_id"]) for ann in coco.dataset.get("annotations", [])}
    empty = set(coco.imgs.keys()) - non_empty
    fp = sum(1 for p in predictions if int(p["image_id"]) in empty and float(p["score"]) >= conf)
    return {"empty_images": float(len(empty)), "empty_fp": float(fp), "empty_fp_per_image": float(fp / max(1, len(empty)))}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    model_specs = []
    if args.m0_weights:
        model_specs.append(("M0_raw_yolov8n_graycopy", args.m0_weights))
    model_specs.append(("M1_graynav_yolov8n_ft", args.m1_weights))
    if args.m2_weights:
        model_specs.append(("M2_graystem_bc", args.m2_weights))

    summary: dict[str, Any] = {"settings": vars(args) | {"corruptions": corruptions}, "models": {}}
    for cidx, corruption in enumerate(corruptions):
        image_dir = prepare_eval_images(args.dataset_root, args.out_dir, corruption)
        for model_name, weights in model_specs:
            pred_json = args.out_dir / "predictions" / f"{model_name}_{corruption}.json"
            predictions, perf = predict_to_coco_json(weights, image_dir, args.annotations, pred_json, args)
            metrics = coco_eval(args.annotations, predictions)
            empty = empty_false_positive_rate(args.annotations, predictions)
            key = f"{model_name}/{corruption}"
            summary["models"][key] = {"metrics": metrics, "performance": perf, "empty_fp": empty, "prediction_count": len(predictions)}
            for name, value in metrics.items():
                writer.add_scalar(f"coco/{model_name}/{corruption}/{name}", value, cidx)
            writer.add_scalar(f"empty_fp/{model_name}/{corruption}", empty["empty_fp_per_image"], cidx)
    writer.flush()
    writer.close()
    out = args.out_dir / "graystem_eval_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"summary={out}")


if __name__ == "__main__":
    main()

