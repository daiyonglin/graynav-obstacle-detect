#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from graynav_dce import register_ultralytics_dce
from graynav_ood22 import COCO_OVERLAP_OOD22, NON_COCO_OOD22, OOD22_NAME_TO_ID, OOD22_NAMES, remap_prediction_class


CORRUPTIONS = ["normal", "low_light", "high_exposure", "low_contrast", "motion_blur", "noise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate M0 YOLOv8n and M1 GrayNav-DCE on OOD22 gray data.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--m0-weights", type=Path, required=True)
    parser.add_argument("--m1-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--corruptions", default=",".join(CORRUPTIONS))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/tensorboard/graynav_dce"))
    return parser.parse_args()


def split_image_dir(dataset_root: Path, split: str) -> Path:
    """Return the requested split image directory, falling back from test to val."""
    d = dataset_root / "images" / split
    if d.exists() and any(d.glob("*.jpg")):
        return d
    if split == "test":
        fallback = dataset_root / "images" / "val"
        if fallback.exists() and any(fallback.glob("*.jpg")):
            return fallback
    raise FileNotFoundError(f"no images found for split={split} under {dataset_root}")


def build_coco_annotations(dataset_root: Path, split: str, out_json: Path) -> Path:
    """Convert YOLO OOD22 labels to COCO annotations for one split."""
    image_dir = split_image_dir(dataset_root, split)
    label_dir = dataset_root / "labels" / image_dir.name
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id, img_path in enumerate(sorted(image_dir.glob("*.jpg")), start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"failed to read image: {img_path}")
        h, w = img.shape[:2]
        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
            x = (cx - bw / 2.0) * w
            y = (cy - bh / 2.0) * h
            ww = bw * w
            hh = bh * h
            if ww <= 0 or hh <= 0 or cls < 0 or cls >= len(OOD22_NAMES):
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls + 1,
                    "bbox": [x, y, ww, hh],
                    "area": ww * hh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [{"id": idx + 1, "name": name, "supercategory": "obstacle"} for idx, name in enumerate(OOD22_NAMES)]
    data = {"images": images, "annotations": annotations, "categories": categories}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out_json


def apply_corruption(gray: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Create deterministic mono-sensor-like robustness inputs."""
    g = gray.astype(np.float32)
    if mode == "normal":
        out = g
    elif mode == "low_light":
        out = g * 0.45
    elif mode == "high_exposure":
        out = np.clip(g * 1.55 + 20.0, 0, 255)
    elif mode == "low_contrast":
        mean = float(g.mean())
        out = np.clip((g - mean) * 0.55 + mean, 0, 255)
    elif mode == "motion_blur":
        kernel = np.ones((1, 7), dtype=np.float32) / 7.0
        out = cv2.filter2D(g, -1, kernel)
    elif mode == "noise":
        rng = np.random.default_rng(seed)
        out = np.clip(g + rng.normal(0, 12, g.shape), 0, 255)
    else:
        raise ValueError(mode)
    return np.round(out).astype(np.uint8)


def prepare_eval_images(image_dir: Path, out_dir: Path, corruption: str) -> Path:
    """Return original or generated corrupted gray3 images."""
    if corruption == "normal":
        return image_dir
    dst = out_dir / "eval_inputs" / corruption
    if dst.exists() and any(dst.glob("*.jpg")):
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for src in tqdm(sorted(image_dir.glob("*.jpg")), desc=f"prepare {corruption}"):
        gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"failed to read image: {src}")
        seed = sum(bytearray(src.name.encode("utf-8"))) & 0xFFFF
        aug = apply_corruption(gray, corruption, seed)
        bgr = cv2.cvtColor(aug, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dst / src.name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return dst


def image_id_by_name(coco: COCO) -> dict[str, int]:
    """Map file names to COCO image ids."""
    return {Path(img["file_name"]).name: int(img_id) for img_id, img in coco.imgs.items()}


def predict_to_ood_json(weights: Path, images: Path, annotations: Path, out_json: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run YOLO prediction and map model-specific classes to OOD22 categories."""
    register_ultralytics_dce()
    coco = COCO(str(annotations))
    ids = image_id_by_name(coco)
    model = YOLO(str(weights))
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
        batch=args.batch,
    )
    count = 0
    for result in tqdm(results, desc=f"predict {weights.name}"):
        count += 1
        image_id = ids.get(Path(result.path).name)
        if image_id is None or result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, scores, cls_ids):
            oid = remap_prediction_class(model.names, int(cls_id))
            if oid is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append({"image_id": image_id, "category_id": oid + 1, "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(predictions), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    return predictions, {"images": float(count), "wall_seconds": elapsed, "fps": float(count / elapsed) if elapsed > 0 else 0.0}


def coco_eval(annotations: Path, predictions: list[dict[str, Any]], category_names: list[str] | None = None) -> dict[str, float]:
    """Run COCO bbox evaluation for all or selected OOD22 categories."""
    coco_gt = COCO(str(annotations))
    if not predictions:
        return {name: 0.0 for name in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100"]}
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if category_names is not None:
        ev.params.catIds = [OOD22_NAME_TO_ID[name] + 1 for name in category_names]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {name: float(value) for name, value in zip(names, ev.stats)}


def empty_false_positive_rate(annotations: Path, predictions: list[dict[str, Any]], conf: float = 0.25) -> dict[str, float]:
    """Compute false positives per empty image."""
    coco = COCO(str(annotations))
    non_empty = {int(ann["image_id"]) for ann in coco.dataset.get("annotations", [])}
    empty = set(coco.imgs.keys()) - non_empty
    fp = sum(1 for p in predictions if int(p["image_id"]) in empty and float(p["score"]) >= conf)
    return {"empty_images": float(len(empty)), "empty_fp": float(fp), "empty_fp_per_image": float(fp / max(1, len(empty)))}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = split_image_dir(args.dataset_root, args.split)
    annotations = build_coco_annotations(args.dataset_root, image_dir.name, args.out_dir / f"annotations_{image_dir.name}.json")
    corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    summary: dict[str, Any] = {
        "settings": vars(args) | {"image_dir": str(image_dir), "annotations": str(annotations), "corruptions": corruptions},
        "models": {},
    }
    specs = [("M0_raw_yolov8n_graycopy", args.m0_weights), ("M1_graynav_dce_yolov8n", args.m1_weights)]
    for cidx, corruption in enumerate(corruptions):
        eval_images = prepare_eval_images(image_dir, args.out_dir, corruption)
        for model_name, weights in specs:
            pred_path = args.out_dir / "predictions" / f"{model_name}_{corruption}.json"
            preds, perf = predict_to_ood_json(weights, eval_images, annotations, pred_path, args)
            all_metrics = coco_eval(annotations, preds)
            overlap_metrics = coco_eval(annotations, preds, COCO_OVERLAP_OOD22)
            non_coco_metrics = coco_eval(annotations, preds, NON_COCO_OOD22)
            empty = empty_false_positive_rate(annotations, preds)
            key = f"{model_name}/{corruption}"
            summary["models"][key] = {
                "ood22_all": all_metrics,
                "coco_overlap": overlap_metrics,
                "non_coco": non_coco_metrics,
                "empty_fp": empty,
                "performance": perf,
                "prediction_count": len(preds),
            }
            writer.add_scalar(f"ood22_all/{model_name}/{corruption}/AP50", all_metrics["AP50"], cidx)
            writer.add_scalar(f"coco_overlap/{model_name}/{corruption}/AP50", overlap_metrics["AP50"], cidx)
            writer.add_scalar(f"non_coco/{model_name}/{corruption}/AP50", non_coco_metrics["AP50"], cidx)
            writer.add_scalar(f"empty_fp/{model_name}/{corruption}", empty["empty_fp_per_image"], cidx)
    writer.flush()
    writer.close()
    out = args.out_dir / "graynav_dce_eval_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"summary={out}")


if __name__ == "__main__":
    main()
