#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO

from graynav_dce import register_ultralytics_dce


CORRUPTIONS = ["normal", "low_light", "high_exposure", "low_contrast", "motion_blur", "noise"]
COCO_TO_GENERIC_ALIASES = {
    "person": ["person"],
    "car": ["car"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle"],
    "dog": ["dog"],
    "bus": ["bus"],
    "truck": ["truck"],
    "train": ["train"],
    "bench": ["bench"],
    "traffic light": ["traffic_light"],
    "fire hydrant": ["fire_hydrant"],
    "stop sign": ["stop_sign", "traffic_sign"],
    "chair": ["chair"],
    "potted plant": ["plant_pot"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate COCO YOLO, DCE YOLO, and optional ablation models on a generic gray obstacle dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--m0-weights", type=Path, required=True)
    parser.add_argument("--m1-weights", type=Path, required=True)
    parser.add_argument("--m0-name", default="M0_raw_yolov8n_graycopy")
    parser.add_argument("--m1-name", default="M1_graynav_dce_yolov8n")
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="NAME=WEIGHTS",
        help="Optional additional model for ablation, for example M2_yolov8n_no_dce=/path/best.pt.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--corruptions", default=",".join(CORRUPTIONS))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/tensorboard/graynav_dce_generic"))
    return parser.parse_args()


def parse_extra_models(items: list[str]) -> list[tuple[str, Path]]:
    """Parse NAME=WEIGHTS model specs used for controlled ablation evaluation."""
    models: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-model must be NAME=WEIGHTS, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"empty extra model name: {item}")
        models.append((name, Path(path).expanduser()))
    return models


def normalize_name(name: str) -> str:
    """Normalize class names for cross-dataset mapping."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def load_names(path: Path) -> list[str]:
    """Load normalized names from an Ultralytics YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]
    return [normalize_name(str(x)) for x in names]


def split_image_dir(root: Path, split: str) -> Path:
    """Return split image directory with test-to-val fallback."""
    d = root / "images" / split
    if d.exists() and any(d.glob("*.jpg")):
        return d
    if split == "test":
        fallback = root / "images" / "val"
        if fallback.exists() and any(fallback.glob("*.jpg")):
            return fallback
    raise FileNotFoundError(f"no images for split={split} under {root}")


def build_annotations(root: Path, split: str, names: list[str], out_json: Path) -> Path:
    """Convert YOLO labels into COCO annotations."""
    image_dir = split_image_dir(root, split)
    label_dir = root / "labels" / image_dir.name
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
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
            if cls < 0 or cls >= len(names):
                continue
            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
            ww, hh = bw * w, bh * h
            x, y = (cx - bw / 2.0) * w, (cy - bh / 2.0) * h
            if ww <= 0 or hh <= 0:
                continue
            anns.append({"id": ann_id, "image_id": image_id, "category_id": cls + 1, "bbox": [x, y, ww, hh], "area": ww * hh, "iscrowd": 0})
            ann_id += 1
    cats = [{"id": idx + 1, "name": name, "supercategory": "obstacle"} for idx, name in enumerate(names)]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}, ensure_ascii=False), encoding="utf-8")
    return out_json


def apply_corruption(gray: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Apply deterministic grayscale corruptions."""
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
        out = cv2.filter2D(g, -1, np.ones((1, 7), dtype=np.float32) / 7.0)
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
        bgr = cv2.cvtColor(apply_corruption(gray, corruption, seed), cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(dst / src.name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return dst


def model_class_to_dataset_id(model_names: dict[int, str] | list[str], cls_id: int, dataset_names: list[str]) -> int | None:
    """Map native model class id to the generic dataset class id."""
    if isinstance(model_names, dict):
        raw = str(model_names.get(int(cls_id), ""))
    else:
        raw = str(model_names[int(cls_id)]) if 0 <= int(cls_id) < len(model_names) else ""
    name = normalize_name(raw)
    dataset_map = {n: i for i, n in enumerate(dataset_names)}
    if name in dataset_map:
        return dataset_map[name]
    for alias in COCO_TO_GENERIC_ALIASES.get(raw.lower(), []):
        if alias in dataset_map:
            return dataset_map[alias]
    for alias in COCO_TO_GENERIC_ALIASES.get(name.replace("_", " "), []):
        if alias in dataset_map:
            return dataset_map[alias]
    return None


def image_id_by_name(coco: COCO) -> dict[str, int]:
    """Map file names to COCO ids."""
    return {Path(img["file_name"]).name: int(img_id) for img_id, img in coco.imgs.items()}


def predict_json(model_name: str, weights: Path, images: Path, annotations: Path, dataset_names: list[str], out_json: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run prediction and map class ids to dataset categories."""
    register_ultralytics_dce()
    coco = COCO(str(annotations))
    ids = image_id_by_name(coco)
    model = YOLO(str(weights))
    preds: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    results = model.predict(source=str(images), imgsz=args.imgsz, device=args.device, conf=args.conf, iou=args.iou, max_det=args.max_det, stream=True, verbose=False, batch=args.batch)
    count = 0
    for result in tqdm(results, desc=f"predict {model_name}"):
        count += 1
        image_id = ids.get(Path(result.path).name)
        if image_id is None or result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, scores, cls_ids):
            did = model_class_to_dataset_id(model.names, int(cls_id), dataset_names)
            if did is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            preds.append({"image_id": image_id, "category_id": did + 1, "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": float(score)})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(preds), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    return preds, {"images": float(count), "wall_seconds": elapsed, "fps": float(count / elapsed) if elapsed > 0 else 0.0}


def coco_eval(annotations: Path, predictions: list[dict[str, Any]], names: list[str], selected: list[str] | None = None) -> dict[str, float]:
    """Run COCO bbox evaluation on all or selected categories."""
    coco_gt = COCO(str(annotations))
    if not predictions:
        return {k: 0.0 for k in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100"]}
    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    if selected is not None:
        name_to_id = {name: idx + 1 for idx, name in enumerate(names)}
        ev.params.catIds = [name_to_id[name] for name in selected if name in name_to_id]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    return {k: float(v) for k, v in zip(keys, ev.stats)}


def empty_fp(annotations: Path, predictions: list[dict[str, Any]], conf: float = 0.25) -> dict[str, float]:
    """Compute false positives per empty image."""
    coco = COCO(str(annotations))
    non_empty = {int(a["image_id"]) for a in coco.dataset.get("annotations", [])}
    empty = set(coco.imgs.keys()) - non_empty
    fp = sum(1 for p in predictions if int(p["image_id"]) in empty and float(p["score"]) >= conf)
    return {"empty_images": float(len(empty)), "empty_fp": float(fp), "empty_fp_per_image": float(fp / max(1, len(empty)))}


def overlap_names(dataset_names: list[str]) -> list[str]:
    """Return dataset classes that M0 COCO can plausibly predict."""
    out = []
    for aliases in COCO_TO_GENERIC_ALIASES.values():
        for alias in aliases:
            if alias in dataset_names and alias not in out:
                out.append(alias)
    return out


def model_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Return ordered models for evaluation and explicit claim decomposition."""
    specs = [
        (args.m0_name, args.m0_weights),
        (args.m1_name, args.m1_weights),
    ]
    specs.extend(parse_extra_models(args.extra_model))
    for name, weights in specs:
        if not weights.exists():
            raise FileNotFoundError(f"{name} weights not found: {weights}")
    return specs


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = load_names(args.data_yaml)
    image_dir = split_image_dir(args.dataset_root, args.split)
    annotations = build_annotations(args.dataset_root, image_dir.name, names, args.out_dir / f"annotations_{image_dir.name}.json")
    corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    overlap = overlap_names(names)
    non_overlap = [name for name in names if name not in overlap]
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    specs = model_specs(args)
    summary: dict[str, Any] = {
        "settings": vars(args)
        | {
            "names": names,
            "overlap_names": overlap,
            "non_overlap_names": non_overlap,
            "corruptions": corruptions,
            "model_specs": [{"name": name, "weights": str(weights)} for name, weights in specs],
            "claim_note": "M0 has COCO classes only; task-trained models have dataset-native classes. Use no-DCE ablation to isolate architecture contribution.",
        },
        "models": {},
    }
    for cidx, corruption in enumerate(corruptions):
        eval_images = prepare_eval_images(image_dir, args.out_dir, corruption)
        for model_name, weights in specs:
            pred_path = args.out_dir / "predictions" / f"{model_name}_{corruption}.json"
            preds, perf = predict_json(model_name, weights, eval_images, annotations, names, pred_path, args)
            all_m = coco_eval(annotations, preds, names)
            overlap_m = coco_eval(annotations, preds, names, overlap) if overlap else {}
            non_overlap_m = coco_eval(annotations, preds, names, non_overlap) if non_overlap else {}
            fp = empty_fp(annotations, preds)
            key = f"{model_name}/{corruption}"
            summary["models"][key] = {"all": all_m, "overlap": overlap_m, "non_overlap": non_overlap_m, "empty_fp": fp, "performance": perf, "prediction_count": len(preds)}
            writer.add_scalar(f"all/{model_name}/{corruption}/AP50", all_m.get("AP50", 0.0), cidx)
            writer.add_scalar(f"overlap/{model_name}/{corruption}/AP50", overlap_m.get("AP50", 0.0), cidx)
            writer.add_scalar(f"non_overlap/{model_name}/{corruption}/AP50", non_overlap_m.get("AP50", 0.0), cidx)
            writer.add_scalar(f"empty_fp/{model_name}/{corruption}", fp["empty_fp_per_image"], cidx)
    writer.flush()
    writer.close()
    out = args.out_dir / "graynav_dce_generic_eval_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"summary={out}")


if __name__ == "__main__":
    main()
