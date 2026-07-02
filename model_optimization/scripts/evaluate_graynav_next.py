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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO

from gmfe_utils import GMFEMeta, gmfe_rgb, gray_to_rgb_copy, read_gray, write_rgb_image
from prepare_graynav_next_dataset import SEMANTIC_NAMES


CORRUPTIONS = ["normal", "low_light", "high_exposure", "low_contrast", "motion_blur", "noise"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GrayNav GGG and GMFE fine-tuned models.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m1-weights", type=Path, required=True)
    parser.add_argument("--m2-weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", default="64")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--corruptions", default=",".join(CORRUPTIONS))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/tensorboard/graynav_next"))
    parser.add_argument("--save-inputs", type=int, default=32)
    return parser.parse_args()


def parse_batch(value: str) -> int | float:
    text = str(value).strip()
    return float(text) if "." in text else int(text)


def apply_corruption(gray: np.ndarray, mode: str, seed: int) -> np.ndarray:
    """Apply deterministic sensor-like grayscale corruption for robustness evaluation."""
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
        out = np.zeros_like(g)
        for offset in range(kernel):
            out += padded[:, offset : offset + g.shape[1]]
        out /= float(kernel)
    elif mode == "noise":
        rng = np.random.default_rng(seed)
        out = np.clip(g + rng.normal(0, 0.035, size=g.shape) + np.sqrt(np.clip(g, 0, 1)) * rng.normal(0, 0.025, size=g.shape), 0, 1)
    else:
        raise ValueError(f"unknown corruption: {mode}")
    return np.round(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def write_eval_yaml(root: Path) -> Path:
    """Write a val-only Ultralytics YAML for a generated robustness dataset."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(SEMANTIC_NAMES))
    path = root / "graynav-eval.yaml"
    path.write_text(f"path: {root.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return path


def prepare_eval_variant(dataset_root: Path, out_dir: Path, encoding: str, corruption: str, meta: GMFEMeta, save_inputs: int) -> Path:
    """Create a temporary val dataset for one encoding/corruption pair."""
    if corruption == "normal":
        return dataset_root / "graynav-obstacle8-ggg.yaml" if encoding == "ggg" else dataset_root / "graynav-obstacle8-gmfe.yaml"

    root = out_dir / "eval_inputs" / f"{encoding}_{corruption}"
    img_dir = root / "images" / "val"
    lbl_dir = root / "labels" / "val"
    if img_dir.exists() and any(img_dir.glob("*.jpg")):
        return write_eval_yaml(root)

    gray_dir = dataset_root / "gray" / "images" / "val"
    src_label_dir = dataset_root / "variants" / "ggg" / "labels" / "val"
    files = sorted(gray_dir.glob("*.jpg"))
    for idx, path in enumerate(tqdm(files, desc=f"prepare {encoding}/{corruption}")):
        gray = read_gray(path)
        seed = zlib_crc(path.name)
        corrupted = apply_corruption(gray, corruption, seed)
        rgb = gray_to_rgb_copy(corrupted) if encoding == "ggg" else gmfe_rgb(corrupted, meta)
        write_rgb_image(img_dir / path.name, rgb)
        src_label = src_label_dir / f"{path.stem}.txt"
        dst_label = lbl_dir / f"{path.stem}.txt"
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_label, dst_label)
        if idx < save_inputs:
            audit_dir = out_dir / "input_samples" / f"{encoding}_{corruption}"
            audit_dir.mkdir(parents=True, exist_ok=True)
            write_rgb_image(audit_dir / path.name, rgb)
    return write_eval_yaml(root)


def zlib_crc(text: str) -> int:
    """Stable small seed for deterministic corruption."""
    import zlib

    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Extract stable scalar metrics from Ultralytics DetMetrics."""
    box = getattr(metrics, "box", None)
    out = {
        "map": float(getattr(box, "map", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map75": float(getattr(box, "map75", 0.0)),
        "maps": [float(x) for x in getattr(box, "maps", [])],
    }
    speed = getattr(metrics, "speed", {}) or {}
    out["speed"] = {str(k): float(v) for k, v in speed.items()}
    return out


def run_val(weights: Path, data: Path, args: argparse.Namespace, name: str) -> dict[str, Any]:
    """Run Ultralytics validation and return scalar metrics plus runtime."""
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        device=args.device,
        batch=parse_batch(args.batch),
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        project=str(args.out_dir / "ultralytics_val"),
        name=name,
        plots=True,
        save_json=True,
        exist_ok=True,
    )
    out = metrics_to_dict(metrics)
    out["wall_seconds"] = time.perf_counter() - t0
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = GMFEMeta.from_path(args.dataset_root / "gmfe_meta.json")
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    corruptions = [x.strip() for x in args.corruptions.split(",") if x.strip()]
    summary: dict[str, Any] = {
        "settings": {
            "dataset_root": str(args.dataset_root),
            "m1_weights": str(args.m1_weights),
            "m2_weights": str(args.m2_weights),
            "imgsz": args.imgsz,
            "batch": args.batch,
            "conf": args.conf,
            "iou": args.iou,
            "corruptions": corruptions,
        },
        "models": {},
    }
    try:
        for step, corruption in enumerate(corruptions):
            ggg_yaml = prepare_eval_variant(args.dataset_root, args.out_dir, "ggg", corruption, meta, args.save_inputs)
            gmfe_yaml = prepare_eval_variant(args.dataset_root, args.out_dir, "gmfe", corruption, meta, args.save_inputs)
            pairs = [
                ("M1_ggg_yolov8n_ft", args.m1_weights, ggg_yaml),
                ("M2_gmfe_yolov8n_ft", args.m2_weights, gmfe_yaml),
            ]
            for model_name, weights, yaml_path in pairs:
                key = f"{model_name}/{corruption}"
                print(f"evaluating {key}")
                result = run_val(weights, yaml_path, args, f"{model_name}_{corruption}")
                summary["models"][key] = result
                for metric in ["map", "map50", "map75", "wall_seconds"]:
                    writer.add_scalar(f"eval/{model_name}/{corruption}/{metric}", float(result.get(metric, 0.0)), step)
        if "M1_ggg_yolov8n_ft/normal" in summary["models"] and "M2_gmfe_yolov8n_ft/normal" in summary["models"]:
            for corruption in corruptions:
                m1 = summary["models"].get(f"M1_ggg_yolov8n_ft/{corruption}", {})
                m2 = summary["models"].get(f"M2_gmfe_yolov8n_ft/{corruption}", {})
                for metric in ["map", "map50", "map75"]:
                    delta = float(m2.get(metric, 0.0)) - float(m1.get(metric, 0.0))
                    writer.add_scalar(f"eval_delta/M2_minus_M1/{corruption}/{metric}", delta, corruptions.index(corruption))
    finally:
        writer.flush()
        writer.close()
    out_path = args.out_dir / "graynav_next_eval_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary={out_path}")


if __name__ == "__main__":
    main()
