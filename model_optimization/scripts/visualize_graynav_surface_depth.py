#!/usr/bin/env python3
"""Create deterministic qualitative visualizations for GrayNav SurfaceDepth.

This is an offline diagnostic for public validation data.  Its colour overlays
are intentionally richer than the grayscale Aurora board presentation and must
not be interpreted as a deployed UI or as calibrated metric ranging evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from segmentation.graynav_surface_depth import (  # noqa: E402
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    SURFACE_CLASS_NAMES,
    GrayNavSurfaceDepth,
    depth_bin_centers,
)


IGNORE = 255
PANEL_SIZE = 320
CLASS_COLORS_BGR = np.asarray(
    (
        (70, 190, 70),    # ground: green
        (55, 55, 225),    # blocked: red
        (0, 190, 255),    # step/drop: orange
    ),
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=("ade20k", "nyuv2", "stairnetv3"),
    )
    return parser.parse_args()


def load_manifest(root: Path) -> list[dict[str, object]]:
    path = root / "manifest_val.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise RuntimeError(f"empty validation manifest: {path}")
    return rows


def select_records(
    rows: list[dict[str, object]], sources: list[str], count: int, seed: int
) -> list[dict[str, object]]:
    if count < 1:
        raise ValueError("samples-per-source must be positive")
    chosen: list[dict[str, object]] = []
    for source_index, source in enumerate(sources):
        candidates = sorted(
            (row for row in rows if str(row.get("source")) == source),
            key=lambda row: str(row["source_id"]),
        )
        if not candidates:
            raise RuntimeError(f"validation source is empty: {source}")
        rng = random.Random(seed + source_index * 1009)
        indices = sorted(rng.sample(range(len(candidates)), min(count, len(candidates))))
        chosen.extend(candidates[index] for index in indices)
    return chosen


def expected_depth(depth_logits: torch.Tensor) -> torch.Tensor:
    centers = depth_bin_centers(depth_logits.device).view(1, -1, 1, 1)
    return (torch.softmax(depth_logits, dim=1) * centers).sum(dim=1)


def put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    scale: float = 0.52,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (10, 10, 10),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def titled_image(image: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    """Render one information type as its own readable image file."""

    resized = cv2.resize(image, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    canvas = np.full((PANEL_SIZE + 58, PANEL_SIZE, 3), 24, dtype=np.uint8)
    canvas[58 : 58 + PANEL_SIZE] = resized
    put_text(canvas, title, (10, 21), scale=0.58, thickness=2)
    if subtitle:
        put_text(canvas, subtitle, (10, 41), scale=0.42, color=(205, 205, 205))
    return canvas


def unavailable_panel(message: str) -> np.ndarray:
    image = np.full((PANEL_SIZE, PANEL_SIZE, 3), 38, dtype=np.uint8)
    put_text(image, message, (28, PANEL_SIZE // 2), scale=0.58)
    return image


def segmentation_view(gray: np.ndarray, labels: np.ndarray, alpha: float) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colour = np.zeros_like(base)
    valid = labels != IGNORE
    for class_index, class_colour in enumerate(CLASS_COLORS_BGR):
        colour[labels == class_index] = class_colour
    blended = base.copy()
    blended[valid] = cv2.addWeighted(base, 1.0 - alpha, colour, alpha, 0)[valid]
    for class_index, class_colour in enumerate(CLASS_COLORS_BGR):
        mask = (labels == class_index).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, tuple(int(x) for x in class_colour), 1)
    return blended


def depth_view(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    clipped = np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M)
    normalized = (
        np.log(clipped) - math.log(DEPTH_MIN_M)
    ) / (math.log(DEPTH_MAX_M) - math.log(DEPTH_MIN_M))
    # Reverse Turbo: near is warm/red and far is cool/blue.
    mapped = np.clip((1.0 - normalized) * 255.0, 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(mapped, cv2.COLORMAP_TURBO)
    if valid is not None:
        colour[~valid] = (35, 35, 35)
    return colour


def corridor_mask(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [
            (int(0.34 * width), height - 1),
            (int(0.66 * width), height - 1),
            (int(0.57 * width), int(0.42 * height)),
            (int(0.43 * width), int(0.42 * height)),
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool), points


def depth_level(value: float) -> str:
    if not math.isfinite(value):
        return "UNKNOWN"
    if value < 1.25:
        return "NEAR"
    if value < 2.20:
        return "MID"
    return "FAR"


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    valid = target != IGNORE
    result: dict[str, float | None] = {}
    for class_index, name in enumerate(SURFACE_CLASS_NAMES):
        intersection = int(np.logical_and(prediction == class_index, target == class_index)[valid].sum())
        union = int(np.logical_or(prediction == class_index, target == class_index)[valid].sum())
        result[name] = intersection / union if union else None
    return {"iou": result, "valid_pixels": int(valid.sum())}


def depth_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    valid = np.isfinite(target) & (target >= DEPTH_MIN_M) & (target <= DEPTH_MAX_M)
    if not np.any(valid):
        return {"absrel": None, "delta1": None, "valid_pixels": 0}
    pred = prediction[valid]
    truth = target[valid]
    ratio = np.maximum(pred / truth, truth / pred)
    return {
        "absrel": float(np.mean(np.abs(pred - truth) / truth)),
        "delta1": float(np.mean(ratio < 1.25)),
        "valid_pixels": int(valid.sum()),
    }


def mean_available(rows: list[dict[str, object]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: object = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty visualization directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    contract = checkpoint.get("contract", {})
    if contract.get("input_shape") != [1, 1, 256, 256]:
        raise RuntimeError(f"not a GrayNav mono checkpoint: {contract}")
    model = GrayNavSurfaceDepth(width_mult=float(contract.get("width_mult", 1.0)))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    records = select_records(
        load_manifest(args.data), list(args.sources), args.samples_per_source, args.seed
    )
    results: list[dict[str, object]] = []

    for record in records:
        source = str(record["source"])
        source_id = str(record["source_id"])
        gray = cv2.imread(str(args.data / str(record["image"])), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot read {record['image']}")
        gray256 = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(gray256.astype(np.float32)[None, None] / 255.0).to(device)
        with torch.inference_mode():
            seg_logits, depth_logits = model(tensor)
            seg_probabilities = torch.softmax(seg_logits, dim=1)
            seg_prediction = seg_probabilities.argmax(1)[0].cpu().numpy().astype(np.uint8)
            seg_confidence = seg_probabilities.max(1).values[0].cpu().numpy()
            predicted_depth = expected_depth(depth_logits)[0].cpu().numpy()

        seg_prediction = cv2.resize(seg_prediction, (256, 256), interpolation=cv2.INTER_NEAREST)
        seg_confidence = cv2.resize(seg_confidence, (256, 256), interpolation=cv2.INTER_LINEAR)
        predicted_depth = cv2.resize(predicted_depth, (256, 256), interpolation=cv2.INTER_LINEAR)
        prediction_overlay = segmentation_view(gray256, seg_prediction, alpha=0.42)
        predicted_depth_view = depth_view(predicted_depth)
        corridor, corridor_points = corridor_mask(256, 256)
        cv2.polylines(predicted_depth_view, [corridor_points], True, (245, 245, 245), 2)
        corridor_depth = float(np.median(predicted_depth[corridor]))
        level = depth_level(corridor_depth)
        put_text(predicted_depth_view, level, (12, 24), scale=0.72, thickness=2)

        ratios = {
            name: float(np.mean(seg_prediction == index))
            for index, name in enumerate(SURFACE_CLASS_NAMES)
        }
        sample: dict[str, object] = {
            "source": source,
            "source_id": source_id,
            "image": str(record["image"]),
            "prediction": {
                "class_ratios": ratios,
                "mean_segmentation_confidence": float(seg_confidence.mean()),
                "center_corridor_depth_m_diagnostic_only": corridor_depth,
                "center_corridor_level": level,
            },
            "segmentation_metrics": None,
            "depth_metrics": None,
        }

        if record.get("seg_mask"):
            seg_target = cv2.imread(
                str(args.data / str(record["seg_mask"])), cv2.IMREAD_GRAYSCALE
            )
            if seg_target is None:
                raise RuntimeError(f"cannot read {record['seg_mask']}")
            seg_target = cv2.resize(seg_target, (256, 256), interpolation=cv2.INTER_NEAREST)
            gt_seg_view = segmentation_view(gray256, seg_target, alpha=0.55)
            sample["segmentation_metrics"] = segmentation_metrics(seg_prediction, seg_target)
        else:
            gt_seg_view = unavailable_panel("NO SEGMENTATION GT")

        if record.get("depth"):
            depth_target = np.load(args.data / str(record["depth"])).astype(np.float32)
            depth_target = cv2.resize(depth_target, (256, 256), interpolation=cv2.INTER_NEAREST)
            valid_depth = (
                np.isfinite(depth_target)
                & (depth_target >= DEPTH_MIN_M)
                & (depth_target <= DEPTH_MAX_M)
            )
            gt_depth_view = depth_view(depth_target, valid_depth)
            sample["depth_metrics"] = depth_metrics(predicted_depth, depth_target)
        else:
            gt_depth_view = unavailable_panel("NO METRIC DEPTH GT")

        mono = cv2.cvtColor(gray256, cv2.COLOR_GRAY2BGR)
        scene_dir = args.output / source / source_id.replace(":", "_")
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_images = {
            "mono_input": titled_image(mono, "MONO MODEL INPUT", source_id),
            "segmentation_gt": titled_image(
                gt_seg_view,
                "SEGMENTATION GROUND TRUTH",
                "green ground / red blocked / orange step",
            ),
            "segmentation_prediction": titled_image(
                prediction_overlay,
                "SEGMENTATION PREDICTION",
                f"mean confidence {float(seg_confidence.mean()):.3f}",
            ),
            "depth_prediction": titled_image(
                predicted_depth_view,
                "DEPTH PREDICTION",
                f"center corridor {level}; diagnostic only",
            ),
            "metric_depth_gt": titled_image(
                gt_depth_view,
                "METRIC DEPTH GROUND TRUTH",
                "near warm / far cool",
            ),
        }
        written: dict[str, str] = {}
        for order, (kind, image) in enumerate(scene_images.items(), start=1):
            destination = scene_dir / f"{order:02d}_{kind}.jpg"
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"cannot write {destination}")
            written[kind] = destination.relative_to(args.output).as_posix()
        sample["visualizations"] = written
        results.append(sample)

    aggregate: dict[str, object] = {}
    for source in args.sources:
        rows = [row for row in results if row["source"] == source]
        aggregate[source] = {
            "samples": len(rows),
            "mean_segmentation_confidence": mean_available(
                rows, ("prediction", "mean_segmentation_confidence")
            ),
            "mean_segmentation_iou_when_available": {
                name: mean_available(rows, ("segmentation_metrics", "iou", name))
                for name in SURFACE_CLASS_NAMES
            },
            "mean_depth_absrel_when_available": mean_available(
                rows, ("depth_metrics", "absrel")
            ),
            "mean_depth_delta1_when_available": mean_available(
                rows, ("depth_metrics", "delta1")
            ),
            "corridor_level_counts": {
                level: sum(
                    row["prediction"]["center_corridor_level"] == level
                    for row in rows
                )
                for level in ("NEAR", "MID", "FAR", "UNKNOWN")
            },
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "input_contract": [1, 1, 256, 256],
        "sampling": {
            "split": "official_validation",
            "method": "fixed-seed random without replacement per source",
            "seed": args.seed,
            "samples_per_source": args.samples_per_source,
        },
        "visualization_only_caveats": [
            "Colour overlays are cloud diagnostics and are not the grayscale Aurora OSD.",
            "Public validation images do not reproduce the SC132GS spectral response.",
            "Displayed metric depth is diagnostic and is not a calibrated board claim.",
            "Each scene and information type is saved separately; no contact sheet is generated.",
        ],
        "aggregate": aggregate,
        "samples": results,
    }
    (args.output / "visualization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "README.txt").write_text(
        "GrayNav public-validation qualitative diagnostics.\n"
        "Green=ground candidate, red=blocked surface, orange=step/drop.\n"
        "Depth uses warm=near and cool=far. Metric values are diagnostic only.\n"
        "Each source/scene directory contains five separate images.\n"
        "These colour panels are not the grayscale Aurora deployment UI.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, indent=2))
    print("GRAYNAV_VISUALIZATION_OK")


if __name__ == "__main__":
    main()
