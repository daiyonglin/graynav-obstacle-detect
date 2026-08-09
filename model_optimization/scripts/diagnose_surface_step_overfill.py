#!/usr/bin/env python3
"""Locate and visualize material STEP overfill on the public validation set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from segmentation.graynav_surface_depth import GrayNavSurfaceDepth  # noqa: E402
from train_graynav_surface_depth import (  # noqa: E402
    IGNORE,
    SurfaceDepthDataset,
    false_whole_frame_step_prediction,
)
from visualize_graynav_surface_depth import (  # noqa: E402
    segmentation_view,
    titled_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-visualizations", type=int, default=24)
    return parser.parse_args()


def ratio(mask: torch.Tensor) -> float:
    return float(mask.float().mean()) if mask.numel() else 0.0


def safe_name(source_id: str) -> str:
    return source_id.replace(":", "_").replace("/", "_")


def write_visualization(
    output: Path,
    data: Path,
    record: dict[str, object],
    prediction64: np.ndarray,
    stats: dict[str, object],
) -> dict[str, str]:
    gray = cv2.imread(str(data / str(record["image"])), cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(str(data / str(record["seg_mask"])), cv2.IMREAD_GRAYSCALE)
    if gray is None or target is None:
        raise RuntimeError(f"cannot visualize {record['source_id']}")
    target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    prediction = cv2.resize(prediction64, (256, 256), interpolation=cv2.INTER_NEAREST)
    sample_dir = output / safe_name(str(record["source_id"]))
    sample_dir.mkdir(parents=True, exist_ok=False)
    subtitle = (
        f"GT={stats['truth_step_ratio']:.3f} "
        f"PRED={stats['prediction_step_ratio']:.3f} "
        f"EXCESS={stats['step_excess']:.3f}"
    )
    files = {
        "mono": sample_dir / "01_mono_input.jpg",
        "ground_truth": sample_dir / "02_segmentation_gt.jpg",
        "prediction": sample_dir / "03_segmentation_prediction.jpg",
    }
    cv2.imwrite(str(files["mono"]), titled_image(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "MONO INPUT", str(record["source_id"])))
    cv2.imwrite(str(files["ground_truth"]), titled_image(segmentation_view(gray, target, 0.55), "SEGMENTATION GT", subtitle))
    cv2.imwrite(str(files["prediction"]), titled_image(segmentation_view(gray, prediction, 0.55), "SEGMENTATION PREDICTION", subtitle))
    return {name: str(path.relative_to(output)) for name, path in files.items()}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"diagnostic output already exists: {args.output}")
    if args.max_visualizations < 0:
        raise ValueError("max_visualizations must be non-negative")
    payload = torch.load(args.checkpoint, map_location="cpu")
    contract = payload.get("contract", {})
    model = GrayNavSurfaceDepth(
        width_mult=float(contract.get("width_mult", 1.0)),
        detail64=bool(contract.get("detail64", False)),
        num_surface_classes=int(contract.get("seg_shape", [1, 4])[1]),
    )
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model.to(device).eval()
    dataset = SurfaceDepthDataset(args.data, "val", str(contract.get("experiment", "e1")))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    manifest_by_id = {
        str(record["source_id"]): record for record in dataset.records
    }
    findings: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    for batch in tqdm(loader, desc="step-overfill", unit="batch", dynamic_ncols=True):
        images = batch["image"].to(device)
        truth = F.interpolate(
            batch["seg"][:, None].float(), (64, 64), mode="nearest"
        )[:, 0].long()
        seg_logits, _ = model(images)
        probabilities = torch.softmax(seg_logits, dim=1).cpu()
        confidence, prediction = probabilities.max(1)
        for index, source_id in enumerate(batch["source_id"]):
            target = truth[index]
            guess = prediction[index]
            if not false_whole_frame_step_prediction(guess, target):
                continue
            valid = target != IGNORE
            target_valid = target[valid]
            guess_valid = guess[valid]
            truth_step = ratio(target_valid == 2)
            prediction_step = ratio(guess_valid == 2)
            step_true_positive = int(((target_valid == 2) & (guess_valid == 2)).sum())
            predicted_step_pixels = int((guess_valid == 2).sum())
            truth_step_pixels = int((target_valid == 2).sum())
            union = int(((target_valid == 2) | (guess_valid == 2)).sum())
            unknown = target_valid == 3
            item: dict[str, object] = {
                "source_id": str(source_id),
                "source": str(batch["source"][index]),
                "truth_step_ratio": truth_step,
                "prediction_step_ratio": prediction_step,
                "step_excess": prediction_step - truth_step,
                "raw_prediction_step_ratio": ratio(guess == 2),
                "step_iou": step_true_positive / max(1, union),
                "step_precision": step_true_positive / max(1, predicted_step_pixels),
                "step_recall": step_true_positive / max(1, truth_step_pixels),
                "unknown_to_step_rate": (
                    int(((guess_valid == 2) & unknown).sum()) / max(1, int(unknown.sum()))
                ),
                "mean_confidence": float(confidence[index][valid].mean()),
            }
            findings.append(item)
            predictions[str(source_id)] = guess.numpy().astype(np.uint8)
    findings.sort(key=lambda item: float(item["step_excess"]), reverse=True)
    args.output.mkdir(parents=True, exist_ok=False)
    visualization_root = args.output / "top_overfill_visualizations"
    visualization_root.mkdir()
    for item in findings[: args.max_visualizations]:
        source_id = str(item["source_id"])
        record = manifest_by_id[source_id]
        item["visualizations"] = write_visualization(
            visualization_root,
            args.data,
            record,
            predictions[source_id],
            item,
        )
    excess = np.asarray([float(item["step_excess"]) for item in findings], dtype=np.float64)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "definition": (
            "valid-pixel step ratio > 0.60 and prediction exceeds truth by > 0.20"
        ),
        "count": len(findings),
        "count_by_source": dict(Counter(str(item["source"]) for item in findings)),
        "step_excess_quantiles": {
            name: (float(np.quantile(excess, value)) if excess.size else None)
            for name, value in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("max", 1.0))
        },
        "visualized": min(args.max_visualizations, len(findings)),
        "samples": findings,
    }
    report_path = args.output / "step_overfill_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "count": report["count"],
        "count_by_source": report["count_by_source"],
        "step_excess_quantiles": report["step_excess_quantiles"],
        "visualized": report["visualized"],
        "top_samples": [
            {
                key: item[key]
                for key in (
                    "source_id", "truth_step_ratio", "prediction_step_ratio",
                    "step_excess", "step_iou", "unknown_to_step_rate",
                )
            }
            for item in findings[:10]
        ],
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    print("GRAYNAV_STEP_OVERFILL_DIAGNOSTIC_OK")


if __name__ == "__main__":
    main()
