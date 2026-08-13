#!/usr/bin/env python3
"""Re-evaluate unified checkpoints on one identical, deterministic validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from unified.graynav_unified_perception import build_unified_from_yolo_weights  # noqa: E402
from train_graynav_surface_depth import SurfaceDepthDataset  # noqa: E402
from train_graynav_unified_indoor import (  # noqa: E402
    IndoorDetectionDataset,
    collate_detection,
    evaluate_detection,
    evaluate_scene,
    interleaved_source_indices,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    candidates = [
        args.checkpoints / name
        for name in (
            "best_safety.pt",
            "best_overall.pt",
            "best_detection.pt",
            "best_scene.pt",
            "best_stair.pt",
            "last.pt",
        )
        if (args.checkpoints / name).is_file()
    ]
    if not candidates:
        raise RuntimeError(f"no unified checkpoints found in {args.checkpoints}")

    val_detection = IndoorDetectionDataset(args.coco, "val", 42, 0.0, 0.0)
    partial_detection = IndoorDetectionDataset(args.coco, "val", 42, 1.0, 0.0)
    val_scene = SurfaceDepthDataset(args.scene, "val", experiment="e3")
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    detection_loader = DataLoader(
        val_detection, shuffle=False, collate_fn=collate_detection, **common
    )
    partial_loader = DataLoader(
        partial_detection, shuffle=False, collate_fn=collate_detection, **common
    )
    scene_loader = DataLoader(
        val_scene,
        sampler=interleaved_source_indices(val_scene.records),
        **common,
    )
    detection_batches = math.ceil(len(val_detection) / args.batch_size)
    scene_batches = math.ceil(len(val_scene) / args.batch_size)

    results: list[dict[str, object]] = []
    for checkpoint in candidates:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model, _ = build_unified_from_yolo_weights(args.yolo_weights)
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval()
        metrics = evaluate_scene(model, scene_loader, device, scene_batches)
        metrics.update(
            evaluate_detection(model, detection_loader, device, detection_batches)
        )
        partial = evaluate_detection(
            model, partial_loader, device, detection_batches
        )
        metrics["partial_person_recall"] = partial["person_recall"]
        scene_score = 0.5 * metrics["ground_iou"] + 0.5 * metrics["blocked_iou"]
        detection_score = (
            0.55 * metrics["map50"]
            + 0.25 * metrics["person_ap50"]
            + 0.20 * metrics["partial_person_recall"]
        )
        safety_score = (
            0.45 * metrics["step_f1"]
            + 0.20 * metrics["edge_f1"]
            + 0.20 * scene_score
            + 0.15 * metrics["near_far_order_accuracy"]
            - 0.35 * metrics["no_step_false_image_rate"]
        )
        normal_capability = (
            metrics["person_ap50"] >= 0.50
            and metrics["partial_person_recall"] >= 0.70
            and metrics["ground_iou"] >= 0.50
            and metrics["blocked_iou"] >= 0.50
            and metrics["step_f1"] >= 0.60
            and metrics["near_far_order_accuracy"] >= 0.70
            and metrics["no_step_false_image_rate"] <= 0.20
        )
        combined_score = (
            0.30 * detection_score + 0.25 * scene_score + 0.45 * safety_score
        )
        result = {
            "checkpoint": checkpoint.name,
            "epoch": int(payload.get("epoch", -1)),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
            "metrics": metrics,
            "detection_score": detection_score,
            "scene_score": scene_score,
            "safety_score": safety_score,
            "combined_score": combined_score,
            "normal_capability_gate": normal_capability,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    eligible = [row for row in results if row["normal_capability_gate"]]
    ranked = sorted(
        eligible or results,
        key=lambda row: float(row["combined_score"]),
        reverse=True,
    )
    report = {
        "validation": {
            "detection_images": len(val_detection),
            "partial_person_images": len(partial_detection),
            "scene_images": len(val_scene),
            "deterministic": True,
        },
        "selected_checkpoint": ranked[0]["checkpoint"],
        "selected_by_normal_capability_gate": bool(eligible),
        "ranking": [row["checkpoint"] for row in ranked],
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("GRAYNAV_UNIFIED_CANDIDATE_EVALUATION_OK")


if __name__ == "__main__":
    main()
