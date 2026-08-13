#!/usr/bin/env python3
"""Create separate, fixed visual evidence for unified detection and scene outputs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified.graynav_unified_perception import (  # noqa: E402
    INDOOR_CLASS_NAMES,
    build_unified_from_yolo_weights,
)
from segmentation.graynav_surface_depth import depth_bin_centers  # noqa: E402


COLORS = np.asarray(((60, 180, 75), (220, 60, 60), (255, 190, 40), (80, 120, 220)), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def input_tensor(
    path: Path, device: torch.device
) -> tuple[np.ndarray, torch.Tensor, float, int, int]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read {path}")
    scale = min(384.0 / gray.shape[1], 384.0 / gray.shape[0])
    resized_w = int(round(gray.shape[1] * scale))
    resized_h = int(round(gray.shape[0] * scale))
    resized = cv2.resize(
        gray, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    pad_x = (384 - resized_w) // 2
    pad_y = (384 - resized_h) // 2
    canvas = np.full((384, 384), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    tensor = torch.from_numpy(
        canvas[None, None].astype(np.float32) / 255.0
    ).to(device)
    return gray, tensor, scale, pad_x, pad_y


def unletterbox_grid(
    grid: np.ndarray,
    gray_shape: tuple[int, int],
    scale: float,
    pad_x: int,
    pad_y: int,
) -> np.ndarray:
    """Crop model padding in grid coordinates before restoring image size."""

    grid_h, grid_w = grid.shape[:2]
    resized_w = int(round(gray_shape[1] * scale))
    resized_h = int(round(gray_shape[0] * scale))
    x1 = int(round(pad_x * grid_w / 384.0))
    y1 = int(round(pad_y * grid_h / 384.0))
    x2 = int(round((pad_x + resized_w) * grid_w / 384.0))
    y2 = int(round((pad_y + resized_h) * grid_h / 384.0))
    return grid[max(0, y1) : min(grid_h, y2), max(0, x1) : min(grid_w, x2)]


def save_scene_outputs(
    root: Path,
    stem: str,
    gray: np.ndarray,
    scene: torch.Tensor,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    seg = scene[0, :4].argmax(0).cpu().numpy().astype(np.uint8)
    seg = unletterbox_grid(seg, gray.shape, scale, pad_x, pad_y)
    seg = cv2.resize(COLORS[seg], (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = (torch.softmax(scene[0, 4:20], 0) * depth_bin_centers(scene.device)[:, None, None]).sum(0).cpu().numpy()
    depth = unletterbox_grid(depth, gray.shape, scale, pad_x, pad_y)
    depth = cv2.resize(depth, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    depth_u8 = np.clip((depth - 0.3) / 7.7 * 255.0, 0, 255).astype(np.uint8)
    edge = torch.sigmoid(scene[0, 20]).cpu().numpy()
    edge = unletterbox_grid(edge, gray.shape, scale, pad_x, pad_y)
    edge = cv2.resize(edge, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(root / f"{stem}_mono.png"), gray)
    cv2.imwrite(str(root / f"{stem}_seg.png"), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / f"{stem}_depth.png"), cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO))
    cv2.imwrite(str(root / f"{stem}_stair_edge.png"), np.clip(edge * 255.0, 0, 255).astype(np.uint8))


def save_scene_ground_truth(
    root: Path, stem: str, gray: np.ndarray, row: dict[str, object], data: Path
) -> None:
    seg_path = row.get("seg_mask")
    if seg_path:
        mask = cv2.imread(str(data / str(seg_path)), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            view = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_id, color in enumerate(COLORS):
                view[mask == class_id] = color
            view[mask == 255] = (24, 24, 24)
            cv2.imwrite(
                str(root / f"{stem}_seg_gt.png"),
                cv2.cvtColor(view, cv2.COLOR_RGB2BGR),
            )
    depth_path = row.get("depth")
    if depth_path:
        depth = np.load(data / str(depth_path)).astype(np.float32)
        valid = np.isfinite(depth) & (depth >= 0.3) & (depth <= 8.0)
        depth_u8 = np.zeros_like(depth, dtype=np.uint8)
        depth_u8[valid] = np.clip(
            (depth[valid] - 0.3) / 7.7 * 255.0, 0, 255
        ).astype(np.uint8)
        colored = cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        cv2.imwrite(str(root / f"{stem}_depth_gt.png"), colored)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    model, init = build_unified_from_yolo_weights(args.yolo_weights)
    payload = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    rng = random.Random(args.seed)
    report: dict[str, object] = {"checkpoint": str(args.checkpoint), "epoch": payload.get("epoch"), "initialization": init, "samples": []}

    scene_rows = load_rows(args.scene / "manifest_val.jsonl")
    for source in ("ade20k", "stairnetv3", "nyuv2"):
        rows = [row for row in scene_rows if row["source"] == source]
        selected = rng.sample(rows, min(args.samples_per_source, len(rows)))
        for row in selected:
            path = args.scene / str(row["image"])
            gray, tensor, scale, pad_x, pad_y = input_tensor(path, device)
            scene = model.forward_scene(tensor)
            stem = str(row["source_id"]).replace(":", "_").replace("/", "_")
            folder = args.output / source
            save_scene_outputs(
                folder, stem, gray, scene, scale, pad_x, pad_y
            )
            save_scene_ground_truth(folder, stem, gray, row, args.scene)
            report["samples"].append({"source": source, "source_id": row["source_id"], "stem": stem})

    # Detection evidence uses raw annotations as a stable visual reference; the
    # full mAP evaluator is intentionally kept separate from image generation.
    coco_rows = load_rows(args.coco / "manifest_val.jsonl")
    selected = rng.sample(coco_rows, min(args.samples_per_source, len(coco_rows)))
    for row in selected:
        gray, tensor, scale, pad_x, pad_y = input_tensor(
            Path(str(row["image"])), device
        )
        outputs = model.forward_detection(tensor)
        features = [torch.cat((outputs[1], outputs[0]), 1), torch.cat((outputs[3], outputs[2]), 1), torch.cat((outputs[5], outputs[4]), 1)]
        prediction = model.detect_head._inference(features)
        from ultralytics.utils.ops import non_max_suppression
        detections = non_max_suppression(prediction, conf_thres=0.20, iou_thres=0.60, nc=len(INDOOR_CLASS_NAMES))[0].cpu()
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gt_canvas = canvas.copy()
        for bx, by, bw, bh, cls in row["boxes_xywh"]:
            cv2.rectangle(
                gt_canvas,
                (int(bx), int(by)),
                (int(bx + bw), int(by + bh)),
                (255, 255, 255),
                2,
            )
            cv2.putText(
                gt_canvas,
                INDOOR_CLASS_NAMES[int(cls)],
                (int(bx), max(16, int(by))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        for x1, y1, x2, y2, score, cls in detections[:10].tolist():
            ox1 = int(np.clip((x1 - pad_x) / scale, 0, gray.shape[1] - 1))
            oy1 = int(np.clip((y1 - pad_y) / scale, 0, gray.shape[0] - 1))
            ox2 = int(np.clip((x2 - pad_x) / scale, 0, gray.shape[1] - 1))
            oy2 = int(np.clip((y2 - pad_y) / scale, 0, gray.shape[0] - 1))
            cv2.rectangle(canvas, (ox1, oy1), (ox2, oy2), (255, 255, 255), 2)
            cv2.putText(canvas, f"{INDOOR_CLASS_NAMES[int(cls)]} {score:.2f}", (ox1, max(16, oy1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        stem = str(row["source_id"]).replace(":", "_")
        folder = args.output / "detection"
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{stem}_mono.png"), gray)
        cv2.imwrite(str(folder / f"{stem}_detections_gt.png"), gt_canvas)
        cv2.imwrite(str(folder / f"{stem}_detections.png"), canvas)
        report["samples"].append({
            "source": row.get("source", "detection"),
            "source_id": row["source_id"],
            "stem": stem,
        })

    (args.output / "visualization_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": len(report["samples"])}, ensure_ascii=False, indent=2))
    print("GRAYNAV_UNIFIED_VISUALIZATION_OK")


if __name__ == "__main__":
    main()
