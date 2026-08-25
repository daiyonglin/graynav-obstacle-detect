#!/usr/bin/env python3
"""Compare all seven PyTorch and ONNX outputs on deterministic mono inputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_a1_calibration import letterbox_gray  # noqa: E402
from models.graynav_unified_perception import (  # noqa: E402
    OUTPUT_NAMES,
    build_unified_from_yolo_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def cosine(one: np.ndarray, two: np.ndarray) -> float:
    left = one.astype(np.float64, copy=False).reshape(-1)
    right = two.astype(np.float64, copy=False).reshape(-1)
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-20:
        return 1.0 if np.allclose(left, right) else 0.0
    return float((left @ right) / denominator)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    model, _ = build_unified_from_yolo_weights(args.yolo_weights)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    if [item.name for item in session.get_outputs()] != list(OUTPUT_NAMES):
        raise RuntimeError("ONNX output names/order do not match the deployment contract")
    rows = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("empty validation manifest")

    cosine_values: dict[str, list[float]] = {name: [] for name in OUTPUT_NAMES}
    max_abs = {name: 0.0 for name in OUTPUT_NAMES}
    cls_matches = [0, 0, 0]
    cls_cells = [0, 0, 0]
    seg_matches = depth_matches = edge_matches = scene_cells = 0
    for row in rows:
        tensor = letterbox_gray(Path(str(row["image"])))
        with torch.no_grad():
            torch_outputs = [item.numpy() for item in model(torch.from_numpy(tensor))]
        onnx_outputs = session.run(list(OUTPUT_NAMES), {"images": tensor})
        for name, torch_value, onnx_value in zip(
            OUTPUT_NAMES, torch_outputs, onnx_outputs
        ):
            cosine_values[name].append(cosine(torch_value, onnx_value))
            max_abs[name] = max(
                max_abs[name], float(np.max(np.abs(torch_value - onnx_value)))
            )
        for scale in range(3):
            torch_cls = torch_outputs[scale * 2].argmax(1)
            onnx_cls = onnx_outputs[scale * 2].argmax(1)
            cls_matches[scale] += int((torch_cls == onnx_cls).sum())
            cls_cells[scale] += int(torch_cls.size)
        torch_scene, onnx_scene = torch_outputs[-1], onnx_outputs[-1]
        seg_matches += int(
            (torch_scene[:, :4].argmax(1) == onnx_scene[:, :4].argmax(1)).sum()
        )
        depth_matches += int(
            (torch_scene[:, 4:20].argmax(1) == onnx_scene[:, 4:20].argmax(1)).sum()
        )
        edge_matches += int(
            ((torch_scene[:, 20] >= 0) == (onnx_scene[:, 20] >= 0)).sum()
        )
        scene_cells += int(torch_scene.shape[2] * torch_scene.shape[3])

    report = {
        "samples": len(rows),
        "output_cosine_min": {
            name: min(values) for name, values in cosine_values.items()
        },
        "output_max_abs_error": max_abs,
        "cls_grid_agreement": {
            f"p{index + 3}": cls_matches[index] / max(1, cls_cells[index])
            for index in range(3)
        },
        "seg_grid_agreement": seg_matches / max(1, scene_cells),
        "depth_level_agreement": depth_matches / max(1, scene_cells),
        "stair_edge_sign_agreement": edge_matches / max(1, scene_cells),
    }
    report["passed"] = (
        min(report["output_cosine_min"].values()) >= 0.999
        and report["seg_grid_agreement"] >= 0.99
        and report["depth_level_agreement"] >= 0.99
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)
    print("GRAYNAV_UNIFIED_ONNX_CONSISTENCY_OK")


if __name__ == "__main__":
    main()
