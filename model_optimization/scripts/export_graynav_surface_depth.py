#!/usr/bin/env python3
"""Export the static two-output SurfaceDepth deployment graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from segmentation.graynav_surface_depth import GrayNavSurfaceDepth  # noqa: E402


def strip_identity_nodes(path: Path) -> int:
    """Remove exporter-only aliases so the deployment graph stays A1 minimal."""

    graph = onnx.load(str(path))
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
    }
    if not aliases:
        return 0

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    kept = []
    for node in graph.graph.node:
        if node.op_type == "Identity":
            continue
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        kept.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(kept)
    for output in graph.graph.output:
        output.name = resolve(output.name)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))
    return len(aliases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--width-mult", type=float, choices=(1.0, 0.75), default=1.0)
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()
    if not args.random_init and args.checkpoint is None:
        raise RuntimeError("provide --checkpoint or explicitly use --random-init for A1 preflight")
    model = GrayNavSurfaceDepth(width_mult=args.width_mult)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu")
        contract = payload.get("contract", {}) if isinstance(payload, dict) else {}
        expected_width = float(contract.get("width_mult", args.width_mult))
        if expected_width != args.width_mult:
            raise RuntimeError(f"checkpoint width_mult={expected_width}, requested={args.width_mult}")
        model.load_state_dict(payload.get("model", payload), strict=True)
    model.eval()
    dummy = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        seg, depth = model(dummy)
    if tuple(seg.shape) != (1, 3, 64, 64) or tuple(depth.shape) != (1, 16, 64, 64):
        raise RuntimeError(f"unexpected outputs seg={tuple(seg.shape)} depth={tuple(depth.shape)}")
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.onnx,
        input_names=["images"], output_names=["seg_logits", "depth_logits"],
        opset_version=12, do_constant_folding=True, dynamic_axes=None,
    )
    removed_identities = strip_identity_nodes(args.onnx)
    contract = {
        "model": "graynav_surface_depth_gray1",
        "input_name": "images",
        "input_shape": [1, 1, 256, 256],
        "outputs": {
            "seg_logits": [1, 3, 64, 64],
            "depth_logits": [1, 16, 64, 64],
        },
        "surface_classes": ["ground_candidate", "blocked_surface", "step_or_drop"],
        "depth_bins": 16,
        "depth_range_m": [0.3, 8.0],
        "postprocess": "CPU argmax, softmax/expectation, medians, temporal and geometry fusion",
        "random_init_preflight": bool(args.random_init),
        "removed_identity_nodes": removed_identities,
    }
    args.onnx.with_suffix(".contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
