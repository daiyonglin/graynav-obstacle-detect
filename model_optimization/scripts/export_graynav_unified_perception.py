#!/usr/bin/env python3
"""Export the single-model GrayNav random/pretrained A1 feasibility graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified.graynav_unified_perception import (  # noqa: E402
    OUTPUT_NAMES,
    SURFACE_CLASSES,
    build_random_unified_yolov8n,
    build_unified_from_yolo_weights,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_identity_nodes(path: Path) -> int:
    """Remove exporter-only aliases before the strict A1 graph audit."""

    model = onnx.load(str(path))
    aliases = {
        node.output[0]: node.input[0]
        for node in model.graph.node
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
    for node in model.graph.node:
        if node.op_type == "Identity":
            continue
        for index, name in enumerate(node.input):
            node.input[index] = resolve(name)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    for output in model.graph.output:
        output.name = resolve(output.name)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return len(aliases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--surface-e3", type=Path)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    if args.opset != 12:
        raise ValueError("the first A1 contract fixes opset 12")
    if args.onnx.exists():
        raise FileExistsError(args.onnx)

    torch.manual_seed(42)
    if args.weights is None:
        model = build_random_unified_yolov8n()
        init_report: dict[str, object] = {
            "detector_initialization": "random",
            "a1_safe_depthwise_detection_heads": True,
            "static_c2f_splits": True,
            "one_channel_first_conv_initialized": True,
            "rgb_input_used": False,
        }
    else:
        model, init_report = build_unified_from_yolo_weights(args.weights)
        init_report["detector_initialization"] = "folded_official_coco80"
    if args.surface_e3 is not None:
        init_report["surface_e3_import"] = model.import_surface_e3_heads(args.surface_e3)

    model.eval()
    sample = torch.zeros(1, 1, 384, 384, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(sample)
    output_shapes = {
        name: list(tensor.shape) for name, tensor in zip(OUTPUT_NAMES, outputs)
    }
    expected = {
        "cls_p3": [1, 80, 48, 48], "reg_p3": [1, 64, 48, 48],
        "cls_p4": [1, 80, 24, 24], "reg_p4": [1, 64, 24, 24],
        "cls_p5": [1, 80, 12, 12], "reg_p5": [1, 64, 12, 12],
        "seg_logits": [1, 4, 48, 48], "depth_logits": [1, 16, 48, 48],
    }
    if output_shapes != expected:
        raise RuntimeError(f"unified output contract mismatch: {output_shapes}")

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        str(args.onnx),
        input_names=["images"],
        output_names=list(OUTPUT_NAMES),
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,
        # PyTorch 2.12 defaults to the dynamo exporter, which first emits
        # opset 18 and then attempts a lossy down-conversion.  GrayNav's A1
        # flow is intentionally pinned to the proven static opset-12 path.
        dynamo=False,
    )
    removed_identities = strip_identity_nodes(args.onnx)
    contract = {
        "model": "graynav_unified_perception_gray1",
        "random_init_preflight": args.weights is None,
        "input": {"images": [1, 1, 384, 384]},
        "outputs": expected,
        "surface_classes": list(SURFACE_CLASSES),
        "depth_bins": 16,
        "depth_range_m": [0.3, 8.0],
        "cpu_postprocess": "sigmoid, DFL, NMS, argmax, depth grouping, temporal fusion",
        "initialization": init_report,
        "removed_identity_nodes": removed_identities,
        "onnx": {
            "bytes": args.onnx.stat().st_size,
            "sha256": sha256(args.onnx),
            "opset": args.opset,
        },
    }
    contract_path = args.onnx.with_suffix(".contract.json")
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print("GRAYNAV_UNIFIED_EXPORT_OK")


if __name__ == "__main__":
    main()
