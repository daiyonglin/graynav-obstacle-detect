#!/usr/bin/env python3
"""Fail-fast static contract and A1 operator audit for the unified graph."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx


ALLOWED = {
    "Conv", "AveragePool", "MaxPool", "BatchNormalization", "Add", "Mul",
    "Concat", "Split", "Relu", "Sigmoid", "Resize", "Constant",
}
FORBIDDEN = {
    "Softmax", "ArgMax", "Sub", "Div", "Transpose", "NonMaxSuppression",
    "Reshape", "Gather", "Slice", "Shape",
}
EXPECTED_INPUTS = {"images": [1, 1, 384, 384]}
EXPECTED_OUTPUTS = {
    "cls_p3": [1, 80, 48, 48], "reg_p3": [1, 64, 48, 48],
    "cls_p4": [1, 80, 24, 24], "reg_p4": [1, 64, 24, 24],
    "cls_p5": [1, 80, 12, 12], "reg_p5": [1, 64, 12, 12],
    "seg_logits": [1, 4, 48, 48], "depth_logits": [1, 16, 48, 48],
}


def dims(value: onnx.ValueInfoProto) -> list[int]:
    return [int(item.dim_value) for item in value.type.tensor_type.shape.dim]


def ints(node: onnx.NodeProto, name: str) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == name:
            return list(attribute.ints)
    return []


def scalar(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(attribute.i)
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    model = onnx.load(str(args.onnx))
    errors: list[str] = []
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))

    input_shapes = {value.name: dims(value) for value in model.graph.input}
    output_shapes = {value.name: dims(value) for value in model.graph.output}
    if input_shapes != EXPECTED_INPUTS:
        errors.append(f"input contract mismatch: {input_shapes}")
    if output_shapes != EXPECTED_OUTPUTS:
        errors.append(f"output contract mismatch: {output_shapes}")

    counts = Counter(node.op_type for node in model.graph.node)
    unsupported = sorted(op for op in counts if op not in ALLOWED)
    forbidden = sorted(op for op in counts if op in FORBIDDEN)
    if unsupported:
        errors.append(f"unsupported operators: {unsupported}")
    if forbidden:
        errors.append(f"forbidden operators: {forbidden}")

    initializers = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Conv":
            for name in ("kernel_shape", "strides", "pads"):
                values = ints(node, name)
                if values and max(values) > 16:
                    errors.append(
                        f"{node.name or node.output[0]} {name} exceeds 16: {values}"
                    )
            weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
            group = scalar(node, "group", 1)
            if weight is not None and len(weight.dims) == 4:
                cin_per_group = int(weight.dims[1])
                kh, kw = int(weight.dims[2]), int(weight.dims[3])
                cin = cin_per_group * group
                if cin_per_group * kh * kw > 2048:
                    errors.append(
                        f"{node.name or node.output[0]} Kw*Kh*Cin/group exceeds 2048"
                    )
                if group not in (1, cin):
                    errors.append(
                        f"{node.name or node.output[0]} group={group}, Cin={cin}"
                    )
        if node.op_type in {"AveragePool", "MaxPool"}:
            kernel = ints(node, "kernel_shape")
            if kernel and max(kernel) > 8:
                errors.append(f"{node.name or node.output[0]} pool exceeds 8: {kernel}")

    report = {
        "onnx": str(args.onnx),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "op_counts": dict(sorted(counts.items())),
        "unsupported_ops": unsupported,
        "forbidden_ops": forbidden,
        "errors": errors,
        "a1_precheck_passed": not errors,
        "note": "Static precheck only; the official A1 compiler remains authoritative.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)
    print("GRAYNAV_UNIFIED_A1_PRECHECK_OK")


if __name__ == "__main__":
    main()
