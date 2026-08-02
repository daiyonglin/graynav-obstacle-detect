#!/usr/bin/env python3
"""Fail-fast A1 contract and operator audit for the surface model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import onnx


ALLOWED = {
    "Conv", "AveragePool", "GlobalAveragePool", "MaxPool", "BatchNormalization",
    "Add", "Mul", "Concat", "Relu", "Resize", "Constant", "Identity",
}
FORBIDDEN = {
    "Softmax", "ArgMax", "Transpose", "Div", "Sub", "Reshape", "Gather", "Slice",
    "NonMaxSuppression",
}


def dims(value: onnx.ValueInfoProto) -> list[int]:
    return [int(dim.dim_value) for dim in value.type.tensor_type.shape.dim]


def attribute_ints(node: onnx.NodeProto, name: str) -> list[int]:
    for attribute in node.attribute:
        if attribute.name == name:
            return list(attribute.ints)
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.onnx))
    errors: list[str] = []
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        errors.append(str(exc))
    input_shape = dims(model.graph.input[0]) if model.graph.input else []
    output_shape = dims(model.graph.output[0]) if model.graph.output else []
    if input_shape != [1, 1, 256, 256]:
        errors.append(f"input shape mismatch: {input_shape}")
    if output_shape != [1, 4, 32, 32]:
        errors.append(f"output shape mismatch: {output_shape}")
    counts = Counter(node.op_type for node in model.graph.node)
    unsupported = sorted(op for op in counts if op not in ALLOWED)
    forbidden = sorted(op for op in counts if op in FORBIDDEN)
    if unsupported:
        errors.append(f"unsupported operators: {unsupported}")
    if forbidden:
        errors.append(f"forbidden operators: {forbidden}")
    for node in model.graph.node:
        if node.op_type == "Conv":
            for name in ("kernel_shape", "strides", "pads"):
                values = attribute_ints(node, name)
                if values and max(values) > 16:
                    errors.append(f"{node.name or node.output[0]} {name} exceeds 16: {values}")
            weight_name = node.input[1] if len(node.input) > 1 else ""
            weight = next((item for item in model.graph.initializer if item.name == weight_name), None)
            if weight is not None and len(weight.dims) == 4:
                per_kernel = int(weight.dims[1] * weight.dims[2] * weight.dims[3])
                if per_kernel > 2048:
                    errors.append(
                        f"{node.name or node.output[0]} kernel Cin volume exceeds 2048: {per_kernel}"
                    )
        if node.op_type in {"AveragePool", "MaxPool"}:
            kernel = attribute_ints(node, "kernel_shape")
            if kernel and max(kernel) > 8:
                errors.append(f"{node.name or node.output[0]} pool exceeds 8: {kernel}")
    report = {
        "onnx": str(args.onnx),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "op_counts": dict(sorted(counts.items())),
        "unsupported_ops": unsupported,
        "forbidden_ops": forbidden,
        "errors": errors,
        "a1_precheck_passed": not errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
