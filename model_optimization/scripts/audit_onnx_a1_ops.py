#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


SUPPORTED_OPS = {
    "Conv",
    "AveragePool",
    "GlobalAveragePool",
    "MaxPool",
    "BatchNormalization",
    "Add",
    "Mul",
    "Concat",
    "Split",
    "Relu",
    "LeakyRelu",
    "Sigmoid",
    "Resize",
    "Constant",
    "Identity",
}
RISKY_OPS = {"Sub", "Div", "Transpose", "Softmax", "NonMaxSuppression", "Reshape", "Gather", "Slice"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ONNX operators against the known A1 AI-Tool deployment constraints.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect-input-channels", type=int, required=True)
    parser.add_argument("--expect-num-classes", type=int, required=True)
    return parser.parse_args()


def attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    """Read a scalar/list ONNX node attribute."""
    for a in node.attribute:
        if a.name == name:
            if a.ints:
                return list(a.ints)
            if a.i:
                return int(a.i)
            if a.f:
                return float(a.f)
            if a.s:
                return a.s.decode("utf-8", errors="ignore")
    return default


def input_channels(model: onnx.ModelProto) -> int | None:
    """Read NCHW input channels from the first graph input when statically known."""
    if not model.graph.input:
        return None
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) < 2:
        return None
    return int(dims[1].dim_value) if dims[1].dim_value else None


def output_channels(value_info: onnx.ValueInfoProto) -> int | None:
    """Read NCHW output channels when statically known."""
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) != 4:
        return None
    return int(dims[1].dim_value) if dims[1].dim_value else None


def audit_conv(node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto]) -> list[str]:
    """Check Conv kernel, stride and padding limits documented for A1."""
    issues: list[str] = []
    kernel = attr(node, "kernel_shape", [])
    strides = attr(node, "strides", [])
    pads = attr(node, "pads", [])
    group = int(attr(node, "group", 1) or 1)
    for label, vals in [("kernel", kernel), ("stride", strides), ("pad", pads)]:
        if vals and any(int(v) > 16 for v in vals):
            issues.append(f"{node.name or node.output[0]} Conv {label} exceeds 16: {vals}")
    if group < 1:
        issues.append(f"{node.name or node.output[0]} Conv invalid group={group}")
    weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
    if weight is not None and len(weight.dims) == 4:
        cin_per_group = int(weight.dims[1])
        kh, kw = int(weight.dims[2]), int(weight.dims[3])
        cin = cin_per_group * group
        if cin_per_group * kh * kw > 2048:
            issues.append(
                f"{node.name or node.output[0]} Kw*Kh*Cin/group exceeds 2048"
            )
        if group not in (1, cin):
            issues.append(f"{node.name or node.output[0]} group={group}, Cin={cin}")
    return issues


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.onnx))
    checker_errors: list[str] = []
    try:
        onnx.checker.check_model(model)
    except Exception as exc:  # noqa: BLE001
        checker_errors.append(str(exc))

    ops = Counter(node.op_type for node in model.graph.node)
    unsupported = sorted(op for op in ops if op not in SUPPORTED_OPS)
    risky = sorted(op for op in ops if op in RISKY_OPS)
    issues: list[str] = []
    initializers = {item.name: item for item in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Conv":
            issues.extend(audit_conv(node, initializers))
        if node.op_type in {"AveragePool", "MaxPool"}:
            kernel = attr(node, "kernel_shape", [])
            if kernel and any(int(v) > 8 for v in kernel):
                issues.append(f"{node.name or node.output[0]} MaxPool kernel exceeds 8: {kernel}")
        if node.op_type == "LeakyRelu":
            alpha = float(attr(node, "alpha", 0.01) or 0.01)
            if alpha not in (0.1, 0.01):
                issues.append(f"{node.name or node.output[0]} LeakyRelu alpha not A1-friendly: {alpha}")

    in_ch = input_channels(model)
    if in_ch != args.expect_input_channels:
        issues.append(f"input channels mismatch: got {in_ch}, expected {args.expect_input_channels}")

    out_channels = [output_channels(o) for o in model.graph.output]
    cls_heads = sum(1 for c in out_channels if c == args.expect_num_classes)
    reg_heads = sum(1 for c in out_channels if c == 64)
    if len(model.graph.output) == 6 and (cls_heads != 3 or reg_heads != 3):
        issues.append(f"head6 output mismatch: cls_heads={cls_heads}, reg_heads={reg_heads}, channels={out_channels}")

    report = {
        "onnx": str(args.onnx),
        "input_channels": in_ch,
        "output_channels": out_channels,
        "op_counts": dict(sorted(ops.items())),
        "unsupported_ops": unsupported,
        "risky_ops": risky,
        "checker_errors": checker_errors,
        "constraint_issues": issues,
        "a1_precheck_passed": not checker_errors and not unsupported and not risky and not issues,
        "notes": [
            "This is a static precheck only; final authority is the A1 conversion compiler.",
            "YOLO DFL/Softmax/NMS must remain outside ONNX and run on CPU postprocess.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["a1_precheck_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
