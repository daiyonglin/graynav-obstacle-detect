#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-name", default="images")
    parser.add_argument(
        "--mode",
        choices=["replicate_exact", "luma_init"],
        default="replicate_exact",
        help=(
            "replicate_exact preserves a model trained with [gray,gray,gray] input "
            "by summing RGB conv weights. luma_init is for initializing a model "
            "from natural RGB pretrained weights before gray fine-tuning."
        ),
    )
    parser.add_argument("--weights", nargs=3, type=float, default=[0.299, 0.587, 0.114])
    return parser.parse_args()


def find_first_conv(model: onnx.ModelProto) -> onnx.NodeProto:
    for node in model.graph.node:
        if node.op_type == "Conv":
            return node
    raise RuntimeError("No Conv node found")


def replace_initializer(model: onnx.ModelProto, name: str, array: np.ndarray) -> None:
    init = numpy_helper.from_array(array.astype(np.float32), name=name)
    for i, old in enumerate(model.graph.initializer):
        if old.name == name:
            model.graph.initializer.remove(old)
            model.graph.initializer.insert(i, init)
            return
    raise RuntimeError(f"initializer not found: {name}")


def main() -> None:
    args = parse_args()
    model = onnx.load(str(args.input))
    conv = find_first_conv(model)
    weight_name = conv.input[1]
    weight_init = next((x for x in model.graph.initializer if x.name == weight_name), None)
    if weight_init is None:
        raise RuntimeError(f"first conv weight initializer not found: {weight_name}")
    w = numpy_helper.to_array(weight_init)
    if w.ndim != 4 or w.shape[1] != 3:
        raise RuntimeError(f"expected first conv weight [out,3,k,k], got {w.shape}")
    if args.mode == "replicate_exact":
        w_gray = np.sum(w.astype(np.float32), axis=1, keepdims=True)
    else:
        coeff = np.array(args.weights, dtype=np.float32).reshape(1, 3, 1, 1)
        w_gray = np.sum(w.astype(np.float32) * coeff, axis=1, keepdims=True)
    replace_initializer(model, weight_name, w_gray)

    for inp in model.graph.input:
        if inp.name == args.input_name:
            shape = inp.type.tensor_type.shape.dim
            if len(shape) != 4:
                raise RuntimeError(f"expected 4D input, got {len(shape)} dims")
            shape[1].dim_value = 1
            break
    else:
        raise RuntimeError(f"input not found: {args.input_name}")

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"mode={args.mode}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
