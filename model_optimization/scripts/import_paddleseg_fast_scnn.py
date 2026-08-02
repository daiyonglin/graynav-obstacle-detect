#!/usr/bin/env python3
"""Import an official PaddleSeg Fast-SCNN checkpoint into the A1-safe PyTorch port.

Paddle and PyTorch both store Conv2D kernels as OIHW.  Parameters are matched in
module order, BatchNorm statistics are preserved, and only the RGB stem is
changed by W_gray = W_R + W_G + W_B.  A mismatch fails closed instead of
silently leaving random layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import paddle
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from segmentation.graynav_fast_scnn import GrayNavFastSCNN  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddle-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-mult", type=float, choices=(1.0, 0.75), default=1.0)
    args = parser.parse_args()

    payload = paddle.load(str(args.paddle_checkpoint))
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    paddle_items = [(name, np.asarray(value)) for name, value in payload.items()]
    model = GrayNavFastSCNN(width_mult=args.width_mult)
    target = model.state_dict()
    target_names = [name for name in target if not name.endswith("num_batches_tracked")]
    if len(paddle_items) != len(target_names):
        raise RuntimeError(
            f"state count mismatch: Paddle={len(paddle_items)} PyTorch={len(target_names)}"
        )

    imported: dict[str, torch.Tensor] = {}
    audit: list[dict[str, object]] = []
    folded = False
    for (source_name, array), target_name in zip(paddle_items, target_names):
        expected = tuple(target[target_name].shape)
        if target_name == "learning_to_downsample.0.0.weight" and array.ndim == 4 and array.shape[1] == 3:
            array = array.sum(axis=1, keepdims=True)
            folded = True
        if array.ndim == len(expected) and all(source >= wanted for source, wanted in zip(array.shape, expected)):
            array = array[tuple(slice(0, wanted) for wanted in expected)]
        if tuple(array.shape) != expected:
            raise RuntimeError(
                f"ordered state mismatch: {source_name} {tuple(array.shape)} -> "
                f"{target_name} {expected}"
            )
        imported[target_name] = torch.from_numpy(array).to(dtype=target[target_name].dtype)
        audit.append({"source": source_name, "target": target_name, "shape": list(expected)})

    for name in target:
        if name.endswith("num_batches_tracked"):
            imported[name] = target[name]
    model.load_state_dict(imported, strict=True)
    if not folded or tuple(model.first_conv.weight.shape)[1] != 1:
        raise RuntimeError("official RGB first convolution was not folded to one channel")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "source": str(args.paddle_checkpoint),
            "one_channel_first_conv_initialized": True,
            "rgb_input_used": False,
            "contract": {
                "input": [1, 1, 256, 256],
                "output": [1, 4, 32, 32],
                "width_mult": args.width_mult,
            },
        },
        args.output,
    )
    args.output.with_suffix(".import.json").write_text(
        json.dumps({"folded": folded, "mapping": audit}, indent=2), encoding="utf-8"
    )
    print("input_shape=1x1x256x256")
    print(f"first_conv_shape={tuple(model.first_conv.weight.shape)}")
    print("one_channel_first_conv_initialized=True")
    print("rgb_input_used=False")


if __name__ == "__main__":
    main()
