#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or validate GrayStem-BC YOLOv8n checkpoints.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["init-from-m1", "tie-bc"], default="tie-bc")
    return parser.parse_args()


def first_conv(model: YOLO) -> torch.nn.Conv2d:
    """Return YOLOv8 first Conv2d layer inside the Ultralytics Conv wrapper."""
    module0 = model.model.model[0]
    conv = getattr(module0, "conv", None)
    if not isinstance(conv, torch.nn.Conv2d):
        raise TypeError(f"unexpected first layer type: {type(module0)} / {type(conv)}")
    if conv.weight.ndim != 4 or conv.weight.shape[1] != 3:
        raise ValueError(f"expected first conv weight [C,3,K,K], got {tuple(conv.weight.shape)}")
    return conv


def tie_first_conv_to_bc(model: YOLO) -> None:
    """Fold any 3-channel first conv into a GrayStem-BC equivalent for [G,G,G] input."""
    conv = first_conv(model)
    with torch.no_grad():
        folded = conv.weight.data.sum(dim=1, keepdim=True) / 3.0
        conv.weight.data.copy_(folded.repeat(1, 3, 1, 1))


def save_model(model: YOLO, path: Path) -> None:
    """Save a modified Ultralytics model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"saved: {path}")


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    tie_first_conv_to_bc(model)
    save_model(model, args.out)


if __name__ == "__main__":
    main()

