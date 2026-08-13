#!/usr/bin/env python3
"""Restore sparsely supervised indoor class rows from official COCO weights."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified.graynav_unified_perception import (  # noqa: E402
    INDOOR_CLASS_NAMES,
    build_unified_from_yolo_weights,
)


RESTORED_CLASSES = ("backpack", "handbag", "suitcase", "bench")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    official, _ = build_unified_from_yolo_weights(args.yolo_weights)
    candidate, _ = build_unified_from_yolo_weights(args.yolo_weights)
    candidate.load_state_dict(payload["model"], strict=True)
    rows = [INDOOR_CLASS_NAMES.index(name) for name in RESTORED_CLASSES]
    before = copy.deepcopy(candidate.state_dict())
    with torch.no_grad():
        for scale in range(3):
            source = official.detect_head.cv3[scale][-1]
            target = candidate.detect_head.cv3[scale][-1]
            indices = torch.tensor(rows, dtype=torch.long)
            target.weight.index_copy_(
                0, indices, source.weight.index_select(0, indices)
            )
            target.bias.index_copy_(
                0, indices, source.bias.index_select(0, indices)
            )

    after = candidate.state_dict()
    changed: list[str] = []
    for name, value in after.items():
        if not torch.equal(value, before[name]):
            changed.append(name)
    expected_suffixes = {
        f"detector.model.22.cv3.{scale}.2.{kind}"
        for scale in range(2)
        for kind in ("weight", "bias")
    } | {
        f"detector.model.22.cv3.2.3.{kind}" for kind in ("weight", "bias")
    }
    if set(changed) != expected_suffixes:
        raise RuntimeError(f"unexpected tensors changed: {changed}")
    for scale in range(3):
        source = official.detect_head.cv3[scale][-1]
        target = candidate.detect_head.cv3[scale][-1]
        if not torch.equal(target.weight[rows], source.weight[rows]):
            raise RuntimeError(f"scale {scale} restored weights do not match")
        common = [index for index in range(len(INDOOR_CLASS_NAMES)) if index not in rows]
        # Common-class rows must remain exactly as trained.
        branch_weight_name = next(
            name
            for name, parameter in candidate.named_parameters()
            if parameter is target.weight
        )
        branch_bias_name = next(
            name
            for name, parameter in candidate.named_parameters()
            if parameter is target.bias
        )
        if not torch.equal(after[branch_weight_name][common], before[branch_weight_name][common]):
            raise RuntimeError(f"scale {scale} common class weights changed")
        if not torch.equal(after[branch_bias_name][common], before[branch_bias_name][common]):
            raise RuntimeError(f"scale {scale} common class biases changed")

    derived = dict(payload)
    derived["model"] = candidate.state_dict()
    derived["derived_candidate"] = {
        "operation": "restore_sparse_class_projection_rows",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "source_weights": str(args.yolo_weights),
        "restored_classes": list(RESTORED_CLASSES),
        "changed_tensors": changed,
        "training_metrics_are_parent_metrics": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, args.output)
    print(json.dumps({
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
        **derived["derived_candidate"],
    }, ensure_ascii=False, indent=2))
    print("GRAYNAV_RARE_COCO_ROWS_RESTORED_OK")


if __name__ == "__main__":
    main()
