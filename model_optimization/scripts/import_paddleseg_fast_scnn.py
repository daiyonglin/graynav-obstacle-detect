#!/usr/bin/env python3
"""Import official PaddleSeg Fast-SCNN weights into the A1-safe PyTorch port.

The two implementations are deliberately not state-for-state identical:

* Paddle ConvBN blocks may contain a convolution bias, while the PyTorch port
  uses bias-free convolutions.  Such a bias is folded into the following BN
  running mean so inference remains equivalent.
* The RGB stem is folded with ``W_gray = W_R + W_G + W_B``.
* PaddleSeg's final PPM fusion convolution is 3x3.  The A1-safe graph uses 1x1;
  summing the spatial kernel preserves the response for locally constant input.
* The Cityscapes classifier and training-only auxiliary head are not useful for
  GrayNav and are intentionally not imported.

Every shared encoder/decoder tensor is mapped by an explicit module name.  Any
unknown checkpoint state fails closed instead of being silently misaligned.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from segmentation.graynav_fast_scnn import GrayNavFastSCNN  # noqa: E402


@dataclass(frozen=True)
class ConvBNMapping:
    source: str
    target: str
    transform: str = "identity"


def _mapping_table() -> list[ConvBNMapping]:
    mappings = [
        ConvBNMapping(
            "learning_to_downsample.conv_bn_relu",
            "learning_to_downsample.0",
            "rgb_to_gray",
        ),
    ]
    for block_index in (1, 2):
        target_index = block_index
        source_block = f"learning_to_downsample.dsconv_bn_relu{block_index}"
        mappings.extend(
            (
                ConvBNMapping(
                    f"{source_block}.depthwise_conv",
                    f"learning_to_downsample.{target_index}.0",
                ),
                ConvBNMapping(
                    f"{source_block}.piontwise_conv",
                    f"learning_to_downsample.{target_index}.1",
                ),
            )
        )

    for stage in (1, 2, 3):
        for block in range(3):
            for conv in range(3):
                mappings.append(
                    ConvBNMapping(
                        f"global_feature_extractor.bottleneck{stage}.{block}.block.{conv}",
                        f"bottleneck{stage}.{block}.block.{conv}",
                    )
                )

    for stage, target_branch in enumerate(("proj8", "proj4", "proj2", "proj1")):
        mappings.append(
            ConvBNMapping(
                f"global_feature_extractor.ppm.stages.{stage}.1",
                f"pyramid_pooling.{target_branch}",
            )
        )
    mappings.append(
        ConvBNMapping(
            "global_feature_extractor.ppm.conv_bn_relu2",
            "pyramid_pooling.fuse",
            "spatial_sum",
        )
    )
    mappings.extend(
        (
            ConvBNMapping("feature_fusion.dwconv", "low_depthwise"),
            ConvBNMapping("feature_fusion.conv_low_res", "low_pointwise"),
            ConvBNMapping("feature_fusion.conv_high_res", "high_pointwise"),
        )
    )
    for decoder_block in (1, 2):
        target_index = decoder_block - 1
        source_block = f"classifier.dsconv{decoder_block}"
        mappings.extend(
            (
                ConvBNMapping(
                    f"{source_block}.depthwise_conv",
                    f"classifier.{target_index}.0",
                ),
                ConvBNMapping(
                    f"{source_block}.piontwise_conv",
                    f"classifier.{target_index}.1",
                ),
            )
        )
    return mappings


def _fit_prefix(array: np.ndarray, expected: tuple[int, ...], name: str) -> np.ndarray:
    if array.ndim != len(expected):
        raise RuntimeError(f"rank mismatch for {name}: {array.shape} -> {expected}")
    if array.ndim == 4 and tuple(array.shape[2:]) != tuple(expected[2:]):
        raise RuntimeError(
            f"kernel mismatch for {name}: {tuple(array.shape)} -> {expected}"
        )
    if any(source < wanted for source, wanted in zip(array.shape, expected)):
        raise RuntimeError(f"state too small for {name}: {array.shape} -> {expected}")
    fitted = array[tuple(slice(0, wanted) for wanted in expected)]
    if tuple(fitted.shape) != expected:
        raise RuntimeError(f"shape mismatch for {name}: {fitted.shape} -> {expected}")
    return np.ascontiguousarray(fitted)


def convert_state(
    payload: dict[str, object], width_mult: float
) -> tuple[GrayNavFastSCNN, dict[str, object]]:
    """Convert an already loaded Paddle state dictionary and return an audit."""

    torch.manual_seed(0)
    model = GrayNavFastSCNN(width_mult=width_mult)
    target = model.state_dict()
    converted = {name: value.clone() for name, value in target.items()}
    consumed: set[str] = set()
    audit: list[dict[str, object]] = []
    folded_biases: list[str] = []

    def source_array(name: str, *, optional: bool = False) -> np.ndarray | None:
        value = payload.get(name)
        if value is None:
            if optional:
                return None
            raise RuntimeError(f"missing Paddle state: {name}")
        consumed.add(name)
        return np.asarray(value)

    for item in _mapping_table():
        source_weight_name = f"{item.source}._conv.weight"
        target_weight_name = f"{item.target}.0.weight"
        weight = source_array(source_weight_name)
        assert weight is not None
        if item.transform == "rgb_to_gray":
            if weight.ndim != 4 or weight.shape[1] != 3:
                raise RuntimeError(
                    f"expected RGB stem at {source_weight_name}, got {weight.shape}"
                )
            weight = weight.sum(axis=1, keepdims=True)
        elif item.transform == "spatial_sum":
            if weight.ndim != 4 or tuple(weight.shape[2:]) != (3, 3):
                raise RuntimeError(
                    f"expected 3x3 PPM fusion at {source_weight_name}, got {weight.shape}"
                )
            weight = weight.sum(axis=(2, 3), keepdims=True)
        elif item.transform != "identity":
            raise RuntimeError(f"unknown transform: {item.transform}")

        expected_weight = tuple(target[target_weight_name].shape)
        weight = _fit_prefix(weight, expected_weight, target_weight_name)
        converted[target_weight_name] = torch.from_numpy(weight).to(
            dtype=target[target_weight_name].dtype
        )
        audit.append(
            {
                "source": source_weight_name,
                "target": target_weight_name,
                "shape": list(expected_weight),
                "transform": item.transform,
            }
        )

        source_bias_name = f"{item.source}._conv.bias"
        conv_bias = source_array(source_bias_name, optional=True)
        bn_suffixes = (
            ("weight", "weight"),
            ("bias", "bias"),
            ("_mean", "running_mean"),
            ("_variance", "running_var"),
        )
        for source_suffix, target_suffix in bn_suffixes:
            source_name = f"{item.source}._batch_norm.{source_suffix}"
            target_name = f"{item.target}.1.{target_suffix}"
            array = source_array(source_name)
            assert array is not None
            if source_suffix == "_mean" and conv_bias is not None:
                # BN(conv(x) + b; mean, var) == BN(conv(x); mean - b, var).
                array = array - conv_bias
                folded_biases.append(source_bias_name)
            expected = tuple(target[target_name].shape)
            array = _fit_prefix(array, expected, target_name)
            converted[target_name] = torch.from_numpy(array).to(
                dtype=target[target_name].dtype
            )
            audit.append(
                {"source": source_name, "target": target_name, "shape": list(expected)}
            )

    permitted_ignored = {
        name
        for name in payload
        if name == "StructuredToParameterName@@"
        or name.startswith("auxlayer.")
        or name.startswith("classifier.conv.")
    }
    unexpected = sorted(set(payload) - consumed - permitted_ignored)
    if unexpected:
        raise RuntimeError(f"unexpected unmapped Paddle states: {unexpected[:10]}")

    model.load_state_dict(converted, strict=True)
    if tuple(model.first_conv.weight.shape)[1] != 1:
        raise RuntimeError("official RGB first convolution was not folded to one channel")
    shared_target_names = {
        name
        for name in target
        if not name.endswith("num_batches_tracked")
        and not name.startswith("classifier.2.")
    }
    imported_target_names = {row["target"] for row in audit}
    missing = sorted(shared_target_names - imported_target_names)
    if missing:
        raise RuntimeError(f"shared PyTorch states were not initialized: {missing[:10]}")

    report: dict[str, object] = {
        "input_shape": [1, 1, 256, 256],
        "first_conv_shape": list(model.first_conv.weight.shape),
        "one_channel_first_conv_initialized": True,
        "rgb_input_used": False,
        "width_mult": width_mult,
        "imported_target_tensors": len(imported_target_names),
        "folded_conv_biases": sorted(set(folded_biases)),
        "intentionally_ignored_source_states": sorted(permitted_ignored),
        "random_task_head_states": ["classifier.2.weight", "classifier.2.bias"],
        "mapping": audit,
    }
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddle-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width-mult", type=float, choices=(1.0, 0.75), default=1.0)
    args = parser.parse_args()

    # Keep Paddle optional when conversion helpers are imported by unit tests.
    import paddle

    payload = paddle.load(str(args.paddle_checkpoint))
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported Paddle checkpoint payload: {type(payload)!r}")
    model, report = convert_state(payload, args.width_mult)

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
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("input_shape=1x1x256x256")
    print(f"first_conv_shape={tuple(model.first_conv.weight.shape)}")
    print("one_channel_first_conv_initialized=True")
    print("rgb_input_used=False")
    print(f"imported_target_tensors={report['imported_target_tensors']}")
    print(f"folded_conv_bias_count={len(report['folded_conv_biases'])}")
    print("GRAYNAV_FASTSCNN_IMPORT_OK")


if __name__ == "__main__":
    main()
