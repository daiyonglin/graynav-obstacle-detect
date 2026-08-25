from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_fast_scnn_weights import _mapping_table, convert_state  # noqa: E402
from models.graynav_fast_scnn import GrayNavFastSCNN  # noqa: E402


def synthetic_paddle_state() -> dict[str, object]:
    target = GrayNavFastSCNN().state_dict()
    payload: dict[str, object] = {}
    for item in _mapping_table():
        target_weight = target[f"{item.target}.0.weight"]
        source_shape = list(target_weight.shape)
        if item.transform == "rgb_to_gray":
            source_shape[1] = 3
        elif item.transform == "spatial_sum":
            source_shape[2:] = [3, 3]
        payload[f"{item.source}._conv.weight"] = np.ones(source_shape, np.float32)
        payload[f"{item.source}._conv.bias"] = np.full(
            source_shape[0], 2.0, np.float32
        )
        for source_suffix, target_suffix, value in (
            ("weight", "weight", 1.0),
            ("bias", "bias", 0.0),
            ("_mean", "running_mean", 5.0),
            ("_variance", "running_var", 4.0),
        ):
            shape = target[f"{item.target}.1.{target_suffix}"].shape
            payload[f"{item.source}._batch_norm.{source_suffix}"] = np.full(
                tuple(shape), value, np.float32
            )
    payload["classifier.conv.weight"] = np.zeros((19, 128, 1, 1), np.float32)
    payload["classifier.conv.bias"] = np.zeros(19, np.float32)
    payload["auxlayer.conv.weight"] = np.zeros((19, 32, 1, 1), np.float32)
    payload["StructuredToParameterName@@"] = {}
    return payload


class PaddleSegFastSCNNImportTest(unittest.TestCase):
    def test_transforms_rgb_bias_and_ppm_kernel(self) -> None:
        model, report = convert_state(synthetic_paddle_state(), 1.0)
        state = model.state_dict()

        self.assertEqual(report["imported_target_tensors"], 220)
        self.assertEqual(tuple(model.first_conv.weight.shape), (32, 1, 3, 3))
        self.assertTrue((model.first_conv.weight == 3.0).all())
        self.assertTrue(
            (state["learning_to_downsample.0.1.running_mean"] == 3.0).all()
        )
        self.assertTrue((state["pyramid_pooling.fuse.0.weight"] == 9.0).all())

    def test_unknown_checkpoint_state_fails_closed(self) -> None:
        payload = synthetic_paddle_state()
        payload["unexpected.weight"] = np.zeros(1, np.float32)
        with self.assertRaisesRegex(RuntimeError, "unexpected unmapped Paddle states"):
            convert_state(payload, 1.0)


if __name__ == "__main__":
    unittest.main()
