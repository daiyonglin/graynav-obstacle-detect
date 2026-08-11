from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from unified.graynav_unified_perception import (  # noqa: E402
        OUTPUT_NAMES,
        build_random_unified_yolov8n,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    if exc.name == "ultralytics":
        build_random_unified_yolov8n = None
        OUTPUT_NAMES = ()
    else:
        raise


@unittest.skipIf(build_random_unified_yolov8n is None, "ultralytics is not installed")
class GrayNavUnifiedPerceptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(42)
        cls.model = build_random_unified_yolov8n().eval()

    def test_true_mono_eight_output_contract(self) -> None:
        with torch.no_grad():
            outputs = self.model(torch.zeros(1, 1, 384, 384))
        expected = (
            (1, 80, 48, 48), (1, 64, 48, 48),
            (1, 80, 24, 24), (1, 64, 24, 24),
            (1, 80, 12, 12), (1, 64, 12, 12),
            (1, 4, 48, 48), (1, 16, 48, 48),
        )
        self.assertEqual(OUTPUT_NAMES, (
            "cls_p3", "reg_p3", "cls_p4", "reg_p4",
            "cls_p5", "reg_p5", "seg_logits", "depth_logits",
        ))
        self.assertEqual(tuple(tuple(item.shape) for item in outputs), expected)
        self.assertTrue(all(bool(torch.isfinite(item).all()) for item in outputs))

    def test_first_conv_is_single_channel(self) -> None:
        self.assertEqual(self.model.first_conv.in_channels, 1)
        self.assertEqual(self.model.first_conv.weight.shape[1], 1)

    def test_all_three_tasks_backpropagate(self) -> None:
        model = build_random_unified_yolov8n().train()
        outputs = model(torch.rand(1, 1, 128, 128))
        loss = sum(item.float().square().mean() for item in outputs)
        loss.backward()
        self.assertIsNotNone(model.first_conv.weight.grad)
        self.assertIsNotNone(model.seg_head.weight.grad)
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.first_conv.weight.grad).all()))


if __name__ == "__main__":
    unittest.main()
