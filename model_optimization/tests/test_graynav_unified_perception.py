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

    def test_true_mono_seven_output_contract(self) -> None:
        with torch.no_grad():
            outputs = self.model(torch.zeros(1, 1, 384, 384))
        expected = (
            (1, 8, 48, 48), (1, 64, 48, 48),
            (1, 8, 24, 24), (1, 64, 24, 24),
            (1, 8, 12, 12), (1, 64, 12, 12),
            (1, 21, 48, 48),
        )
        self.assertEqual(OUTPUT_NAMES, (
            "cls_p3", "reg_p3", "cls_p4", "reg_p4",
            "cls_p5", "reg_p5", "scene_logits",
        ))
        self.assertEqual(tuple(tuple(item.shape) for item in outputs), expected)
        self.assertTrue(all(bool(torch.isfinite(item).all()) for item in outputs))

    def test_first_conv_is_single_channel(self) -> None:
        self.assertEqual(self.model.first_conv.in_channels, 1)
        self.assertEqual(self.model.first_conv.weight.shape[1], 1)

    def test_detection_convolutions_respect_a1_input_limit(self) -> None:
        for branch_group in (self.model.detect_head.cv2, self.model.detect_head.cv3):
            for branch in branch_group:
                for module in branch.modules():
                    if not isinstance(module, torch.nn.Conv2d):
                        continue
                    kh, kw = module.kernel_size
                    self.assertLessEqual(kh * kw * module.in_channels, 2048)
                    self.assertIn(module.groups, (1, module.in_channels))

    def test_all_three_tasks_backpropagate(self) -> None:
        model = build_random_unified_yolov8n().train()
        outputs = model(torch.rand(1, 1, 128, 128))
        loss = sum(item.float().square().mean() for item in outputs)
        loss.backward()
        self.assertIsNotNone(model.first_conv.weight.grad)
        self.assertIsNotNone(model.seg_head.weight.grad)
        self.assertIsNotNone(model.depth_head.weight.grad)
        self.assertIsNotNone(model.stair_edge_head.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.first_conv.weight.grad).all()))

    def test_training_fast_paths_match_full_forward(self) -> None:
        image = torch.rand(1, 1, 128, 128)
        with torch.no_grad():
            full = self.model(image)
            detection = self.model.forward_detection(image)
            scene = self.model.forward_scene(image)
        for expected, actual in zip(full[:6], detection):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(full[-1], scene))


if __name__ == "__main__":
    unittest.main()
