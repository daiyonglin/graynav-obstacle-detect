from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.graynav_unified_perception import build_random_unified_yolov8n  # noqa: E402


class UnifiedDetectionLossTest(unittest.TestCase):
    def test_ultralytics_loss_accepts_raw_unified_outputs(self) -> None:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.loss import v8DetectionLoss

        model = build_random_unified_yolov8n().train()
        model.detector.args = get_cfg()
        outputs = model(torch.rand(2, 1, 128, 128))
        features = [
            torch.cat((outputs[1], outputs[0]), 1),
            torch.cat((outputs[3], outputs[2]), 1),
            torch.cat((outputs[5], outputs[4]), 1),
        ]
        batch = {
            "img": torch.rand(2, 1, 128, 128),
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([
                [0.5, 0.5, 0.4, 0.6],
                [0.5, 0.5, 0.3, 0.3],
            ]),
        }
        loss, parts = v8DetectionLoss(model.detector)(features, batch)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        self.assertTrue(bool(torch.isfinite(parts).all()))
        loss.sum().backward()
        self.assertIsNotNone(model.first_conv.weight.grad)


if __name__ == "__main__":
    unittest.main()
