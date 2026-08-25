from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_surface_depth import (  # noqa: E402
    CLASS_WEIGHTS,
    SurfaceDepthDataset,
    experiment_gates,
    false_whole_frame_step_prediction,
    multitask_loss,
    tensorboard_scalar_metrics,
)


class SurfaceDepthExperimentTest(unittest.TestCase):
    def test_tensorboard_safety_logging_separates_text(self) -> None:
        values = {
            "count": 5,
            "rate": 0.25,
            "definition": "diagnostic text",
        }
        self.assertEqual(
            tensorboard_scalar_metrics(values),
            {"count": 5.0, "rate": 0.25},
        )

    def test_ade_positive_crop_contains_the_selected_step(self) -> None:
        seg = torch.full((512, 512), 3, dtype=torch.uint8).numpy()
        seg[256, 256] = 2
        random.seed(42)
        x, y = SurfaceDepthDataset.crop_origin(
            512, 512, seg, "ade20k", ade_step_center_prob=1.0
        )
        self.assertLessEqual(x, 256)
        self.assertLessEqual(y, 256)
        self.assertGreater(x + 256, 256)
        self.assertGreater(y + 256, 256)

    def test_stair_negative_crop_prefers_unknown_context(self) -> None:
        seg = torch.full((512, 512), 3, dtype=torch.uint8).numpy()
        seg[:300, :300] = 2
        random.seed(7)
        x, y = SurfaceDepthDataset.crop_origin(
            512,
            512,
            seg,
            "stairnetv3",
            stair_step_center_prob=0.0,
            stair_negative_crop_prob=1.0,
            stair_negative_crop_attempts=32,
        )
        selected = seg[y : y + 256, x : x + 256]
        self.assertLess(float((selected == 2).mean()), 0.25)

    def test_e2_loss_is_finite_and_backpropagates(self) -> None:
        seg_logits = torch.randn(2, 4, 64, 64, requires_grad=True)
        depth_logits = torch.randn(2, 16, 64, 64, requires_grad=True)
        seg = torch.randint(0, 4, (2, 256, 256))
        depth = torch.rand(2, 256, 256) * 7.7 + 0.3
        images = torch.rand(2, 1, 256, 256)
        loss, parts = multitask_loss(
            seg_logits,
            depth_logits,
            seg,
            depth,
            CLASS_WEIGHTS,
            images=images,
            experiment="e2",
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(parts["depth_grouped"], 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(seg_logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(depth_logits.grad).all()))

    def test_final_gate_requires_e0_gradient_improvement(self) -> None:
        task = {
            "ground_candidate": 0.70,
            "blocked_surface": 0.75,
            "step_or_drop": 0.80,
            "unknown_other": 0.50,
        }
        metrics = {
            "iou": task,
            "precision": task,
            "recall": task,
            "f1": task,
            "per_source": {
                "stairnetv3": {"precision": task, "recall": task},
            },
            "safety": {
                "stair_false_whole_frame_step_prediction_count": 0,
                "false_whole_frame_step_prediction_count": 0,
                "ade_no_step_bottom_false_image_rate": 0.02,
                "hazard_to_ground_rate": 0.04,
            },
            "depth": {
                "absrel": 0.20,
                "delta1": 0.70,
                "near_far_order_accuracy": 0.85,
                "gradient_mae": 0.84,
            },
        }
        self.assertTrue(experiment_gates(metrics, "e2", e0_gradient_mae=1.0)["passed"])
        self.assertFalse(experiment_gates(metrics, "e2", e0_gradient_mae=None)["passed"])

    def test_whole_frame_step_gate_distinguishes_truth_from_overfill(self) -> None:
        truth = torch.full((64, 64), 2, dtype=torch.long)
        truth[:3] = 3
        all_step = torch.full_like(truth, 2)
        self.assertFalse(false_whole_frame_step_prediction(all_step, truth))

        sparse_truth = torch.full((64, 64), 3, dtype=torch.long)
        sparse_truth[40:] = 2
        self.assertTrue(false_whole_frame_step_prediction(all_step, sparse_truth))

        ignored = torch.full((64, 64), 255, dtype=torch.long)
        self.assertFalse(false_whole_frame_step_prediction(all_step, ignored))


if __name__ == "__main__":
    unittest.main()
