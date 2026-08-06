from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_graynav_surface_depth import (  # noqa: E402
    CLASS_WEIGHTS,
    experiment_gates,
    multitask_loss,
)


class SurfaceDepthExperimentTest(unittest.TestCase):
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
                "stair_whole_frame_step_prediction_count": 0,
                "whole_frame_step_prediction_count": 0,
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


if __name__ == "__main__":
    unittest.main()
