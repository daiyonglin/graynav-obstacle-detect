from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from segmentation.graynav_fast_scnn import GrayNavFastSCNN, fold_rgb_first_conv_to_gray  # noqa: E402


class GrayNavFastSCNNTest(unittest.TestCase):
    def test_contract(self) -> None:
        model = GrayNavFastSCNN().eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(model.first_conv.weight.shape)[1], 1)
        self.assertEqual(tuple(output.shape), (1, 4, 32, 32))

    def test_rgb_fold_is_sum(self) -> None:
        model = GrayNavFastSCNN()
        rgb = torch.arange(model.first_conv.weight.shape[0] * 3 * 3 * 3, dtype=torch.float32)
        rgb = rgb.reshape(model.first_conv.weight.shape[0], 3, 3, 3)
        fold_rgb_first_conv_to_gray(model, rgb)
        self.assertTrue(torch.equal(model.first_conv.weight, rgb.sum(dim=1, keepdim=True)))

    def test_width_075_contract(self) -> None:
        model = GrayNavFastSCNN(width_mult=0.75).eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(output.shape), (1, 4, 32, 32))
        self.assertEqual(model.first_conv.weight.shape[0], 24)


if __name__ == "__main__":
    unittest.main()
