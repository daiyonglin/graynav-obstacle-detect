from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.graynav_surface_depth import (  # noqa: E402
    GrayNavSurfaceDepth,
    depth_bin_centers,
    fold_rgb_first_conv_to_gray,
)


class GrayNavSurfaceDepthTest(unittest.TestCase):
    def test_static_contract(self) -> None:
        model = GrayNavSurfaceDepth().eval()
        with torch.no_grad():
            seg, depth = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(seg.shape), (1, 4, 64, 64))
        self.assertEqual(tuple(depth.shape), (1, 16, 64, 64))
        self.assertEqual(model.first_conv.weight.shape[1], 1)

    def test_detail64_contract(self) -> None:
        model = GrayNavSurfaceDepth(detail64=True).eval()
        with torch.no_grad():
            seg, depth = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(seg.shape), (1, 4, 64, 64))
        self.assertEqual(tuple(depth.shape), (1, 16, 64, 64))
        self.assertTrue(model.detail64)

    def test_fold_is_replicate_exact(self) -> None:
        model = GrayNavSurfaceDepth()
        shape = model.first_conv.weight.shape
        rgb = torch.arange(shape[0] * 3 * shape[2] * shape[3], dtype=torch.float32)
        rgb = rgb.reshape(shape[0], 3, shape[2], shape[3])
        fold_rgb_first_conv_to_gray(model, rgb)
        self.assertTrue(torch.equal(model.first_conv.weight, rgb.sum(dim=1, keepdim=True)))

    def test_depth_centers_are_ordered(self) -> None:
        centers = depth_bin_centers()
        self.assertEqual(centers.numel(), 16)
        self.assertTrue(bool(torch.all(centers[1:] > centers[:-1])))
        self.assertGreater(float(centers[0]), 0.3)
        self.assertLess(float(centers[-1]), 8.0)


if __name__ == "__main__":
    unittest.main()
