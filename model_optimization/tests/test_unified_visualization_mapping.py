from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from visualize_graynav_unified import unletterbox_grid  # noqa: E402


class UnifiedVisualizationMappingTest(unittest.TestCase):
    def test_vertical_letterbox_padding_is_removed(self) -> None:
        grid = np.arange(48 * 48, dtype=np.int32).reshape(48, 48)
        # A 640x320 image becomes 384x192 with 96px top/bottom padding.
        cropped = unletterbox_grid(grid, (320, 640), 0.6, 0, 96)
        self.assertEqual(cropped.shape, (24, 48))
        self.assertTrue(np.array_equal(cropped, grid[12:36]))

    def test_horizontal_letterbox_padding_is_removed(self) -> None:
        grid = np.arange(48 * 48, dtype=np.int32).reshape(48, 48)
        # A 320x640 image becomes 192x384 with 96px side padding.
        cropped = unletterbox_grid(grid, (640, 320), 0.6, 96, 0)
        self.assertEqual(cropped.shape, (48, 24))
        self.assertTrue(np.array_equal(cropped, grid[:, 12:36]))


if __name__ == "__main__":
    unittest.main()
