from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_graynav_surface_depth_dataset import (  # noqa: E402
    BLOCKED,
    GROUND,
    IGNORE,
    STEP,
    paired_stair_files,
    read_depth,
    remap_ade,
)


class SurfaceDepthDatasetMappingTest(unittest.TestCase):
    def test_ade20k_hazard_mapping(self) -> None:
        # floor, road, wall, vegetation, water, stairs, stairway,
        # escalator, step and person (ignored)
        source = np.array(
            [[4, 7, 1, 18, 22, 54, 60, 97, 122, 13]],
            dtype=np.uint8,
        )
        expected = np.array(
            [[GROUND, GROUND, BLOCKED, BLOCKED, BLOCKED,
              STEP, STEP, STEP, STEP, IGNORE]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(remap_ade(source), expected)

    def test_official_stairnet_depthes_directory_is_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            split = Path(temp)
            for name in ("images", "segmentations", "depthes"):
                (split / name).mkdir()
            for relative in (
                "images/sample.png",
                "segmentations/sample.png",
                "depthes/sample.png",
            ):
                (split / relative).touch()
            pairs = paired_stair_files(split)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][2], split / "depthes" / "sample.png")

    def test_stairnet_uint8_depth_visualization_is_loss_masked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "depth.png"
            self.assertTrue(cv2.imwrite(str(path), np.full((8, 8), 127, np.uint8)))
            self.assertIsNone(read_depth(path))

    def test_uint16_depth_is_converted_from_mm_to_m(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "depth.png"
            source = np.full((8, 8), 2500, np.uint16)
            self.assertTrue(cv2.imwrite(str(path), source))
            converted = read_depth(path)
            self.assertIsNotNone(converted)
            np.testing.assert_allclose(converted, 2.5, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
