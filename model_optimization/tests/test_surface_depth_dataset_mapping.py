from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_graynav_surface_depth_dataset import (  # noqa: E402
    BLOCKED,
    GROUND,
    IGNORE,
    STEP,
    paired_stair_files,
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


if __name__ == "__main__":
    unittest.main()
