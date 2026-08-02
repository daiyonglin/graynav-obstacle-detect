from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_graynav_surface_dataset import (  # noqa: E402
    BLOCKED,
    GROUND,
    IGNORE,
    POTHOLE,
    STEP,
    remap_mapillary,
)


class SurfaceDatasetMappingTest(unittest.TestCase):
    def test_mapillary_v12_critical_ids(self) -> None:
        # road, sidewalk, wall, vegetation, curb, pothole, person
        source = np.array([[13, 15, 6, 30, 2, 43, 19]], dtype=np.uint8)
        mapped = remap_mapillary(source)
        expected = np.array(
            [[GROUND, GROUND, BLOCKED, BLOCKED, STEP, POTHOLE, IGNORE]],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(mapped, expected)


if __name__ == "__main__":
    unittest.main()
