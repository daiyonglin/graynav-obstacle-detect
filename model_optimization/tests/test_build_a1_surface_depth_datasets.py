from __future__ import annotations

import unittest

from model_optimization.scripts.build_a1_surface_depth_datasets import (
    allocate_counts,
    select_records,
)


class A1SurfaceDepthDatasetsTest(unittest.TestCase):
    def test_fixed_counts_match_training_source_mix(self) -> None:
        weights = {"ade20k": 0.40, "nyuv2": 0.35, "stairnetv3": 0.25}
        self.assertEqual(
            allocate_counts(160, weights),
            {"ade20k": 64, "nyuv2": 56, "stairnetv3": 40},
        )
        self.assertEqual(
            allocate_counts(40, weights),
            {"ade20k": 16, "nyuv2": 14, "stairnetv3": 10},
        )

    def test_selection_is_deterministic_balanced_and_disjoint(self) -> None:
        records = [
            {"source": source, "source_id": f"{source}:{index:04d}"}
            for source in ("ade20k", "nyuv2", "stairnetv3")
            for index in range(100)
        ]
        calibrate_counts = {"ade20k": 8, "nyuv2": 7, "stairnetv3": 5}
        evaluate_counts = {"ade20k": 4, "nyuv2": 3, "stairnetv3": 3}
        first = select_records(records, calibrate_counts, evaluate_counts, seed=42)
        second = select_records(list(reversed(records)), calibrate_counts, evaluate_counts, seed=42)
        self.assertEqual(first, second)
        calibrated, evaluated = first
        self.assertEqual(len(calibrated), 20)
        self.assertEqual(len(evaluated), 10)
        self.assertFalse(
            {row["source_id"] for row in calibrated}
            & {row["source_id"] for row in evaluated}
        )


if __name__ == "__main__":
    unittest.main()
