from __future__ import annotations

import unittest

from model_optimization.scripts.build_a1_unified_datasets import (
    CALIBRATE_COUNTS,
    EVALUATE_COUNTS,
    SOURCES,
    select_records,
)


class A1UnifiedDatasetsTest(unittest.TestCase):
    def test_fixed_contract_counts(self) -> None:
        self.assertEqual(sum(CALIBRATE_COUNTS.values()), 160)
        self.assertEqual(sum(EVALUATE_COUNTS.values()), 40)
        self.assertEqual(
            CALIBRATE_COUNTS,
            {
                "voc2007": 52,
                "coco128_smoke_only": 12,
                "ade20k": 32,
                "stairnetv3": 40,
                "nyuv2": 24,
            },
        )

    def test_selection_is_deterministic_balanced_and_disjoint(self) -> None:
        records = [
            {"source": source, "source_id": f"{source}:{index:04d}"}
            for source in SOURCES
            for index in range(100)
        ]
        first = select_records(records, 42)
        second = select_records(list(reversed(records)), 42)
        self.assertEqual(first, second)
        calibrate, evaluate = first
        self.assertEqual(len(calibrate), 160)
        self.assertEqual(len(evaluate), 40)
        self.assertFalse(
            {(row["source"], row["source_id"]) for row in calibrate}
            & {(row["source"], row["source_id"]) for row in evaluate}
        )


if __name__ == "__main__":
    unittest.main()
