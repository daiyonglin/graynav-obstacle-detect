from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_graynav_unified_indoor import interleaved_source_indices  # noqa: E402


class UnifiedValidationSamplingTest(unittest.TestCase):
    def test_first_round_contains_all_sources_without_overlap(self) -> None:
        records = (
            [{"source": "ade20k"}] * 5
            + [{"source": "nyuv2"}] * 3
            + [{"source": "stairnetv3"}] * 4
        )
        indices = interleaved_source_indices(records)
        self.assertEqual(len(indices), len(records))
        self.assertEqual(len(set(indices)), len(records))
        self.assertEqual(
            [records[index]["source"] for index in indices[:3]],
            ["ade20k", "nyuv2", "stairnetv3"],
        )


if __name__ == "__main__":
    unittest.main()
