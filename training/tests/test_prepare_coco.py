from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_coco_indoor8 import build_records  # noqa: E402


class PrepareCocoIndoor8Test(unittest.TestCase):
    def test_mapping_and_deterministic_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (root / name).write_bytes(b"x")
            payload = {
                "categories": [
                    {"id": 1, "name": "person"},
                    {"id": 62, "name": "chair"},
                    {"id": 3, "name": "car"},
                ],
                "images": [
                    {"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
                    {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80},
                    {"id": 3, "file_name": "c.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"image_id": 1, "category_id": 1, "bbox": [1, 2, 30, 40], "iscrowd": 0},
                    {"image_id": 1, "category_id": 62, "bbox": [5, 6, 10, 12], "iscrowd": 0},
                    {"image_id": 3, "category_id": 3, "bbox": [1, 2, 30, 40], "iscrowd": 0},
                ],
            }
            records, counts = build_records(payload, root, negative_modulus=2)
            self.assertEqual([row["source_id"] for row in records], ["coco:1", "coco:2"])
            self.assertEqual(records[0]["boxes_xywh"], [[1.0, 2.0, 30.0, 40.0, 0], [5.0, 6.0, 10.0, 12.0, 1]])
            self.assertEqual(counts["person"], 1)
            self.assertEqual(counts["chair"], 1)
            self.assertEqual(counts["negative_images"], 1)


if __name__ == "__main__":
    unittest.main()
