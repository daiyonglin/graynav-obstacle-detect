from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_voc2007_indoor8 import read_voc_split  # noqa: E402


class PrepareVoc2007Indoor8Test(unittest.TestCase):
    def test_mapping_and_difficult_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ImageSets" / "Main").mkdir(parents=True)
            (root / "Annotations").mkdir()
            (root / "JPEGImages").mkdir()
            (root / "ImageSets" / "Main" / "train.txt").write_text(
                "000001\n", encoding="ascii"
            )
            cv2.imwrite(
                str(root / "JPEGImages" / "000001.jpg"),
                np.zeros((100, 200, 3), dtype=np.uint8),
            )
            (root / "Annotations" / "000001.xml").write_text(
                """<annotation>
                <object><name>person</name><difficult>0</difficult><bndbox>
                <xmin>11</xmin><ymin>21</ymin><xmax>111</xmax><ymax>81</ymax>
                </bndbox></object>
                <object><name>chair</name><difficult>1</difficult><bndbox>
                <xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>20</ymax>
                </bndbox></object>
                <object><name>dog</name><difficult>0</difficult><bndbox>
                <xmin>1</xmin><ymin>1</ymin><xmax>20</xmax><ymax>20</ymax>
                </bndbox></object>
                </annotation>""",
                encoding="utf-8",
            )
            records, counts = read_voc_split(root, "train", 0)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["boxes_xywh"], [[10.0, 20.0, 101.0, 61.0, 0]])
            self.assertEqual(counts["person"], 1)
            self.assertNotIn("chair", counts)


if __name__ == "__main__":
    unittest.main()
