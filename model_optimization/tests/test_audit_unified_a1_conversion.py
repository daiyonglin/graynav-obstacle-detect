from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from model_optimization.scripts.audit_unified_a1_conversion import (
    EXPECTED_OUTPUTS,
    audit,
)


def npy(value: np.ndarray) -> bytes:
    output = BytesIO()
    np.save(output, value)
    return output.getvalue()


class UnifiedA1ConversionAuditTest(unittest.TestCase):
    def package(self, similarity: float = 0.99) -> dict[str, bytes]:
        details = {"0000": {name: similarity for name in EXPECTED_OUTPUTS}}
        report = {
            "method": "cosine_similarity",
            "num": 1,
            "similarity": [
                {"output_name": name, "similarity": similarity}
                for name in EXPECTED_OUTPUTS
            ],
            "detail": details,
        }
        files = {
            "model/test.m1model": b"m1",
            "model/test_InputOrderScale.txt": (
                b"InputTensor name: images Scale: 0.0039215689 "
                b"OrderIn M1MODEL: 0\n"
            ),
            "model/test_OutputOrderScale.txt": "".join(
                f"OutputTensor name: {name} Scale: 0.1 "
                f"OrderIn M1MODEL: {index}\n"
                for index, name in enumerate(EXPECTED_OUTPUTS)
            ).encode(),
            "model/evaluate/test_evaluate_report.json": json.dumps(report).encode(),
        }
        original = np.asarray([1.0, 0.0], dtype=np.float32)
        # Construct an exact requested cosine in two dimensions.
        simulated = np.asarray(
            [similarity, np.sqrt(max(0.0, 1.0 - similarity**2))],
            dtype=np.float32,
        )
        for name in EXPECTED_OUTPUTS:
            files[f"model/evaluate/0000.d/{name}.ori.npy"] = npy(original)
            files[f"model/evaluate/0000.d/{name}.sim.npy"] = npy(simulated)
        return files

    def test_complete_seven_output_package_passes(self) -> None:
        files = self.package()
        result = audit(list(files), files.__getitem__, 0.90)
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(len(result["output_order_scale"]), 7)

    def test_missing_output_fails_closed(self) -> None:
        files = self.package()
        report_name = "model/evaluate/test_evaluate_report.json"
        report = json.loads(files[report_name])
        report["similarity"] = report["similarity"][:-1]
        files[report_name] = json.dumps(report).encode()
        result = audit(list(files), files.__getitem__, 0.90)
        self.assertFalse(result["passed"])

    def test_low_per_sample_cosine_fails(self) -> None:
        files = self.package(0.80)
        result = audit(list(files), files.__getitem__, 0.90)
        self.assertFalse(result["passed"])
        self.assertTrue(any("below" in item for item in result["errors"]))


if __name__ == "__main__":
    unittest.main()
