#!/usr/bin/env python3
"""Build the exact A1 datasets.zip NCHW float32 calibration contract."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate", type=int, default=160)
    parser.add_argument("--evaluate", type=int, default=40)
    parser.add_argument("--size", type=int, choices=(256, 384), default=256)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in (args.data / "manifest_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    required = args.calibrate + args.evaluate
    if len(records) < required:
        raise RuntimeError(f"need {required} public validation samples, found {len(records)}")
    root = args.output / "datasets"
    if root.exists():
        shutil.rmtree(root)
    for name, selected in (
        ("calibrate_datasets", records[: args.calibrate]),
        ("evaluate_datasets", records[args.calibrate : required]),
    ):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for index, record in enumerate(selected):
            gray = cv2.imread(str(args.data / str(record["image"])), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"cannot read {record['image']}")
            gray = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            tensor = gray.astype(np.float32)[None, None] / 255.0
            np.save(folder / f"{index:04d}.npy", tensor)
    archive = shutil.make_archive(str(args.output / "datasets"), "zip", args.output, "datasets")
    contract = {
        "archive": archive,
        "input_name": "images",
        "shape": [1, 1, args.size, args.size],
        "dtype": "float32",
        "range": [0.0, 1.0],
        "calibrate": args.calibrate,
        "evaluate": args.evaluate,
        "writer": "numpy.save",
    }
    (args.output / "datasets_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
