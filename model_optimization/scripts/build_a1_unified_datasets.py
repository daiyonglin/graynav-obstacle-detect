#!/usr/bin/env python3
"""Build deterministic, disjoint A1 calibration data for the unified model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SOURCES = ("coco2017", "ade20k", "stairnetv3", "nyuv2")
CALIBRATE_COUNTS = {
    "coco2017": 64, "ade20k": 32, "stairnetv3": 40, "nyuv2": 24,
}
EVALUATE_COUNTS = {
    "coco2017": 16, "ade20k": 8, "stairnetv3": 10, "nyuv2": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def deterministic_order(
    records: list[dict[str, object]], seed: int
) -> list[dict[str, object]]:
    return sorted(
        records,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}:{row['source']}:{row['source_id']}".encode("utf-8")
            ).hexdigest(),
            str(row["source_id"]),
        ),
    )


def select_records(
    records: list[dict[str, object]], seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        source = str(row["source"])
        if source in SOURCES:
            grouped[source].append(row)
    calibrate: list[dict[str, object]] = []
    evaluate: list[dict[str, object]] = []
    for source in SOURCES:
        ordered = deterministic_order(grouped[source], seed)
        needed = CALIBRATE_COUNTS[source] + EVALUATE_COUNTS[source]
        if len(ordered) < needed:
            raise RuntimeError(
                f"source {source} needs {needed} samples, found {len(ordered)}"
            )
        boundary = CALIBRATE_COUNTS[source]
        calibrate.extend(ordered[:boundary])
        evaluate.extend(ordered[boundary:needed])
    return calibrate, evaluate


def letterbox_gray(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read {path}")
    scale = min(384.0 / gray.shape[1], 384.0 / gray.shape[0])
    width = int(round(gray.shape[1] * scale))
    height = int(round(gray.shape[0] * scale))
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((384, 384), 114, dtype=np.uint8)
    x = (384 - width) // 2
    y = (384 - height) // 2
    canvas[y : y + height, x : x + width] = resized
    return canvas.astype(np.float32)[None, None] / 255.0


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    coco = load_jsonl(args.coco / "manifest_val.jsonl")
    scene = load_jsonl(args.scene / "manifest_val.jsonl")
    records = coco + scene
    calibrated, evaluated = select_records(records, args.seed)
    calibrate_ids = {(str(row["source"]), str(row["source_id"])) for row in calibrated}
    evaluate_ids = {(str(row["source"]), str(row["source_id"])) for row in evaluated}
    if calibrate_ids & evaluate_ids:
        raise RuntimeError("calibration/evaluation selections overlap")

    dataset_root = args.output / "datasets"
    rows: list[dict[str, object]] = []
    observed_min, observed_max = 1.0, 0.0
    for split, selected in (
        ("calibrate_datasets", calibrated),
        ("evaluate_datasets", evaluated),
    ):
        folder = dataset_root / split
        folder.mkdir(parents=True, exist_ok=False)
        for index, row in enumerate(selected):
            image = Path(str(row["image"]))
            if not image.is_absolute():
                image = args.scene / image
            tensor = letterbox_gray(image)
            if tensor.shape != (1, 1, 384, 384) or tensor.dtype != np.float32:
                raise RuntimeError(f"invalid tensor contract: {image} {tensor.shape}")
            if not np.isfinite(tensor).all():
                raise RuntimeError(f"non-finite tensor: {image}")
            observed_min = min(observed_min, float(tensor.min()))
            observed_max = max(observed_max, float(tensor.max()))
            filename = f"{index:04d}.npy"
            np.save(folder / filename, tensor)
            rows.append({
                "split": split,
                "file": f"datasets/{split}/{filename}",
                "source": row["source"],
                "source_id": row["source_id"],
                "image": str(image),
            })

    archive = shutil.make_archive(
        str(args.output / "datasets"), "zip", args.output, "datasets"
    )
    source_counts = {
        split: dict(sorted(Counter(
            str(row["source"]) for row in rows if row["split"] == split
        ).items()))
        for split in ("calibrate_datasets", "evaluate_datasets")
    }
    contract = {
        "archive": archive,
        "input_name": "images",
        "shape": [1, 1, 384, 384],
        "dtype": "float32",
        "required_range": [0.0, 1.0],
        "observed_range": [observed_min, observed_max],
        "calibrate": sum(CALIBRATE_COUNTS.values()),
        "evaluate": sum(EVALUATE_COUNTS.values()),
        "source_counts": source_counts,
        "calibration_evaluation_overlap": 0,
        "sampling": f"sha256(seed={args.seed}, source, source_id)",
        "rgb_input_used": False,
    }
    (args.output / "datasets_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "datasets_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print("GRAYNAV_A1_UNIFIED_DATASETS_OK")


if __name__ == "__main__":
    main()
