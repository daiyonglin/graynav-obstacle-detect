#!/usr/bin/env python3
"""Build a deterministic, source-balanced A1 SurfaceDepth calibration archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SOURCES = ("ade20k", "nyuv2", "stairnetv3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate", type=int, default=160)
    parser.add_argument("--evaluate", type=int, default=40)
    parser.add_argument("--size", type=int, choices=(256, 384), default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-ade20k", type=float, default=0.40)
    parser.add_argument("--sampling-nyuv2", type=float, default=0.35)
    parser.add_argument("--sampling-stairnetv3", type=float, default=0.25)
    return parser.parse_args()


def allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate an exact total with deterministic largest-remainder rounding."""

    if total < 0:
        raise RuntimeError(f"sample count must be non-negative: {total}")
    if set(weights) != set(SOURCES):
        raise RuntimeError(f"source weights must cover {SOURCES}: {weights}")
    if any(value < 0.0 for value in weights.values()):
        raise RuntimeError(f"source weights must be non-negative: {weights}")
    weight_sum = sum(weights.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"source weights must sum to 1.0: {weights}")
    raw = {source: total * weights[source] for source in SOURCES}
    counts = {source: math.floor(raw[source]) for source in SOURCES}
    remaining = total - sum(counts.values())
    order = sorted(
        SOURCES,
        key=lambda source: (-(raw[source] - counts[source]), SOURCES.index(source)),
    )
    for source in order[:remaining]:
        counts[source] += 1
    if sum(counts.values()) != total:
        raise RuntimeError(f"allocation failed: total={total}, counts={counts}")
    return counts


def deterministic_order(records: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    """Order records by a stable hash without depending on manifest ordering."""

    def key(record: dict[str, object]) -> tuple[str, str]:
        source_id = str(record["source_id"])
        digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()
        return digest, source_id

    return sorted(records, key=key)


def select_records(
    records: list[dict[str, object]],
    calibrate_counts: dict[str, int],
    evaluate_counts: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        source = str(record.get("source"))
        if source in SOURCES:
            grouped[source].append(record)
    calibrated: list[dict[str, object]] = []
    evaluated: list[dict[str, object]] = []
    for source in SOURCES:
        ordered = deterministic_order(grouped[source], seed)
        needed = calibrate_counts[source] + evaluate_counts[source]
        if len(ordered) < needed:
            raise RuntimeError(
                f"source {source} needs {needed} validation records, found {len(ordered)}"
            )
        split = calibrate_counts[source]
        calibrated.extend(ordered[:split])
        evaluated.extend(ordered[split:needed])
    return calibrated, evaluated


def main() -> None:
    args = parse_args()
    weights = {
        "ade20k": args.sampling_ade20k,
        "nyuv2": args.sampling_nyuv2,
        "stairnetv3": args.sampling_stairnetv3,
    }
    calibrate_counts = allocate_counts(args.calibrate, weights)
    evaluate_counts = allocate_counts(args.evaluate, weights)
    records = [
        json.loads(line)
        for line in (args.data / "manifest_val.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    calibrated, evaluated = select_records(
        records, calibrate_counts, evaluate_counts, args.seed
    )
    calibrate_ids = {str(record["source_id"]) for record in calibrated}
    evaluate_ids = {str(record["source_id"]) for record in evaluated}
    overlap = calibrate_ids & evaluate_ids
    if overlap:
        raise RuntimeError(f"calibration/evaluation overlap: {sorted(overlap)[:10]}")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite calibration output: {args.output}")
    root = args.output / "datasets"
    manifest_rows: list[dict[str, object]] = []
    observed_min = 1.0
    observed_max = 0.0
    for split_name, selected in (
        ("calibrate_datasets", calibrated),
        ("evaluate_datasets", evaluated),
    ):
        folder = root / split_name
        folder.mkdir(parents=True, exist_ok=False)
        for index, record in enumerate(selected):
            image_path = args.data / str(record["image"])
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"cannot read {record['image']}")
            gray = cv2.resize(gray, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            tensor = gray.astype(np.float32)[None, None] / 255.0
            if tensor.shape != (1, 1, args.size, args.size):
                raise RuntimeError(f"unexpected tensor shape for {image_path}: {tensor.shape}")
            if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
                raise RuntimeError(f"invalid tensor for {image_path}")
            observed_min = min(observed_min, float(tensor.min()))
            observed_max = max(observed_max, float(tensor.max()))
            filename = f"{index:04d}.npy"
            np.save(folder / filename, tensor)
            manifest_rows.append({
                "split": split_name,
                "file": f"datasets/{split_name}/{filename}",
                "source": record["source"],
                "source_id": record["source_id"],
                "prepared_image": record["image"],
            })
    archive = shutil.make_archive(str(args.output / "datasets"), "zip", args.output, "datasets")
    split_sources = {
        split_name: dict(sorted(Counter(
            str(row["source"]) for row in manifest_rows if row["split"] == split_name
        ).items()))
        for split_name in ("calibrate_datasets", "evaluate_datasets")
    }
    contract = {
        "archive": archive,
        "input_name": "images",
        "shape": [1, 1, args.size, args.size],
        "dtype": "float32",
        "required_range": [0.0, 1.0],
        "observed_range": [observed_min, observed_max],
        "calibrate": args.calibrate,
        "evaluate": args.evaluate,
        "source_weights": weights,
        "source_counts": split_sources,
        "sampling": f"sha256(seed={args.seed}, source_id), source-stratified",
        "calibration_evaluation_overlap": len(overlap),
        "rgb_input_used": False,
        "writer": "numpy.save",
    }
    (args.output / "datasets_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "datasets_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print("GRAYNAV_A1_SURFACE_DEPTH_DATASETS_OK")


if __name__ == "__main__":
    main()
