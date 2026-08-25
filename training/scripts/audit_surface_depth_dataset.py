#!/usr/bin/env python3
"""Audit prepared-v2 labels and emit deterministic, separate label views."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


IGNORE = 255
CLASS_NAMES = ("ground_candidate", "blocked_surface", "step_or_drop", "unknown_other")
COLORS = np.asarray(((70, 190, 70), (55, 55, 225), (0, 190, 255), (180, 90, 30)), np.uint8)
FIXED_SUFFIXES = {
    "ade20k": ("000285", "000501"),
    "stairnetv3": ("000006", "000014", "000027", "000068", "000299", "000382", "000414", "000485"),
    "nyuv2": ("000118", "000649", "000663", "001088"),
}


def read_manifest(root: Path, split: str) -> list[dict[str, object]]:
    path = root / f"manifest_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_view(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    colour = np.zeros_like(base)
    valid = mask != IGNORE
    for index, value in enumerate(COLORS):
        colour[mask == index] = value
    result = base.copy()
    result[valid] = cv2.addWeighted(base, 0.45, colour, 0.55, 0)[valid]
    result[~valid] = (35, 35, 35)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pixel_counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(5, np.int64))
    step_ratios: dict[str, list[float]] = defaultdict(list)
    whole_frame_labels: list[str] = []
    all_rows: dict[str, list[dict[str, object]]] = {}
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "val"):
        rows = read_manifest(args.data, split)
        all_rows[split] = rows
        split_ids[split] = {str(row["source_id"]) for row in rows}
        for row in rows:
            if not row.get("seg_mask"):
                continue
            mask = cv2.imread(str(args.data / str(row["seg_mask"])), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"cannot read {row['seg_mask']}")
            source = str(row["source"])
            for index in range(4):
                pixel_counts[source][index] += int((mask == index).sum())
            pixel_counts[source][4] += int((mask == IGNORE).sum())
            ratio = float((mask == 2).mean())
            step_ratios[source].append(ratio)
            if ratio > 0.95:
                whole_frame_labels.append(str(row["source_id"]))
    overlap = sorted(split_ids["train"] & split_ids["val"])
    sources: dict[str, object] = {}
    for source, counts in sorted(pixel_counts.items()):
        total = max(1, int(counts.sum()))
        ratios = step_ratios[source]
        sources[source] = {
            "pixel_counts": {
                **{name: int(counts[index]) for index, name in enumerate(CLASS_NAMES)},
                "ignore": int(counts[4]),
            },
            "pixel_ratios": {
                **{name: float(counts[index] / total) for index, name in enumerate(CLASS_NAMES)},
                "ignore": float(counts[4] / total),
            },
            "step_area_distribution": {
                "samples": len(ratios),
                "positive_samples": sum(value > 0 for value in ratios),
                "p50": float(np.quantile(ratios, 0.50)) if ratios else 0.0,
                "p90": float(np.quantile(ratios, 0.90)) if ratios else 0.0,
                "p99": float(np.quantile(ratios, 0.99)) if ratios else 0.0,
                "max": max(ratios, default=0.0),
            },
        }
    visual_root = args.output / "fixed_labels"
    visual_count = 0
    val_rows = all_rows["val"]
    for source, suffixes in FIXED_SUFFIXES.items():
        candidates = [row for row in val_rows if row["source"] == source]
        for suffix in suffixes:
            matches = [row for row in candidates if str(row["source_id"]).endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(f"fixed audit id {source}:{suffix} matched {len(matches)} rows")
            row = matches[0]
            scene = visual_root / source / str(row["source_id"]).replace(":", "_")
            scene.mkdir(parents=True, exist_ok=True)
            gray = cv2.imread(str(args.data / str(row["image"])), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"cannot read {row['image']}")
            cv2.imwrite(str(scene / "01_mono.png"), gray)
            if row.get("seg_mask"):
                mask = cv2.imread(str(args.data / str(row["seg_mask"])), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(scene / "02_label.png"), label_view(gray, mask))
            visual_count += 1
    report = {
        "contract": {
            "classes": list(CLASS_NAMES),
            "input_channels": 1,
            "rgb_input_used": False,
            "stair_whole_frame_definition": "step pixel ratio > 0.95",
        },
        "sources": sources,
        "whole_frame_step_label_count": len(whole_frame_labels),
        "whole_frame_step_label_ids": whole_frame_labels,
        "train_val_source_id_overlap": len(overlap),
        "fixed_visualization_scenes": visual_count,
        "manifest_sha256": {
            split: file_sha256(args.data / f"manifest_{split}.jsonl") for split in ("train", "val")
        },
        "passed": not overlap and not whole_frame_labels,
    }
    destination = args.output / "prepared_v2_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)
    print("GRAYNAV_PREPARED_V2_AUDIT_OK")


if __name__ == "__main__":
    main()
