#!/usr/bin/env python3
"""Convert public Mapillary/StairNet labels into GrayNav grayscale masks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


GROUND = 0
BLOCKED = 1
STEP = 2
POTHOLE = 3
IGNORE = 255

# Mapillary Vistas v1.2 semantic ids from config_v1.2.json.  Do not reuse
# v2.0 ids: its 124-class taxonomy has different numeric assignments.
MAPILLARY_GROUND = {8, 9, 11, 13, 14, 15}
MAPILLARY_BLOCKED = {3, 4, 5, 6, 17, 29, 30, 31}
MAPILLARY_STEP = {2}
MAPILLARY_POTHOLE = {43}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapillary-root", type=Path)
    parser.add_argument("--stair-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    return None


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def remap_mapillary(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    for value in MAPILLARY_GROUND:
        out[mask == value] = GROUND
    for value in MAPILLARY_BLOCKED:
        out[mask == value] = BLOCKED
    for value in MAPILLARY_STEP:
        out[mask == value] = STEP
    for value in MAPILLARY_POTHOLE:
        out[mask == value] = POTHOLE
    return out


def validate_mapillary_config(root: Path) -> None:
    config_path = root / "config_v1.2.json"
    if not config_path.is_file():
        raise RuntimeError(f"official Mapillary config_v1.2.json is required: {config_path}")
    labels = json.loads(config_path.read_text(encoding="utf-8")).get("labels", [])
    expected = {
        2: "curb", 3: "fence", 4: "guard rail", 5: "barrier", 6: "wall",
        8: "crosswalk", 9: "curb cut", 11: "pedestrian area", 13: "road",
        14: "service lane", 15: "sidewalk", 17: "building", 29: "terrain",
        30: "vegetation", 31: "water", 43: "pothole",
    }
    for class_id, token in expected.items():
        if class_id >= len(labels):
            raise RuntimeError(f"Mapillary v1.2 class id missing: {class_id}")
        entry = labels[class_id]
        name = str(entry.get("readable", entry.get("name", ""))).lower().replace("-", " ")
        if token not in name:
            raise RuntimeError(
                f"Mapillary taxonomy mismatch at id {class_id}: expected {token!r}, got {name!r}"
            )


def write_pair(
    image_path: Path,
    raw_mask: np.ndarray,
    output: Path,
    split: str,
    stem: str,
    source: str,
) -> dict[str, object]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cannot read image: {image_path}")
    # OpenCV COLOR_BGR2GRAY uses its fixed BT.601-compatible coefficients.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if raw_mask.shape[:2] != gray.shape[:2]:
        raw_mask = cv2.resize(raw_mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    image_out = output / "images" / split / f"{stem}.png"
    mask_out = output / "masks" / split / f"{stem}.png"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_out), gray):
        raise RuntimeError(f"cannot write image: {image_out}")
    if not cv2.imwrite(str(mask_out), raw_mask):
        raise RuntimeError(f"cannot write mask: {mask_out}")
    values, counts = np.unique(raw_mask, return_counts=True)
    hist = {str(int(v)): int(c) for v, c in zip(values, counts)}
    rare_y, rare_x = np.where((raw_mask == STEP) | (raw_mask == POTHOLE))
    rare_center = None
    if rare_x.size:
        rare_center = [float(rare_x.mean() / raw_mask.shape[1]), float(rare_y.mean() / raw_mask.shape[0])]
    return {
        "image": image_out.relative_to(output).as_posix(),
        "mask": mask_out.relative_to(output).as_posix(),
        "source": source,
        "source_id": f"{source}:{image_path.stem}",
        "rare": bool(np.any(raw_mask == STEP) or np.any(raw_mask == POTHOLE)),
        "rare_center": rare_center,
        "gray_rule": "OpenCV COLOR_BGR2GRAY (BT.601-compatible)",
        "class_histogram": hist,
    }


def convert_mapillary(root: Path, output: Path, split: str) -> list[dict[str, object]]:
    validate_mapillary_config(root)
    aliases = (split, "training" if split == "train" else "validation")
    split_root = find_dir(root, aliases)
    if split_root is None:
        return []
    images_dir = find_dir(split_root, ("images",))
    labels_dir = find_dir(split_root, ("v1.2/labels", "labels"))
    if images_dir is None or labels_dir is None:
        raise RuntimeError(f"Mapillary {split} must contain images and v1.2/labels")
    records: list[dict[str, object]] = []
    labels_by_stem = {p.stem: p for p in image_files(labels_dir)}
    for image_path in image_files(images_dir):
        label_path = labels_by_stem.get(image_path.stem)
        if label_path is None:
            continue
        mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"cannot read label: {label_path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mapped = remap_mapillary(mask)
        records.append(
            write_pair(image_path, mapped, output, split, f"mapillary_{image_path.stem}", "mapillary")
        )
    return records


def pair_stair_files(split_root: Path) -> list[tuple[Path, Path]]:
    images_dir = find_dir(split_root, ("images", "image"))
    masks_dir = find_dir(split_root, ("segmentations", "segmentation", "labels", "masks"))
    if images_dir is None or masks_dir is None:
        return []
    masks = {path.stem: path for path in image_files(masks_dir)}
    return [(path, masks[path.stem]) for path in image_files(images_dir) if path.stem in masks]


def convert_stairs(root: Path, output: Path, split: str) -> list[dict[str, object]]:
    split_root = find_dir(root, (split, "training" if split == "train" else "validation"))
    if split_root is None:
        return []
    records: list[dict[str, object]] = []
    for image_path, mask_path in pair_stair_files(split_root):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"cannot read stair label: {mask_path}")
        if mask.ndim == 3:
            positive = np.any(mask != 0, axis=2)
        else:
            positive = mask != 0
        mapped = np.full(positive.shape, IGNORE, dtype=np.uint8)
        mapped[positive] = STEP
        records.append(
            write_pair(image_path, mapped, output, split, f"stair_{image_path.stem}", "stair")
        )
    return records


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "classes": ["ground_candidate", "blocked_surface", "step_or_drop", "pothole"],
        "gray_rule": "OpenCV COLOR_BGR2GRAY (BT.601-compatible)",
        "input_channels": 1,
        "rgb_input_used": False,
    }
    split_records: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "val"):
        records: list[dict[str, object]] = []
        if args.mapillary_root:
            records.extend(convert_mapillary(args.mapillary_root, args.output, split))
        if args.stair_root:
            records.extend(convert_stairs(args.stair_root, args.output, split))
        manifest = args.output / f"manifest_{split}.jsonl"
        manifest.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        summary[split] = {"samples": len(records), "manifest": manifest.name}
        split_records[split] = records
    train_ids = {str(record["source_id"]) for record in split_records["train"]}
    val_ids = {str(record["source_id"]) for record in split_records["val"]}
    overlap = sorted(train_ids & val_ids)
    if overlap:
        raise RuntimeError(f"train/validation source overlap detected: {overlap[:20]}")
    summary["split_audit"] = {"source_id_overlap": 0, "passed": True}
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
