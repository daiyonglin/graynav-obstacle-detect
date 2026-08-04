#!/usr/bin/env python3
"""Prepare ADE20K, NYUv2 and StairNetV3 for mono multi-task training.

The converter never invents labels. A sample may carry segmentation, depth, or
both; the training loader masks whichever task is absent.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.io import loadmat


GROUND = 0
BLOCKED = 1
STEP = 2
IGNORE = 255

# MIT Scene Parsing 150 uses one-based ids and zero for unlabeled pixels.
ADE_GROUND = {4, 7, 12, 14, 29, 30, 47, 53, 55}
ADE_BLOCKED = {1, 2, 15, 33, 39, 43}
ADE_STEP = {54, 60}

# NYUv2 labelled release uses the official NYU40 ids.  Stairs are folded into
# "other structure" by NYU40, so NYU contributes ground/blocked and metric
# depth while StairNetV3 supplies the explicit step/drop supervision.
NYU_GROUND = {2, 20}  # floor, floor mat
NYU_BLOCKED = {
    1, 3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 17, 19, 22, 24, 25,
    29, 30, 32, 33, 34, 36, 38, 39,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ade-root", type=Path)
    parser.add_argument("--nyu-mat", type=Path)
    parser.add_argument("--nyu-splits", type=Path)
    parser.add_argument("--stair-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def files(root: Path, suffixes: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def find_dir(root: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.is_dir():
            return path
    return None


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def write_sample(
    output: Path,
    split: str,
    stem: str,
    source: str,
    gray: np.ndarray,
    seg: np.ndarray | None,
    depth: np.ndarray | None,
) -> dict[str, object]:
    image_path = output / "images" / split / f"{stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), gray):
        raise RuntimeError(f"cannot write {image_path}")
    record: dict[str, object] = {
        "image": image_path.relative_to(output).as_posix(),
        "source": source,
        "source_id": f"{source}:{stem}",
        "seg_mask": None,
        "depth": None,
        "gray_rule": "OpenCV BT.601-compatible grayscale",
    }
    if seg is not None:
        seg_path = output / "segmentation" / split / f"{stem}.png"
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        if seg.shape != gray.shape:
            seg = cv2.resize(seg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(seg_path), seg.astype(np.uint8)):
            raise RuntimeError(f"cannot write {seg_path}")
        record["seg_mask"] = seg_path.relative_to(output).as_posix()
        record["has_step"] = bool(np.any(seg == STEP))
    if depth is not None:
        if depth.shape != gray.shape:
            depth = cv2.resize(depth, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth_path = output / "depth" / split / f"{stem}.npy"
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, depth.astype(np.float32))
        record["depth"] = depth_path.relative_to(output).as_posix()
    return record


def remap_ade(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    for value in ADE_GROUND:
        out[mask == value] = GROUND
    for value in ADE_BLOCKED:
        out[mask == value] = BLOCKED
    for value in ADE_STEP:
        out[mask == value] = STEP
    return out


def remap_nyu(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    for value in NYU_GROUND:
        out[mask == value] = GROUND
    for value in NYU_BLOCKED:
        out[mask == value] = BLOCKED
    return out


def convert_ade(root: Path, output: Path, split: str) -> list[dict[str, object]]:
    official = "training" if split == "train" else "validation"
    image_root = find_dir(root, (f"images/{official}", official))
    mask_root = find_dir(root, (f"annotations/{official}", f"annotations_detectron2/{official}"))
    if image_root is None or mask_root is None:
        raise RuntimeError("ADE20K root must contain images/{training,validation} and annotations")
    mask_by_relative = {
        path.relative_to(mask_root).with_suffix("").as_posix(): path
        for path in files(mask_root, {".png"})
    }
    records: list[dict[str, object]] = []
    for index, image_path in enumerate(files(image_root, {".jpg", ".jpeg", ".png"})):
        key = image_path.relative_to(image_root).with_suffix("").as_posix()
        mask_path = mask_by_relative.get(key)
        if mask_path is None:
            continue
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or raw_mask is None:
            raise RuntimeError(f"cannot read ADE pair {image_path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        records.append(write_sample(
            output, split, f"ade_{split}_{index:06d}", "ade20k", gray,
            remap_ade(raw_mask), None,
        ))
    return records


def split_indices(splits_path: Path | None, count: int) -> tuple[set[int], set[int]]:
    if splits_path is None:
        boundary = int(round(count * 0.80))
        return set(range(boundary)), set(range(boundary, count))
    payload = loadmat(str(splits_path))
    train = {int(value) - 1 for value in np.asarray(payload["trainNdxs"]).reshape(-1)}
    test = {int(value) - 1 for value in np.asarray(payload["testNdxs"]).reshape(-1)}
    return train, test


def convert_nyu(mat_path: Path, splits_path: Path | None, output: Path) -> dict[str, list[dict[str, object]]]:
    records = {"train": [], "val": []}
    with h5py.File(mat_path, "r") as handle:
        images = handle["images"]
        depths = handle["depths"]
        labels = handle.get("labels")
        train, val = split_indices(splits_path, images.shape[0])
        for index in range(images.shape[0]):
            split = "train" if index in train else ("val" if index in val else "")
            if not split:
                continue
            rgb = np.asarray(images[index]).transpose(2, 1, 0)
            depth = np.asarray(depths[index]).T.astype(np.float32)
            seg = None if labels is None else remap_nyu(np.asarray(labels[index]).T)
            gray = to_gray(rgb)
            records[split].append(write_sample(
                output, split, f"nyu_{index:06d}", "nyuv2", gray, seg, depth,
            ))
    return records


def paired_stair_files(split_root: Path) -> list[tuple[Path, Path, Path | None]]:
    image_root = find_dir(split_root, ("images", "image", "rgb"))
    mask_root = find_dir(split_root, ("segmentations", "segmentation", "labels", "masks"))
    depth_root = find_dir(split_root, ("depth", "depths", "depth_maps"))
    if image_root is None or mask_root is None:
        return []
    masks = {path.stem: path for path in files(mask_root, {".png", ".jpg", ".bmp"})}
    depths = {} if depth_root is None else {
        path.stem: path for path in files(depth_root, {".npy", ".png", ".tiff", ".tif"})
    }
    return [
        (path, masks[path.stem], depths.get(path.stem))
        for path in files(image_root, {".jpg", ".jpeg", ".png", ".bmp"})
        if path.stem in masks
    ]


def read_depth(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cannot read depth {path}")
    depth = raw.astype(np.float32)
    # StairNet exports may be millimetres; values already in metres remain unchanged.
    if float(np.nanpercentile(depth, 95)) > 100.0:
        depth *= 0.001
    return depth


def convert_stairs(root: Path, output: Path, split: str) -> list[dict[str, object]]:
    split_root = find_dir(root, (split, "training" if split == "train" else "validation"))
    if split_root is None:
        return []
    records: list[dict[str, object]] = []
    for index, (image_path, mask_path, depth_path) in enumerate(paired_stair_files(split_root)):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise RuntimeError(f"cannot read StairNet pair {image_path}")
        positive = np.any(mask != 0, axis=2) if mask.ndim == 3 else mask != 0
        band = max(3, min(15, (min(positive.shape) // 80) | 1))
        positive = cv2.dilate(positive.astype(np.uint8), np.ones((band, band), np.uint8)) > 0
        seg = np.full(positive.shape, IGNORE, dtype=np.uint8)
        seg[positive] = STEP
        records.append(write_sample(
            output, split, f"stair_{split}_{index:06d}", "stairnetv3",
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), seg, read_depth(depth_path),
        ))
    return records


def main() -> None:
    args = parse_args()
    if args.output.exists() and args.overwrite:
        resolved = args.output.resolve()
        protected = {Path("/").resolve(), Path.home().resolve(), Path("/root/autodl-tmp").resolve()}
        if resolved in protected or len(resolved.parts) < 4:
            raise RuntimeError(f"refusing to overwrite broad output path: {resolved}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    if args.ade_root:
        for split in by_split:
            by_split[split].extend(convert_ade(args.ade_root, args.output, split))
    if args.nyu_mat:
        nyu = convert_nyu(args.nyu_mat, args.nyu_splits, args.output)
        for split in by_split:
            by_split[split].extend(nyu[split])
    if args.stair_root:
        for split in by_split:
            by_split[split].extend(convert_stairs(args.stair_root, args.output, split))
    if not any(by_split.values()):
        raise RuntimeError("no public dataset was provided or recognized")
    summary: dict[str, object] = {
        "classes": ["ground_candidate", "blocked_surface", "step_or_drop"],
        "depth_bins": 16,
        "depth_range_m": [0.3, 8.0],
        "input_channels": 1,
        "rgb_input_used": False,
    }
    ids: dict[str, set[str]] = {}
    for split, records in by_split.items():
        manifest = args.output / f"manifest_{split}.jsonl"
        manifest.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        ids[split] = {str(record["source_id"]) for record in records}
        summary[split] = {
            "samples": len(records),
            "segmentation_samples": sum(record["seg_mask"] is not None for record in records),
            "depth_samples": sum(record["depth"] is not None for record in records),
        }
    overlap = ids.get("train", set()) & ids.get("val", set())
    if overlap:
        raise RuntimeError(f"train/val overlap: {sorted(overlap)[:10]}")
    summary["split_audit"] = {"source_id_overlap": 0, "passed": True}
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
