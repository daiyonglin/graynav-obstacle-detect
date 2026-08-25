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
UNKNOWN = 3
IGNORE = 255
MAX_STAIR_STEP_RATIO = 0.95

# MIT Scene Parsing 150 uses one-based ids and zero for unlabeled pixels.
# Keep this mapping conservative: ground is a traversable-shape candidate, while
# water, vegetation and vertical structures are blocked.  The four explicit
# stair-like labels in ADE150 are folded into one deployment hazard class.
ADE_GROUND = {4, 7, 12, 14, 29, 30, 47, 53, 55, 92, 95}
ADE_BLOCKED = {
    1, 2, 5, 10, 15, 18, 22, 27, 33, 35, 39, 43, 49, 52, 61,
    69, 73, 80, 85, 96, 105, 110, 114, 129,
}
ADE_STEP = {54, 60, 97, 122}


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
    # ADE ids 1..150 are valid annotated pixels.  Unmapped semantic content is
    # explicitly UNKNOWN instead of ignore, so it can suppress false hazards.
    out = np.full(mask.shape[:2], UNKNOWN, dtype=np.uint8)
    out[mask == 0] = IGNORE
    for value in ADE_GROUND:
        out[mask == value] = GROUND
    for value in ADE_BLOCKED:
        out[mask == value] = BLOCKED
    for value in ADE_STEP:
        out[mask == value] = STEP
    return out


def remap_stair_mask(mask: np.ndarray, boundary_pixels: int = 5) -> np.ndarray:
    """Map StairNet to STEP/UNKNOWN with an ignored anti-aliased boundary band."""

    if mask.ndim == 3:
        positive = np.any(mask >= 128, axis=2)
    else:
        positive = mask >= 128
    kernel_size = max(1, int(boundary_pixels) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded = cv2.dilate(positive.astype(np.uint8), kernel) > 0
    contracted = cv2.erode(positive.astype(np.uint8), kernel) > 0
    boundary = expanded & ~contracted
    seg = np.full(positive.shape, UNKNOWN, dtype=np.uint8)
    seg[positive & ~boundary] = STEP
    seg[boundary] = IGNORE
    return seg


def is_full_frame_stair_label(seg: np.ndarray) -> bool:
    """Reject StairNet labels that cannot provide meaningful negative pixels."""

    return float(np.mean(seg == STEP)) > MAX_STAIR_STEP_RATIO


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
    """Convert official labeled NYUv2 as grayscale metric-depth supervision.

    The MAT file's ``labels`` tensor contains the original fine-grained NYU
    category ids, not NYU40 ids.  Applying NYU40 class numbers directly would
    silently create incorrect ground/blocked masks, so segmentation remains
    loss-masked here.  ADE20K and StairNetV3 provide segmentation supervision.
    """

    records = {"train": [], "val": []}
    with h5py.File(mat_path, "r") as handle:
        images = handle["images"]
        depths = handle["depths"]
        train, val = split_indices(splits_path, images.shape[0])
        for index in range(images.shape[0]):
            split = "train" if index in train else ("val" if index in val else "")
            if not split:
                continue
            rgb = np.asarray(images[index]).transpose(2, 1, 0)
            depth = np.asarray(depths[index]).T.astype(np.float32)
            gray = to_gray(rgb)
            records[split].append(write_sample(
                output, split, f"nyu_{index:06d}", "nyuv2", gray, None, depth,
            ))
    return records


def paired_stair_files(split_root: Path) -> list[tuple[Path, Path, Path | None]]:
    image_root = find_dir(split_root, ("images", "image", "rgb"))
    mask_root = find_dir(split_root, ("segmentations", "segmentation", "labels", "masks"))
    # The official StairNetV3 archive intentionally spells this directory
    # "depthes".  Retain aliases for repackaged mirrors, but recognize the
    # published layout without requiring the user to rename any files.
    depth_root = find_dir(split_root, ("depthes", "depth", "depths", "depth_maps"))
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
    # StairNetV3 train/val publishes 8-bit depth visualizations.  No metric
    # scale or per-image conversion is provided for those PNGs, so treating
    # their 0..255 intensities as metres or millimetres would silently corrupt
    # the ordinal-depth target.  Keep them loss-masked; NYUv2 supplies the
    # reliable metric-depth supervision.  Repackaged uint16 depth maps are
    # accepted as millimetres, while float arrays are expected through .npy.
    if raw.dtype == np.uint8:
        return None
    depth = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        depth *= 0.001
    return depth


def convert_stairs(
    root: Path, output: Path, split: str
) -> tuple[list[dict[str, object]], int]:
    split_root = find_dir(root, (split, "training" if split == "train" else "validation"))
    if split_root is None:
        return [], 0
    records: list[dict[str, object]] = []
    filtered_full_frame = 0
    for index, (image_path, mask_path, depth_path) in enumerate(paired_stair_files(split_root)):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or mask is None:
            raise RuntimeError(f"cannot read StairNet pair {image_path}")
        seg = remap_stair_mask(mask, boundary_pixels=5)
        if is_full_frame_stair_label(seg):
            filtered_full_frame += 1
            continue
        records.append(write_sample(
            output, split, f"stair_{split}_{index:06d}", "stairnetv3",
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), seg, read_depth(depth_path),
        ))
    return records, filtered_full_frame


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
    stair_full_frame_filtered = {"train": 0, "val": 0}
    if args.ade_root:
        for split in by_split:
            by_split[split].extend(convert_ade(args.ade_root, args.output, split))
    if args.nyu_mat:
        nyu = convert_nyu(args.nyu_mat, args.nyu_splits, args.output)
        for split in by_split:
            by_split[split].extend(nyu[split])
    if args.stair_root:
        for split in by_split:
            stair_records, filtered = convert_stairs(args.stair_root, args.output, split)
            by_split[split].extend(stair_records)
            stair_full_frame_filtered[split] = filtered
    if not any(by_split.values()):
        raise RuntimeError("no public dataset was provided or recognized")
    summary: dict[str, object] = {
        "classes": [
            "ground_candidate", "blocked_surface", "step_or_drop", "unknown_other"
        ],
        "depth_bins": 16,
        "depth_range_m": [0.3, 8.0],
        "input_channels": 1,
        "rgb_input_used": False,
        "automatic_filters": {
            "stair_step_ratio_gt_0_95": stair_full_frame_filtered,
        },
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
