#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from graynav_obstacle8 import SEMANTIC_NAMES, category_id_to_semantic, semantic_categories


DEFAULT_QUOTAS = {
    0: 12000,
    1: 8000,
    2: 7000,
    3: 5000,
    4: 7000,
    5: 8000,
    6: 6000,
    7: 5000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-balanced GrayNav-Obstacle8 GGG dataset from COCO2017.")
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-images", type=int, default=60000)
    parser.add_argument("--hard-negative", type=int, default=2500)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--min-box", type=float, default=4.0)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_json_items(path: Path, prefix: str):
    """Stream large COCO arrays with ijson, falling back to json for small files."""
    try:
        import ijson
    except ImportError:
        key = prefix.split(".")[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from data.get(key, [])
        return
    with path.open("rb") as f:
        yield from ijson.items(f, prefix)


def read_images(path: Path, max_images: int = 0) -> dict[int, dict[str, Any]]:
    """Read COCO image metadata; the image table is small enough for memory."""
    items = [dict(x) for x in iter_json_items(path, "images.item")]
    items.sort(key=lambda x: str(x["file_name"]))
    if max_images > 0:
        items = items[:max_images]
    return {int(x["id"]): x for x in items}


def read_categories(path: Path) -> list[dict[str, Any]]:
    """Read COCO categories from an annotation file."""
    return [dict(x) for x in iter_json_items(path, "categories.item")]


def collect_boxes(path: Path, cat_to_sem: dict[int, int], min_box: float) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """Stream COCO annotations and keep GrayNav-mapped boxes grouped by image id."""
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for ann in tqdm(iter_json_items(path, "annotations.item"), desc=f"scan {path.name}"):
        if int(ann.get("iscrowd", 0)):
            continue
        sem = cat_to_sem.get(int(ann["category_id"]))
        if sem is None:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w >= min_box and h >= min_box:
            boxes[int(ann["image_id"])].append((sem, (x, y, w, h)))
    return dict(boxes)


def select_balanced_images(
    images: dict[int, dict[str, Any]],
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    target_images: int,
    hard_negative: int,
    seed: int,
) -> list[int]:
    """Select a deterministic class-balanced image subset plus hard negatives."""
    rng = random.Random(seed)
    pools: dict[int, list[int]] = {idx: [] for idx in range(len(SEMANTIC_NAMES))}
    for image_id, image_boxes in boxes.items():
        if image_id not in images:
            continue
        for sem in sorted({x[0] for x in image_boxes}):
            pools[sem].append(image_id)
    for pool in pools.values():
        rng.shuffle(pool)

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_sem_counts = {idx: 0 for idx in range(len(SEMANTIC_NAMES))}
    for sem, quota in DEFAULT_QUOTAS.items():
        for image_id in pools[sem]:
            if image_id not in selected_set:
                selected.append(image_id)
                selected_set.add(image_id)
                for image_sem in {x[0] for x in boxes.get(image_id, [])}:
                    selected_sem_counts[image_sem] += 1
            if selected_sem_counts[sem] >= quota:
                break

    negatives = [image_id for image_id in images if image_id not in boxes]
    rng.shuffle(negatives)
    for image_id in negatives[:hard_negative]:
        if image_id not in selected_set:
            selected.append(image_id)
            selected_set.add(image_id)

    all_positive = [image_id for image_id in images if image_id in boxes and image_id not in selected_set]
    rng.shuffle(all_positive)
    for image_id in all_positive:
        if len(selected) >= target_images:
            break
        selected.append(image_id)
        selected_set.add(image_id)

    if len(selected) < target_images:
        rest = [image_id for image_id in images if image_id not in selected_set]
        rng.shuffle(rest)
        selected.extend(rest[: target_images - len(selected)])

    rng.shuffle(selected)
    return selected[:target_images]


def gray3_from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Convert a color image to exact three-channel grayscale for YOLO training."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def decode_zip_image(zf: zipfile.ZipFile, file_name: str) -> np.ndarray:
    """Decode one COCO train image from train2017.zip."""
    with zf.open(f"train2017/{file_name}") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode train image: {file_name}")
    return image


def write_yolo_label(path: Path, boxes: list[tuple[int, tuple[float, float, float, float]]], width: int, height: int) -> list[int]:
    """Write YOLO labels and return class instance counts."""
    counts = [0 for _ in SEMANTIC_NAMES]
    lines: list[str] = []
    for sem, (x, y, w, h) in boxes:
        cx = (x + 0.5 * w) / max(1, width)
        cy = (y + 0.5 * h) / max(1, height)
        nw = w / max(1, width)
        nh = h / max(1, height)
        if nw > 0 and nh > 0:
            lines.append(f"{sem} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            counts[sem] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts


def write_split(
    split: str,
    image_ids: list[int],
    images: dict[int, dict[str, Any]],
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    out: Path,
    jpg_quality: int,
    train_zip: Path | None = None,
    val_dir: Path | None = None,
) -> dict[str, Any]:
    """Write exact GGG images and labels for one train/val split."""
    counts = [0 for _ in SEMANTIC_NAMES]
    empty = 0
    writer = zipfile.ZipFile(train_zip) if train_zip else None
    try:
        for image_id in tqdm(image_ids, desc=f"write {split}"):
            rec = images[image_id]
            file_name = Path(str(rec["file_name"])).name
            if writer:
                image = decode_zip_image(writer, file_name)
            else:
                src = val_dir / file_name
                image = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read val image: {src}")
            gray3 = gray3_from_bgr(image)
            img_out = out / "images" / split / file_name
            img_out.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(img_out), gray3, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            if not ok:
                raise RuntimeError(f"failed to write image: {img_out}")
            label_boxes = boxes.get(image_id, [])
            if not label_boxes:
                empty += 1
            label_out = out / "labels" / split / f"{Path(file_name).stem}.txt"
            per_class = write_yolo_label(label_out, label_boxes, int(rec["width"]), int(rec["height"]))
            counts = [a + b for a, b in zip(counts, per_class)]
    finally:
        if writer:
            writer.close()
    return {
        "images": len(image_ids),
        "empty_images": empty,
        "class_instances": {name: counts[idx] for idx, name in enumerate(SEMANTIC_NAMES)},
    }


def to_coco_obstacle8(
    image_ids: list[int],
    images: dict[int, dict[str, Any]],
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    out_path: Path,
) -> None:
    """Write COCO-style GrayNav-Obstacle8 annotations for fair M0/M1/M2 evaluation."""
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id in image_ids:
        rec = images[image_id]
        coco_images.append(
            {
                "id": image_id,
                "file_name": Path(str(rec["file_name"])).name,
                "width": int(rec["width"]),
                "height": int(rec["height"]),
            }
        )
        for sem, (x, y, w, h) in boxes.get(image_id, []):
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": sem + 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "info": {"description": "GrayNav-Obstacle8 remapped COCO annotations"},
                "licenses": [],
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": semantic_categories(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_yaml(root: Path) -> Path:
    """Write Ultralytics dataset YAML for GrayNav-Obstacle8."""
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(SEMANTIC_NAMES))
    path = root / "graynav8.yaml"
    path.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    train_categories = read_categories(args.train_annotations)
    val_categories = read_categories(args.val_annotations)
    train_cat_to_sem = category_id_to_semantic(train_categories)
    val_cat_to_sem = category_id_to_semantic(val_categories)
    train_images = read_images(args.train_annotations)
    val_images = read_images(args.val_annotations, args.val_max_images)
    train_boxes = collect_boxes(args.train_annotations, train_cat_to_sem, args.min_box)
    val_boxes = collect_boxes(args.val_annotations, val_cat_to_sem, args.min_box)

    train_ids = select_balanced_images(train_images, train_boxes, args.target_images, args.hard_negative, args.seed)
    val_ids = list(val_images.keys())
    train_stats = write_split("train", train_ids, train_images, train_boxes, args.out, args.jpg_quality, train_zip=args.train_zip)
    val_stats = write_split("val", val_ids, val_images, val_boxes, args.out, args.jpg_quality, val_dir=args.val_images)
    yaml_path = write_yaml(args.out)
    to_coco_obstacle8(train_ids, train_images, train_boxes, args.out / "annotations" / "instances_train_obstacle8.json")
    to_coco_obstacle8(val_ids, val_images, val_boxes, args.out / "annotations" / "instances_val_obstacle8.json")

    manifest = {
        "dataset": "GrayNav-Obstacle8-GrayStem",
        "train_zip": str(args.train_zip),
        "train_annotations": str(args.train_annotations),
        "val_images": str(args.val_images),
        "val_annotations": str(args.val_annotations),
        "yaml": str(yaml_path),
        "seed": args.seed,
        "target_images": args.target_images,
        "hard_negative": args.hard_negative,
        "semantic_names": SEMANTIC_NAMES,
        "splits": {"train": train_stats, "val": val_stats},
    }
    (args.out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
