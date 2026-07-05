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

from graynav_obstacle8 import (
    COCO80_NAMES,
    SEMANTIC_NAMES,
    category_id_to_semantic,
    category_id_to_yolo80,
    semantic_categories,
)


DEFAULT_NAV_QUOTAS = {
    0: 9000,
    1: 5000,
    2: 4000,
    3: 3000,
    4: 4000,
    5: 5000,
    6: 4000,
    7: 3000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a gray [G,G,G] COCO80 fine-tuning dataset for YOLOv8n.")
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-images", type=int, default=30000)
    parser.add_argument("--context-images", type=int, default=3000)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--min-box", type=float, default=4.0)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_json_items(path: Path, prefix: str):
    """Stream large COCO JSON arrays using ijson when available."""
    try:
        import ijson
    except ImportError:
        key = prefix.split(".")[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from data.get(key, [])
        return
    with path.open("rb") as f:
        yield from ijson.items(f, prefix)


def read_categories(path: Path) -> list[dict[str, Any]]:
    """Read COCO categories."""
    return [dict(x) for x in iter_json_items(path, "categories.item")]


def read_images(path: Path, max_images: int = 0) -> dict[int, dict[str, Any]]:
    """Read COCO image metadata."""
    images = [dict(x) for x in iter_json_items(path, "images.item")]
    images.sort(key=lambda x: str(x["file_name"]))
    if max_images > 0:
        images = images[:max_images]
    return {int(x["id"]): x for x in images}


def collect_boxes(
    path: Path,
    cat_to_yolo80: dict[int, int],
    cat_to_semantic: dict[int, int],
    min_box: float,
) -> tuple[dict[int, list[tuple[int, tuple[float, float, float, float]]]], dict[int, set[int]]]:
    """Collect COCO80 labels and GrayNav semantic presence for sampling."""
    labels: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    semantic_presence: dict[int, set[int]] = defaultdict(set)
    for ann in tqdm(iter_json_items(path, "annotations.item"), desc=f"scan {path.name}"):
        if int(ann.get("iscrowd", 0)):
            continue
        cat_id = int(ann["category_id"])
        cls = cat_to_yolo80.get(cat_id)
        if cls is None:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w < min_box or h < min_box:
            continue
        image_id = int(ann["image_id"])
        labels[image_id].append((cls, (x, y, w, h)))
        sem = cat_to_semantic.get(cat_id)
        if sem is not None:
            semantic_presence[image_id].add(sem)
    return dict(labels), dict(semantic_presence)


def select_images(
    images: dict[int, dict[str, Any]],
    labels: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    semantic_presence: dict[int, set[int]],
    target_images: int,
    context_images: int,
    seed: int,
) -> list[int]:
    """Select a navigation-biased subset while preserving COCO80 labels."""
    rng = random.Random(seed)
    selected: list[int] = []
    selected_set: set[int] = set()
    pools = {sem: [image_id for image_id, sems in semantic_presence.items() if sem in sems] for sem in range(len(SEMANTIC_NAMES))}
    for pool in pools.values():
        rng.shuffle(pool)

    sem_counts = {sem: 0 for sem in range(len(SEMANTIC_NAMES))}
    for sem, quota in DEFAULT_NAV_QUOTAS.items():
        for image_id in pools[sem]:
            if image_id not in images:
                continue
            if image_id not in selected_set:
                selected.append(image_id)
                selected_set.add(image_id)
                for present in semantic_presence.get(image_id, set()):
                    sem_counts[present] += 1
            if sem_counts[sem] >= quota:
                break

    context_pool = [image_id for image_id in images if image_id in labels and image_id not in semantic_presence and image_id not in selected_set]
    rng.shuffle(context_pool)
    for image_id in context_pool[:context_images]:
        selected.append(image_id)
        selected_set.add(image_id)

    rest = [image_id for image_id in images if image_id in labels and image_id not in selected_set]
    rng.shuffle(rest)
    for image_id in rest:
        if len(selected) >= target_images:
            break
        selected.append(image_id)
        selected_set.add(image_id)
    rng.shuffle(selected)
    return selected[:target_images]


def gray3_from_bgr(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to exact gray replicated in three channels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def decode_zip_image(zf: zipfile.ZipFile, file_name: str) -> np.ndarray:
    """Decode a train2017 image from zip without extracting the full archive."""
    with zf.open(f"train2017/{file_name}") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {file_name}")
    return image


def write_label(path: Path, boxes: list[tuple[int, tuple[float, float, float, float]]], width: int, height: int) -> list[int]:
    """Write YOLO COCO80 labels."""
    counts = [0 for _ in COCO80_NAMES]
    lines: list[str] = []
    for cls, (x, y, w, h) in boxes:
        cx = (x + 0.5 * w) / max(1, width)
        cy = (y + 0.5 * h) / max(1, height)
        nw = w / max(1, width)
        nh = h / max(1, height)
        if nw > 0 and nh > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            counts[cls] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts


def write_split(
    split: str,
    image_ids: list[int],
    images: dict[int, dict[str, Any]],
    labels: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    out: Path,
    jpg_quality: int,
    train_zip: Path | None = None,
    val_dir: Path | None = None,
) -> dict[str, Any]:
    """Write gray three-channel images and COCO80 labels for one split."""
    counts = [0 for _ in COCO80_NAMES]
    with zipfile.ZipFile(train_zip) if train_zip else nullcontext(None) as zf:
        for image_id in tqdm(image_ids, desc=f"write {split}"):
            rec = images[image_id]
            file_name = Path(str(rec["file_name"])).name
            if zf:
                image = decode_zip_image(zf, file_name)
            else:
                image = cv2.imread(str(val_dir / file_name), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to read val image: {val_dir / file_name}")
            out_image = out / "images" / split / file_name
            out_image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_image), gray3_from_bgr(image), [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
            out_label = out / "labels" / split / f"{Path(file_name).stem}.txt"
            per_class = write_label(out_label, labels.get(image_id, []), int(rec["width"]), int(rec["height"]))
            counts = [a + b for a, b in zip(counts, per_class)]
    nonzero = {COCO80_NAMES[idx]: value for idx, value in enumerate(counts) if value > 0}
    return {"images": len(image_ids), "class_instances_nonzero": nonzero, "total_instances": int(sum(counts))}


class nullcontext:
    """Local null context for Python 3.10 compatibility."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_args):
        return False


def write_yaml(root: Path) -> Path:
    """Write Ultralytics COCO80 YAML for gray-domain fine-tuning."""
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(COCO80_NAMES))
    path = root / "gray_coco80.yaml"
    path.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return path


def write_graynav_eval_annotations(
    image_ids: list[int],
    images: dict[int, dict[str, Any]],
    semantic_labels: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    out_path: Path,
) -> None:
    """Write 8-class GrayNav COCO annotations used only for evaluation."""
    out_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for image_id in image_ids:
        rec = images[image_id]
        out_images.append({"id": image_id, "file_name": Path(str(rec["file_name"])).name, "width": int(rec["width"]), "height": int(rec["height"])})
        for sem, (x, y, w, h) in semantic_labels.get(image_id, []):
            annotations.append(
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
            {"info": {"description": "GrayNav eval annotations from COCO80 dataset"}, "licenses": [], "images": out_images, "annotations": annotations, "categories": semantic_categories()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def semantic_boxes(path: Path, cat_to_semantic: dict[int, int], min_box: float) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """Collect only GrayNav semantic boxes for the evaluation annotation file."""
    out: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = defaultdict(list)
    for ann in tqdm(iter_json_items(path, "annotations.item"), desc=f"semantic {path.name}"):
        if int(ann.get("iscrowd", 0)):
            continue
        sem = cat_to_semantic.get(int(ann["category_id"]))
        if sem is None:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w >= min_box and h >= min_box:
            out[int(ann["image_id"])].append((sem, (x, y, w, h)))
    return dict(out)


def main() -> None:
    args = parse_args()
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    train_categories = read_categories(args.train_annotations)
    val_categories = read_categories(args.val_annotations)
    train_cat_to_yolo80 = category_id_to_yolo80(train_categories)
    val_cat_to_yolo80 = category_id_to_yolo80(val_categories)
    train_cat_to_sem = category_id_to_semantic(train_categories)
    val_cat_to_sem = category_id_to_semantic(val_categories)
    train_images = read_images(args.train_annotations)
    val_images = read_images(args.val_annotations, args.val_max_images)
    train_labels, train_sem_presence = collect_boxes(args.train_annotations, train_cat_to_yolo80, train_cat_to_sem, args.min_box)
    val_labels, _ = collect_boxes(args.val_annotations, val_cat_to_yolo80, val_cat_to_sem, args.min_box)
    train_ids = select_images(train_images, train_labels, train_sem_presence, args.target_images, args.context_images, args.seed)
    val_ids = list(val_images.keys())
    train_stats = write_split("train", train_ids, train_images, train_labels, args.out, args.jpg_quality, train_zip=args.train_zip)
    val_stats = write_split("val", val_ids, val_images, val_labels, args.out, args.jpg_quality, val_dir=args.val_images)
    yaml_path = write_yaml(args.out)
    train_semantic = semantic_boxes(args.train_annotations, train_cat_to_sem, args.min_box)
    val_semantic = semantic_boxes(args.val_annotations, val_cat_to_sem, args.min_box)
    write_graynav_eval_annotations(train_ids, train_images, train_semantic, args.out / "annotations" / "instances_train_graynav8.json")
    write_graynav_eval_annotations(val_ids, val_images, val_semantic, args.out / "annotations" / "instances_val_graynav8.json")
    manifest = {
        "dataset": "GrayCOCO80-YOLOv8n-GrayNavEval",
        "purpose": "Keep YOLOv8n COCO80 detection head during grayscale fine-tuning; remap predictions to GrayNav classes only during evaluation.",
        "train_zip": str(args.train_zip),
        "train_annotations": str(args.train_annotations),
        "val_images": str(args.val_images),
        "val_annotations": str(args.val_annotations),
        "target_images": args.target_images,
        "context_images": args.context_images,
        "semantic_sampling_quotas": {SEMANTIC_NAMES[idx]: quota for idx, quota in DEFAULT_NAV_QUOTAS.items()},
        "yaml": str(yaml_path),
        "splits": {"train": train_stats, "val": val_stats},
    }
    (args.out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

