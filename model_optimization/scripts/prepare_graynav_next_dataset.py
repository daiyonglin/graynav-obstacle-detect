#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

from gmfe_utils import (
    channel_correlation,
    estimate_gmfe_meta,
    gmfe_rgb,
    gray_to_rgb_copy,
    read_gray,
    save_gmfe_audit,
    write_rgb_image,
)


SEMANTIC_NAMES = [
    "person",
    "chair/seat",
    "table/desk",
    "sofa/bed",
    "bag/suitcase",
    "small_object",
    "vehicle/bicycle",
    "generic_obstacle",
]

NAME_TO_SEMANTIC = {
    "person": 0,
    "chair": 1,
    "bench": 1,
    "dining table": 2,
    "table": 2,
    "desk": 2,
    "couch": 3,
    "sofa": 3,
    "bed": 3,
    "backpack": 4,
    "handbag": 4,
    "suitcase": 4,
    "bottle": 5,
    "cup": 5,
    "book": 5,
    "laptop": 5,
    "keyboard": 5,
    "cell phone": 5,
    "remote": 5,
    "bicycle": 6,
    "motorcycle": 6,
    "car": 6,
    "bus": 6,
    "truck": 6,
    "traffic cone": 7,
    "potted plant": 7,
    "trash can": 7,
    "refrigerator": 7,
    "toilet": 7,
    "tv": 7,
    "microwave": 7,
    "oven": 7,
    "sink": 7,
    "vase": 7,
    "umbrella": 7,
    "sports ball": 7,
    "skateboard": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GrayNav-Obstacle8 GGG and GMFE datasets from COCO.")
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-max-images", type=int, default=30000)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--nav-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--min-box", type=float, default=4.0)
    parser.add_argument("--gmfe-scale-sample", type=int, default=512)
    parser.add_argument("--audit-samples", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_json_items(path: Path, prefix: str):
    """Stream large COCO JSON arrays without loading whole annotations into memory."""
    try:
        import ijson
    except ImportError:
        key = prefix.split(".")[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        yield from data.get(key, [])
        return
    with path.open("rb") as f:
        yield from ijson.items(f, prefix)


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def category_map(path: Path) -> dict[int, int]:
    """Map COCO category ids to GrayNav-Obstacle8 semantic ids."""
    out: dict[int, int] = {}
    for cat in iter_json_items(path, "categories.item"):
        name = normalize_name(cat.get("name", ""))
        if name in NAME_TO_SEMANTIC:
            out[int(cat["id"])] = NAME_TO_SEMANTIC[name]
    return out


def read_images(path: Path, max_images: int = 0) -> list[dict[str, Any]]:
    """Read COCO image metadata; image records are small enough to keep in memory."""
    images = [dict(item) for item in iter_json_items(path, "images.item")]
    images.sort(key=lambda x: str(x["file_name"]))
    return images[:max_images] if max_images > 0 else images


def nav_image_ids(path: Path, cat_to_sem: dict[int, int]) -> set[int]:
    """Collect images containing at least one mapped navigation object."""
    out: set[int] = set()
    for ann in tqdm(iter_json_items(path, "annotations.item"), desc="scan nav annotations"):
        if int(ann.get("iscrowd", 0)):
            continue
        if int(ann["category_id"]) in cat_to_sem:
            out.add(int(ann["image_id"]))
    return out


def select_train_images(images: list[dict[str, Any]], nav_ids: set[int], max_images: int, nav_ratio: float, seed: int) -> list[dict[str, Any]]:
    """Select a fixed navigation-biased COCO train subset."""
    rng = random.Random(seed)
    nav = [x for x in images if int(x["id"]) in nav_ids]
    context = [x for x in images if int(x["id"]) not in nav_ids]
    rng.shuffle(nav)
    rng.shuffle(context)
    want_nav = min(len(nav), int(round(max_images * nav_ratio)))
    selected = nav[:want_nav] + context[: max_images - want_nav]
    if len(selected) < max_images:
        used = {int(x["id"]) for x in selected}
        rest = [x for x in images if int(x["id"]) not in used]
        rng.shuffle(rest)
        selected.extend(rest[: max_images - len(selected)])
    rng.shuffle(selected)
    return selected


def grouped_boxes(path: Path, selected_ids: set[int], cat_to_sem: dict[int, int], min_box: float) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    """Stream annotations and keep only boxes for selected images."""
    grouped: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {image_id: [] for image_id in selected_ids}
    for ann in tqdm(iter_json_items(path, "annotations.item"), desc="group annotations"):
        image_id = int(ann["image_id"])
        if image_id not in selected_ids or int(ann.get("iscrowd", 0)):
            continue
        sem = cat_to_sem.get(int(ann["category_id"]))
        if sem is None:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        if w >= min_box and h >= min_box:
            grouped[image_id].append((sem, (x, y, w, h)))
    return grouped


def write_label(path: Path, boxes: list[tuple[int, tuple[float, float, float, float]]], width: int, height: int) -> list[int]:
    """Write a YOLO label file and return per-class instance counts."""
    counts = [0 for _ in SEMANTIC_NAMES]
    lines: list[str] = []
    for sem, (x, y, w, h) in boxes:
        cx = (x + 0.5 * w) / max(1, width)
        cy = (y + 0.5 * h) / max(1, height)
        nw = w / max(1, width)
        nh = h / max(1, height)
        if nw > 0.0 and nh > 0.0:
            lines.append(f"{sem} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            counts[sem] += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts


def write_yaml(root: Path, variant: str) -> Path:
    """Write an Ultralytics YAML for one encoded dataset variant."""
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(SEMANTIC_NAMES))
    path = root / f"graynav-obstacle8-{variant}.yaml"
    text = f"path: {(root / 'variants' / variant).as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    path.write_text(text, encoding="utf-8")
    return path


def train_zip_member_bytes(zf: zipfile.ZipFile, split: str, file_name: str) -> bytes:
    with zf.open(f"{split}/{file_name}") as f:
        return f.read()


def decode_gray(data: bytes) -> Any:
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError("failed to decode image from zip bytes")
    return gray


def write_encoded_images(gray_path: Path, ggg_path: Path, gmfe_path: Path, meta) -> list[list[float]]:
    """Write GGG and GMFE encoded images from one gray source."""
    gray = read_gray(gray_path)
    ggg = gray_to_rgb_copy(gray)
    gmfe = gmfe_rgb(gray, meta)
    write_rgb_image(ggg_path, ggg)
    write_rgb_image(gmfe_path, gmfe)
    return channel_correlation(gmfe)


def copy_labels(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_split(
    split: str,
    images: list[dict[str, Any]],
    boxes: dict[int, list[tuple[int, tuple[float, float, float, float]]]],
    out: Path,
    meta,
    source_zip: Path | None = None,
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Write gray source images, GGG/GMFE variants, and labels for one split."""
    counts = [0 for _ in SEMANTIC_NAMES]
    empty = 0
    corr_samples: list[list[list[float]]] = []
    with zipfile.ZipFile(source_zip) if source_zip else nullcontext(None) as zf:
        for rec in tqdm(images, desc=f"write {split}"):
            image_id = int(rec["id"])
            file_name = Path(str(rec["file_name"])).name
            gray_path = out / "gray" / "images" / split / file_name
            if source_zip:
                data = train_zip_member_bytes(zf, "train2017", file_name)
                gray = decode_gray(data)
                gray_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(gray_path), gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    raise RuntimeError(f"failed to write gray image: {gray_path}")
            elif source_dir:
                src = source_dir / file_name
                gray = read_gray(src)
                gray_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(gray_path), gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if not ok:
                    raise RuntimeError(f"failed to write gray image: {gray_path}")
            else:
                raise ValueError("source_zip or source_dir is required")

            ggg_img = out / "variants" / "ggg" / "images" / split / file_name
            gmfe_img = out / "variants" / "gmfe" / "images" / split / file_name
            corr = write_encoded_images(gray_path, ggg_img, gmfe_img, meta)
            if len(corr_samples) < 128:
                corr_samples.append(corr)

            split_boxes = boxes.get(image_id, [])
            if not split_boxes:
                empty += 1
            label_name = f"{Path(file_name).stem}.txt"
            gray_label = out / "gray" / "labels" / split / label_name
            per_class = write_label(gray_label, split_boxes, int(rec["width"]), int(rec["height"]))
            counts = [a + b for a, b in zip(counts, per_class)]
            copy_labels(gray_label, out / "variants" / "ggg" / "labels" / split / label_name)
            copy_labels(gray_label, out / "variants" / "gmfe" / "labels" / split / label_name)

    return {
        "images": len(images),
        "empty_images": empty,
        "class_instances": {name: counts[i] for i, name in enumerate(SEMANTIC_NAMES)},
        "gmfe_corr_samples": corr_samples[:16],
    }


class nullcontext:
    """Small local nullcontext to keep Python 3.10 compatibility explicit."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_args):
        return False


def save_audits(out: Path, meta, limit: int, seed: int) -> None:
    """Save visual panels proving that GMFE inputs are derived from grayscale."""
    files = sorted((out / "gray" / "images" / "train").glob("*.jpg"))
    random.Random(seed).shuffle(files)
    audit_dir = out / "audit" / "gmfe_inputs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(files[:limit]):
        gray = read_gray(path)
        save_gmfe_audit(audit_dir / f"audit_{idx:04d}_{path.name}", gray, gmfe_rgb(gray, meta))


def main() -> None:
    args = parse_args()
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    train_cat = category_map(args.train_annotations)
    val_cat = category_map(args.val_annotations)
    train_images_all = read_images(args.train_annotations)
    nav_ids = nav_image_ids(args.train_annotations, train_cat)
    train_images = select_train_images(train_images_all, nav_ids, args.train_max_images, args.nav_ratio, args.seed)
    val_images = read_images(args.val_annotations, args.val_max_images)

    train_ids = {int(x["id"]) for x in train_images}
    val_ids = {int(x["id"]) for x in val_images}
    train_boxes = grouped_boxes(args.train_annotations, train_ids, train_cat, args.min_box)
    val_boxes = grouped_boxes(args.val_annotations, val_ids, val_cat, args.min_box)

    # Write gray train images first so GMFE normalization comes from actual training inputs.
    train_gray_dir = args.out / "gray" / "images" / "train"
    with zipfile.ZipFile(args.train_zip) as zf:
        for rec in tqdm(train_images, desc="write train gray for GMFE scale"):
            file_name = Path(str(rec["file_name"])).name
            gray = decode_gray(train_zip_member_bytes(zf, "train2017", file_name))
            dst = train_gray_dir / file_name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), gray, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    meta = estimate_gmfe_meta(sorted(train_gray_dir.glob("*.jpg")), limit=args.gmfe_scale_sample)
    (args.out / "gmfe_meta.json").write_text(meta.to_json(), encoding="utf-8")

    train_stats = prepare_split("train", train_images, train_boxes, args.out, meta, source_zip=args.train_zip)
    val_stats = prepare_split("val", val_images, val_boxes, args.out, meta, source_dir=args.val_images)
    save_audits(args.out, meta, args.audit_samples, args.seed)

    ggg_yaml = write_yaml(args.out, "ggg")
    gmfe_yaml = write_yaml(args.out, "gmfe")
    manifest = {
        "dataset": "GrayNav-Obstacle8-v2",
        "train_zip": str(args.train_zip),
        "train_annotations": str(args.train_annotations),
        "val_images": str(args.val_images),
        "val_annotations": str(args.val_annotations),
        "train_max_images": args.train_max_images,
        "val_max_images": args.val_max_images,
        "nav_ratio": args.nav_ratio,
        "semantic_names": SEMANTIC_NAMES,
        "gmfe_meta": json.loads(meta.to_json()),
        "yaml": {"ggg": str(ggg_yaml), "gmfe": str(gmfe_yaml)},
        "splits": {"train": train_stats, "val": val_stats},
    }
    (args.out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
