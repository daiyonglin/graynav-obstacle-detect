#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm


NAV_LABELS = {
    "person",
    "chair",
    "bench",
    "couch",
    "bed",
    "dining table",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "book",
    "laptop",
    "keyboard",
    "cell phone",
    "remote",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a balanced COCO image subset directly from train2017.zip.")
    parser.add_argument("--zip", required=True, type=Path, help="COCO train2017.zip path.")
    parser.add_argument("--annotations", required=True, type=Path, help="instances_train2017.json path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output subset directory.")
    parser.add_argument("--split", default="train2017", help="Zip folder name, usually train2017.")
    parser.add_argument("--max-images", type=int, default=20000)
    parser.add_argument("--nav-ratio", type=float, default=0.75, help="Target fraction of navigation-relevant images.")
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def iter_json_items(path: Path, prefix: str):
    """Stream large COCO JSON arrays without loading the whole file."""
    try:
        import ijson
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install ijson") from exc

    with path.open("rb") as f:
        yield from ijson.items(f, prefix)


def read_categories(path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    """Read COCO categories and derive the navigation-relevant category ids."""
    categories = [dict(item) for item in iter_json_items(path, "categories.item")]
    nav_cat_ids = {int(c["id"]) for c in categories if str(c["name"]) in NAV_LABELS}
    if not nav_cat_ids:
        raise RuntimeError("no navigation category ids matched COCO categories")
    return categories, nav_cat_ids


def read_nav_image_ids(path: Path, nav_cat_ids: set[int]) -> set[int]:
    """Collect image ids containing at least one navigation-relevant annotation."""
    nav_image_ids: set[int] = set()
    for ann in tqdm(iter_json_items(path, "annotations.item"), desc="scan nav annotations"):
        if int(ann.get("iscrowd", 0)):
            continue
        if int(ann["category_id"]) in nav_cat_ids:
            nav_image_ids.add(int(ann["image_id"]))
    return nav_image_ids


def choose_images(annotation_path: Path, nav_image_ids: set[int], max_images: int, nav_ratio: float, seed: int) -> list[dict[str, Any]]:
    """Sample images with a controlled bias toward navigation-relevant categories."""
    nav_images: list[dict[str, Any]] = []
    context_images: list[dict[str, Any]] = []
    for item in tqdm(iter_json_items(annotation_path, "images.item"), desc="scan images"):
        im = dict(item)
        if int(im["id"]) in nav_image_ids:
            nav_images.append(im)
        else:
            context_images.append(im)

    rng = random.Random(seed)
    rng.shuffle(nav_images)
    rng.shuffle(context_images)

    wanted_nav = min(len(nav_images), int(round(max_images * nav_ratio)))
    wanted_context = max(0, max_images - wanted_nav)
    selected = nav_images[:wanted_nav] + context_images[:wanted_context]
    if len(selected) < max_images:
        selected_ids = {int(im["id"]) for im in selected}
        remaining = [im for im in nav_images + context_images if int(im["id"]) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_images - len(selected)])
    rng.shuffle(selected)
    return selected


def write_subset_annotations(annotation_path: Path, categories: list[dict[str, Any]], selected: list[dict[str, Any]], out_path: Path) -> None:
    """Write a compact COCO annotation file for future supervised experiments."""
    image_ids = {int(im["id"]) for im in selected}
    subset_annotations = []
    for ann in tqdm(iter_json_items(annotation_path, "annotations.item"), desc="write subset annotations"):
        if int(ann["image_id"]) in image_ids:
            subset_annotations.append(dict(ann))
    subset = {
        "info": {"description": "COCO train2017 sampled subset for GrayNav experiments"},
        "licenses": [],
        "images": selected,
        "annotations": subset_annotations,
        "categories": categories,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")


def extract_images(zip_path: Path, selected: list[dict[str, Any]], split: str, image_dir: Path) -> list[str]:
    """Extract selected images without expanding the whole COCO archive."""
    image_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for im in tqdm(selected, desc="extract subset"):
            member = f"{split}/{im['file_name']}"
            target = image_dir / im["file_name"]
            if target.exists():
                continue
            try:
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
            except KeyError:
                missing.append(member)
                continue
    return missing


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(args.zip)
    if not args.annotations.exists():
        raise FileNotFoundError(args.annotations)

    categories, nav_cat_ids = read_categories(args.annotations)
    nav_image_ids = read_nav_image_ids(args.annotations, nav_cat_ids)
    selected = choose_images(args.annotations, nav_image_ids, args.max_images, args.nav_ratio, args.seed)
    image_dir = args.out_dir / "images"
    missing = extract_images(args.zip, selected, args.split, image_dir)
    write_subset_annotations(args.annotations, categories, selected, args.out_dir / "annotations" / f"instances_{args.split}_subset.json")

    manifest = {
        "zip": str(args.zip),
        "annotations": str(args.annotations),
        "split": args.split,
        "seed": args.seed,
        "max_images": args.max_images,
        "selected_images": len(selected),
        "extracted_images": len(list(image_dir.glob('*.jpg'))),
        "nav_image_candidates": len(nav_image_ids),
        "nav_ratio_target": args.nav_ratio,
        "nav_category_ids": sorted(nav_cat_ids),
        "missing": missing[:50],
        "missing_count": len(missing),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "subset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
