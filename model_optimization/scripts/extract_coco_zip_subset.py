#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import defaultdict
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_images(coco: dict[str, Any], max_images: int, nav_ratio: float, seed: int) -> tuple[list[dict[str, Any]], set[int]]:
    """Sample images with a controlled bias toward navigation-relevant categories."""
    categories = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    nav_cat_ids = {cid for cid, name in categories.items() if name in NAV_LABELS}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[int(ann["image_id"])].append(ann)

    images = list(coco["images"])
    nav_images = [im for im in images if any(int(a["category_id"]) in nav_cat_ids for a in anns_by_image[int(im["id"])])]
    context_images = [im for im in images if im not in nav_images]

    rng = random.Random(seed)
    rng.shuffle(nav_images)
    rng.shuffle(context_images)

    wanted_nav = min(len(nav_images), int(round(max_images * nav_ratio)))
    wanted_context = max(0, max_images - wanted_nav)
    selected = nav_images[:wanted_nav] + context_images[:wanted_context]
    if len(selected) < max_images:
        selected_ids = {int(im["id"]) for im in selected}
        remaining = [im for im in images if int(im["id"]) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_images - len(selected)])
    rng.shuffle(selected)
    return selected, nav_cat_ids


def write_subset_annotations(coco: dict[str, Any], selected: list[dict[str, Any]], out_path: Path) -> None:
    """Write a compact COCO annotation file for future supervised experiments."""
    image_ids = {int(im["id"]) for im in selected}
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": selected,
        "annotations": [ann for ann in coco["annotations"] if int(ann["image_id"]) in image_ids],
        "categories": coco["categories"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(subset, ensure_ascii=False), encoding="utf-8")


def extract_images(zip_path: Path, selected: list[dict[str, Any]], split: str, image_dir: Path) -> list[str]:
    """Extract selected images without expanding the whole COCO archive."""
    image_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for im in tqdm(selected, desc="extract subset"):
            member = f"{split}/{im['file_name']}"
            target = image_dir / im["file_name"]
            if target.exists():
                continue
            if member not in names:
                missing.append(member)
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
    return missing


def main() -> None:
    args = parse_args()
    if not args.zip.exists():
        raise FileNotFoundError(args.zip)
    if not args.annotations.exists():
        raise FileNotFoundError(args.annotations)

    coco = load_json(args.annotations)
    selected, nav_cat_ids = choose_images(coco, args.max_images, args.nav_ratio, args.seed)
    image_dir = args.out_dir / "images"
    missing = extract_images(args.zip, selected, args.split, image_dir)
    write_subset_annotations(coco, selected, args.out_dir / "annotations" / f"instances_{args.split}_subset.json")

    manifest = {
        "zip": str(args.zip),
        "annotations": str(args.annotations),
        "split": args.split,
        "seed": args.seed,
        "max_images": args.max_images,
        "selected_images": len(selected),
        "extracted_images": len(list(image_dir.glob('*.jpg'))),
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
