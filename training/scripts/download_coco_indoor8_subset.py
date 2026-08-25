#!/usr/bin/env python3
"""Select and download a deterministic compact COCO Indoor8 rehearsal set.

The unified model starts from official COCO YOLOv8n weights, so it does not
need all 118k COCO train images.  This builder keeps explicit per-class minima,
fills the remaining budget with target-positive images, and adds a small set of
true class-negative images.  It writes a filtered COCO JSON that is consumed by
``prepare_coco_indoor8.py`` without inventing any labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm.auto import tqdm


INDOOR_NAMES = (
    "person",
    "chair",
    "dining table",
    "backpack",
    "handbag",
    "suitcase",
    "couch",
    "bench",
)

TRAIN_QUOTAS = {
    "person": 3500,
    "chair": 1800,
    "dining table": 1000,
    "backpack": 900,
    "handbag": 900,
    "suitcase": 700,
    "couch": 700,
    "bench": 700,
}

VAL_QUOTAS = {
    "person": 500,
    "chair": 300,
    "dining table": 200,
    "backpack": 180,
    "handbag": 180,
    "suitcase": 140,
    "couch": 140,
    "bench": 140,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--filtered-annotations", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--max-images", type=int, required=True)
    parser.add_argument("--negative-images", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stable_key(seed: int, image_id: int) -> str:
    return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()


def select_subset(
    payload: dict[str, object],
    quotas: dict[str, int],
    max_images: int,
    negative_images: int,
    seed: int,
) -> tuple[set[int], Counter[str], set[int]]:
    categories = {
        int(row["id"]): str(row["name"])
        for row in payload["categories"]  # type: ignore[index]
    }
    target_ids = {
        category_id: name
        for category_id, name in categories.items()
        if name in INDOOR_NAMES
    }
    image_classes: dict[int, set[str]] = defaultdict(set)
    for row in payload["annotations"]:  # type: ignore[index]
        category_id = int(row["category_id"])
        if category_id in target_ids and int(row.get("iscrowd", 0)) == 0:
            image_classes[int(row["image_id"])].add(target_ids[category_id])

    by_class: dict[str, list[int]] = {name: [] for name in INDOOR_NAMES}
    for image_id, names in image_classes.items():
        for name in names:
            by_class[name].append(image_id)
    for values in by_class.values():
        values.sort(key=lambda image_id: stable_key(seed, image_id))

    selected: set[int] = set()
    counts: Counter[str] = Counter()
    # Rare classes are satisfied first so the person category cannot consume
    # the entire compact budget.
    for name in sorted(INDOOR_NAMES, key=lambda item: (len(by_class[item]), item)):
        for image_id in by_class[name]:
            if counts[name] >= quotas[name]:
                break
            if image_id not in selected:
                selected.add(image_id)
                counts.update(image_classes[image_id])

    positive_budget = max_images - negative_images
    all_positive = sorted(image_classes, key=lambda image_id: stable_key(seed + 1, image_id))
    for image_id in all_positive:
        if len(selected) >= positive_budget:
            break
        if image_id not in selected:
            selected.add(image_id)
            counts.update(image_classes[image_id])

    all_image_ids = {int(row["id"]) for row in payload["images"]}  # type: ignore[index]
    negative_candidates = sorted(
        all_image_ids.difference(image_classes),
        key=lambda image_id: stable_key(seed + 2, image_id),
    )
    negatives = set(negative_candidates[:negative_images])
    selected.update(negatives)

    for name, quota in quotas.items():
        if counts[name] < quota:
            raise RuntimeError(f"COCO quota not met for {name}: {counts[name]} < {quota}")
    if len(selected) > max_images:
        raise RuntimeError(f"selected {len(selected)} images beyond budget {max_images}")
    return selected, counts, negatives


def download_one(url: str, destination: Path, attempts: int = 5) -> None:
    if destination.is_file() and destination.stat().st_size > 1024:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            request = Request(url, headers={"User-Agent": "GrayNav/1.0"})
            with urlopen(request, timeout=60) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if partial.stat().st_size <= 1024:
                raise RuntimeError("downloaded file is unexpectedly small")
            os.replace(partial, destination)
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == attempts:
                raise
            time.sleep(float(attempt))


def main() -> None:
    args = parse_args()
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    quotas = TRAIN_QUOTAS if args.split == "train" else VAL_QUOTAS
    selected, counts, negatives = select_subset(
        payload, quotas, args.max_images, args.negative_images, args.seed
    )
    selected_images = [
        row for row in payload["images"]  # type: ignore[index]
        if int(row["id"]) in selected
    ]
    selected_annotations = [
        row for row in payload["annotations"]  # type: ignore[index]
        if int(row["image_id"]) in selected
    ]
    image_urls: list[tuple[str, Path]] = []
    for row in selected_images:
        file_name = str(row["file_name"])
        url = f"http://images.cocodataset.org/{args.split}2017/{file_name}"
        image_urls.append((url, args.images / file_name))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, url, destination): destination
            for url, destination in image_urls
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"COCO {args.split} subset"
        ):
            future.result()

    filtered = {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": payload["categories"],
    }
    args.filtered_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.filtered_annotations.write_text(
        json.dumps(filtered, ensure_ascii=False), encoding="utf-8"
    )
    report = {
        "split": args.split,
        "seed": args.seed,
        "selected_images": len(selected_images),
        "negative_images": len(negatives),
        "class_image_counts": dict(counts),
        "quotas": quotas,
        "filtered_annotations": str(args.filtered_annotations.resolve()),
        "images": str(args.images.resolve()),
    }
    report_path = args.filtered_annotations.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("GRAYNAV_COCO_INDOOR8_SUBSET_READY")


if __name__ == "__main__":
    main()
