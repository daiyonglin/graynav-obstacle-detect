#!/usr/bin/env python3
"""Build deterministic COCO manifests for GrayNav's eight indoor classes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


INDOOR = {
    "person": 0,
    "chair": 1,
    "dining table": 2,
    "backpack": 3,
    "handbag": 4,
    "suitcase": 5,
    "couch": 6,
    "bench": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--negative-modulus", type=int, default=10,
        help="Keep one deterministic class-negative image per N images; 0 disables negatives.",
    )
    return parser.parse_args()


def build_records(payload: dict[str, object], images_root: Path, negative_modulus: int) -> tuple[list[dict[str, object]], Counter[str]]:
    categories = {
        int(row["id"]): str(row["name"])
        for row in payload["categories"]  # type: ignore[index]
    }
    selected = {
        category_id: INDOOR[name]
        for category_id, name in categories.items()
        if name in INDOOR
    }
    annotations: dict[int, list[list[float | int]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in payload["annotations"]:  # type: ignore[index]
        category_id = int(row["category_id"])
        if category_id not in selected or int(row.get("iscrowd", 0)) != 0:
            continue
        x, y, width, height = (float(value) for value in row["bbox"])
        if width < 2.0 or height < 2.0:
            continue
        cls = selected[category_id]
        annotations[int(row["image_id"])].append([x, y, width, height, cls])
        counts[tuple(INDOOR)[cls]] += 1

    records: list[dict[str, object]] = []
    for image in sorted(payload["images"], key=lambda item: int(item["id"])):  # type: ignore[index]
        image_id = int(image["id"])
        boxes = annotations.get(image_id, [])
        keep_negative = negative_modulus > 0 and image_id % negative_modulus == 0
        if not boxes and not keep_negative:
            continue
        path = images_root / str(image["file_name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({
            "source": "coco2017",
            "source_id": f"coco:{image_id}",
            "image": str(path.resolve()),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "boxes_xywh": boxes,
        })
    counts["images"] = len(records)
    counts["negative_images"] = sum(not row["boxes_xywh"] for row in records)
    return records, counts


def main() -> None:
    args = parse_args()
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    records, counts = build_records(payload, args.images, args.negative_modulus)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / f"manifest_{args.split}.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    report = {
        "split": args.split,
        "class_names": list(INDOOR),
        "counts": dict(counts),
        "annotations": str(args.annotations.resolve()),
        "images": str(args.images.resolve()),
        "manifest": str(manifest.resolve()),
    }
    (args.output / f"audit_{args.split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("GRAYNAV_COCO_INDOOR8_PREPARED")


if __name__ == "__main__":
    main()
