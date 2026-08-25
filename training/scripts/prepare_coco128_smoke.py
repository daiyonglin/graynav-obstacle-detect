#!/usr/bin/env python3
"""Convert an existing YOLO-format COCO128 copy into smoke-only manifests.

This dataset is intentionally too small for model selection.  It exists only
to validate the unified training loop and the 8 GiB local-GPU configuration
before downloading a compact formal detection subset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


COCO_TO_INDOOR = {
    0: 0,   # person
    56: 1,  # chair
    60: 2,  # dining table
    24: 3,  # backpack
    26: 4,  # handbag
    28: 5,  # suitcase
    57: 6,  # couch
    13: 7,  # bench
}
CLASS_NAMES = (
    "person", "chair", "dining_table", "backpack",
    "handbag", "suitcase", "couch", "bench",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-modulus", type=int, default=5)
    return parser.parse_args()


def convert(dataset: Path, val_modulus: int) -> tuple[dict[str, list[dict]], dict]:
    image_root = dataset / "images" / "train2017"
    label_root = dataset / "labels" / "train2017"
    if val_modulus < 2:
        raise ValueError("val modulus must be at least two")
    records: dict[str, list[dict]] = {"train": [], "val": []}
    counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    for index, image_path in enumerate(sorted(image_root.glob("*.jpg"))):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        height, width = image.shape
        boxes = []
        label_path = label_root / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                source_class = int(values[0])
                if source_class not in COCO_TO_INDOOR:
                    continue
                cx, cy, box_width, box_height = map(float, values[1:])
                x = (cx - box_width * 0.5) * width
                y = (cy - box_height * 0.5) * height
                target_class = COCO_TO_INDOOR[source_class]
                boxes.append([
                    x, y, box_width * width, box_height * height, target_class,
                ])
                counts["val" if index % val_modulus == 0 else "train"][CLASS_NAMES[target_class]] += 1
        split = "val" if index % val_modulus == 0 else "train"
        records[split].append({
            "source": "coco128_smoke_only",
            "source_id": f"coco128:{image_path.stem}",
            "image": str(image_path.resolve()),
            "width": width,
            "height": height,
            "boxes_xywh": boxes,
        })
    for split in records:
        counts[split]["images"] = len(records[split])
        counts[split]["negative_images"] = sum(
            not row["boxes_xywh"] for row in records[split]
        )
    return records, {
        "smoke_only": True,
        "model_selection_allowed": False,
        "source": str(dataset.resolve()),
        "classes": list(CLASS_NAMES),
        "splits": {key: dict(value) for key, value in counts.items()},
    }


def main() -> None:
    args = parse_args()
    records, audit = convert(args.dataset, args.val_modulus)
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in records.items():
        (args.output / f"manifest_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    (args.output / "SMOKE_ONLY.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("GRAYNAV_COCO128_SMOKE_READY")


if __name__ == "__main__":
    main()
