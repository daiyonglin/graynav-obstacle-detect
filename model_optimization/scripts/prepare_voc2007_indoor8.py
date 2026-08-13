#!/usr/bin/env python3
"""Prepare compact indoor manifests from Pascal VOC 2007 train/val.

VOC supplies the high-value classroom classes without requiring the full COCO
archive.  The missing bag/bench classes remain initialized from COCO weights
and are replayed separately from the tiny local COCO128 smoke copy.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2


VOC_TO_INDOOR = {
    "person": 0,
    "chair": 1,
    "diningtable": 2,
    "sofa": 6,
}
CLASS_NAMES = (
    "person", "chair", "dining_table", "backpack",
    "handbag", "suitcase", "couch", "bench",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-modulus", type=int, default=20)
    parser.add_argument("--coco128-replay", type=Path)
    return parser.parse_args()


def read_voc_split(voc_root: Path, split: str, negative_modulus: int) -> tuple[list[dict], Counter[str]]:
    ids = (voc_root / "ImageSets" / "Main" / f"{split}.txt").read_text(
        encoding="ascii"
    ).split()
    records: list[dict] = []
    counts: Counter[str] = Counter()
    for ordinal, image_id in enumerate(ids):
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        root = ET.parse(xml_path).getroot()
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        height, width = image.shape
        boxes: list[list[float | int]] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in VOC_TO_INDOOR:
                continue
            if int(obj.findtext("difficult") or 0) != 0:
                continue
            bounds = obj.find("bndbox")
            if bounds is None:
                continue
            x1 = max(0.0, float(bounds.findtext("xmin") or 1) - 1.0)
            y1 = max(0.0, float(bounds.findtext("ymin") or 1) - 1.0)
            x2 = min(float(width), float(bounds.findtext("xmax") or width))
            y2 = min(float(height), float(bounds.findtext("ymax") or height))
            if x2 - x1 < 2.0 or y2 - y1 < 2.0:
                continue
            cls = VOC_TO_INDOOR[name]
            boxes.append([x1, y1, x2 - x1, y2 - y1, cls])
            counts[CLASS_NAMES[cls]] += 1
        keep_negative = negative_modulus > 0 and ordinal % negative_modulus == 0
        if not boxes and not keep_negative:
            continue
        records.append({
            "source": "voc2007",
            "source_id": f"voc2007:{image_id}",
            "image": str(image_path.resolve()),
            "width": width,
            "height": height,
            "boxes_xywh": boxes,
        })
    counts["images"] = len(records)
    counts["negative_images"] = sum(not row["boxes_xywh"] for row in records)
    return records, counts


def load_replay(root: Path | None, split: str) -> list[dict]:
    if root is None:
        return []
    path = root / f"manifest_{split}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = {
        "dataset": "voc2007_indoor8_with_coco128_replay",
        "formal_detection_source_is_not_coco_full": True,
        "voc_root": str(args.voc_root.resolve()),
        "coco128_replay": None if args.coco128_replay is None else str(args.coco128_replay.resolve()),
        "classes": list(CLASS_NAMES),
        "splits": {},
    }
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    for split in ("train", "val"):
        records, counts = read_voc_split(args.voc_root, split, args.negative_modulus)
        replay = load_replay(args.coco128_replay, split)
        records.extend(replay)
        ids = {str(row["source_id"]) for row in records}
        (train_ids if split == "train" else val_ids).update(ids)
        (args.output / f"manifest_{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        audit["splits"][split] = {
            "voc": dict(counts),
            "coco128_replay_images": len(replay),
            "total_images": len(records),
        }
    overlap = sorted(train_ids & val_ids)
    audit["source_id_overlap"] = len(overlap)
    if overlap:
        raise RuntimeError(f"train/val overlap: {overlap[:5]}")
    (args.output / "dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("GRAYNAV_VOC2007_INDOOR8_READY")


if __name__ == "__main__":
    main()
