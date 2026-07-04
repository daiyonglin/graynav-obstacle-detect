#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable


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

COCO_NAME_TO_SEMANTIC = {
    "person": 0,
    "chair": 1,
    "bench": 1,
    "dining table": 2,
    "couch": 3,
    "bed": 3,
    "backpack": 4,
    "handbag": 4,
    "suitcase": 4,
    "bottle": 5,
    "cup": 5,
    "bowl": 5,
    "book": 5,
    "laptop": 5,
    "keyboard": 5,
    "mouse": 5,
    "cell phone": 5,
    "remote": 5,
    "bicycle": 6,
    "motorcycle": 6,
    "car": 6,
    "bus": 6,
    "truck": 6,
    "potted plant": 7,
    "umbrella": 7,
    "sports ball": 7,
    "skateboard": 7,
    "tv": 7,
    "microwave": 7,
    "oven": 7,
    "sink": 7,
    "refrigerator": 7,
    "toilet": 7,
    "vase": 7,
}


def normalize_name(name: str) -> str:
    """Normalize dataset/model class names before GrayNav-Obstacle8 mapping."""
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def category_id_to_semantic(categories: Iterable[dict]) -> dict[int, int]:
    """Build a COCO category-id to GrayNav semantic-id mapping."""
    out: dict[int, int] = {}
    for cat in categories:
        sem = COCO_NAME_TO_SEMANTIC.get(normalize_name(cat.get("name", "")))
        if sem is not None:
            out[int(cat["id"])] = sem
    return out


def yolo_class_to_semantic(names: dict | list) -> dict[int, int]:
    """Map Ultralytics model class ids to GrayNav semantic ids by class name."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    out: dict[int, int] = {}
    for idx, name in items:
        normalized = normalize_name(name)
        if normalized in [normalize_name(x) for x in SEMANTIC_NAMES]:
            sem = [normalize_name(x) for x in SEMANTIC_NAMES].index(normalized)
        else:
            sem = COCO_NAME_TO_SEMANTIC.get(normalized)
        if sem is not None:
            out[int(idx)] = sem
    return out


def semantic_categories() -> list[dict]:
    """Return COCO-style category records for GrayNav-Obstacle8 evaluation."""
    return [{"id": idx + 1, "name": name, "supercategory": "graynav"} for idx, name in enumerate(SEMANTIC_NAMES)]
