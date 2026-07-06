from __future__ import annotations

from typing import Mapping


OOD22_NAMES = [
    "person",
    "car",
    "tree",
    "spherical_roadblock",
    "warning_column",
    "waste_container",
    "street_light",
    "fire_hydrant",
    "traffic_light",
    "stop_sign",
    "pole",
    "bench",
    "curb",
    "stairs",
    "bicycle",
    "motorcycle",
    "dog",
    "bus",
    "truck",
    "train",
    "bus_stop",
    "crutch",
]

OOD22_NAME_TO_ID = {name: idx for idx, name in enumerate(OOD22_NAMES)}

COCO_TO_OOD22 = {
    "person": "person",
    "car": "car",
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
    "bench": "bench",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "dog": "dog",
    "bus": "bus",
    "truck": "truck",
    "train": "train",
}

COCO_OVERLAP_OOD22 = [COCO_TO_OOD22[name] for name in COCO_TO_OOD22]
NON_COCO_OOD22 = [name for name in OOD22_NAMES if name not in set(COCO_TO_OOD22.values())]


def normalize_name(name: str) -> str:
    """Normalize class names from Roboflow/YOLO YAML files."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def remap_prediction_class(model_names: Mapping[int, str] | list[str], cls_id: int) -> int | None:
    """Map either COCO80 or OOD22 model class ids to OOD22 ids."""
    if isinstance(model_names, Mapping):
        name = model_names.get(int(cls_id))
    else:
        name = model_names[int(cls_id)] if 0 <= int(cls_id) < len(model_names) else None
    if name is None:
        return None
    raw = str(name).strip()
    normalized = normalize_name(raw)
    if normalized in OOD22_NAME_TO_ID:
        return OOD22_NAME_TO_ID[normalized]
    mapped = COCO_TO_OOD22.get(raw.lower())
    if mapped is None:
        mapped = COCO_TO_OOD22.get(raw.lower().replace("_", " "))
    if mapped is None:
        return None
    return OOD22_NAME_TO_ID[mapped]
