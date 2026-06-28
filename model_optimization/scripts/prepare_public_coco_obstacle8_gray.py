#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from tqdm import tqdm


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
    "pedestrian": 0,
    "man": 0,
    "woman": 0,
    "child": 0,
    "chair": 1,
    "bench": 1,
    "seat": 1,
    "stool": 1,
    "armchair": 1,
    "wheelchair": 1,
    "dining table": 2,
    "table": 2,
    "desk": 2,
    "coffee table": 2,
    "sofa": 3,
    "couch": 3,
    "bed": 3,
    "backpack": 4,
    "handbag": 4,
    "suitcase": 4,
    "bag": 4,
    "briefcase": 4,
    "bottle": 5,
    "cup": 5,
    "book": 5,
    "laptop": 5,
    "keyboard": 5,
    "cell phone": 5,
    "mobile phone": 5,
    "remote": 5,
    "remote control": 5,
    "bicycle": 6,
    "motorcycle": 6,
    "motorbike": 6,
    "car": 6,
    "vehicle": 6,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--min-box", type=float, default=4.0)
    parser.add_argument("--empty-keep-prob", type=float, default=0.02)
    parser.add_argument("--copy-rgb", action="store_true", help="Copy source RGB instead of writing gray 3-channel JPG")
    parser.add_argument("--prefix", default="", help="Prefix output names to merge multiple datasets safely")
    parser.add_argument("--unknown-policy", choices=["ignore", "generic"], default="ignore")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def category_map(coco: dict, unknown_policy: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for cat in coco.get("categories", []):
        name = normalize_name(str(cat.get("name", "")))
        if name in NAME_TO_SEMANTIC:
            out[int(cat["id"])] = NAME_TO_SEMANTIC[name]
        elif unknown_policy == "generic":
            out[int(cat["id"])] = 7
    return out


def grouped_annotations(
    coco: dict,
    cat_to_sem: Dict[int, int],
    min_box: float,
) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    grouped: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w < min_box or h < min_box:
            continue
        sem = cat_to_sem.get(int(ann["category_id"]))
        if sem is None:
            continue
        grouped.setdefault(int(ann["image_id"]), []).append((sem, (x, y, w, h)))
    return grouped


def find_image(images_root: Path, file_name: str) -> Optional[Path]:
    direct = images_root / file_name
    if direct.exists():
        return direct
    by_name = list(images_root.rglob(Path(file_name).name))
    return by_name[0] if by_name else None


def write_gray_3ch(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def write_label(
    dst: Path,
    boxes: Iterable[Tuple[int, Tuple[float, float, float, float]]],
    width: int,
    height: int,
) -> None:
    lines: List[str] = []
    for sem, (x, y, w, h) in boxes:
        cx = (x + 0.5 * w) / max(1.0, float(width))
        cy = (y + 0.5 * h) / max(1.0, float(height))
        nw = w / max(1.0, float(width))
        nh = h / max(1.0, float(height))
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        nw = min(1.0, max(0.0, nw))
        nh = min(1.0, max(0.0, nh))
        if nw > 0.0 and nh > 0.0:
            lines.append(f"{sem} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_dataset_yaml(out: Path) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(SEMANTIC_NAMES))
    text = f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    (out / "obstacle8-gray.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in coco.get("images", [])}
    anns = grouped_annotations(coco, category_map(coco, args.unknown_policy), args.min_box)
    ids = list(images.keys())
    random.Random(args.seed).shuffle(ids)
    if args.limit > 0:
        ids = ids[: args.limit]

    counts = [0 for _ in SEMANTIC_NAMES]
    kept = 0
    skipped = 0
    for image_id in tqdm(ids, desc=f"convert {args.split}"):
        rec = images[image_id]
        boxes = anns.get(image_id, [])
        if not boxes and random.random() > args.empty_keep_prob:
            skipped += 1
            continue
        src = find_image(args.images, str(rec["file_name"]))
        if src is None:
            skipped += 1
            continue
        stem = f"{args.prefix}{Path(rec['file_name']).stem}"
        dst_img = args.out / "images" / args.split / f"{stem}.jpg"
        dst_lbl = args.out / "labels" / args.split / f"{stem}.txt"
        if args.copy_rgb:
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_img)
        elif not write_gray_3ch(src, dst_img):
            skipped += 1
            continue
        write_label(dst_lbl, boxes, int(rec["width"]), int(rec["height"]))
        for sem, _ in boxes:
            counts[sem] += 1
        kept += 1

    write_dataset_yaml(args.out)
    print(f"out={args.out}")
    print(f"kept={kept} skipped={skipped}")
    for i, name in enumerate(SEMANTIC_NAMES):
        print(f"{i}: {name}: {counts[i]}")


if __name__ == "__main__":
    main()
