#!/usr/bin/env python3
"""
Build an 8-class grayscale obstacle-detection dataset from public COCO-style
annotations.

The script does not download datasets. Point it at already downloaded public
datasets such as COCO, Objects365, LVIS, or exported Open Images COCO-format
annotations, and it writes a YOLO dataset with the semantic classes used by the
board-side postprocess.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2


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
    "seat": 1,
    "stool": 1,
    "table": 2,
    "desk": 2,
    "dining table": 2,
    "couch": 3,
    "sofa": 3,
    "bed": 3,
    "backpack": 4,
    "handbag": 4,
    "suitcase": 4,
    "bag": 4,
    "bottle": 5,
    "cup": 5,
    "book": 5,
    "laptop": 5,
    "keyboard": 5,
    "cell phone": 5,
    "mobile phone": 5,
    "remote": 5,
    "bicycle": 6,
    "motorcycle": 6,
    "motorbike": 6,
    "car": 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Dataset image directory")
    parser.add_argument("--annotations", type=Path, required=True, help="COCO-format annotation json")
    parser.add_argument("--out", type=Path, required=True, help="Output YOLO dataset directory")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink/write converted images")
    parser.add_argument("--gray", action="store_true", default=True, help="Write grayscale 3-channel jpgs")
    parser.add_argument("--empty-keep-prob", type=float, default=0.03, help="Keep a small number of empty scenes")
    return parser.parse_args()


def category_map(coco: dict) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for cat in coco.get("categories", []):
        cid = int(cat["id"])
        name = str(cat.get("name", "")).strip().lower()
        out[cid] = NAME_TO_SEMANTIC.get(name, 7)
    return out


def image_records(coco: dict) -> Dict[int, dict]:
    return {int(img["id"]): img for img in coco.get("images", [])}


def grouped_annotations(coco: dict, cat_to_sem: Dict[int, int]) -> Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]]:
    grouped: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w < 4 or h < 4:
            continue
        sem = cat_to_sem.get(int(ann["category_id"]), 7)
        grouped.setdefault(int(ann["image_id"]), []).append((sem, (x, y, w, h)))
    return grouped


def find_image(images_root: Path, file_name: str) -> Optional[Path]:
    p = images_root / file_name
    if p.exists():
        return p
    matches = list(images_root.rglob(Path(file_name).name))
    return matches[0] if matches else None


def write_gray_image(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def write_label(path: Path, boxes: Iterable[Tuple[int, Tuple[float, float, float, float]]], w: int, h: int) -> None:
    lines: List[str] = []
    for sem, (x, y, bw, bh) in boxes:
        cx = (x + bw * 0.5) / max(1.0, float(w))
        cy = (y + bh * 0.5) / max(1.0, float(h))
        nw = bw / max(1.0, float(w))
        nh = bh / max(1.0, float(h))
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        nw = min(1.0, max(0.0, nw))
        nh = min(1.0, max(0.0, nh))
        lines.append(f"{sem} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml(out: Path) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(SEMANTIC_NAMES))
    text = f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    (out / "obstacle8-gray.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    coco = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = image_records(coco)
    anns = grouped_annotations(coco, category_map(coco))
    ids = list(images.keys())
    random.Random(20260616).shuffle(ids)
    if args.limit > 0:
        ids = ids[: args.limit]

    kept = 0
    skipped = 0
    for image_id in ids:
        rec = images[image_id]
        boxes = anns.get(image_id, [])
        if not boxes and random.random() > args.empty_keep_prob:
            skipped += 1
            continue
        src = find_image(args.images, rec["file_name"])
        if src is None:
            skipped += 1
            continue
        stem = Path(rec["file_name"]).stem
        dst_img = args.out / "images" / args.split / f"{stem}.jpg"
        dst_lbl = args.out / "labels" / args.split / f"{stem}.txt"
        if args.gray:
            if not write_gray_image(src, dst_img):
                skipped += 1
                continue
        else:
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_img)
        write_label(dst_lbl, boxes, int(rec["width"]), int(rec["height"]))
        kept += 1

    write_yaml(args.out)
    print(f"kept={kept} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()

