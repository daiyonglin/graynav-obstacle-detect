#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
from tqdm import tqdm
from ultralytics import YOLO


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

COCO_TO_SEMANTIC: Dict[int, int] = {
    0: 0,
    13: 1,
    56: 1,
    60: 2,
    57: 3,
    59: 3,
    24: 4,
    26: 4,
    28: 4,
    39: 5,
    41: 5,
    63: 5,
    65: 5,
    66: 5,
    67: 5,
    73: 5,
    1: 6,
    2: 6,
    3: 6,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default="yolov8l.pt")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="0")
    parser.add_argument("--gray", action="store_true", default=True)
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS]


def write_gray(src: Path, dst: Path) -> bool:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))


def main() -> None:
    args = parse_args()
    model = YOLO(args.teacher)
    images = list_images(args.source)
    out_img_dir = args.out / "images" / args.split
    out_lbl_dir = args.out / "labels" / args.split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    for path in tqdm(images, desc="teacher pseudo-label"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        lines: List[str] = []
        if result.boxes is not None:
            for box in result.boxes:
                raw_cls = int(box.cls.item())
                if raw_cls not in COCO_TO_SEMANTIC:
                    continue
                sem = COCO_TO_SEMANTIC[raw_cls]
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if bw < 6.0 or bh < 6.0:
                    continue
                cx = (x1 + 0.5 * bw) / max(1.0, float(w))
                cy = (y1 + 0.5 * bh) / max(1.0, float(h))
                lines.append(f"{sem} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        if not lines:
            continue
        stem = path.stem
        dst_img = out_img_dir / f"pseudo_{stem}.jpg"
        dst_lbl = out_lbl_dir / f"pseudo_{stem}.txt"
        if args.gray:
            if not write_gray(path, dst_img):
                continue
        else:
            shutil.copy2(path, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

