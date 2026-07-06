#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--calib-num", type=int, default=80)
    parser.add_argument("--eval-num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--input-mode", choices=["gray1", "gray3", "color3"], default="gray3")
    return parser.parse_args()


def letterbox_gray_to_3ch(gray: np.ndarray, imgsz: int, color: int = 114) -> Tuple[np.ndarray, float, int, int]:
    h, w = gray.shape[:2]
    scale = min(imgsz / float(h), imgsz / float(w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz), color, dtype=np.uint8)
    dx = (imgsz - new_w) // 2
    dy = (imgsz - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    img3 = np.stack([canvas, canvas, canvas], axis=2)
    return img3, scale, dx, dy


def letterbox_gray_to_1ch(gray: np.ndarray, imgsz: int, color: int = 114) -> Tuple[np.ndarray, float, int, int]:
    """Letterbox a grayscale image as a true one-channel model input."""
    h, w = gray.shape[:2]
    scale = min(imgsz / float(h), imgsz / float(w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz), color, dtype=np.uint8)
    dx = (imgsz - new_w) // 2
    dy = (imgsz - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas[None, :, :], scale, dx, dy


def letterbox_color_to_3ch(img: np.ndarray, imgsz: int, color: int = 114) -> np.ndarray:
    """Letterbox an existing 3-channel input without collapsing GMFE channels."""
    h, w = img.shape[:2]
    scale = min(imgsz / float(h), imgsz / float(w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    dx = (imgsz - new_w) // 2
    dy = (imgsz - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w] = resized
    return canvas


def preprocess(path: Path, imgsz: int, input_mode: str) -> np.ndarray:
    """Build an A1 calibration tensor matching the trained model input encoding."""
    if input_mode == "color3":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {path}")
        img3 = letterbox_color_to_3ch(img, imgsz)
    elif input_mode == "gray3":
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"failed to read {path}")
        img3, _, _, _ = letterbox_gray_to_3ch(gray, imgsz)
        x = img3.astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[None, ...]
    else:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"failed to read {path}")
        img1, _, _, _ = letterbox_gray_to_1ch(gray, imgsz)
        return (img1.astype(np.float32) / 255.0)[None, ...]


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))


def main() -> None:
    args = parse_args()
    files = [p for p in args.input_dir.rglob("*") if p.suffix.lower() in VALID_EXTS]
    need = args.calib_num + args.eval_num
    if len(files) < need:
        raise SystemExit(f"Need at least {need} images, found {len(files)}")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    selected = files[:need]

    datasets_root = args.output_root / "datasets"
    calib_dir = datasets_root / "calibrate_datasets"
    eval_dir = datasets_root / "evaluate_datasets"
    calib_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    for p in list(calib_dir.glob("*.npy")) + list(eval_dir.glob("*.npy")):
        p.unlink()

    for i, path in enumerate(selected[: args.calib_num]):
        np.save(calib_dir / f"calib_{i:04d}.npy", preprocess(path, args.imgsz, args.input_mode))
    for i, path in enumerate(selected[args.calib_num :]):
        np.save(eval_dir / f"eval_{i:04d}.npy", preprocess(path, args.imgsz, args.input_mode))

    zip_path = args.output_root / "datasets.zip"
    if zip_path.exists():
        zip_path.unlink()
    zip_dir(datasets_root, zip_path)
    print(f"calib={args.calib_num} eval={args.eval_num}")
    print(f"zip={zip_path}")


if __name__ == "__main__":
    main()
