#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Callable, List

import cv2
import numpy as np
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="YOLO dataset root")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--quality", type=int, default=94)
    return parser.parse_args()


def motion_blur(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= float(k)
    return cv2.filter2D(gray, -1, kernel)


def gaussian_noise(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(4.0, 16.0)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0.0, sigma, gray.shape)
    return np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def gamma_shift(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma = rng.uniform(0.55, 1.85)
    table = np.array([(i / 255.0) ** gamma * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def exposure_shift(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.55, 1.45)
    beta = rng.uniform(-38.0, 38.0)
    return np.clip(gray.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def low_contrast(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    mean = float(np.mean(gray))
    scale = rng.uniform(0.35, 0.75)
    return np.clip((gray.astype(np.float32) - mean) * scale + mean, 0, 255).astype(np.uint8)


def overexpose_patch(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    out = gray.copy()
    h, w = out.shape[:2]
    pw = rng.randint(max(8, w // 8), max(9, w // 3))
    ph = rng.randint(max(8, h // 8), max(9, h // 3))
    x = rng.randint(0, max(0, w - pw))
    y = rng.randint(0, max(0, h - ph))
    out[y : y + ph, x : x + pw] = np.clip(out[y : y + ph, x : x + pw].astype(np.float32) + rng.uniform(35, 95), 0, 255)
    return out.astype(np.uint8)


AUGS: List[Callable[[np.ndarray, random.Random], np.ndarray]] = [
    motion_blur,
    gaussian_noise,
    gamma_shift,
    exposure_shift,
    low_contrast,
    overexpose_patch,
]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in VALID_EXTS and "_aug" not in p.stem]
    if not images:
        raise SystemExit(f"No source images found in {image_dir}")

    for img_path in tqdm(images, desc="augment"):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for copy_idx in range(args.copies):
            aug = img.copy()
            for fn in rng.sample(AUGS, rng.randint(1, 3)):
                aug = fn(aug, rng)
            bgr = cv2.cvtColor(aug, cv2.COLOR_GRAY2BGR)
            out_stem = f"{img_path.stem}_aug{copy_idx}"
            out_img = image_dir / f"{out_stem}.jpg"
            out_lbl = label_dir / f"{out_stem}.txt"
            cv2.imwrite(str(out_img), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            shutil.copy2(label_path, out_lbl)


if __name__ == "__main__":
    main()

