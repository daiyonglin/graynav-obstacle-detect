#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether a YOLO image dataset is exact grayscale replicated to 3 channels.")
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="Image directories to audit.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--sample-images", type=int, default=1000, help="Maximum files per root to inspect.")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--channel-tolerance", type=int, default=2, help="Allowed max per-pixel channel delta after JPEG compression.")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    """Collect image files under a root with deterministic ordering."""
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS)


def audit_image(path: Path, tolerance: int) -> dict[str, Any]:
    """Measure channel equality and gray consistency for one image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"path": str(path), "read_ok": False}
    b, g, r = cv2.split(img)
    bg = np.abs(b.astype(np.int16) - g.astype(np.int16))
    gr = np.abs(g.astype(np.int16) - r.astype(np.int16))
    br = np.abs(b.astype(np.int16) - r.astype(np.int16))
    max_delta = int(max(bg.max(initial=0), gr.max(initial=0), br.max(initial=0)))
    bad_pixels = int(((bg > tolerance) | (gr > tolerance) | (br > tolerance)).sum())
    total_pixels = int(img.shape[0] * img.shape[1])
    return {
        "path": str(path),
        "read_ok": True,
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "max_channel_delta": max_delta,
        "bad_pixel_ratio": float(bad_pixels / max(1, total_pixels)),
        "is_gray3": bool(max_delta <= tolerance and bad_pixels == 0),
    }


def summarize_root(root: Path, sample_images: int, seed: int, tolerance: int) -> dict[str, Any]:
    """Audit a directory and summarize whether it contains gray-replicated images."""
    files = collect_images(root)
    rng = random.Random(seed + sum(bytearray(str(root).encode("utf-8"))))
    sampled = list(files)
    rng.shuffle(sampled)
    sampled = sampled[:sample_images] if sample_images > 0 else sampled

    per_image = [audit_image(path, tolerance) for path in tqdm(sampled, desc=f"audit {root.name}")]
    readable = [item for item in per_image if item.get("read_ok")]
    failures = [item for item in readable if not item.get("is_gray3")]
    max_delta = max((int(item["max_channel_delta"]) for item in readable), default=0)
    max_bad_ratio = max((float(item["bad_pixel_ratio"]) for item in readable), default=0.0)
    return {
        "root": str(root),
        "total_images": len(files),
        "sampled_images": len(sampled),
        "readable_images": len(readable),
        "failed_images": len(failures),
        "max_channel_delta": max_delta,
        "max_bad_pixel_ratio": max_bad_ratio,
        "passed": bool(len(readable) == len(sampled) and not failures),
        "failure_examples": failures[:20],
    }


def main() -> None:
    args = parse_args()
    report = {
        "settings": {
            "sample_images": args.sample_images,
            "seed": args.seed,
            "channel_tolerance": args.channel_tolerance,
        },
        "roots": [summarize_root(root, args.sample_images, args.seed, args.channel_tolerance) for root in args.roots],
    }
    report["passed"] = all(item["passed"] for item in report["roots"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit_report={args.out}")
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
