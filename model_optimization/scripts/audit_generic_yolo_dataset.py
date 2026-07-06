#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a prepared gray YOLO dataset for split leakage, label validity, and grayscale consistency.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gray-sample", type=int, default=2000)
    parser.add_argument("--phash-sample", type=int, default=0, help="Optional per-split perceptual hash sample; 0 checks all images.")
    return parser.parse_args()


def load_names(data_yaml: Path | None, root: Path) -> list[str]:
    """Load class names from a data YAML or dataset manifest."""
    if data_yaml is not None and data_yaml.exists():
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        names = data.get("names", [])
        if isinstance(names, dict):
            return [str(names[idx]) for idx in sorted(names)]
        return [str(x) for x in names]
    manifest = root / "dataset_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return [str(x) for x in data.get("names", [])]
    raise FileNotFoundError("provide --data-yaml or keep dataset_manifest.json under dataset root")


def image_paths(root: Path, split: str) -> list[Path]:
    """List image files for one YOLO split."""
    image_dir = root / "images" / split
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def file_sha256(path: Path) -> str:
    """Compute a stable content hash for exact duplicate detection."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> str | None:
    """Compute a compact grayscale difference hash for obvious near-duplicate clues."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def gray_stats(paths: list[Path], sample: int) -> dict[str, Any]:
    """Verify that sampled images are exact gray replicated to three channels."""
    selected = paths if sample <= 0 else paths[:sample]
    checked = 0
    bad = []
    max_diff = 0
    for path in tqdm(selected, desc="audit gray", leave=False):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            bad.append({"path": str(path), "reason": "unreadable"})
            continue
        diff_bg = int(np.max(np.abs(img[:, :, 0].astype(np.int16) - img[:, :, 1].astype(np.int16))))
        diff_gr = int(np.max(np.abs(img[:, :, 1].astype(np.int16) - img[:, :, 2].astype(np.int16))))
        cur = max(diff_bg, diff_gr)
        max_diff = max(max_diff, cur)
        if cur != 0 and len(bad) < 20:
            bad.append({"path": str(path), "reason": "channels differ", "max_channel_diff": cur})
        checked += 1
    return {"checked": checked, "max_channel_diff": max_diff, "bad_examples": bad}


def audit_labels(root: Path, split: str, names: list[str]) -> dict[str, Any]:
    """Check YOLO label validity and collect per-class instance counts."""
    label_dir = root / "labels" / split
    images = image_paths(root, split)
    counts = Counter()
    invalid: list[dict[str, Any]] = []
    empty_images = 0
    total_lines = 0
    for image in tqdm(images, desc=f"audit labels {split}", leave=False):
        label = label_dir / f"{image.stem}.txt"
        lines = label.read_text(encoding="utf-8", errors="ignore").splitlines() if label.exists() else []
        valid_in_image = 0
        for line_no, line in enumerate(lines, start=1):
            parts = line.strip().split()
            total_lines += 1
            if len(parts) != 5:
                if len(invalid) < 50:
                    invalid.append({"file": str(label), "line": line_no, "reason": "not detection xywh format", "raw": line[:160]})
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = [float(v) for v in parts[1:]]
            except ValueError:
                if len(invalid) < 50:
                    invalid.append({"file": str(label), "line": line_no, "reason": "non numeric", "raw": line[:160]})
                continue
            if cls < 0 or cls >= len(names) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                if len(invalid) < 50:
                    invalid.append({"file": str(label), "line": line_no, "reason": "out of range", "raw": line[:160]})
                continue
            counts[names[cls]] += 1
            valid_in_image += 1
        if valid_in_image == 0:
            empty_images += 1
    return {
        "images": len(images),
        "labels": len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0,
        "empty_images": empty_images,
        "raw_label_lines": total_lines,
        "valid_instances": int(sum(counts.values())),
        "class_instances": dict(counts),
        "invalid_examples": invalid,
    }


def duplicate_audit(root: Path, phash_sample: int) -> dict[str, Any]:
    """Find exact content duplicates and identical perceptual hashes across splits."""
    by_name: dict[str, list[str]] = defaultdict(list)
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_dhash: dict[str, list[str]] = defaultdict(list)
    for split in SPLITS:
        paths = image_paths(root, split)
        selected = paths if phash_sample <= 0 else paths[:phash_sample]
        for path in tqdm(paths, desc=f"sha {split}", leave=False):
            by_name[path.name].append(split)
            by_sha[file_sha256(path)].append(f"{split}/{path.name}")
        for path in tqdm(selected, desc=f"dhash {split}", leave=False):
            value = dhash(path)
            if value is not None:
                by_dhash[value].append(f"{split}/{path.name}")
    cross_name = {k: v for k, v in by_name.items() if len(set(v)) > 1}
    cross_sha = {k: v for k, v in by_sha.items() if len({x.split('/')[0] for x in v}) > 1}
    cross_dhash = {k: v[:20] for k, v in by_dhash.items() if len({x.split('/')[0] for x in v}) > 1}
    return {
        "cross_split_same_filename_count": len(cross_name),
        "cross_split_same_filename_examples": dict(list(cross_name.items())[:20]),
        "cross_split_exact_duplicate_count": len(cross_sha),
        "cross_split_exact_duplicate_examples": dict(list(cross_sha.items())[:20]),
        "cross_split_identical_dhash_count": len(cross_dhash),
        "cross_split_identical_dhash_examples": dict(list(cross_dhash.items())[:20]),
    }


def main() -> None:
    args = parse_args()
    names = load_names(args.data_yaml, args.dataset_root)
    split_stats = {split: audit_labels(args.dataset_root, split, names) for split in SPLITS if (args.dataset_root / "images" / split).exists()}
    gray_paths = [p for split in SPLITS for p in image_paths(args.dataset_root, split)]
    report = {
        "dataset_root": str(args.dataset_root),
        "data_yaml": str(args.data_yaml) if args.data_yaml else None,
        "nc": len(names),
        "names": names,
        "splits": split_stats,
        "gray_channel_audit": gray_stats(gray_paths, args.gray_sample),
        "duplicate_audit": duplicate_audit(args.dataset_root, args.phash_sample),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit={args.out}")


if __name__ == "__main__":
    main()
