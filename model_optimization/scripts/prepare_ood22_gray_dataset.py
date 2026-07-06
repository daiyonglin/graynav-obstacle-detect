#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import yaml
from tqdm import tqdm

from graynav_ood22 import OOD22_NAMES, normalize_name


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a gray [G,G,G] OOD22 YOLO dataset from Roboflow/YOLO exports.")
    parser.add_argument("--source", type=Path, required=True, help="Roboflow YOLOv8 zip or extracted dataset directory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-ratio", default="0.8,0.1,0.1", help="Fallback train,val,test ratio if no split dirs exist.")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a Roboflow/Ultralytics dataset YAML."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]
    data["names"] = [normalize_name(str(x)) for x in names]
    return data


def source_root(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return an extracted source directory and the temporary owner if needed."""
    if source.is_dir():
        return source, None
    if source.suffix.lower() != ".zip":
        raise SystemExit(f"source must be a directory or zip: {source}")
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(source) as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name), tmp


def find_dataset_yaml(root: Path) -> Path | None:
    """Find the most likely dataset YAML from a Roboflow export."""
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("*.yaml"))
    candidates = [p for p in candidates if p.name not in {"graynav_ood22_gray.yaml"}]
    return candidates[0] if candidates else None


def find_split_dir(root: Path, split: str) -> Path | None:
    """Locate a split directory using common train/valid/val/test names."""
    for alias in SPLIT_ALIASES[split]:
        direct = root / alias
        if direct.exists():
            return direct
        for p in root.rglob(alias):
            if p.is_dir() and ((p / "images").exists() or any(x.suffix.lower() in IMAGE_EXTS for x in p.rglob("*"))):
                return p
    return None


def image_label_pairs(split_dir: Path) -> list[tuple[Path, Path | None]]:
    """Collect image paths and matching YOLO label paths from one split."""
    image_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    label_dir = split_dir / "labels" if (split_dir / "labels").exists() else split_dir
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    pairs: list[tuple[Path, Path | None]] = []
    for img in images:
        label = label_dir / f"{img.stem}.txt"
        pairs.append((img, label if label.exists() else None))
    return pairs


def collect_all_pairs(root: Path) -> list[tuple[Path, Path | None]]:
    """Collect all image/label pairs for fallback split generation."""
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    pairs: list[tuple[Path, Path | None]] = []
    for img in images:
        possible = list(root.rglob(f"{img.stem}.txt"))
        label = possible[0] if possible else None
        pairs.append((img, label))
    return pairs


def gray3_from_image(path: Path) -> Any:
    """Read one image and return exact three-channel grayscale BGR."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def remap_label(src: Path | None, dst: Path, src_names: list[str]) -> dict[str, int]:
    """Copy a YOLO label file while remapping class ids to canonical OOD22 ids."""
    counts = {name: 0 for name in OOD22_NAMES}
    lines_out: list[str] = []
    if src and src.exists():
        for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            old_id = int(float(parts[0]))
            if not (0 <= old_id < len(src_names)):
                continue
            name = normalize_name(src_names[old_id])
            if name not in OOD22_NAMES:
                continue
            new_id = OOD22_NAMES.index(name)
            lines_out.append(" ".join([str(new_id)] + parts[1:]))
            counts[name] += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return counts


def write_split(split: str, pairs: list[tuple[Path, Path | None]], out: Path, src_names: list[str], jpg_quality: int) -> dict[str, Any]:
    """Write gray images and remapped labels for a split."""
    counts = {name: 0 for name in OOD22_NAMES}
    for img, label in tqdm(pairs, desc=f"write {split}"):
        dst_img = out / "images" / split / f"{img.stem}.jpg"
        dst_lbl = out / "labels" / split / f"{img.stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_img), gray3_from_image(img), [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        per = remap_label(label, dst_lbl, src_names)
        counts = {name: counts[name] + per[name] for name in OOD22_NAMES}
    return {"images": len(pairs), "instances": int(sum(counts.values())), "class_instances": counts}


def split_pairs(root: Path, seed: int, ratio: str) -> dict[str, list[tuple[Path, Path | None]]]:
    """Use official split directories if present; otherwise create a fixed random split."""
    official: dict[str, list[tuple[Path, Path | None]]] = {}
    for split in ["train", "val", "test"]:
        d = find_split_dir(root, split)
        if d:
            official[split] = image_label_pairs(d)
    if official.get("train") and official.get("val"):
        official.setdefault("test", [])
        return official

    pairs = collect_all_pairs(root)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    a, b, c = [float(x) for x in ratio.split(",")]
    total = max(1e-9, a + b + c)
    n = len(pairs)
    n_train = int(n * a / total)
    n_val = int(n * b / total)
    return {"train": pairs[:n_train], "val": pairs[n_train : n_train + n_val], "test": pairs[n_train + n_val :]}


def write_dataset_yaml(out: Path) -> Path:
    """Write the canonical Ultralytics YAML for OOD22."""
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(OOD22_NAMES))
    path = out / "graynav_ood22_gray.yaml"
    path.write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    root, tmp = source_root(args.source)
    try:
        yaml_path = find_dataset_yaml(root)
        yaml_data = load_yaml(yaml_path) if yaml_path else {}
        src_names = yaml_data.get("names") or OOD22_NAMES
        missing = [name for name in OOD22_NAMES if name not in set(src_names)]
        if missing:
            print(f"WARNING: source YAML missing OOD22 names: {missing}")

        splits = split_pairs(root, args.seed, args.split_ratio)
        stats = {split: write_split(split, pairs, args.out, src_names, args.jpg_quality) for split, pairs in splits.items()}
        out_yaml = write_dataset_yaml(args.out)
        manifest = {
            "dataset": "GrayNav-OOD22-Gray",
            "source": str(args.source),
            "source_yaml": str(yaml_path) if yaml_path else None,
            "source_names": src_names,
            "canonical_names": OOD22_NAMES,
            "seed": args.seed,
            "gray_conversion": "cv2 BGR2GRAY then GRAY2BGR",
            "yaml": str(out_yaml),
            "splits": stats,
        }
        (args.out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
