#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import yaml
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a generic YOLO dataset as gray detection data.")
    parser.add_argument("--source", type=Path, required=True, help="YOLO dataset zip or extracted directory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--output-mode", choices=["gray3", "gray1"], default="gray3")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    """Normalize dataset class names for stable YAML and evaluation mapping."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def source_root(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return an extracted source directory and a temporary owner when needed."""
    if source.is_dir():
        return source, None
    if source.suffix.lower() != ".zip":
        raise SystemExit(f"source must be a zip or directory: {source}")
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(source) as zf:
        zf.extractall(tmp.name)
    return Path(tmp.name), tmp


def find_dataset_yaml(root: Path) -> Path:
    """Find a data.yaml file inside a YOLO dataset export."""
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"no dataset yaml found under {root}")
    return candidates[0]


def load_names(path: Path) -> list[str]:
    """Load and normalize class names from a YOLO data YAML."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]
    if not names:
        raise ValueError(f"dataset YAML has no names: {path}")
    return [normalize_name(str(x)) for x in names]


def find_split(root: Path, split: str) -> Path | None:
    """Find train/valid/test split directories in common YOLO layouts."""
    aliases = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}[split]
    for alias in aliases:
        for p in [root / alias, *root.rglob(alias)]:
            if p.is_dir() and (p / "images").exists():
                return p
    return None


def gray_from_image(path: Path, output_mode: str) -> Any:
    """Read one image and return either exact gray [G,G,G] or true one-channel gray."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if output_mode == "gray1":
        return gray
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def segment_to_box(coords: list[float]) -> tuple[float, float, float, float] | None:
    """Convert normalized polygon coordinates to a normalized xywh bbox."""
    if len(coords) < 4 or len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return x1 + w / 2.0, y1 + h / 2.0, w, h


def normalize_yolo_label_line(line: str, num_classes: int) -> tuple[int, str] | None:
    """Return a detection-format YOLO label line from bbox or polygon input."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    if cls < 0 or cls >= num_classes:
        return None
    vals = [float(x) for x in parts[1:]]
    if len(vals) == 4:
        cx, cy, w, h = vals
    else:
        box = segment_to_box(vals)
        if box is None:
            return None
        cx, cy, w, h = box
    if w <= 0 or h <= 0:
        return None
    cx, cy, w, h = [min(1.0, max(0.0, v)) for v in (cx, cy, w, h)]
    return cls, f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_split(split: str, split_dir: Path, out: Path, names: list[str], jpg_quality: int, output_mode: str) -> dict[str, Any]:
    """Write one split as gray images and detection-format labels."""
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    counts = [0 for _ in names]
    polygon_lines = 0
    bbox_lines = 0
    for img in tqdm(images, desc=f"write {split}"):
        dst_img = out / "images" / split / f"{img.stem}.jpg"
        dst_label = out / "labels" / split / f"{img.stem}.txt"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_img), gray_from_image(img, output_mode), [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

        lines_out: list[str] = []
        src_label = label_dir / f"{img.stem}.txt"
        if src_label.exists():
            for raw in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = raw.strip().split()
                if len(parts) > 5:
                    polygon_lines += 1
                elif len(parts) == 5:
                    bbox_lines += 1
                normalized = normalize_yolo_label_line(raw, len(names))
                if normalized is None:
                    continue
                cls, out_line = normalized
                counts[cls] += 1
                lines_out.append(out_line)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return {
        "images": len(images),
        "instances": int(sum(counts)),
        "bbox_label_lines": bbox_lines,
        "polygon_label_lines_converted": polygon_lines,
        "class_instances": {names[idx]: value for idx, value in enumerate(counts)},
    }


def write_yaml(out: Path, names: list[str], output_mode: str) -> Path:
    """Write a canonical Ultralytics data YAML."""
    body = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    path = out / "gray_dataset.yaml"
    channels = "channels: 1\n" if output_mode == "gray1" else ""
    path.write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n{channels}names:\n{body}\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.out.exists() and args.overwrite:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    root, tmp = source_root(args.source)
    try:
        yaml_path = find_dataset_yaml(root)
        names = load_names(yaml_path)
        splits: dict[str, Path] = {}
        for split in ["train", "val", "test"]:
            d = find_split(root, split)
            if d is not None:
                splits[split] = d
        if "train" not in splits or "val" not in splits:
            raise FileNotFoundError(f"expected train and valid/val splits under {root}")
        splits.setdefault("test", splits["val"])
        stats = {split: write_split(split, d, args.out, names, args.jpg_quality, args.output_mode) for split, d in splits.items()}
        out_yaml = write_yaml(args.out, names, args.output_mode)
        manifest = {
            "dataset": "Generic-Obstacle-Gray-YOLO",
            "source": str(args.source),
            "source_yaml": str(yaml_path),
            "names": names,
            "nc": len(names),
            "input_mode": args.output_mode,
            "gray_conversion": "cv2 BGR2GRAY" if args.output_mode == "gray1" else "cv2 BGR2GRAY then GRAY2BGR",
            "label_conversion": "YOLO bbox preserved; YOLO polygon converted to bbox by min/max coordinates",
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
