#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from graynav_dce import register_ultralytics_dce


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8n-family model on a prepared gray-copy obstacle dataset.")
    parser.add_argument("--model-yaml", type=Path, default=Path("configs/graynav_dce_yolov8n.yaml"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/graynav_dce_state"))
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="graynav_dce_yolov8n_ood22")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", default="64")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--cache", nargs="?", const="ram", default="none", choices=["none", "ram", "disk"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="AdamW", choices=["SGD", "Adam", "AdamW", "auto"])
    parser.add_argument("--lr0", type=float, default=0.0003)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N model layers for controlled fine-tuning.")
    parser.add_argument("--mosaic", type=float, default=0.25)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--final-weights-file", type=Path)
    return parser.parse_args()


def parse_batch(value: str) -> int | float:
    """Support both fixed integer batch and Ultralytics auto-batch fractions."""
    text = str(value).strip()
    return float(text) if "." in text else int(text)


def parse_cache(value: str) -> bool | str:
    """Map CLI cache mode to Ultralytics train argument."""
    if value in ("", "none", "False", "false", "0"):
        return False
    return value


def best_path(model: YOLO, project: str, name: str) -> Path:
    """Locate the best.pt emitted by Ultralytics."""
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        best = Path(save_dir) / "weights" / "best.pt"
        if best.exists():
            return best
    candidates = list(Path(project).rglob(f"{name}/weights/best.pt")) if Path(project).exists() else []
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise RuntimeError(f"best.pt not found for {project}/{name}, save_dir={save_dir}")


def dataset_nc(data_yaml: Path) -> int:
    """Read class count from an Ultralytics data YAML."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return len(names)
    return len(list(names))


def materialize_model_yaml(model_yaml: Path, nc: int, out_dir: Path) -> Path:
    """Write a temporary model YAML with nc matched to the active dataset."""
    data = yaml.safe_load(model_yaml.read_text(encoding="utf-8")) or {}
    data["nc"] = int(nc)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_yaml.stem}_nc{nc}.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"base weights not found: {args.weights}")
    register_ultralytics_dce()

    nc = dataset_nc(args.data)
    active_yaml = materialize_model_yaml(args.model_yaml, nc, args.out_dir)
    model = YOLO(str(active_yaml))
    model.load(str(args.weights))

    train_kwargs: dict[str, Any] = dict(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=parse_batch(args.batch),
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        amp=True,
        seed=args.seed,
        cache=parse_cache(str(args.cache)),
        pretrained=False,
        single_cls=False,
        rect=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        degrees=0.0,
        translate=0.08,
        scale=0.50,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=args.mosaic,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=args.close_mosaic,
        plots=True,
        val=True,
        exist_ok=True,
        verbose=args.verbose,
    )
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    print("train_graynav_dce_yolov8n")
    print(f"model_yaml={args.model_yaml}")
    print(f"active_model_yaml={active_yaml}")
    print(f"dataset_nc={nc}")
    print(f"data={args.data}")
    print(f"weights_init={args.weights}")
    print(f"epochs={args.epochs} batch={args.batch} imgsz={args.imgsz}")
    print(f"optimizer={args.optimizer} lr0={args.lr0} lrf={args.lrf} freeze={args.freeze}")
    model.train(**train_kwargs)
    best = best_path(model, args.project, args.name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    final_file = args.final_weights_file or args.out_dir / "M1_dce_final_weights.txt"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    final_file.write_text(str(best), encoding="utf-8")
    print(f"final weights: {best}")
    print(f"final weights file: {final_file}")


if __name__ == "__main__":
    main()
