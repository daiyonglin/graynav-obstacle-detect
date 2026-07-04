#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml
from ultralytics import YOLO


STAGE_ORDER = ["warmup", "adapt", "stabilize"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training_stages.yaml"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", default="128", help="Integer batch size or Ultralytics auto-batch fraction, e.g. 0.85")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project")
    parser.add_argument("--name-prefix")
    parser.add_argument("--base-weights")
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER)
    parser.add_argument("--cache", nargs="?", const="ram", default="none", choices=["none", "ram", "disk"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--single-stage", action="store_true", help="Train only the first requested stage from its input weights")
    parser.add_argument("--final-weights-file", type=Path, help="Optional file that receives the final best.pt path.")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_batch(value: str) -> int | float:
    """Preserve Ultralytics support for fixed batch sizes and auto-batch fractions."""
    text = str(value).strip()
    if "." in text:
        return float(text)
    return int(text)


def parse_cache(value: str) -> bool | str:
    """Map a CLI cache mode to the Ultralytics train() cache argument."""
    if value in ("", "none", "False", "false", "0"):
        return False
    return value


def stage_best_path(model: YOLO, project: str, run_name: str) -> Path:
    """Return the actual best.pt path from Ultralytics, with fallback search."""
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        best = Path(save_dir) / "weights" / "best.pt"
        if best.exists():
            return best

    fallback_roots = [Path(project), Path.cwd()]
    candidates: list[Path] = []
    for root in fallback_roots:
        if root.exists():
            candidates.extend(root.rglob(f"{run_name}/weights/best.pt"))
    candidates = [p for p in candidates if p.exists()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    expected = Path(project) / run_name / "weights" / "best.pt"
    raise RuntimeError(f"stage completed but best.pt not found; expected={expected}, save_dir={save_dir}")


def train_one_stage(
    stage_name: str,
    weights: Path | str,
    data: Path,
    cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    weights_path = Path(str(weights))
    if not weights_path.exists():
        raise FileNotFoundError(f"training weights not found, refusing Ultralytics auto-download: {weights_path}")
    model_cfg = cfg["model"]
    stage_cfg = cfg["stages"][stage_name]
    project = args.project or model_cfg.get("project", "runs/detect")
    name_prefix = args.name_prefix or model_cfg.get("name_prefix", "obstacle8_gray_yolov8n")
    workers = args.workers if args.workers is not None else int(model_cfg.get("workers", 8))
    run_name = f"{name_prefix}_{stage_name}"

    train_kwargs: Dict[str, Any] = dict(
        data=str(data),
        imgsz=int(model_cfg.get("imgsz", 384)),
        epochs=int(stage_cfg["epochs"]),
        batch=parse_batch(args.batch),
        device=args.device,
        workers=workers,
        project=project,
        name=run_name,
        patience=int(model_cfg.get("patience", 25)),
        pretrained=True,
        single_cls=False,
        rect=False,
        cos_lr=True,
        optimizer="auto",
        amp=True,
        cache=parse_cache(str(args.cache)),
        verbose=bool(args.verbose),
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=float(stage_cfg.get("hsv_v", 0.25)),
        degrees=2.0,
        translate=0.08,
        scale=0.45,
        shear=0.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=float(stage_cfg.get("mosaic", 0.0)),
        mixup=float(stage_cfg.get("mixup", 0.0)),
        copy_paste=0.0,
        erasing=0.15,
        close_mosaic=int(stage_cfg.get("close_mosaic", 0)),
        lr0=float(stage_cfg["lr0"]),
        lrf=float(stage_cfg["lrf"]),
        warmup_epochs=3.0,
        plots=True,
        val=True,
        exist_ok=True,
    )
    freeze = int(stage_cfg.get("freeze", 0))
    if freeze > 0:
        train_kwargs["freeze"] = freeze

    print("=" * 90)
    print(f"stage={stage_name}")
    print(f"weights={weights}")
    print(f"run={Path(project) / run_name}")
    print(f"epochs={train_kwargs['epochs']} batch={train_kwargs['batch']} freeze={freeze}")
    print("=" * 90)

    model = YOLO(str(weights_path))
    model.train(**train_kwargs)
    return stage_best_path(model, project, run_name)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    weights: Path | str = args.base_weights or cfg["model"].get("base_weights", "yolov8n.pt")

    requested: List[str] = args.stages
    if args.single_stage:
        requested = requested[:1]

    for stage in requested:
        weights = train_one_stage(stage, weights, args.data, cfg, args)

    print("=" * 90)
    print("final weights:", weights)
    if args.final_weights_file:
        args.final_weights_file.parent.mkdir(parents=True, exist_ok=True)
        args.final_weights_file.write_text(str(weights), encoding="utf-8")
        print("final weights file:", args.final_weights_file)


if __name__ == "__main__":
    main()
