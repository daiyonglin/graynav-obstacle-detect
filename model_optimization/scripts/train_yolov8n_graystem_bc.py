#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from ultralytics import YOLO


STAGE_ORDER = ["warmup", "adapt", "stabilize"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8n checkpoint with a constrained GrayStem-BC first layer.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", default="0.85")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name-prefix", default="M2_graystem_bc_yolov8n80")
    parser.add_argument("--cache", nargs="?", const="ram", default="none", choices=["none", "ram", "disk"])
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER, default=STAGE_ORDER)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--final-weights-file", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load the stage schedule used by the constrained GrayStem trainer."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_batch(value: str) -> int | float:
    """Pass integer batch sizes and Ultralytics auto-batch fractions through correctly."""
    text = str(value).strip()
    return float(text) if "." in text else int(text)


def parse_cache(value: str) -> bool | str:
    """Map CLI cache strings to the values expected by Ultralytics."""
    if value in ("", "none", "False", "false", "0"):
        return False
    return value


def first_conv_from_module(module: torch.nn.Module) -> torch.nn.Conv2d:
    """Find the YOLOv8 first Conv2d inside the Ultralytics Conv wrapper."""
    model_layers = getattr(module, "model", None)
    if model_layers is None or len(model_layers) == 0:
        raise TypeError("model has no layer list")
    wrapper = model_layers[0]
    conv = getattr(wrapper, "conv", None)
    if not isinstance(conv, torch.nn.Conv2d):
        raise TypeError(f"unexpected first layer: {type(wrapper)} / {type(conv)}")
    if conv.weight.ndim != 4 or conv.weight.shape[1] != 3:
        raise ValueError(f"expected first conv [out,3,k,k], got {tuple(conv.weight.shape)}")
    return conv


def tie_first_conv_to_graystem_bc(module: torch.nn.Module) -> None:
    """Project the first conv to the GrayStem-BC subspace while preserving gray-copy behavior."""
    conv = first_conv_from_module(module)
    with torch.no_grad():
        folded = conv.weight.data.sum(dim=1, keepdim=True) / 3.0
        conv.weight.data.copy_(folded.repeat(1, 3, 1, 1))


def graycopy_equivalence_error(module: torch.nn.Module) -> float:
    """Return max first-layer output error between current conv and its GrayStem-BC projection."""
    conv = first_conv_from_module(module)
    with torch.no_grad():
        sample = torch.rand(1, 1, 64, 64, device=conv.weight.device, dtype=conv.weight.dtype)
        gray3 = sample.repeat(1, 3, 1, 1)
        before = conv(gray3)
        original = conv.weight.data.clone()
        tie_first_conv_to_graystem_bc(module)
        after = conv(gray3)
        conv.weight.data.copy_(original)
    return float((before - after).abs().max().detach().cpu())


def add_graystem_callbacks(model: YOLO) -> None:
    """Keep the first convolution in GrayStem-BC form throughout training."""

    def _tie(trainer: Any) -> None:
        train_model = getattr(trainer, "model", None)
        if train_model is not None:
            tie_first_conv_to_graystem_bc(train_model)

    model.add_callback("on_train_start", _tie)
    model.add_callback("on_train_epoch_start", _tie)
    model.add_callback("on_train_batch_end", _tie)
    model.add_callback("on_fit_epoch_end", _tie)
    model.add_callback("on_train_end", _tie)


def stage_best_path(model: YOLO, project: str, run_name: str) -> Path:
    """Locate the best checkpoint from the actual Ultralytics save directory."""
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        best = Path(save_dir) / "weights" / "best.pt"
        if best.exists():
            return best
    candidates: list[Path] = []
    for root in [Path(project), Path.cwd()]:
        if root.exists():
            candidates.extend(root.rglob(f"{run_name}/weights/best.pt"))
    candidates = [p for p in candidates if p.exists()]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise RuntimeError(f"stage completed but best.pt not found: project={project} run={run_name} save_dir={save_dir}")


def train_one_stage(stage: str, weights: Path, data: Path, cfg: Dict[str, Any], args: argparse.Namespace) -> Path:
    """Train one Ultralytics stage with GrayStem-BC projection callbacks enabled."""
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    model_cfg = cfg["model"]
    stage_cfg = cfg["stages"][stage]
    workers = args.workers if args.workers is not None else int(model_cfg.get("workers", 8))
    run_name = f"{args.name_prefix}_{stage}"

    model = YOLO(str(weights))
    initial_error = graycopy_equivalence_error(model.model)
    tie_first_conv_to_graystem_bc(model.model)
    add_graystem_callbacks(model)

    freeze = int(stage_cfg.get("freeze", 0))
    train_kwargs: Dict[str, Any] = dict(
        data=str(data),
        imgsz=int(model_cfg.get("imgsz", 384)),
        epochs=int(stage_cfg["epochs"]),
        batch=parse_batch(args.batch),
        device=args.device,
        workers=workers,
        project=args.project,
        name=run_name,
        patience=int(model_cfg.get("patience", 20)),
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
        hsv_v=float(stage_cfg.get("hsv_v", 0.12)),
        degrees=float(stage_cfg.get("degrees", 1.0)),
        translate=float(stage_cfg.get("translate", 0.05)),
        scale=float(stage_cfg.get("scale", 0.35)),
        shear=0.0,
        perspective=float(stage_cfg.get("perspective", 0.0003)),
        flipud=0.0,
        fliplr=0.5,
        mosaic=float(stage_cfg.get("mosaic", 0.0)),
        mixup=0.0,
        copy_paste=0.0,
        erasing=float(stage_cfg.get("erasing", 0.08)),
        close_mosaic=int(stage_cfg.get("close_mosaic", 0)),
        lr0=float(stage_cfg["lr0"]),
        lrf=float(stage_cfg["lrf"]),
        warmup_epochs=float(stage_cfg.get("warmup_epochs", 2.0)),
        plots=True,
        val=True,
        exist_ok=True,
    )
    if freeze > 0:
        train_kwargs["freeze"] = freeze

    print("=" * 90)
    print(f"graystem_bc_stage={stage}")
    print(f"weights={weights}")
    print(f"run={Path(args.project) / run_name}")
    print(f"epochs={train_kwargs['epochs']} batch={train_kwargs['batch']} freeze={freeze}")
    print(f"initial_graycopy_equivalence_error={initial_error:.8g}")
    print("=" * 90)

    model.train(**train_kwargs)
    best = stage_best_path(model, args.project, run_name)
    final_model = YOLO(str(best))
    tie_first_conv_to_graystem_bc(final_model.model)
    final_model.save(str(best))
    return best


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    weights: Path = args.base_weights
    requested: List[str] = args.stages
    for stage in requested:
        weights = train_one_stage(stage, weights, args.data, cfg, args)
    args.final_weights_file.parent.mkdir(parents=True, exist_ok=True)
    args.final_weights_file.write_text(str(weights), encoding="utf-8")
    print("=" * 90)
    print("final graystem-bc weights:", weights)
    print("final weights file:", args.final_weights_file)


if __name__ == "__main__":
    main()
