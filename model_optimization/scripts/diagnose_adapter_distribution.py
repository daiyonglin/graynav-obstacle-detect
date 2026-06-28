#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from gray_adapter import load_adapter_bundle


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose pseudo-RGB gray adapter channel and first-conv activation statistics.")
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--adapter", type=Path, help="gray_adapter.pt. Omit for baseline gray-copy.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def collect_images(path: Path, max_images: int) -> list[Path]:
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in VALID_EXTS)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"no images found: {path}")
    return images


def baseline(gray: np.ndarray) -> torch.Tensor:
    arr = np.stack([gray, gray, gray], axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def adapt(gray: np.ndarray, adapter: torch.nn.Module | None) -> torch.Tensor:
    if adapter is None:
        return baseline(gray)
    x = torch.from_numpy(gray.astype(np.float32) / 255.0).view(1, 1, gray.shape[0], gray.shape[1])
    with torch.no_grad():
        return adapter(x).clamp(0.0, 1.0).cpu()


def corrcoef_channels(x: torch.Tensor) -> np.ndarray:
    arr = x.squeeze(0).numpy().reshape(3, -1)
    return np.corrcoef(arr)


def first_conv(model: torch.nn.Module) -> torch.nn.Module:
    # Ultralytics YOLOv8 first module is usually Conv(c1=3,c2=16,k=3,s=2).
    first = model.model[0]
    return first


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(args.weights)).model.to(device).eval()
    conv0 = first_conv(yolo).to(device).eval()
    adapter = load_adapter_bundle(args.adapter, map_location="cpu").eval() if args.adapter else None
    paths = collect_images(args.images, args.max_images)

    channel_means: list[list[float]] = []
    channel_stds: list[list[float]] = []
    corr01: list[float] = []
    corr02: list[float] = []
    corr12: list[float] = []
    activation_means: list[float] = []
    activation_stds: list[float] = []
    activation_abs_means: list[float] = []

    for path in tqdm(paths, desc="diagnose adapter distribution"):
        gray = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        x = adapt(gray, adapter)
        ch = x.squeeze(0).numpy().reshape(3, -1)
        channel_means.append([float(v) for v in ch.mean(axis=1)])
        channel_stds.append([float(v) for v in ch.std(axis=1)])
        corr = corrcoef_channels(x)
        corr01.append(float(corr[0, 1]))
        corr02.append(float(corr[0, 2]))
        corr12.append(float(corr[1, 2]))
        with torch.no_grad():
            act = conv0(x.to(device))
        activation_means.append(float(act.mean().detach().cpu()))
        activation_stds.append(float(act.std().detach().cpu()))
        activation_abs_means.append(float(act.abs().mean().detach().cpu()))

    cm = np.asarray(channel_means, dtype=np.float64)
    cs = np.asarray(channel_stds, dtype=np.float64)
    result = {
        "images": str(args.images),
        "weights": str(args.weights),
        "adapter": str(args.adapter) if args.adapter else "baseline_gray_copy",
        "image_count": len(paths),
        "channel_mean": {f"ch{i}": summarize(cm[:, i].tolist()) for i in range(3)},
        "channel_std": {f"ch{i}": summarize(cs[:, i].tolist()) for i in range(3)},
        "channel_corr": {
            "ch0_ch1": summarize(corr01),
            "ch0_ch2": summarize(corr02),
            "ch1_ch2": summarize(corr12),
        },
        "first_conv_activation": {
            "mean": summarize(activation_means),
            "std": summarize(activation_stds),
            "abs_mean": summarize(activation_abs_means),
        },
        "interpretation": {
            "baseline_expected": "Gray-copy baseline should have channel correlations close to 1.0.",
            "adapter_goal": "A useful pseudo-RGB adapter should reduce exact channel redundancy and shift first-conv activation statistics without destroying detection metrics.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
