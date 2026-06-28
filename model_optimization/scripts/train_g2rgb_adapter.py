#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO

from distill_loss import FeatureHookSet, feature_distill_loss, matched_tensor_distill
from gray_adapter import G2RGBResidualAdapter, save_adapter_bundle
from mono_sim import MonoSim


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train G2RGB residual adapter with frozen YOLOv8n distillation.")
    parser.add_argument("--source", type=Path, required=True, help="RGB training image directory")
    parser.add_argument("--weights", type=Path, required=True, help="YOLOv8n .pt weights")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/g2rgb_adapter"))
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--use-5x5-branch", action="store_true")
    parser.add_argument("--feature-layers", default="0,2,4,15,18", help="Comma-separated YOLO module indices for hooks")
    parser.add_argument("--head-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=2.0)
    parser.add_argument("--luma-weight", type=float, default=0.1)
    parser.add_argument("--residual-weight", type=float, default=0.01)
    parser.add_argument("--strong-monosim", action="store_true", help="Use stronger gray corruption for robust stage")
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/tensorboard/g2rgb_adapter"))
    return parser.parse_args()


def letterbox_rgb(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB source with YOLO-style constant-padding letterbox."""
    h, w = img_bgr.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


class RGBImageDataset(Dataset):
    """Load RGB images and apply fixed 384-style letterbox preprocessing."""

    def __init__(self, source: Path, imgsz: int, max_images: int, seed: int) -> None:
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in VALID_EXTS)
        if not images:
            raise SystemExit(f"no images found under {source}")
        rng = random.Random(seed)
        rng.shuffle(images)
        self.images = images[:max_images] if max_images > 0 else images
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        path = self.images[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read image: {path}")
        rgb = letterbox_rgb(img, self.imgsz).astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        return rgb_t, str(path)


def parse_feature_layers(value: str) -> List[int]:
    """Parse and validate a comma-separated feature-hook layer list."""
    layers = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not layers:
        raise ValueError("feature layer list cannot be empty")
    return layers


def save_checkpoint(adapter: G2RGBResidualAdapter, args: argparse.Namespace, name: str, metrics: dict) -> None:
    """Save adapter state plus metadata compatible with existing export scripts."""
    save_adapter_bundle(
        adapter,
        args.out_dir / name,
        {
            "weights": str(args.weights),
            "imgsz": args.imgsz,
            "input_channels": 1,
            "output_channels": 3,
            "teacher": "frozen_yolov8n_rgb",
            "student": "monosim_gray_to_g2rgb_to_frozen_yolov8n",
            "loss": "head_distill + feature_distill + luminance + residual",
            "feature_layers": parse_feature_layers(args.feature_layers),
            "strong_monosim": bool(args.strong_monosim),
            **metrics,
        },
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    dataset = RGBImageDataset(args.source, args.imgsz, args.max_images, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")

    yolo = YOLO(str(args.weights)).model.to(device).eval()
    for param in yolo.parameters():
        param.requires_grad_(False)

    adapter = G2RGBResidualAdapter(
        hidden=args.hidden,
        blocks=args.blocks,
        alpha=args.alpha,
        use_5x5_branch=args.use_5x5_branch,
    ).to(device)
    monosim = MonoSim(strong=args.strong_monosim).to(device).train()
    hooks = FeatureHookSet(yolo, parse_feature_layers(args.feature_layers))

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    writer = SummaryWriter(log_dir=str(args.tensorboard_dir))
    print(f"TensorBoard logdir: {args.tensorboard_dir}")

    best = float("inf")
    global_step = 0
    accumulate = max(1, int(args.accumulate))
    try:
        for epoch in range(args.epochs):
            adapter.train()
            running = 0.0
            count = 0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(loader, desc=f"g2rgb epoch {epoch + 1}/{args.epochs}")
            for batch_idx, (rgb, _paths) in enumerate(pbar):
                rgb = rgb.to(device, non_blocking=True)
                gray = monosim(rgb)

                hooks.clear()
                with torch.no_grad():
                    teacher_out = yolo(rgb)
                    teacher_features = hooks.snapshot(detach=True)

                hooks.clear()
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    adapted = adapter(gray)
                    student_out = yolo(adapted)
                    student_features = hooks.snapshot(detach=False)
                    head_loss = matched_tensor_distill(student_out, teacher_out, loss_type="smooth_l1")
                    feat_loss = feature_distill_loss(student_features, teacher_features)
                    luma_loss = adapter.luminance_loss(gray)
                    residual_loss = adapter.regularization_loss(gray)
                    loss = (
                        args.head_weight * head_loss
                        + args.feature_weight * feat_loss
                        + args.luma_weight * luma_loss
                        + args.residual_weight * residual_loss
                    )
                    backward_loss = loss / float(accumulate)

                scaler.scale(backward_loss).backward()
                if ((batch_idx + 1) % accumulate == 0) or ((batch_idx + 1) == len(loader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                batch_loss = float(loss.detach().cpu())
                running += batch_loss * rgb.shape[0]
                count += rgb.shape[0]
                writer.add_scalar("train/loss_total_step", batch_loss, global_step)
                writer.add_scalar("train/loss_head_step", float(head_loss.detach().cpu()), global_step)
                writer.add_scalar("train/loss_feature_step", float(feat_loss.detach().cpu()), global_step)
                writer.add_scalar("train/loss_luminance_step", float(luma_loss.detach().cpu()), global_step)
                writer.add_scalar("train/loss_residual_step", float(residual_loss.detach().cpu()), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                global_step += 1
                pbar.set_postfix(loss=running / max(1, count))

            scheduler.step()
            epoch_loss = running / max(1, count)
            metrics = {"epoch": epoch + 1, "loss": epoch_loss}
            writer.add_scalar("train/loss_total_epoch", epoch_loss, epoch + 1)
            save_checkpoint(adapter, args, "g2rgb_adapter_last", metrics)
            if epoch_loss < best:
                best = epoch_loss
                save_checkpoint(adapter, args, "g2rgb_adapter_best", {**metrics, "best_loss": best})
                writer.add_scalar("train/best_loss", best, epoch + 1)
    finally:
        hooks.close()
        writer.flush()
        writer.close()

    print(f"best loss: {best:.6f}")
    print(f"best adapter: {args.out_dir / 'g2rgb_adapter_best.pt'}")
    print(f"last adapter: {args.out_dir / 'g2rgb_adapter_last.pt'}")


if __name__ == "__main__":
    main()
