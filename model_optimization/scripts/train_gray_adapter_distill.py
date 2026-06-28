#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics import YOLO

from gray_adapter import build_adapter, save_adapter_bundle


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="RGB scene image directory")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/gray_adapter"))
    parser.add_argument("--adapter", choices=["ggg", "lut", "conv"], default="lut")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps for memory-limited GPUs")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--pixel-loss-weight", type=float, default=0.05)
    parser.add_argument("--reg-loss-weight", type=float, default=0.02)
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("runs/tensorboard/gray_adapter"))
    return parser.parse_args()


def letterbox_rgb(img_bgr: np.ndarray, size: int) -> np.ndarray:
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


class SceneDataset(Dataset):
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
        gray = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        gray_t = torch.from_numpy(gray).unsqueeze(0).contiguous()
        return gray_t, rgb_t, str(path)


def flatten_tensors(x) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if torch.is_tensor(x):
        if x.dtype.is_floating_point:
            out.append(x)
    elif isinstance(x, (list, tuple)):
        for v in x:
            out.extend(flatten_tensors(v))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(flatten_tensors(v))
    return out


def distill_loss(student_out, teacher_out) -> torch.Tensor:
    student = flatten_tensors(student_out)
    teacher = flatten_tensors(teacher_out)
    pairs = [(s, t) for s, t in zip(student, teacher) if tuple(s.shape) == tuple(t.shape)]
    if not pairs:
        raise RuntimeError("YOLO forward returned no matching tensors for distillation")
    loss = torch.zeros((), device=pairs[0][0].device)
    for s, t in pairs:
        loss = loss + F.mse_loss(s.float(), t.detach().float())
    return loss / len(pairs)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    ds = SceneDataset(args.source, args.imgsz, args.max_images, args.seed)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")

    yolo = YOLO(args.weights).model.to(device).eval()
    for p in yolo.parameters():
        p.requires_grad_(False)

    adapter = build_adapter(args.adapter).to(device)
    if args.adapter == "ggg":
        print("GGG baseline adapter has no trainable parameters; saving metadata only.")
        save_adapter_bundle(adapter, args.out_dir / "gray_adapter", {
            "weights": args.weights,
            "imgsz": args.imgsz,
            "input_channels": 1,
            "output_channels": 3,
            "training": "fixed_replicate_baseline",
        })
        return

    accumulate = max(1, int(args.accumulate))
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    tb_dir = args.tensorboard_dir / args.adapter
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logdir: {tb_dir}")

    best = float("inf")
    global_step = 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(args.epochs):
            adapter.train()
            running = 0.0
            count = 0
            pbar = tqdm(loader, desc=f"adapter {args.adapter} epoch {epoch + 1}/{args.epochs}")
            opt.zero_grad(set_to_none=True)
            for batch_idx, (gray, rgb, _paths) in enumerate(pbar):
                gray = gray.to(device, non_blocking=True)
                rgb = rgb.to(device, non_blocking=True)
                with torch.no_grad():
                    teacher = yolo(rgb)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    adapted = adapter(gray)
                    student = yolo(adapted)
                    det_loss = distill_loss(student, teacher)
                    pix_loss = F.l1_loss(adapted, rgb)
                    reg_loss = adapter.regularization_loss() if hasattr(adapter, "regularization_loss") else torch.zeros((), device=device)
                    loss = det_loss + args.pixel_loss_weight * pix_loss + args.reg_loss_weight * reg_loss
                    backward_loss = loss / float(accumulate)
                scaler.scale(backward_loss).backward()
                should_step = ((batch_idx + 1) % accumulate == 0) or ((batch_idx + 1) == len(loader))
                if should_step:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                batch_loss = float(loss.detach().cpu())
                running += batch_loss * gray.shape[0]
                count += gray.shape[0]
                writer.add_scalar("train/loss_total_step", batch_loss, global_step)
                writer.add_scalar("train/loss_distill_step", float(det_loss.detach().cpu()), global_step)
                writer.add_scalar("train/loss_pixel_step", float(pix_loss.detach().cpu()), global_step)
                writer.add_scalar("train/loss_regularization_step", float(reg_loss.detach().cpu()), global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/accumulate", accumulate, global_step)
                global_step += 1
                pbar.set_postfix(loss=running / max(1, count))
            epoch_loss = running / max(1, count)
            writer.add_scalar("train/loss_total_epoch", epoch_loss, epoch + 1)
            if epoch_loss < best:
                best = epoch_loss
                writer.add_scalar("train/best_loss", best, epoch + 1)
                save_adapter_bundle(adapter, args.out_dir / "gray_adapter", {
                    "weights": args.weights,
                    "imgsz": args.imgsz,
                    "input_channels": 1,
                    "output_channels": 3,
                    "teacher": "frozen_yolov8n_rgb",
                    "distillation": "raw_forward_tensor_mse_plus_pixel_l1",
                    "best_loss": best,
                    "tensorboard_dir": str(tb_dir),
                })
    finally:
        writer.flush()
        writer.close()

    print(f"best adapter loss: {best:.6f}")
    print(f"adapter bundle: {args.out_dir / 'gray_adapter.pt'}")
    print(f"adapter metadata: {args.out_dir / 'gray_adapter.json'}")
    print(f"tensorboard logdir: {tb_dir}")


if __name__ == "__main__":
    main()
