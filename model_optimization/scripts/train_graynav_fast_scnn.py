#!/usr/bin/env python3
"""Train GrayNav's true one-channel Fast-SCNN without distillation."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from segmentation.graynav_fast_scnn import (  # noqa: E402
    GrayNavFastSCNN,
    SURFACE_CLASS_NAMES,
    fold_rgb_first_conv_to_gray,
)


IGNORE = 255
CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 3.0, 4.0], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width-mult", type=float, choices=(1.0, 0.75), default=1.0)
    parser.add_argument("--amp", action="store_true", help="enable CUDA mixed precision")
    return parser.parse_args()


def read_manifest(root: Path, split: str) -> list[dict[str, object]]:
    path = root / f"manifest_{split}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"empty manifest: {path}")
    return records


class SurfaceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.records = read_manifest(root, split)
        self.training = split == "train"

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _augment(gray: np.ndarray) -> np.ndarray:
        image = gray.astype(np.float32) / 255.0
        image = np.clip(image ** random.uniform(0.65, 1.55), 0.0, 1.0)
        image = np.clip(image * random.uniform(0.65, 1.35) + random.uniform(-0.12, 0.12), 0.0, 1.0)
        if random.random() < 0.35:
            sigma = random.uniform(0.005, 0.04)
            image = np.clip(image + np.random.normal(0.0, sigma, image.shape), 0.0, 1.0)
        if random.random() < 0.20:
            image = np.random.poisson(image * 80.0).astype(np.float32) / 80.0
            image = np.clip(image, 0.0, 1.0)
        if random.random() < 0.25:
            image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.1, 1.0))
        if random.random() < 0.25:
            yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
            cx = random.uniform(0, image.shape[1])
            cy = random.uniform(0, image.shape[0])
            radius = max(image.shape) * random.uniform(0.45, 0.85)
            shade = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius, 0.0, 1.0)
            image *= 1.0 - shade * random.uniform(0.0, 0.35)
        return np.clip(image, 0.0, 1.0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        gray = cv2.imread(str(self.root / str(record["image"])), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root / str(record["mask"])), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None:
            raise RuntimeError(f"cannot read prepared pair at index {index}")
        if self.training:
            scale = random.uniform(0.75, 1.50)
            width = max(256, int(round(gray.shape[1] * scale)))
            height = max(256, int(round(gray.shape[0] * scale)))
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            rare_center = record.get("rare_center")
            if record.get("rare") and isinstance(rare_center, list) and len(rare_center) == 2:
                center_x = float(rare_center[0]) * width + random.uniform(-48.0, 48.0)
                center_y = float(rare_center[1]) * height + random.uniform(-48.0, 48.0)
                x = int(np.clip(round(center_x - 128), 0, width - 256))
                y = int(np.clip(round(center_y - 128), 0, height - 256))
            else:
                x = random.randint(0, width - 256)
                y = random.randint(0, height - 256)
            gray = gray[y : y + 256, x : x + 256]
            mask = mask[y : y + 256, x : x + 256]
            if random.random() < 0.5:
                gray = np.ascontiguousarray(gray[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            image = self._augment(gray)
        else:
            gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            image = gray.astype(np.float32) / 255.0
        return torch.from_numpy(image[None].astype(np.float32)), torch.from_numpy(mask.astype(np.int64))


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=4).permute(0, 3, 1, 2).float()
    valid_f = valid[:, None].float()
    probs = torch.softmax(logits, dim=1) * valid_f
    one_hot *= valid_f
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = (probs + one_hot).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def load_pretrained(model: GrayNavFastSCNN, checkpoint: Path | None) -> bool:
    if checkpoint is None:
        return False
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    rgb_key = next((key for key in state if key.endswith("learning_to_downsample.0.0.weight")), None)
    rgb_weight = state.pop(rgb_key) if rgb_key is not None and state[rgb_key].shape[1] == 3 else None
    model.load_state_dict(state, strict=False)
    if rgb_weight is not None:
        fold_rgb_first_conv_to_gray(model, rgb_weight)
        return True
    return model.first_conv.weight.shape[1] == 1


@torch.no_grad()
def export_error_samples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output: Path,
    limit: int = 16,
) -> None:
    palette = np.array(
        [[64, 200, 64], [200, 64, 64], [40, 180, 240], [200, 40, 220]],
        dtype=np.uint8,
    )
    worst: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for images, target in loader:
        logits = F.interpolate(model(images.to(device)), size=(256, 256), mode="nearest")
        pred = logits.argmax(dim=1).cpu().numpy()
        target_np = target.numpy()
        image_np = (images[:, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        for gray, truth, guess in zip(image_np, target_np, pred):
            valid = truth != IGNORE
            hazard = valid & (truth > 0)
            false_safe = hazard & (guess == 0)
            wrong = valid & (truth != guess)
            score = float(false_safe.sum() / max(1, hazard.sum())) + 0.1 * float(
                wrong.sum() / max(1, valid.sum())
            )
            worst.append((score, gray.copy(), truth.copy(), guess.copy()))
            worst.sort(key=lambda item: item[0], reverse=True)
            del worst[limit:]
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.png"):
        old.unlink()
    for index, (score, gray, truth, guess) in enumerate(worst):
        truth_color = np.zeros((*truth.shape, 3), dtype=np.uint8)
        valid = truth != IGNORE
        truth_color[valid] = palette[truth[valid]]
        pred_color = palette[guess]
        panel = np.concatenate([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), truth_color, pred_color], axis=1)
        cv2.imwrite(str(output / f"{index:02d}_score_{score:.4f}.png"), panel)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        logits = F.interpolate(model(images), size=(256, 256), mode="nearest")
        pred = logits.argmax(dim=1)
        valid = target != IGNORE
        encoded = target[valid] * 4 + pred[valid]
        confusion += torch.bincount(encoded.cpu(), minlength=16).reshape(4, 4)
    tp = confusion.diag().float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    iou = tp / (tp + fp + fn).clamp_min(1.0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)
    hazard_total = confusion[1:, :].sum().item()
    false_safe = confusion[1:, 0].sum().item() / max(1, hazard_total)
    return {
        "iou": {name: float(iou[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "f1": {name: float(f1[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "hazard_macro_f1": float(f1[1:].mean()),
        "hazard_false_safe_rate": float(false_safe),
        "confusion": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.pretrained is not None and args.resume is not None:
        raise RuntimeError("--pretrained and --resume are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    train_set = SurfaceDataset(args.data, "train")
    val_set = SurfaceDataset(args.data, "val")
    groups = [
        "stair" if record.get("source") == "stair" else
        ("mapillary_rare" if record.get("rare") else "mapillary_normal")
        for record in train_set.records
    ]
    target_fraction = {"mapillary_normal": 0.50, "mapillary_rare": 0.25, "stair": 0.25}
    group_count = {name: groups.count(name) for name in target_fraction}
    missing = [name for name, count in group_count.items() if count == 0]
    if missing:
        raise RuntimeError(f"cannot satisfy fixed 50/25/25 sampling; empty groups: {missing}")
    weights = [target_fraction[name] / group_count[name] for name in groups]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = GrayNavFastSCNN(width_mult=args.width_mult).to(device)
    if args.pretrained is None and args.resume is None and not args.allow_random_init:
        raise RuntimeError(
            "--pretrained is required for the planned first-conv fold; "
            "use --allow-random-init only for architecture smoke tests"
        )
    folded = load_pretrained(model, args.pretrained) if args.resume is None else True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    warmup_epochs = min(5, args.epochs)

    start_epoch = 0
    best_score = (-math.inf, -math.inf)
    history: list[dict[str, object]] = []
    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu")
        contract = payload.get("contract", {})
        if contract.get("input") != [1, 1, 256, 256] or contract.get("width_mult") != args.width_mult:
            raise RuntimeError(f"resume checkpoint contract mismatch: {contract}")
        if "optimizer" not in payload:
            raise RuntimeError("resume checkpoint lacks optimizer state; start from --pretrained instead")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler") and amp_enabled:
            scaler.load_state_dict(payload["scaler"])
        start_epoch = int(payload.get("epoch", 0))
        history = list(payload.get("history", []))
        saved_score = payload.get("best_score")
        if isinstance(saved_score, (list, tuple)) and len(saved_score) == 2:
            best_score = (float(saved_score[0]), float(saved_score[1]))
        print(f"resumed_from={args.resume} start_epoch={start_epoch}")

    print("input_shape=1x1x256x256")
    print(f"first_conv_shape={tuple(model.first_conv.weight.shape)}")
    print(f"one_channel_first_conv_initialized={folded}")
    print("rgb_input_used=False")
    print(f"amp_enabled={amp_enabled}")
    if args.pretrained is not None and not folded:
        raise RuntimeError("pretrained checkpoint did not provide a valid folded one-channel stem")

    for epoch in range(start_epoch, args.epochs):
        if epoch < warmup_epochs:
            scale = float(epoch + 1) / float(warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs - 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale

        model.train()
        total_loss = 0.0
        for images, target in train_loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                raw_logits = model(images)
                logits = F.interpolate(raw_logits, size=(256, 256), mode="nearest")
            logits_for_loss = logits.float()
            ce = F.cross_entropy(
                logits_for_loss,
                target,
                weight=CLASS_WEIGHTS.to(device),
                ignore_index=IGNORE,
            )
            loss = 0.7 * ce + 0.3 * dice_loss(logits_for_loss, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())

        model.eval()
        metrics = evaluate(model, val_loader, device)
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_loader)),
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        score = (
            float(metrics["hazard_macro_f1"]),
            -float(metrics["hazard_false_safe_rate"]),
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "epoch": epoch + 1,
            "metrics": metrics,
            "history": history,
            "best_score": list(max(best_score, score)),
            "contract": {
                "input": [1, 1, 256, 256],
                "output": [1, 4, 32, 32],
                "width_mult": args.width_mult,
            },
        }
        torch.save(checkpoint, args.output / "last.pt")
        if score > best_score:
            best_score = score
            checkpoint["best_score"] = list(best_score)
            torch.save(checkpoint, args.output / "best.pt")
        (args.output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not (args.output / "best.pt").is_file():
        raise RuntimeError("best.pt is missing; resume output directory must contain the selected checkpoint")
    best = torch.load(args.output / "best.pt", map_location="cpu")
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    export_error_samples(model, val_loader, device, args.output / "error_samples")


if __name__ == "__main__":
    main()
