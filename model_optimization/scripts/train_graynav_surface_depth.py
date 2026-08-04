#!/usr/bin/env python3
"""Train GrayNav SurfaceDepth from public grayscale-converted supervision."""

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

from segmentation.graynav_surface_depth import (  # noqa: E402
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    NUM_DEPTH_BINS,
    NUM_SURFACE_CLASSES,
    SURFACE_CLASS_NAMES,
    GrayNavSurfaceDepth,
    depth_bin_centers,
)


IGNORE = 255
CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 3.0], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width-mult", type=float, choices=(1.0, 0.75), default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--pretrained-fastscnn",
        type=Path,
        help="official PaddleSeg Fast-SCNN after true-mono folding/import",
    )
    return parser.parse_args()


def manifest(root: Path, split: str) -> list[dict[str, object]]:
    path = root / f"manifest_{split}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise RuntimeError(f"empty manifest {path}")
    return records


class SurfaceDepthDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: Path, split: str) -> None:
        self.root = root
        self.records = manifest(root, split)
        self.training = split == "train"

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def augment(image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32) / 255.0
        out = np.clip(out ** random.uniform(0.65, 1.55), 0.0, 1.0)
        out = np.clip(out * random.uniform(0.65, 1.35) + random.uniform(-0.12, 0.12), 0.0, 1.0)
        if random.random() < 0.35:
            out = np.clip(out + np.random.normal(0.0, random.uniform(0.005, 0.04), out.shape), 0, 1)
        if random.random() < 0.25:
            out = cv2.GaussianBlur(out, (3, 3), random.uniform(0.1, 1.0))
        if random.random() < 0.25:
            yy, xx = np.mgrid[: out.shape[0], : out.shape[1]]
            cx, cy = random.uniform(0, out.shape[1]), random.uniform(0, out.shape[0])
            radius = max(out.shape) * random.uniform(0.45, 0.85)
            shade = np.clip(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius, 0, 1)
            out *= 1.0 - shade * random.uniform(0.0, 0.35)
        return np.clip(out, 0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        gray = cv2.imread(str(self.root / str(record["image"])), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot read image for record {index}")
        seg = np.full(gray.shape, IGNORE, dtype=np.uint8)
        if record.get("seg_mask"):
            loaded = cv2.imread(str(self.root / str(record["seg_mask"])), cv2.IMREAD_GRAYSCALE)
            if loaded is None:
                raise RuntimeError(f"cannot read segmentation for record {index}")
            seg = loaded
        depth = np.zeros(gray.shape, dtype=np.float32)
        if record.get("depth"):
            depth = np.load(self.root / str(record["depth"])).astype(np.float32)
        if seg.shape != gray.shape:
            seg = cv2.resize(seg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        if depth.shape != gray.shape:
            depth = cv2.resize(depth, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.training:
            scale = random.uniform(0.75, 1.5)
            width = max(256, int(round(gray.shape[1] * scale)))
            height = max(256, int(round(gray.shape[0] * scale)))
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            x = random.randint(0, width - 256)
            y = random.randint(0, height - 256)
            gray, seg, depth = (
                value[y : y + 256, x : x + 256] for value in (gray, seg, depth)
            )
            if random.random() < 0.5:
                gray = np.ascontiguousarray(gray[:, ::-1])
                seg = np.ascontiguousarray(seg[:, ::-1])
                depth = np.ascontiguousarray(depth[:, ::-1])
            image = self.augment(gray)
        else:
            image = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            seg = cv2.resize(seg, (256, 256), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_NEAREST)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "seg": torch.from_numpy(seg.astype(np.int64)),
            "depth": torch.from_numpy(depth.astype(np.float32)),
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE
    safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe, NUM_SURFACE_CLASSES).permute(0, 3, 1, 2).float()
    weights = valid[:, None].float()
    probs = torch.softmax(logits, dim=1) * weights
    one_hot *= weights
    intersection = (probs * one_hot).sum((0, 2, 3))
    denominator = (probs + one_hot).sum((0, 2, 3))
    return 1.0 - ((2 * intersection + 1) / (denominator + 1)).mean()


def depth_targets(depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(depth) & (depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M)
    normalized = (
        (torch.log(depth.clamp_min(DEPTH_MIN_M)) - math.log(DEPTH_MIN_M))
        / (math.log(DEPTH_MAX_M) - math.log(DEPTH_MIN_M))
    )
    bins = torch.floor(normalized * NUM_DEPTH_BINS).long().clamp(0, NUM_DEPTH_BINS - 1)
    return bins, valid


def expected_depth(logits: torch.Tensor) -> torch.Tensor:
    centers = depth_bin_centers(logits.device).view(1, -1, 1, 1)
    return (torch.softmax(logits, dim=1) * centers).sum(dim=1)


def initialize_shared_encoder(model: GrayNavSurfaceDepth, checkpoint: Path) -> int:
    """Load every shared Fast-SCNN tensor and leave only the two heads random."""

    payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "model" not in payload:
        raise RuntimeError("pretrained Fast-SCNN checkpoint must contain a model state")
    if payload.get("rgb_input_used") is not False or not payload.get(
        "one_channel_first_conv_initialized", False
    ):
        raise RuntimeError("pretrained checkpoint did not prove true-mono first-conv folding")
    source = payload["model"]
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        mapped = name
        if name.startswith("classifier.0."):
            mapped = "shared_decoder.0." + name[len("classifier.0.") :]
        elif name.startswith("classifier.1."):
            mapped = "shared_decoder.1." + name[len("classifier.1.") :]
        elif name.startswith("classifier.2."):
            continue
        if mapped in target and tuple(value.shape) == tuple(target[mapped].shape):
            loaded[mapped] = value
    required = [
        name for name in target
        if not name.startswith("seg_head.") and not name.startswith("depth_head.")
    ]
    missing = [name for name in required if name not in loaded]
    if missing:
        raise RuntimeError(f"pretrained shared encoder is incomplete: {missing[:10]}")
    target.update(loaded)
    model.load_state_dict(target, strict=True)
    if tuple(model.first_conv.weight.shape)[1] != 1:
        raise RuntimeError("pretrained first convolution is not single-channel")
    return len(loaded)


def multitask_loss(
    seg_logits: torch.Tensor,
    depth_logits: torch.Tensor,
    seg: torch.Tensor,
    depth: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    seg64 = F.interpolate(seg[:, None].float(), size=(64, 64), mode="nearest")[:, 0].long()
    depth64 = F.interpolate(depth[:, None], size=(64, 64), mode="nearest")[:, 0]
    seg_valid = bool((seg64 != IGNORE).any())
    if seg_valid:
        ce = F.cross_entropy(seg_logits, seg64, weight=class_weights, ignore_index=IGNORE)
        dloss = dice_loss(seg_logits, seg64)
        seg_loss = 0.7 * ce + 0.3 * dloss
    else:
        seg_loss = seg_logits.sum() * 0.0
    bins, valid = depth_targets(depth64)
    if bool(valid.any()):
        pixel_ce = F.cross_entropy(depth_logits, bins, reduction="none")
        probabilities = torch.softmax(depth_logits, dim=1)
        cumulative = probabilities.cumsum(dim=1)
        thresholds = torch.arange(NUM_DEPTH_BINS, device=depth.device).view(1, -1, 1, 1)
        target_cumulative = (thresholds >= bins[:, None]).float()
        ordinal_emd = torch.abs(cumulative - target_cumulative).mean(dim=1)
        ordinal = pixel_ce[valid].mean() + 0.25 * ordinal_emd[valid].mean()
        predicted = expected_depth(depth_logits)
        dx_valid = valid[:, :, 1:] & valid[:, :, :-1]
        dy_valid = valid[:, 1:, :] & valid[:, :-1, :]
        smooth_parts = []
        if bool(dx_valid.any()):
            smooth_parts.append(torch.abs(predicted[:, :, 1:] - predicted[:, :, :-1])[dx_valid].mean())
        if bool(dy_valid.any()):
            smooth_parts.append(torch.abs(predicted[:, 1:, :] - predicted[:, :-1, :])[dy_valid].mean())
        smooth = torch.stack(smooth_parts).mean() if smooth_parts else ordinal * 0.0
        depth_loss = 0.4 * ordinal + 0.1 * smooth
    else:
        ordinal = depth_logits.sum() * 0.0
        smooth = depth_logits.sum() * 0.0
        depth_loss = depth_logits.sum() * 0.0
    total = seg_loss + depth_loss
    return total, {
        "seg": float(seg_loss.detach()),
        "depth_ordinal": float(ordinal.detach()),
        "depth_smooth": float(smooth.detach()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    absrel_sum = 0.0
    delta1_hits = 0
    depth_count = 0
    sample_depth_pairs: list[tuple[float, float]] = []
    for batch in loader:
        images = batch["image"].to(device)
        seg = F.interpolate(batch["seg"][:, None].float(), (64, 64), mode="nearest")[:, 0].long()
        depth = F.interpolate(batch["depth"][:, None], (64, 64), mode="nearest")[:, 0].to(device)
        seg_logits, depth_logits = model(images)
        pred = seg_logits.argmax(1).cpu()
        valid_seg = seg != IGNORE
        encoded = seg[valid_seg] * 3 + pred[valid_seg]
        confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)
        estimate = expected_depth(depth_logits)
        _, valid_depth = depth_targets(depth)
        if bool(valid_depth.any()):
            truth = depth[valid_depth]
            guess = estimate[valid_depth]
            absrel_sum += float((torch.abs(guess - truth) / truth).sum())
            ratio = torch.maximum(guess / truth, truth / guess.clamp_min(1e-3))
            delta1_hits += int((ratio < 1.25).sum())
            depth_count += int(truth.numel())
            for sample in range(depth.shape[0]):
                sample_valid = valid_depth[sample]
                if bool(sample_valid.any()):
                    truth_median = float(depth[sample][sample_valid].median())
                    guess_median = float(estimate[sample][sample_valid].median())
                    if math.isfinite(truth_median) and math.isfinite(guess_median):
                        sample_depth_pairs.append((truth_median, guess_median))
    order_hits = order_pairs = 0
    # Cap the all-pairs calculation while preserving the official validation order.
    ordered_samples = sample_depth_pairs[:512]
    for i in range(len(ordered_samples)):
        for j in range(i + 1, len(ordered_samples)):
            truth_delta = ordered_samples[i][0] - ordered_samples[j][0]
            if abs(truth_delta) < 0.25:
                continue
            guess_delta = ordered_samples[i][1] - ordered_samples[j][1]
            order_hits += int(truth_delta * guess_delta > 0.0)
            order_pairs += 1
    tp = confusion.diag().float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    iou = tp / (tp + fp + fn).clamp_min(1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return {
        "iou": {name: float(iou[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "f1": {name: float(f1[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "hazard_macro_f1": float(f1[1:].mean()),
        "depth_absrel": absrel_sum / max(1, depth_count),
        "depth_delta1": delta1_hits / max(1, depth_count),
        "near_far_order_accuracy": order_hits / max(1, order_pairs),
        "near_far_order_pairs": order_pairs,
        "depth_pixels": depth_count,
        "confusion": confusion.tolist(),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    train_set = SurfaceDepthDataset(args.data, "train")
    val_set = SurfaceDepthDataset(args.data, "val")
    fractions = {"ade20k": 0.40, "nyuv2": 0.35, "stairnetv3": 0.25}
    groups = [str(record["source"]) for record in train_set.records]
    counts = {name: groups.count(name) for name in fractions}
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"required public source is empty: {missing}")
    sample_weights = [fractions[name] / counts[name] for name in groups]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = GrayNavSurfaceDepth(width_mult=args.width_mult).to(device)
    if args.pretrained_fastscnn and args.resume:
        raise RuntimeError("use either --resume or --pretrained-fastscnn, not both")
    if args.pretrained_fastscnn:
        initialized = initialize_shared_encoder(model, args.pretrained_fastscnn)
        print(f"pretrained_shared_tensors={initialized}")
        print(f"first_conv_shape={tuple(model.first_conv.weight.shape)}")
        print("one_channel_first_conv_initialized=True")
        print("rgb_input_used=False")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    history_path = args.output / "history.json"
    history: list[dict[str, object]] = []
    best_score = -1e9
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                best_score = max(float(row["selection_score"]) for row in history)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    class_weights = CLASS_WEIGHTS.to(device)
    for epoch in range(start_epoch, args.epochs):
        if epoch < 5:
            epoch_lr = args.lr * float(epoch + 1) / 5.0
        else:
            progress = float(epoch - 5) / max(1, args.epochs - 5)
            epoch_lr = args.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        model.train()
        running = 0.0
        parts = {"seg": 0.0, "depth_ordinal": 0.0, "depth_smooth": 0.0}
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                seg_logits, depth_logits = model(images)
                loss, detail = multitask_loss(seg_logits, depth_logits, seg, depth, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            for key in parts:
                parts[key] += detail[key]
        model.eval()
        metrics = evaluate(model, val_loader, device)
        score = float(metrics["hazard_macro_f1"]) - 0.25 * float(metrics["depth_absrel"])
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "loss": running / max(1, len(train_loader)),
            **{key: value / max(1, len(train_loader)) for key, value in parts.items()},
            "metrics": metrics,
            "selection_score": score,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        contract = {
            "model": "graynav_surface_depth_gray1",
            "input_shape": [1, 1, 256, 256],
            "seg_shape": [1, 3, 64, 64],
            "depth_shape": [1, 16, 64, 64],
            "depth_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "width_mult": args.width_mult,
            "rgb_input_used": False,
        }
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "contract": contract,
        }
        torch.save(checkpoint, args.output / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, args.output / "best.pt")
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
