#!/usr/bin/env python3
"""Train GrayNav SurfaceDepth from public grayscale-converted supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(MODEL_ROOT))

from models.graynav_surface_depth import (  # noqa: E402
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    NUM_DEPTH_BINS,
    NUM_SURFACE_CLASSES,
    SURFACE_CLASS_NAMES,
    GrayNavSurfaceDepth,
    depth_bin_centers,
)


IGNORE = 255
CLASS_WEIGHTS = torch.tensor([1.0, 1.5, 3.0, 0.5], dtype=torch.float32)
NEAR_MID_FAR_EDGES_M = (1.25, 2.20)


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
    parser.add_argument(
        "--experiment", choices=("e1", "e2", "e3"), default="e1",
        help="E1 label baseline, E2 depth-loss repair, or E3 true-64 detail fusion",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument(
        "--e0-metrics", type=Path,
        help="prepared-v2 E0 evaluation used for the required 15%% gradient improvement",
    )
    parser.add_argument(
        "--pretrained-fastscnn",
        type=Path,
        help="official PaddleSeg Fast-SCNN after true-mono folding/import",
    )
    parser.add_argument("--sampling-ade20k", type=float, default=0.40)
    parser.add_argument("--sampling-nyuv2", type=float, default=0.35)
    parser.add_argument("--sampling-stairnetv3", type=float, default=0.25)
    parser.add_argument(
        "--ade-step-center-prob",
        type=float,
        default=0.0,
        help="probability of centering an ADE20K training crop on a mapped step pixel",
    )
    parser.add_argument(
        "--stair-step-center-prob",
        type=float,
        default=0.70,
        help="probability of centering a StairNet crop on a step pixel",
    )
    parser.add_argument(
        "--stair-negative-crop-prob",
        type=float,
        default=0.0,
        help="probability of choosing the lowest-STEP StairNet crop from random candidates",
    )
    parser.add_argument("--stair-negative-crop-attempts", type=int, default=12)
    return parser.parse_args()


def manifest(root: Path, split: str) -> list[dict[str, object]]:
    path = root / f"manifest_{split}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise RuntimeError(f"empty manifest {path}")
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SurfaceDepthDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        root: Path,
        split: str,
        experiment: str = "e1",
        ade_step_center_prob: float = 0.0,
        stair_step_center_prob: float = 0.70,
        stair_negative_crop_prob: float = 0.0,
        stair_negative_crop_attempts: int = 12,
    ) -> None:
        self.root = root
        self.records = manifest(root, split)
        self.training = split == "train"
        self.experiment = experiment
        if not 0.0 <= ade_step_center_prob <= 1.0:
            raise ValueError("ade_step_center_prob must be in [0, 1]")
        if not 0.0 <= stair_step_center_prob <= 1.0:
            raise ValueError("stair_step_center_prob must be in [0, 1]")
        if not 0.0 <= stair_negative_crop_prob <= 1.0:
            raise ValueError("stair_negative_crop_prob must be in [0, 1]")
        if stair_step_center_prob + stair_negative_crop_prob > 1.0:
            raise ValueError("StairNet crop probabilities must sum to at most 1")
        if stair_negative_crop_attempts < 1:
            raise ValueError("stair_negative_crop_attempts must be positive")
        self.ade_step_center_prob = ade_step_center_prob
        self.stair_step_center_prob = stair_step_center_prob
        self.stair_negative_crop_prob = stair_negative_crop_prob
        self.stair_negative_crop_attempts = stair_negative_crop_attempts

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

    @staticmethod
    def crop_origin(
        width: int,
        height: int,
        seg: np.ndarray,
        source: str,
        ade_step_center_prob: float = 0.0,
        stair_step_center_prob: float = 0.70,
        stair_negative_crop_prob: float = 0.0,
        stair_negative_crop_attempts: int = 12,
    ) -> tuple[int, int]:
        draw = random.random()
        center_probability = (
            stair_step_center_prob if source == "stairnetv3"
            else ade_step_center_prob if source == "ade20k"
            else 0.0
        )
        if center_probability > 0.0 and draw < center_probability:
            ys, xs = np.where(seg == 2)
            if len(xs):
                selected = random.randrange(len(xs))
                x = int(np.clip(xs[selected] - random.randint(64, 192), 0, width - 256))
                y = int(np.clip(ys[selected] - random.randint(64, 192), 0, height - 256))
                return x, y
        if (
            source == "stairnetv3"
            and draw < stair_step_center_prob + stair_negative_crop_prob
        ):
            candidates = [
                (
                    random.randint(0, width - 256),
                    random.randint(0, height - 256),
                )
                for _ in range(stair_negative_crop_attempts)
            ]
            return min(
                candidates,
                key=lambda origin: float(
                    (seg[
                        origin[1] : origin[1] + 256,
                        origin[0] : origin[0] + 256,
                    ] == 2).mean()
                ),
            )
        return random.randint(0, width - 256), random.randint(0, height - 256)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        source = str(record["source"])
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
            if source == "nyuv2" and self.experiment in ("e2", "e3"):
                scale = 288.0 / min(gray.shape)
            else:
                scale = random.uniform(0.75, 1.5)
                scale = max(scale, 256.0 / min(gray.shape))
            width = max(256, int(round(gray.shape[1] * scale)))
            height = max(256, int(round(gray.shape[0] * scale)))
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
            seg = cv2.resize(seg, (width, height), interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
            x, y = self.crop_origin(
                width,
                height,
                seg,
                source,
                self.ade_step_center_prob,
                self.stair_step_center_prob,
                self.stair_negative_crop_prob,
                self.stair_negative_crop_attempts,
            )
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
            "source": source,
            "source_id": str(record["source_id"]),
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE
    safe = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe, NUM_SURFACE_CLASSES).permute(0, 3, 1, 2).float()
    weights = valid[:, None].float()
    probs = torch.softmax(logits, dim=1) * weights
    one_hot *= weights
    # UNKNOWN participates in CE but cannot dominate the task-class Dice.
    intersection = (probs * one_hot).sum((0, 2, 3))[:3]
    denominator = (probs + one_hot).sum((0, 2, 3))[:3]
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
        if not name.startswith((
            "seg_head.", "depth_head.", "detail_projection.",
            "semantic_projection.", "detail_refinement.",
        ))
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
    images: torch.Tensor | None = None,
    experiment: str = "e1",
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
        smooth_parts: list[torch.Tensor] = []
        if bool(dx_valid.any()):
            smooth_parts.append(torch.abs(predicted[:, :, 1:] - predicted[:, :, :-1])[dx_valid].mean())
        if bool(dy_valid.any()):
            smooth_parts.append(torch.abs(predicted[:, 1:, :] - predicted[:, :-1, :])[dy_valid].mean())
        smooth = torch.stack(smooth_parts).mean() if smooth_parts else ordinal * 0.0
        log_l1 = silog = gradient = grouped = ordinal * 0.0
        if experiment == "e1":
            depth_loss = 0.4 * ordinal + 0.1 * smooth
        else:
            log_error = torch.log(predicted.clamp_min(1e-3)) - torch.log(depth64.clamp_min(1e-3))
            valid_log_error = log_error[valid]
            log_l1 = valid_log_error.abs().mean()
            silog = torch.sqrt(
                (valid_log_error.square().mean() - 0.85 * valid_log_error.mean().square())
                .clamp_min(1e-8)
            )
            gradient_parts: list[torch.Tensor] = []
            edge_smooth_parts: list[torch.Tensor] = []
            image64 = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)[:, 0]
            if bool(dx_valid.any()):
                pred_dx = predicted[:, :, 1:] - predicted[:, :, :-1]
                truth_dx = depth64[:, :, 1:] - depth64[:, :, :-1]
                image_dx = torch.abs(image64[:, :, 1:] - image64[:, :, :-1])
                gradient_parts.append(torch.abs(pred_dx - truth_dx)[dx_valid].mean())
                edge_smooth_parts.append(
                    (torch.abs(pred_dx) * torch.exp(-10.0 * image_dx))[dx_valid].mean()
                )
            if bool(dy_valid.any()):
                pred_dy = predicted[:, 1:, :] - predicted[:, :-1, :]
                truth_dy = depth64[:, 1:, :] - depth64[:, :-1, :]
                image_dy = torch.abs(image64[:, 1:, :] - image64[:, :-1, :])
                gradient_parts.append(torch.abs(pred_dy - truth_dy)[dy_valid].mean())
                edge_smooth_parts.append(
                    (torch.abs(pred_dy) * torch.exp(-10.0 * image_dy))[dy_valid].mean()
                )
            gradient = torch.stack(gradient_parts).mean() if gradient_parts else ordinal * 0.0
            smooth = torch.stack(edge_smooth_parts).mean() if edge_smooth_parts else ordinal * 0.0
            grouped_target = torch.zeros_like(depth64, dtype=torch.long)
            grouped_target[depth64 >= NEAR_MID_FAR_EDGES_M[0]] = 1
            grouped_target[depth64 >= NEAR_MID_FAR_EDGES_M[1]] = 2
            centers = depth_bin_centers(depth_logits.device)
            near = centers < NEAR_MID_FAR_EDGES_M[0]
            mid = (centers >= NEAR_MID_FAR_EDGES_M[0]) & (centers < NEAR_MID_FAR_EDGES_M[1])
            far = centers >= NEAR_MID_FAR_EDGES_M[1]
            probabilities = torch.softmax(depth_logits, dim=1)
            grouped_prob = torch.stack(
                (probabilities[:, near].sum(1), probabilities[:, mid].sum(1), probabilities[:, far].sum(1)),
                dim=1,
            ).clamp_min(1e-7)
            grouped = F.nll_loss(torch.log(grouped_prob), grouped_target, reduction="none")[valid].mean()
            depth_core = (
                0.35 * ordinal
                + 0.20 * log_l1
                + 0.15 * silog
                + 0.15 * gradient
                + 0.05 * smooth
                + 0.10 * grouped
            )
            depth_loss = 0.6 * depth_core
    else:
        ordinal = smooth = depth_logits.sum() * 0.0
        log_l1 = silog = gradient = grouped = depth_logits.sum() * 0.0
        depth_loss = depth_logits.sum() * 0.0
    total = seg_loss + depth_loss
    return total, {
        "seg": float(seg_loss.detach()),
        "depth_ordinal": float(ordinal.detach()),
        "depth_smooth": float(smooth.detach()),
        "depth_log_l1": float(log_l1.detach()),
        "depth_silog": float(silog.detach()),
        "depth_gradient": float(gradient.detach()),
        "depth_grouped": float(grouped.detach()),
    }


def class_metrics(confusion: torch.Tensor) -> dict[str, object]:
    tp = confusion.diag().float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    iou = tp / (tp + fp + fn).clamp_min(1)
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "iou": {name: float(iou[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "precision": {name: float(precision[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "recall": {name: float(recall[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "f1": {name: float(f1[i]) for i, name in enumerate(SURFACE_CLASS_NAMES)},
        "confusion": confusion.tolist(),
    }


def depth_level(values: torch.Tensor) -> torch.Tensor:
    levels = torch.zeros_like(values, dtype=torch.long)
    levels[values >= NEAR_MID_FAR_EDGES_M[0]] = 1
    levels[values >= NEAR_MID_FAR_EDGES_M[1]] = 2
    return levels


def false_whole_frame_step_prediction(
    guess: torch.Tensor,
    truth: torch.Tensor,
    prediction_threshold: float = 0.60,
    excess_threshold: float = 0.20,
) -> bool:
    """Flag material step overfill without penalizing truly stair-filled scenes.

    StairNet contains legitimate images whose ground-truth stair area exceeds
    60 percent.  A raw prediction-area threshold therefore cannot distinguish
    those samples from the original all-frame false-positive failure.  Compare
    prediction and truth over labelled pixels and only flag a prediction that
    is both mostly STEP and overfills truth by more than 20 percentage points.
    """

    valid = truth != IGNORE
    if not bool(valid.any()):
        return False
    prediction_ratio = float((guess[valid] == 2).float().mean())
    truth_ratio = float((truth[valid] == 2).float().mean())
    return (
        prediction_ratio > prediction_threshold
        and prediction_ratio - truth_ratio > excess_threshold
    )


def tensorboard_scalar_metrics(
    metrics: dict[str, object],
) -> dict[str, float]:
    """Return only numeric metrics accepted by SummaryWriter.add_scalar."""

    numeric = (int, float, np.integer, np.floating)
    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, numeric)
    }


def experiment_gates(
    metrics: dict[str, object], experiment: str, e0_gradient_mae: float | None = None
) -> dict[str, object]:
    iou = metrics["iou"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    safety = metrics["safety"]
    depth = metrics["depth"]
    e1_checks = {
        "stair_false_whole_frame_step_predictions_zero": (
            safety["stair_false_whole_frame_step_prediction_count"] == 0
        ),
        "stair_step_precision_ge_065": metrics["per_source"]["stairnetv3"]["precision"]["step_or_drop"] >= 0.65,
        "stair_step_recall_ge_065": metrics["per_source"]["stairnetv3"]["recall"]["step_or_drop"] >= 0.65,
        "ade_bottom_step_false_image_rate_le_010": safety["ade_no_step_bottom_false_image_rate"] <= 0.10,
        "blocked_iou_ge_070": iou["blocked_surface"] >= 0.70,
        "hazard_to_ground_le_010": safety["hazard_to_ground_rate"] <= 0.10,
    }
    final_checks = {
        "ground_iou_ge_065": iou["ground_candidate"] >= 0.65,
        "blocked_iou_ge_070": iou["blocked_surface"] >= 0.70,
        "step_precision_ge_070": precision["step_or_drop"] >= 0.70,
        "step_recall_ge_070": recall["step_or_drop"] >= 0.70,
        "step_f1_ge_070": f1["step_or_drop"] >= 0.70,
        "hazard_to_ground_le_008": safety["hazard_to_ground_rate"] <= 0.08,
        "false_whole_frame_step_predictions_zero": (
            safety["false_whole_frame_step_prediction_count"] == 0
        ),
        "ade_bottom_step_false_image_rate_le_005": safety["ade_no_step_bottom_false_image_rate"] <= 0.05,
        "depth_absrel_le_025": depth["absrel"] <= 0.25,
        "depth_delta1_ge_060": depth["delta1"] >= 0.60,
        "near_far_order_ge_080": depth["near_far_order_accuracy"] >= 0.80,
        "depth_gradient_improves_15pct_vs_e0": (
            e0_gradient_mae is not None
            and depth["gradient_mae"] <= 0.85 * e0_gradient_mae
        ),
    }
    checks = e1_checks if experiment == "e1" else final_checks
    return {"passed": all(checks.values()), "checks": checks, "final_checks": final_checks}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    confusion = torch.zeros((NUM_SURFACE_CLASSES, NUM_SURFACE_CLASSES), dtype=torch.int64)
    source_confusions = {
        source: torch.zeros_like(confusion) for source in ("ade20k", "nyuv2", "stairnetv3")
    }
    bottom_confusion = torch.zeros_like(confusion)
    corridor_confusion = torch.zeros_like(confusion)
    calibration_count = torch.zeros(10, dtype=torch.int64)
    calibration_correct = torch.zeros(10, dtype=torch.float64)
    calibration_confidence = torch.zeros(10, dtype=torch.float64)
    stair_outside_fp = stair_outside_pixels = 0
    ade_no_step = ade_false_images = whole_frame_step = stair_whole_frame_step = 0
    false_whole_frame_step = stair_false_whole_frame_step = 0
    hazard_ground = hazard_pixels = 0
    depth_count = delta1 = delta2 = delta3 = 0
    absrel_sum = squared_error_sum = squared_log_error_sum = 0.0
    gradient_error_sum = 0.0
    gradient_count = 0
    depth_level_confusion = torch.zeros((3, 3), dtype=torch.int64)
    sample_depth_pairs: list[tuple[float, float]] = []
    corridor_level_hits = corridor_level_count = 0
    for batch in tqdm(loader, desc="validation", unit="batch", dynamic_ncols=True, leave=False):
        images = batch["image"].to(device)
        seg = F.interpolate(batch["seg"][:, None].float(), (64, 64), mode="nearest")[:, 0].long()
        depth = F.interpolate(batch["depth"][:, None], (64, 64), mode="nearest")[:, 0].to(device)
        seg_logits, depth_logits = model(images)
        probabilities = torch.softmax(seg_logits, dim=1).cpu()
        confidence, pred = probabilities.max(1)
        sources = list(batch["source"])
        for sample, source in enumerate(sources):
            truth = seg[sample]
            guess = pred[sample]
            valid = truth != IGNORE
            if bool(valid.any()):
                encoded = truth[valid] * NUM_SURFACE_CLASSES + guess[valid]
                sample_confusion = torch.bincount(
                    encoded, minlength=NUM_SURFACE_CLASSES ** 2
                ).reshape(NUM_SURFACE_CLASSES, NUM_SURFACE_CLASSES)
                confusion += sample_confusion
                source_confusions[str(source)] += sample_confusion
                bottom_valid = valid[19:, :]
                if bool(bottom_valid.any()):
                    encoded_bottom = truth[19:, :][bottom_valid] * NUM_SURFACE_CLASSES + guess[19:, :][bottom_valid]
                    bottom_confusion += torch.bincount(
                        encoded_bottom, minlength=NUM_SURFACE_CLASSES ** 2
                    ).reshape(NUM_SURFACE_CLASSES, NUM_SURFACE_CLASSES)
                corridor_valid = valid[19:, 21:43]
                if bool(corridor_valid.any()):
                    encoded_corridor = truth[19:, 21:43][corridor_valid] * NUM_SURFACE_CLASSES + guess[19:, 21:43][corridor_valid]
                    corridor_confusion += torch.bincount(
                        encoded_corridor, minlength=NUM_SURFACE_CLASSES ** 2
                    ).reshape(NUM_SURFACE_CLASSES, NUM_SURFACE_CLASSES)
                conf_valid = confidence[sample][valid]
                correct_valid = (truth[valid] == guess[valid]).float()
                bins = torch.clamp((conf_valid * 10).long(), max=9)
                calibration_count += torch.bincount(bins, minlength=10)
                calibration_correct.scatter_add_(0, bins, correct_valid.double())
                calibration_confidence.scatter_add_(0, bins, conf_valid.double())
                hazards = (truth == 1) | (truth == 2)
                hazard_ground += int(((guess == 0) & hazards).sum())
                hazard_pixels += int(hazards.sum())
            step_ratio = float((guess == 2).float().mean())
            whole_frame_step += int(step_ratio > 0.60)
            false_whole = false_whole_frame_step_prediction(guess, truth)
            false_whole_frame_step += int(false_whole)
            if source == "stairnetv3":
                stair_whole_frame_step += int(step_ratio > 0.60)
                stair_false_whole_frame_step += int(false_whole)
                outside = truth == 3
                stair_outside_fp += int(((guess == 2) & outside).sum())
                stair_outside_pixels += int(outside.sum())
            if source == "ade20k" and not bool((truth == 2).any()):
                ade_no_step += 1
                ade_false_images += int(float((guess[19:, :] == 2).float().mean()) > 0.03)
        estimate = expected_depth(depth_logits)
        _, valid_depth = depth_targets(depth)
        if bool(valid_depth.any()):
            truth = depth[valid_depth]
            guess = estimate[valid_depth]
            error = guess - truth
            absrel_sum += float((error.abs() / truth).sum())
            squared_error_sum += float(error.square().sum())
            squared_log_error_sum += float((torch.log(guess.clamp_min(1e-3)) - torch.log(truth)).square().sum())
            ratio = torch.maximum(guess / truth, truth / guess.clamp_min(1e-3))
            delta1 += int((ratio < 1.25).sum())
            delta2 += int((ratio < 1.25 ** 2).sum())
            delta3 += int((ratio < 1.25 ** 3).sum())
            depth_count += int(truth.numel())
            encoded_levels = depth_level(truth) * 3 + depth_level(guess)
            depth_level_confusion += torch.bincount(encoded_levels.cpu(), minlength=9).reshape(3, 3)
            dx_valid = valid_depth[:, :, 1:] & valid_depth[:, :, :-1]
            dy_valid = valid_depth[:, 1:, :] & valid_depth[:, :-1, :]
            if bool(dx_valid.any()):
                gradient_error_sum += float(torch.abs(
                    (estimate[:, :, 1:] - estimate[:, :, :-1])
                    - (depth[:, :, 1:] - depth[:, :, :-1])
                )[dx_valid].sum())
                gradient_count += int(dx_valid.sum())
            if bool(dy_valid.any()):
                gradient_error_sum += float(torch.abs(
                    (estimate[:, 1:, :] - estimate[:, :-1, :])
                    - (depth[:, 1:, :] - depth[:, :-1, :])
                )[dy_valid].sum())
                gradient_count += int(dy_valid.sum())
            for sample in range(depth.shape[0]):
                sample_valid = valid_depth[sample]
                if bool(sample_valid.any()):
                    sample_depth_pairs.append((
                        float(depth[sample][sample_valid].median()),
                        float(estimate[sample][sample_valid].median()),
                    ))
                corridor_valid = valid_depth[sample, 19:, 21:43]
                if bool(corridor_valid.any()):
                    corridor_truth = depth[sample, 19:, 21:43][corridor_valid].median()
                    corridor_guess = estimate[sample, 19:, 21:43][corridor_valid].median()
                    corridor_level_hits += int(
                        int(depth_level(corridor_truth).item())
                        == int(depth_level(corridor_guess).item())
                    )
                    corridor_level_count += 1
    order_hits = order_pairs = 0
    for i, first in enumerate(sample_depth_pairs[:512]):
        for second in sample_depth_pairs[i + 1 : 512]:
            truth_delta = first[0] - second[0]
            if abs(truth_delta) < 0.25:
                continue
            order_hits += int(truth_delta * (first[1] - second[1]) > 0.0)
            order_pairs += 1
    overall = class_metrics(confusion)
    level_tp = depth_level_confusion.diag().float()
    level_precision = level_tp / depth_level_confusion.sum(0).float().clamp_min(1)
    level_recall = level_tp / depth_level_confusion.sum(1).float().clamp_min(1)
    level_f1 = 2 * level_precision * level_recall / (level_precision + level_recall).clamp_min(1e-8)
    valid_cal = calibration_count > 0
    ece = float((
        calibration_count[valid_cal].double() / max(1, int(calibration_count.sum()))
        * torch.abs(
            calibration_correct[valid_cal] / calibration_count[valid_cal]
            - calibration_confidence[valid_cal] / calibration_count[valid_cal]
        )
    ).sum()) if bool(valid_cal.any()) else 0.0
    metrics: dict[str, object] = {
        **overall,
        "hazard_macro_f1": float(np.mean([
            overall["f1"]["blocked_surface"], overall["f1"]["step_or_drop"]
        ])),
        "per_source": {name: class_metrics(value) for name, value in source_confusions.items()},
        "bottom70": class_metrics(bottom_confusion),
        "central_corridor": class_metrics(corridor_confusion),
        "safety": {
            "stair_outside_false_positive_rate": stair_outside_fp / max(1, stair_outside_pixels),
            "ade_no_step_bottom_false_image_rate": ade_false_images / max(1, ade_no_step),
            "ade_no_step_image_count": ade_no_step,
            "whole_frame_step_prediction_count": whole_frame_step,
            "stair_whole_frame_step_prediction_count": stair_whole_frame_step,
            "false_whole_frame_step_prediction_count": false_whole_frame_step,
            "stair_false_whole_frame_step_prediction_count": stair_false_whole_frame_step,
            "false_whole_frame_step_definition": (
                "valid-pixel step ratio > 0.60 and prediction exceeds truth by > 0.20"
            ),
            "hazard_to_ground_rate": hazard_ground / max(1, hazard_pixels),
            "calibration_ece": ece,
        },
        "depth": {
            "absrel": absrel_sum / max(1, depth_count),
            "rmse": math.sqrt(squared_error_sum / max(1, depth_count)),
            "log_rmse": math.sqrt(squared_log_error_sum / max(1, depth_count)),
            "delta1": delta1 / max(1, depth_count),
            "delta2": delta2 / max(1, depth_count),
            "delta3": delta3 / max(1, depth_count),
            "near_mid_far_macro_f1": float(level_f1.mean()),
            "near_mid_far_f1": {
                name: float(level_f1[i]) for i, name in enumerate(("near", "mid", "far"))
            },
            "near_mid_far_confusion": depth_level_confusion.tolist(),
            "near_far_order_accuracy": order_hits / max(1, order_pairs),
            "near_far_order_pairs": order_pairs,
            "gradient_mae": gradient_error_sum / max(1, gradient_count),
            "central_corridor_level_accuracy": corridor_level_hits / max(1, corridor_level_count),
            "central_corridor_samples": corridor_level_count,
            "pixels": depth_count,
            "corridor_temporal_stability_available": False,
        },
    }
    # Backward-readable aliases ease E0/E1 report generation.
    metrics["depth_absrel"] = metrics["depth"]["absrel"]
    metrics["depth_delta1"] = metrics["depth"]["delta1"]
    metrics["near_far_order_accuracy"] = metrics["depth"]["near_far_order_accuracy"]
    return metrics


def main() -> None:
    args = parse_args()
    fractions = {
        "ade20k": args.sampling_ade20k,
        "nyuv2": args.sampling_nyuv2,
        "stairnetv3": args.sampling_stairnetv3,
    }
    if any(value < 0.0 for value in fractions.values()):
        raise RuntimeError(f"sampling fractions must be non-negative: {fractions}")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"sampling fractions must sum to 1.0: {fractions}")
    if not 0.0 <= args.stair_step_center_prob <= 1.0:
        raise RuntimeError("stair-step-center-prob must be in [0, 1]")
    if not 0.0 <= args.stair_negative_crop_prob <= 1.0:
        raise RuntimeError("stair-negative-crop-prob must be in [0, 1]")
    if args.stair_step_center_prob + args.stair_negative_crop_prob > 1.0:
        raise RuntimeError("StairNet crop probabilities must sum to at most 1")
    if args.stair_negative_crop_attempts < 1:
        raise RuntimeError("stair-negative-crop-attempts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    e0_gradient_mae: float | None = None
    if args.e0_metrics:
        e0_payload = json.loads(args.e0_metrics.read_text(encoding="utf-8"))
        e0_gradient_mae = float(e0_payload["metrics"]["depth"]["gradient_mae"])
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=MODEL_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    config = {
        "experiment": args.experiment,
        "git_commit": git_commit,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": 5,
        "schedule": "cosine_to_5pct",
        "amp": args.amp,
        "sampling": fractions,
        "ade_step_center_prob": args.ade_step_center_prob,
        "stair_crop": {
            "step_center_prob": args.stair_step_center_prob,
            "negative_crop_prob": args.stair_negative_crop_prob,
            "random_crop_prob": 1.0 - (
                args.stair_step_center_prob + args.stair_negative_crop_prob
            ),
            "negative_crop_attempts": args.stair_negative_crop_attempts,
        },
        "segmentation": {
            "classes": list(SURFACE_CLASS_NAMES),
            "class_weights": CLASS_WEIGHTS.tolist(),
            "loss": "0.7 weighted CE + 0.3 Dice(task classes 0..2 only)",
        },
        "depth": {
            "bins": NUM_DEPTH_BINS,
            "range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "loss_profile": "legacy" if args.experiment == "e1" else "repaired_e2",
        },
        "detail64": args.experiment == "e3",
        "manifest_sha256": {
            split: sha256(args.data / f"manifest_{split}.jsonl") for split in ("train", "val")
        },
        "e0_metrics": None if args.e0_metrics is None else str(args.e0_metrics),
    }
    (args.output / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_set = SurfaceDepthDataset(
        args.data,
        "train",
        args.experiment,
        args.ade_step_center_prob,
        args.stair_step_center_prob,
        args.stair_negative_crop_prob,
        args.stair_negative_crop_attempts,
    )
    val_set = SurfaceDepthDataset(args.data, "val", args.experiment)
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
    model = GrayNavSurfaceDepth(
        width_mult=args.width_mult, detail64=args.experiment == "e3"
    ).to(device)
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
    best = {"overall": -1e9, "seg": -1e9, "step": -1e9, "depth": -1e9}
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                for row in history:
                    selections = row.get("checkpoint_scores", {})
                    for name in best:
                        if name == "overall" and not row.get("gates", {}).get("passed", False):
                            continue
                        best[name] = max(best[name], float(selections.get(name, -1e9)))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    class_weights = CLASS_WEIGHTS.to(device)
    log_dir = args.log_dir or (args.output / "tensorboard")
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"tensorboard_log_dir={log_dir}")
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
        parts = {
            name: 0.0 for name in (
                "seg", "depth_ordinal", "depth_smooth", "depth_log_l1",
                "depth_silog", "depth_gradient", "depth_grouped",
            )
        }
        progress = tqdm(
            train_loader,
            desc=f"train {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        for batch_index, batch in enumerate(progress):
            images = batch["image"].to(device, non_blocking=True)
            seg = batch["seg"].to(device, non_blocking=True)
            depth = batch["depth"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                seg_logits, depth_logits = model(images)
                loss, detail = multitask_loss(
                    seg_logits, depth_logits, seg, depth, class_weights,
                    images=images, experiment=args.experiment,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            for key in parts:
                parts[key] += detail[key]
            global_step = epoch * len(train_loader) + batch_index
            writer.add_scalar("train/loss_step", float(loss.detach()), global_step)
            writer.add_scalar("train/learning_rate_step", epoch_lr, global_step)
            progress.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                seg=f"{detail['seg']:.4f}",
                depth=f"{detail['depth_ordinal']:.4f}",
                lr=f"{epoch_lr:.2e}",
            )
        model.eval()
        metrics = evaluate(model, val_loader, device)
        gates = experiment_gates(metrics, args.experiment, e0_gradient_mae)
        iou = metrics["iou"]
        f1 = metrics["f1"]
        depth_metrics = metrics["depth"]
        clipped_absrel = min(1.0, max(0.0, float(depth_metrics["absrel"])))
        overall_score = (
            0.20 * float(iou["ground_candidate"])
            + 0.20 * float(iou["blocked_surface"])
            + 0.25 * float(f1["step_or_drop"])
            + 0.15 * float(depth_metrics["delta1"])
            + 0.15 * float(depth_metrics["near_far_order_accuracy"])
            + 0.05 * (1.0 - clipped_absrel)
        )
        checkpoint_scores = {
            "overall": overall_score,
            "seg": float(np.mean([
                f1["ground_candidate"], f1["blocked_surface"], f1["step_or_drop"]
            ])),
            "step": float(f1["step_or_drop"]),
            "depth": (
                float(depth_metrics["delta1"])
                + float(depth_metrics["near_far_order_accuracy"])
                - clipped_absrel
            ),
        }
        row = {
            "epoch": epoch,
            "lr": epoch_lr,
            "loss": running / max(1, len(train_loader)),
            **{key: value / max(1, len(train_loader)) for key, value in parts.items()},
            "metrics": metrics,
            "gates": gates,
            "checkpoint_scores": checkpoint_scores,
            "selection_score": overall_score,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        writer.add_scalar("train/loss_epoch", row["loss"], epoch)
        writer.add_scalar("train/seg_loss_epoch", row["seg"], epoch)
        writer.add_scalar("train/depth_ordinal_epoch", row["depth_ordinal"], epoch)
        writer.add_scalar("train/depth_smooth_epoch", row["depth_smooth"], epoch)
        writer.add_scalar("train/learning_rate_epoch", epoch_lr, epoch)
        writer.add_scalar("val/hazard_macro_f1", metrics["hazard_macro_f1"], epoch)
        writer.add_scalar("val/depth_absrel", metrics["depth_absrel"], epoch)
        writer.add_scalar("val/depth_delta1", metrics["depth_delta1"], epoch)
        writer.add_scalar("val/near_far_order_accuracy", metrics["near_far_order_accuracy"], epoch)
        writer.add_scalar("val/depth_rmse", metrics["depth"]["rmse"], epoch)
        writer.add_scalar("val/depth_log_rmse", metrics["depth"]["log_rmse"], epoch)
        writer.add_scalar("val/depth_gradient_mae", metrics["depth"]["gradient_mae"], epoch)
        writer.add_scalar("val/depth_near_mid_far_macro_f1", metrics["depth"]["near_mid_far_macro_f1"], epoch)
        writer.add_scalar("val/step_precision", metrics["precision"]["step_or_drop"], epoch)
        writer.add_scalar("val/step_recall", metrics["recall"]["step_or_drop"], epoch)
        for name, value in tensorboard_scalar_metrics(metrics["safety"]).items():
            writer.add_scalar(f"val_safety/{name}", value, epoch)
        for name, value in metrics["safety"].items():
            if isinstance(value, str):
                writer.add_text(f"val_safety/{name}", value, epoch)
        writer.add_scalar("val/selection_score", overall_score, epoch)
        writer.add_scalar("val/gate_passed", int(gates["passed"]), epoch)
        for name, value in metrics["iou"].items():
            writer.add_scalar(f"val_iou/{name}", value, epoch)
        for name, value in metrics["f1"].items():
            writer.add_scalar(f"val_f1/{name}", value, epoch)
        writer.flush()
        contract = {
            "model": "graynav_surface_depth_gray1",
            "input_shape": [1, 1, 256, 256],
            "seg_shape": [1, 4, 64, 64],
            "depth_shape": [1, 16, 64, 64],
            "depth_range_m": [DEPTH_MIN_M, DEPTH_MAX_M],
            "width_mult": args.width_mult,
            "experiment": args.experiment,
            "detail64": args.experiment == "e3",
            "rgb_input_used": False,
        }
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "gates": gates,
            "checkpoint_scores": checkpoint_scores,
            "contract": contract,
        }
        torch.save(checkpoint, args.output / "last.pt")
        for name in ("seg", "step", "depth"):
            if checkpoint_scores[name] > best[name]:
                best[name] = checkpoint_scores[name]
                torch.save(checkpoint, args.output / f"best_{name}.pt")
        if gates["passed"] and checkpoint_scores["overall"] > best["overall"]:
            best["overall"] = checkpoint_scores["overall"]
            torch.save(checkpoint, args.output / "best_overall.pt")
        history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    writer.close()


if __name__ == "__main__":
    main()
