#!/usr/bin/env python3
"""Train GrayNav's true-mono indoor detector and packed scene head.

The loader alternates fully supervised detection and scene/depth batches.  A
missing task is masked instead of receiving a fabricated label.  tqdm reports
the live batch state and every scalar is mirrored to TensorBoard.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.graynav_unified_perception import (  # noqa: E402
    DEPTH_BINS,
    INDOOR_CLASS_NAMES,
    GrayNavUnifiedPerception,
    build_unified_from_yolo_weights,
)
from models.graynav_surface_depth import (  # noqa: E402
    DEPTH_MAX_M,
    DEPTH_MIN_M,
    depth_bin_centers,
)
from train_surface_depth import (  # noqa: E402
    CLASS_WEIGHTS,
    IGNORE,
    SurfaceDepthDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True, help="indoor8 manifest directory")
    parser.add_argument("--scene", type=Path, required=True, help="prepared-v2 directory")
    parser.add_argument("--yolo-weights", type=Path, required=True)
    parser.add_argument("--surface-e3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--scene-warmup-epochs",
        type=int,
        default=5,
        help="scene-only adapter recovery before low-LR detection-head replay",
    )
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="micro-batches per optimizer update; use 4 with batch 8 on an 8 GiB GPU",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--partial-person-prob", type=float, default=0.40)
    parser.add_argument("--occlusion-prob", type=float, default=0.20)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument(
        "--train-detector-backbone",
        action="store_true",
        help="opt in to shared YOLO backbone updates; disabled by default to prevent detection forgetting",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _clip_boxes(boxes: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    if not len(boxes):
        return boxes.reshape(0, 5)
    x1, y1, x2, y2 = crop
    out = boxes.copy()
    out[:, 0] = np.maximum(out[:, 0], x1) - x1
    out[:, 1] = np.maximum(out[:, 1], y1) - y1
    out[:, 2] = np.minimum(out[:, 2], x2) - x1
    out[:, 3] = np.minimum(out[:, 3], y2) - y1
    width = np.maximum(out[:, 2] - out[:, 0], 0.0)
    height = np.maximum(out[:, 3] - out[:, 1], 0.0)
    original_area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1.0)
    keep = (width >= 4.0) & (height >= 8.0) & ((width * height) / original_area >= 0.20)
    return out[keep]


def choose_partial_person_crop(
    width: int,
    height: int,
    boxes: np.ndarray,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    """Create upper/lower/side truncation while retaining useful context."""

    people = boxes[boxes[:, 4] == 0]
    if not len(people):
        return None
    px1, py1, px2, py2, _ = people[rng.randrange(len(people))]
    pw, ph = px2 - px1, py2 - py1
    retained = rng.uniform(0.45, 0.80)
    mode = rng.choice(("upper", "lower", "left", "right"))
    margin_x, margin_y = 0.30 * pw, 0.25 * ph
    if mode == "upper":
        cx1, cy1, cx2, cy2 = px1 - margin_x, py1 - margin_y, px2 + margin_x, py1 + retained * ph
    elif mode == "lower":
        cx1, cy1, cx2, cy2 = px1 - margin_x, py2 - retained * ph, px2 + margin_x, py2 + margin_y
    elif mode == "left":
        cx1, cy1, cx2, cy2 = px1 - margin_x, py1 - margin_y, px1 + retained * pw, py2 + margin_y
    else:
        cx1, cy1, cx2, cy2 = px2 - retained * pw, py1 - margin_y, px2 + margin_x, py2 + margin_y
    crop_w = max(cx2 - cx1, 0.55 * max(pw, ph))
    crop_h = max(cy2 - cy1, 0.55 * max(pw, ph))
    center_x, center_y = 0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2)
    side = max(crop_w, crop_h)
    x1 = int(round(np.clip(center_x - side / 2, 0, max(0, width - side))))
    y1 = int(round(np.clip(center_y - side / 2, 0, max(0, height - side))))
    x2 = int(round(min(width, x1 + side)))
    y2 = int(round(min(height, y1 + side)))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


class IndoorDetectionDataset(Dataset[dict[str, object]]):
    def __init__(self, root: Path, split: str, seed: int, partial_prob: float, occlusion_prob: float) -> None:
        self.rows = load_jsonl(root / f"manifest_{split}.jsonl")
        self.training = split == "train"
        self.seed = seed
        self.partial_prob = partial_prob
        self.occlusion_prob = occlusion_prob

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = cv2.imread(str(row["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"cannot read {row['image']}")
        boxes_xywh = np.asarray(row["boxes_xywh"], dtype=np.float32).reshape(-1, 5)
        boxes = boxes_xywh.copy()
        if len(boxes):
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
        augmentation_nonce = random.randint(0, 2**20) if self.training else 0
        rng = random.Random(self.seed * 1_000_003 + index + augmentation_nonce)
        if self.partial_prob > 0.0 and rng.random() < self.partial_prob:
            crop = choose_partial_person_crop(image.shape[1], image.shape[0], boxes, rng)
            if crop is not None:
                boxes = _clip_boxes(boxes, crop)
                x1, y1, x2, y2 = crop
                image = image[y1:y2, x1:x2]

        scale = min(384.0 / image.shape[1], 384.0 / image.shape[0])
        resized_w, resized_h = int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((384, 384), 114, dtype=np.uint8)
        pad_x, pad_y = (384 - resized_w) // 2, (384 - resized_h) // 2
        canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
        if len(boxes):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
        if self.training and rng.random() < 0.5:
            canvas = np.ascontiguousarray(canvas[:, ::-1])
            if len(boxes):
                old_x1 = boxes[:, 0].copy()
                boxes[:, 0] = 384.0 - boxes[:, 2]
                boxes[:, 2] = 384.0 - old_x1
        if self.training:
            gamma = rng.uniform(0.65, 1.55)
            canvas = np.clip((canvas.astype(np.float32) / 255.0) ** gamma * rng.uniform(0.75, 1.25), 0, 1)
            if rng.random() < self.occlusion_prob and len(boxes):
                person = boxes[boxes[:, 4] == 0]
                if len(person):
                    x1, y1, x2, y2, _ = person[rng.randrange(len(person))]
                    ow = max(4, int((x2 - x1) * rng.uniform(0.15, 0.40)))
                    oh = max(4, int((y2 - y1) * rng.uniform(0.15, 0.40)))
                    ox = int(rng.uniform(x1, max(x1, x2 - ow)))
                    oy = int(rng.uniform(y1, max(y1, y2 - oh)))
                    canvas[oy : oy + oh, ox : ox + ow] = rng.uniform(0.1, 0.8)
        else:
            canvas = canvas.astype(np.float32) / 255.0

        cls = boxes[:, 4:5].astype(np.float32) if len(boxes) else np.zeros((0, 1), np.float32)
        xywh = np.zeros((len(boxes), 4), dtype=np.float32)
        if len(boxes):
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / 384.0
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / 384.0
            xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / 384.0
            xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / 384.0
        return {
            "img": torch.from_numpy(canvas[None].astype(np.float32)),
            "cls": torch.from_numpy(cls),
            "bboxes": torch.from_numpy(xywh),
            "source_id": str(row["source_id"]),
        }


def collate_detection(rows: list[dict[str, object]]) -> dict[str, object]:
    batch_idx, classes, boxes = [], [], []
    for index, row in enumerate(rows):
        count = int(row["cls"].shape[0])  # type: ignore[union-attr]
        if count:
            batch_idx.append(torch.full((count,), index, dtype=torch.long))
            classes.append(row["cls"])
            boxes.append(row["bboxes"])
    return {
        "img": torch.stack([row["img"] for row in rows]),
        "batch_idx": torch.cat(batch_idx) if batch_idx else torch.zeros(0, dtype=torch.long),
        "cls": torch.cat(classes) if classes else torch.zeros((0, 1), dtype=torch.float32),
        "bboxes": torch.cat(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "source_id": [row["source_id"] for row in rows],
    }


def infinite(loader: DataLoader) -> Iterator[dict[str, object]]:
    while True:
        yield from loader


def interleaved_source_indices(records: list[dict[str, object]]) -> list[int]:
    """Round-robin validation sources so bounded audits cover every task."""

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(str(row["source"]), []).append(index)
    ordered_sources = [
        source for source in ("ade20k", "nyuv2", "stairnetv3")
        if source in groups
    ]
    ordered_sources.extend(sorted(set(groups).difference(ordered_sources)))
    offsets = {source: 0 for source in ordered_sources}
    result: list[int] = []
    while True:
        appended = False
        for source in ordered_sources:
            offset = offsets[source]
            if offset >= len(groups[source]):
                continue
            result.append(groups[source][offset])
            offsets[source] = offset + 1
            appended = True
        if not appended:
            return result


def scene_loss(scene: torch.Tensor, batch: dict[str, object]) -> tuple[torch.Tensor, dict[str, float]]:
    seg_logits, depth_logits, edge_logits = scene[:, :4], scene[:, 4:20], scene[:, 20:21]
    seg = F.interpolate(batch["seg"][:, None].float(), size=(48, 48), mode="nearest")[:, 0].long()
    depth = F.interpolate(batch["depth"][:, None], size=(48, 48), mode="nearest")[:, 0]
    valid_seg = seg != IGNORE
    if bool(valid_seg.any()):
        ce = F.cross_entropy(seg_logits, seg, weight=CLASS_WEIGHTS.to(scene.device), ignore_index=IGNORE)
        safe = seg.masked_fill(~valid_seg, 0)
        one_hot = F.one_hot(safe, 4).permute(0, 3, 1, 2).float() * valid_seg[:, None]
        probs = torch.softmax(seg_logits, 1) * valid_seg[:, None]
        intersection = (probs * one_hot).sum((0, 2, 3))[:3]
        denominator = (probs + one_hot).sum((0, 2, 3))[:3]
        dice = 1.0 - ((2 * intersection + 1) / (denominator + 1)).mean()
        seg_value = 0.7 * ce + 0.3 * dice
        step = (seg == 2).float()[:, None]
        dilated = F.max_pool2d(step, 3, 1, 1)
        eroded = -F.max_pool2d(-step, 3, 1, 1)
        edge_target = ((dilated - eroded) > 0).float() * valid_seg[:, None]
        bce = F.binary_cross_entropy_with_logits(edge_logits, edge_target, reduction="none")
        focal = ((torch.sigmoid(edge_logits) - edge_target).abs().pow(2) * bce)[valid_seg[:, None]].mean()
        edge_prob = torch.sigmoid(edge_logits) * valid_seg[:, None]
        edge_dice = 1.0 - (2 * (edge_prob * edge_target).sum() + 1) / (edge_prob.sum() + edge_target.sum() + 1)
        edge_value = 0.7 * 8.0 * focal + 0.3 * edge_dice
    else:
        seg_value = seg_logits.sum() * 0.0
        edge_value = edge_logits.sum() * 0.0

    valid_depth = torch.isfinite(depth) & (depth >= DEPTH_MIN_M) & (depth <= DEPTH_MAX_M)
    if bool(valid_depth.any()):
        normalized = (torch.log(depth.clamp_min(DEPTH_MIN_M)) - math.log(DEPTH_MIN_M)) / (math.log(DEPTH_MAX_M) - math.log(DEPTH_MIN_M))
        bins = torch.floor(normalized * DEPTH_BINS).long().clamp(0, DEPTH_BINS - 1)
        ordinal = F.cross_entropy(depth_logits, bins, reduction="none")[valid_depth].mean()
        centers = depth_bin_centers(depth_logits.device).view(1, -1, 1, 1)
        estimate = (torch.softmax(depth_logits, 1) * centers).sum(1)
        log_error = torch.log(estimate.clamp_min(1e-3)) - torch.log(depth.clamp_min(1e-3))
        selected = log_error[valid_depth]
        log_l1 = selected.abs().mean()
        silog = torch.sqrt((selected.square().mean() - 0.85 * selected.mean().square()).clamp_min(1e-8))
        gradients = []
        for pred_delta, gt_delta, mask in (
            (estimate[:, :, 1:] - estimate[:, :, :-1], depth[:, :, 1:] - depth[:, :, :-1], valid_depth[:, :, 1:] & valid_depth[:, :, :-1]),
            (estimate[:, 1:, :] - estimate[:, :-1, :], depth[:, 1:, :] - depth[:, :-1, :], valid_depth[:, 1:, :] & valid_depth[:, :-1, :]),
        ):
            if bool(mask.any()):
                gradients.append((pred_delta - gt_delta).abs()[mask].mean())
        gradient = torch.stack(gradients).mean() if gradients else ordinal * 0.0
        image = F.interpolate(
            batch["image"], size=(48, 48), mode="bilinear", align_corners=False
        )[:, 0]
        smooth_terms = []
        for pred_delta, image_delta, mask in (
            (
                estimate[:, :, 1:] - estimate[:, :, :-1],
                image[:, :, 1:] - image[:, :, :-1],
                valid_depth[:, :, 1:] & valid_depth[:, :, :-1],
            ),
            (
                estimate[:, 1:, :] - estimate[:, :-1, :],
                image[:, 1:, :] - image[:, :-1, :],
                valid_depth[:, 1:, :] & valid_depth[:, :-1, :],
            ),
        ):
            if bool(mask.any()):
                edge_weight = torch.exp(-10.0 * image_delta.abs())
                smooth_terms.append((pred_delta.abs() * edge_weight)[mask].mean())
        smooth = torch.stack(smooth_terms).mean() if smooth_terms else ordinal * 0.0

        # Grouping the 16 ordinal bins into the public NEAR/MID/FAR contract
        # directly supervises the board decision boundary without pretending
        # that public-camera depth is calibrated metric distance on SC132GS.
        bin_values = depth_bin_centers(depth_logits.device)
        near_end = int((bin_values < 1.25).sum())
        mid_end = int((bin_values < 2.20).sum())
        grouped_logits = torch.stack(
            (
                torch.logsumexp(depth_logits[:, :near_end], dim=1),
                torch.logsumexp(depth_logits[:, near_end:mid_end], dim=1),
                torch.logsumexp(depth_logits[:, mid_end:], dim=1),
            ),
            dim=1,
        )
        grouped_target = torch.where(
            depth < 1.25,
            torch.zeros_like(bins),
            torch.where(depth < 2.20, torch.ones_like(bins), torch.full_like(bins, 2)),
        )
        grouped_ce = F.cross_entropy(
            grouped_logits, grouped_target, reduction="none"
        )[valid_depth].mean()
        depth_value = (
            0.35 * ordinal
            + 0.20 * log_l1
            + 0.15 * silog
            + 0.15 * gradient
            + 0.05 * smooth
            + 0.10 * grouped_ce
        )
    else:
        depth_value = depth_logits.sum() * 0.0
    total = seg_value + 0.4 * depth_value + 1.2 * edge_value
    return total, {"seg": float(seg_value.detach()), "depth": float(depth_value.detach()), "edge": float(edge_value.detach())}


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    result = boxes.clone()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    result[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    result[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    result[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return result


def _box_iou(one: torch.Tensor, many: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(one[:2], many[:, :2])
    bottom_right = torch.minimum(one[2:], many[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(1)
    one_area = (one[2:] - one[:2]).clamp_min(0).prod()
    many_area = (many[:, 2:] - many[:, :2]).clamp_min(0).prod(1)
    return intersection / (one_area + many_area - intersection).clamp_min(1e-9)


def _average_precision(records: list[tuple[float, bool]], positives: int) -> float:
    if positives <= 0:
        return float("nan")
    if not records:
        return 0.0
    ordered = sorted(records, key=lambda item: item[0], reverse=True)
    tp = np.cumsum([int(item[1]) for item in ordered], dtype=np.float64)
    fp = np.cumsum([int(not item[1]) for item in ordered], dtype=np.float64)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1.0)
    return float(np.mean([
        precision[recall >= threshold].max(initial=0.0)
        for threshold in np.linspace(0.0, 1.0, 101)
    ]))


@torch.no_grad()
def evaluate_detection(
    model: GrayNavUnifiedPerception,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    from ultralytics.utils.ops import non_max_suppression

    model.eval()
    records: list[list[tuple[float, bool]]] = [
        [] for _ in INDOOR_CLASS_NAMES
    ]
    positives = [0 for _ in INDOOR_CLASS_NAMES]
    matched_people = 0
    total_people = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        images = batch["img"].to(device, non_blocking=True)
        outputs = model(images)
        features = [
            torch.cat((outputs[1], outputs[0]), 1),
            torch.cat((outputs[3], outputs[2]), 1),
            torch.cat((outputs[5], outputs[4]), 1),
        ]
        prediction = model.detect_head._inference(features)
        detections = non_max_suppression(
            prediction,
            conf_thres=0.001,
            iou_thres=0.60,
            max_det=100,
            nc=len(INDOOR_CLASS_NAMES),
        )
        batch_indices = batch["batch_idx"]
        classes = batch["cls"][:, 0].long()
        boxes = _xywh_to_xyxy(batch["bboxes"]) * 384.0
        for image_index, detected in enumerate(detections):
            mask = batch_indices == image_index
            truth_classes = classes[mask]
            truth_boxes = boxes[mask]
            used = torch.zeros(len(truth_boxes), dtype=torch.bool)
            total_people += int((truth_classes == 0).sum())
            for row in detected.detach().cpu():
                cls_id = int(row[5])
                candidates = torch.where((truth_classes == cls_id) & ~used)[0]
                is_true = False
                if len(candidates):
                    ious = _box_iou(row[:4], truth_boxes[candidates])
                    best_local = int(torch.argmax(ious))
                    if float(ious[best_local]) >= 0.50:
                        matched = int(candidates[best_local])
                        used[matched] = True
                        is_true = True
                        if cls_id == 0:
                            matched_people += 1
                records[cls_id].append((float(row[4]), is_true))
            for cls_id in range(len(INDOOR_CLASS_NAMES)):
                positives[cls_id] += int((truth_classes == cls_id).sum())
    aps = [_average_precision(records[index], positives[index]) for index in range(len(records))]
    valid_aps = [value for value in aps if math.isfinite(value)]
    return {
        "map50": float(np.mean(valid_aps)) if valid_aps else 0.0,
        "person_ap50": 0.0 if not math.isfinite(aps[0]) else aps[0],
        "person_recall": matched_people / max(1, total_people),
        **{f"ap50_{name}": 0.0 if not math.isfinite(aps[index]) else aps[index]
           for index, name in enumerate(INDOOR_CLASS_NAMES)},
    }


@torch.no_grad()
def evaluate_scene(model: GrayNavUnifiedPerception, loader: DataLoader, device: torch.device, max_batches: int = 100) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    edge_tp = edge_fp = edge_fn = 0
    depth_absrel_sum = 0.0
    depth_delta1_count = 0
    depth_pixel_count = 0
    depth_medians: list[tuple[float, float]] = []
    no_step_images = false_step_images = whole_step_images = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        images = F.interpolate(batch["image"].to(device), size=(384, 384), mode="bilinear", align_corners=False)
        scene = model(images)[-1]
        target = F.interpolate(batch["seg"][:, None].float(), size=(48, 48), mode="nearest")[:, 0].long()
        prediction = scene[:, :4].argmax(1).cpu()
        valid = target != IGNORE
        encoded = target[valid].cpu() * 4 + prediction[valid]
        confusion += torch.bincount(encoded, minlength=16).reshape(4, 4)
        step = (target == 2).float()[:, None]
        truth_edge = (F.max_pool2d(step, 3, 1, 1) + F.max_pool2d(-step, 3, 1, 1) > 0) & valid[:, None]
        pred_edge = (torch.sigmoid(scene[:, 20:21]).cpu() >= 0.5) & valid[:, None]
        edge_tp += int((pred_edge & truth_edge).sum())
        edge_fp += int((pred_edge & ~truth_edge & valid[:, None]).sum())
        edge_fn += int((~pred_edge & truth_edge).sum())
        for sample_index in range(len(prediction)):
            sample_valid = valid[sample_index]
            if bool(sample_valid.any()):
                truth_has_step = bool((target[sample_index][sample_valid] == 2).any())
                predicted_step_ratio = float(
                    (prediction[sample_index][sample_valid] == 2).float().mean()
                )
                whole_step_images += int(predicted_step_ratio > 0.60)
                if not truth_has_step:
                    no_step_images += 1
                    false_step_images += int(predicted_step_ratio > 0.03)

        depth = F.interpolate(
            batch["depth"][:, None].to(device),
            size=(48, 48),
            mode="nearest",
        )[:, 0]
        valid_depth = (
            torch.isfinite(depth)
            & (depth >= DEPTH_MIN_M)
            & (depth <= DEPTH_MAX_M)
        )
        centers = depth_bin_centers(device).view(1, -1, 1, 1)
        estimate = (torch.softmax(scene[:, 4:20], 1) * centers).sum(1)
        if bool(valid_depth.any()):
            ratio = torch.maximum(
                estimate[valid_depth] / depth[valid_depth],
                depth[valid_depth] / estimate[valid_depth].clamp_min(1e-6),
            )
            depth_absrel_sum += float(
                ((estimate[valid_depth] - depth[valid_depth]).abs()
                 / depth[valid_depth].clamp_min(1e-6)).sum()
            )
            depth_delta1_count += int((ratio < 1.25).sum())
            depth_pixel_count += int(valid_depth.sum())
        for sample_index in range(len(estimate)):
            sample_valid = valid_depth[sample_index]
            if int(sample_valid.sum()) >= 32:
                depth_medians.append((
                    float(depth[sample_index][sample_valid].median()),
                    float(estimate[sample_index][sample_valid].median()),
                ))
    tp = confusion.diag().float()
    fp = confusion.sum(0).float() - tp
    fn = confusion.sum(1).float() - tp
    iou = tp / (tp + fp + fn).clamp_min(1)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    edge_f1 = 2 * edge_tp / max(1, 2 * edge_tp + edge_fp + edge_fn)
    ordered_pairs = 0
    correct_pairs = 0
    for index in range(1, len(depth_medians)):
        truth_delta = depth_medians[index][0] - depth_medians[index - 1][0]
        pred_delta = depth_medians[index][1] - depth_medians[index - 1][1]
        if abs(truth_delta) >= 0.25:
            ordered_pairs += 1
            correct_pairs += int(truth_delta * pred_delta > 0)
    return {
        "ground_iou": float(iou[0]),
        "blocked_iou": float(iou[1]),
        "step_f1": float(f1[2]),
        "edge_f1": edge_f1,
        "no_step_false_image_rate": false_step_images / max(1, no_step_images),
        "whole_step_image_count": float(whole_step_images),
        "depth_absrel": depth_absrel_sum / max(1, depth_pixel_count),
        "depth_delta1": depth_delta1_count / max(1, depth_pixel_count),
        "near_far_order_accuracy": correct_pairs / max(1, ordered_pairs),
        "near_far_order_pairs": float(ordered_pairs),
    }


def checkpoint_payload(model: GrayNavUnifiedPerception, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float], config: dict[str, object]) -> dict[str, object]:
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics, "contract": config, "rgb_input_used": False, "one_channel_first_conv_initialized": True}


def main() -> None:
    args = parse_args()
    if not 0 <= args.scene_warmup_epochs < args.epochs:
        raise ValueError("scene warm-up must be shorter than the full run")
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("batch size and accumulation steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, init_report = build_unified_from_yolo_weights(args.yolo_weights)
    init_report["surface_e3_import"] = model.import_surface_e3_heads(args.surface_e3)
    model.to(device)

    from ultralytics.cfg import get_cfg
    from ultralytics.utils.loss import v8DetectionLoss
    # A standalone DetectionModel loaded from a checkpoint may carry a plain
    # YAML dictionary in ``args``.  The loss implementation expects the
    # attribute-based default configuration normally attached by Trainer.
    model.detector.args = get_cfg()
    criterion = v8DetectionLoss(model.detector)
    detector_head_parameters = list(model.detect_head.parameters())
    detector_head_ids = {id(parameter) for parameter in detector_head_parameters}
    scene_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in detector_head_ids
        and not any(
            id(parameter) == id(detector_parameter)
            for detector_parameter in model.detector.parameters()
        )
    ]
    parameter_groups = [
        {"params": scene_parameters, "lr_scale": 1.0},
        {"params": detector_head_parameters, "lr_scale": 0.2},
    ]
    if args.train_detector_backbone:
        backbone_parameters = [
            parameter
            for parameter in model.detector.parameters()
            if id(parameter) not in detector_head_ids
        ]
        parameter_groups.append({"params": backbone_parameters, "lr_scale": 0.1})
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1

    train_det = IndoorDetectionDataset(args.coco, "train", args.seed, args.partial_person_prob, args.occlusion_prob)
    val_det = IndoorDetectionDataset(args.coco, "val", args.seed, 0.0, 0.0)
    val_partial_person = IndoorDetectionDataset(
        args.coco, "val", args.seed, 1.0, 0.0
    )
    train_scene = SurfaceDepthDataset(args.scene, "train", experiment="e3", ade_step_center_prob=0.5, stair_step_center_prob=0.6, stair_negative_crop_prob=0.2)
    val_scene = SurfaceDepthDataset(args.scene, "val", experiment="e3")
    source_target = {"ade20k": 0.20, "stairnetv3": 0.20, "nyuv2": 0.15}
    counts = Counter(str(row["source"]) for row in train_scene.records)
    weights = [source_target[str(row["source"])] / counts[str(row["source"])] for row in train_scene.records]
    micro_steps_per_epoch = args.steps_per_epoch * args.accumulation_steps
    scene_samples = micro_steps_per_epoch * args.batch_size
    scene_sampler = WeightedRandomSampler(weights, num_samples=scene_samples, replacement=True)
    det_loader = DataLoader(train_det, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_detection, drop_last=True)
    scene_loader = DataLoader(train_scene, batch_size=args.batch_size, sampler=scene_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_det_loader = DataLoader(val_det, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_detection)
    val_partial_loader = DataLoader(val_partial_person, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_detection)
    val_scene_loader = DataLoader(
        val_scene,
        batch_size=args.batch_size,
        sampler=interleaved_source_indices(val_scene.records),
        num_workers=args.workers,
        pin_memory=True,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": "graynav_unified_indoor8_scene21",
        "input": [1, 1, 384, 384],
        "outputs": 7,
        "indoor_classes": list(INDOOR_CLASS_NAMES),
        "sampling": {"compact_detection": 0.45, **source_target},
        "epochs": args.epochs,
        "scene_warmup_epochs": args.scene_warmup_epochs,
        "detection_head_lr_scale": 0.2,
        "detector_backbone_lr_scale": 0.1 if args.train_detector_backbone else 0.0,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": args.batch_size * args.accumulation_steps,
        "optimizer_steps_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
        "detector_backbone_frozen": not args.train_detector_backbone,
        "initialization": init_report,
    }
    (args.output / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    writer = SummaryWriter(str(args.log_dir))
    history_path = args.output / "history.json"
    history: list[dict[str, object]] = []
    if start_epoch > 0 and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if len(history) != start_epoch:
            raise RuntimeError(
                f"resume history/checkpoint mismatch: history={len(history)} start={start_epoch}"
            )
    best = {
        "overall": -float("inf"),
        "detection": -float("inf"),
        "stair": -float("inf"),
        "scene": -float("inf"),
        "safety": -float("inf"),
    }
    for previous in history:
        previous_metrics = previous["metrics"]
        previous_scene = 0.5 * previous_metrics["ground_iou"] + 0.5 * previous_metrics["blocked_iou"]
        previous_stair = 0.6 * previous_metrics["step_f1"] + 0.4 * previous_metrics["edge_f1"]
        previous_detection = (
            0.55 * previous_metrics["map50"]
            + 0.25 * previous_metrics["person_ap50"]
            + 0.20 * previous_metrics["partial_person_recall"]
        )
        previous_safety = (
            0.45 * previous_metrics["step_f1"]
            + 0.20 * previous_metrics["edge_f1"]
            + 0.20 * previous_scene
            + 0.15 * previous_metrics["near_far_order_accuracy"]
            - 0.35 * previous_metrics["no_step_false_image_rate"]
        )
        best["scene"] = max(best["scene"], previous_scene)
        best["stair"] = max(best["stair"], previous_stair)
        best["detection"] = max(best["detection"], previous_detection)
        best["safety"] = max(best["safety"], previous_safety)
        best["overall"] = max(best["overall"], previous["selection_score"])
    # Older/resumed runs may predate the dedicated safety checkpoint.  In
    # that case, force the first newly evaluated epoch to materialize one;
    # otherwise the historical scalar maximum could prevent best_safety.pt
    # from ever being written even though selection is enabled now.
    if history and not (args.output / "best_safety.pt").is_file():
        best["safety"] = -float("inf")
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(start_epoch, args.epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if epoch < args.scene_warmup_epochs:
            learning_rate = args.lr * float(epoch + 1) / max(1, args.scene_warmup_epochs)
        else:
            progress_ratio = (epoch - args.scene_warmup_epochs) / max(1, args.epochs - args.scene_warmup_epochs - 1)
            learning_rate = args.lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress_ratio)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))
        scene_warmup = epoch < args.scene_warmup_epochs
        for parameter in scene_parameters:
            parameter.requires_grad_(True)
        for parameter in detector_head_parameters:
            parameter.requires_grad_(not scene_warmup)
        if not args.train_detector_backbone:
            for parameter in model.detector.parameters():
                parameter.requires_grad_(False)
            for parameter in detector_head_parameters:
                parameter.requires_grad_(not scene_warmup)
        model.train()
        if not args.train_detector_backbone:
            # Freezing parameters alone does not freeze BatchNorm running
            # statistics.  Keep the shared detector graph in eval mode while
            # allowing the task heads and scene branch to train normally.
            model.detector.eval()
            model.detect_head.train(not scene_warmup)
        det_iter, scene_iter = infinite(det_loader), infinite(scene_loader)
        totals = Counter()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            range(micro_steps_per_epoch),
            desc=f"unified {epoch + 1}/{args.epochs}",
        )
        for step in progress:
            detection_step = (
                False if scene_warmup else random.random() < (0.45 / 1.0)
            )
            with torch.cuda.amp.autocast(enabled=args.amp):
                if detection_step:
                    batch = next(det_iter)
                    batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    outputs = model.forward_detection(batch["img"])
                    features = [torch.cat((outputs[1], outputs[0]), 1), torch.cat((outputs[3], outputs[2]), 1), torch.cat((outputs[5], outputs[4]), 1)]
                    loss, parts = criterion(features, batch)
                    loss = loss.sum()
                    detail = {"detect": float(loss.detach()), "seg": 0.0, "depth": 0.0, "edge": 0.0}
                else:
                    batch = next(scene_iter)
                    batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    images = F.interpolate(batch["image"], size=(384, 384), mode="bilinear", align_corners=False)
                    scene = model.forward_scene(images)
                    loss, scene_parts = scene_loss(scene, batch)
                    detail = {"detect": 0.0, **scene_parts}
            scaler.scale(loss / args.accumulation_steps).backward()
            if (step + 1) % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            totals["loss"] += float(loss.detach())
            totals["det_steps" if detection_step else "scene_steps"] += 1
            for key, value in detail.items():
                totals[key] += value
            progress.set_postfix(loss=f"{totals['loss'] / (step + 1):.4f}", task="det" if detection_step else "scene")

        metrics = evaluate_scene(
            model, val_scene_loader, device, max_batches=args.validation_batches
        )
        detection_metrics = evaluate_detection(
            model, val_det_loader, device, max_batches=args.validation_batches
        )
        partial_metrics = evaluate_detection(
            model, val_partial_loader, device, max_batches=args.validation_batches
        )
        metrics.update(detection_metrics)
        metrics["partial_person_recall"] = partial_metrics["person_recall"]
        epoch_loss = totals["loss"] / micro_steps_per_epoch
        detection_loss = totals["detect"] / max(1, totals["det_steps"])
        scene_loss_mean = (
            totals["seg"] + totals["depth"] + totals["edge"]
        ) / max(1, totals["scene_steps"])
        scene_score = 0.5 * metrics["ground_iou"] + 0.5 * metrics["blocked_iou"]
        stair_score = 0.6 * metrics["step_f1"] + 0.4 * metrics["edge_f1"]
        detection_score = (
            0.55 * metrics["map50"]
            + 0.25 * metrics["person_ap50"]
            + 0.20 * metrics["partial_person_recall"]
        )
        overall = 0.35 * scene_score + 0.30 * stair_score + 0.35 * detection_score
        safety_score = (
            0.45 * metrics["step_f1"]
            + 0.20 * metrics["edge_f1"]
            + 0.20 * scene_score
            + 0.15 * metrics["near_far_order_accuracy"]
            - 0.35 * metrics["no_step_false_image_rate"]
        )
        record = {
            "epoch": epoch,
            "lr": learning_rate,
            "loss": epoch_loss,
            "detection_loss": detection_loss,
            "scene_loss": scene_loss_mean,
            "metrics": metrics,
            "selection_score": overall,
            "safety_score": safety_score,
        }
        if device.type == "cuda":
            record["gpu_peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated(device) / 1024**2
            )
            record["gpu_peak_reserved_mib"] = (
                torch.cuda.max_memory_reserved(device) / 1024**2
            )
        history.append(record)
        for key, value in {"lr": learning_rate, "loss/total": epoch_loss, "loss/detection": detection_loss, **{f"metrics/{k}": v for k, v in metrics.items()}, "metrics/selection_score": overall, **{f"resources/{k}": v for k, v in record.items() if k.startswith("gpu_peak_")}}.items():
            writer.add_scalar(key, value, epoch)
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            {
                **metrics,
                "detection_loss": detection_loss,
                "selection_score": overall,
                "safety_score": safety_score,
            },
            config,
        )
        torch.save(payload, args.output / "last.pt")
        if detection_score > best["detection"]:
            best["detection"] = detection_score
            torch.save(payload, args.output / "best_detection.pt")
        if scene_score > best["scene"]:
            best["scene"] = scene_score
            torch.save(payload, args.output / "best_scene.pt")
        if stair_score > best["stair"]:
            best["stair"] = stair_score
            torch.save(payload, args.output / "best_stair.pt")
        if safety_score > best["safety"]:
            best["safety"] = safety_score
            torch.save(payload, args.output / "best_safety.pt")
        if overall > best["overall"]:
            best["overall"] = overall
            torch.save(payload, args.output / "best_overall.pt")
        (args.output / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False))
    writer.close()
    print("GRAYNAV_UNIFIED_INDOOR_TRAINING_OK")


if __name__ == "__main__":
    main()
