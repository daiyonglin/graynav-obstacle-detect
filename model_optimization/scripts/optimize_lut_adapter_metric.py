#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from gray_adapter import lut_uint8_to_adapter, save_adapter_bundle


COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

NAV_LABELS = {
    "person",
    "chair",
    "bench",
    "couch",
    "bed",
    "dining table",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "book",
    "laptop",
    "keyboard",
    "cell phone",
    "remote",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
}


@dataclass
class Candidate:
    strength: float
    gamma: list[float]
    contrast: list[float]
    bias: list[float]
    knot_offsets: list[list[float]]


@dataclass
class CandidateScore:
    candidate_id: int
    score: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    prediction_count: int
    params: Candidate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric-driven search for a deployable 3x256 gray LUT adapter.")
    p.add_argument("--images", required=True, type=Path, help="COCO val/train image directory.")
    p.add_argument("--annotations", required=True, type=Path, help="instances_*.json.")
    p.add_argument("--weights", required=True, type=Path, help="YOLOv8 .pt weights.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-images", type=int, default=300)
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--population", type=int, default=20)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-nms", type=float, default=0.70)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--recall-weight", type=float, default=0.20, help="Objective = F1 + recall_weight * Recall - fp_weight * FP/image.")
    p.add_argument("--fp-weight", type=float, default=0.01)
    p.add_argument("--knot-count", type=int, default=7, help="Per-channel piecewise tone offset knots. More knots increase search space.")
    p.add_argument("--knot-range", type=float, default=0.045, help="Initial absolute range for knot offsets in normalized pixel units.")
    p.add_argument("--seed", type=int, default=20260622)
    p.add_argument("--save-every-generation", action="store_true")
    return p.parse_args()


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_images(coco: dict[str, Any], image_dir: Path, max_images: int, seed: int) -> list[dict[str, Any]]:
    images = [im for im in coco["images"] if (image_dir / im["file_name"]).exists()]
    images = sorted(images, key=lambda x: x["file_name"])
    rng = random.Random(seed)
    rng.shuffle(images)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"no annotated images found in {image_dir}")
    return images


def nav_category_ids(coco: dict[str, Any]) -> set[int]:
    name_to_id = {c["name"]: int(c["id"]) for c in coco["categories"]}
    return {name_to_id[name] for name in NAV_LABELS if name in name_to_id}


def build_gt_index(coco: dict[str, Any], image_ids: set[int], nav_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in image_ids or int(ann.get("iscrowd", 0)) != 0:
            continue
        cat_id = int(ann["category_id"])
        if cat_id not in nav_ids:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        out[image_id].append({"category_id": cat_id, "bbox": [x, y, x + w, y + h]})
    return out


def identity_candidate(knot_count: int) -> Candidate:
    return Candidate(
        strength=0.0,
        gamma=[1.0, 1.0, 1.0],
        contrast=[1.0, 1.0, 1.0],
        bias=[0.0, 0.0, 0.0],
        knot_offsets=[[0.0 for _ in range(knot_count)] for _ in range(3)],
    )


def random_candidate(rng: random.Random, knot_count: int, knot_range: float) -> Candidate:
    return Candidate(
        strength=rng.uniform(0.15, 0.85),
        gamma=[rng.uniform(0.75, 1.30) for _ in range(3)],
        contrast=[rng.uniform(0.85, 1.18) for _ in range(3)],
        bias=[rng.uniform(-0.045, 0.045) for _ in range(3)],
        knot_offsets=[[rng.uniform(-knot_range, knot_range) for _ in range(knot_count)] for _ in range(3)],
    )


def mutate(parent: Candidate, rng: random.Random, scale: float) -> Candidate:
    def clamp(v: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, v))

    return Candidate(
        strength=clamp(parent.strength + rng.gauss(0.0, 0.18 * scale), 0.0, 1.0),
        gamma=[clamp(v + rng.gauss(0.0, 0.12 * scale), 0.55, 1.70) for v in parent.gamma],
        contrast=[clamp(v + rng.gauss(0.0, 0.08 * scale), 0.65, 1.45) for v in parent.contrast],
        bias=[clamp(v + rng.gauss(0.0, 0.025 * scale), -0.12, 0.12) for v in parent.bias],
        knot_offsets=[
            [clamp(v + rng.gauss(0.0, 0.025 * scale), -0.16, 0.16) for v in channel]
            for channel in parent.knot_offsets
        ],
    )


def candidate_to_lut(c: Candidate) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    knot_x = np.linspace(0.0, 1.0, len(c.knot_offsets[0]), dtype=np.float32)
    table = []
    for ch_idx, (gamma, contrast, bias) in enumerate(zip(c.gamma, c.contrast, c.bias)):
        y = np.power(x, gamma)
        y = (y - 0.5) * contrast + 0.5 + bias
        offset = np.interp(x, knot_x, np.asarray(c.knot_offsets[ch_idx], dtype=np.float32))
        y = y + offset
        y = (1.0 - c.strength) * x + c.strength * y
        y = np.clip(y, 0.0, 1.0)
        # Enforce a monotonic LUT. This avoids unstable inversions and keeps board behavior explainable.
        y = np.maximum.accumulate(y)
        table.append(np.round(y * 255.0).astype(np.uint8))
    return np.stack(table, axis=0)


def apply_lut(gray: np.ndarray, lut: np.ndarray) -> Image.Image:
    rgb = np.stack([lut[0][gray], lut[1][gray], lut[2][gray]], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_candidate(
    candidate_id: int,
    candidate: Candidate,
    model: YOLO,
    images: list[dict[str, Any]],
    image_dir: Path,
    gt: dict[int, list[dict[str, Any]]],
    nav_ids: set[int],
    args: argparse.Namespace,
) -> CandidateScore:
    lut = candidate_to_lut(candidate)
    tp = fp = fn = prediction_count = 0
    for offset in range(0, len(images), args.batch):
        metas = images[offset : offset + args.batch]
        batch = []
        for im in metas:
            gray = np.array(Image.open(image_dir / im["file_name"]).convert("L"), dtype=np.uint8)
            batch.append(apply_lut(gray, lut))
        results = model.predict(
            source=batch,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou_nms,
            device=args.device,
            batch=len(batch),
            verbose=False,
            save=False,
        )
        for im, res in zip(metas, results):
            image_id = int(im["id"])
            gts = gt.get(image_id, [])
            matched: set[int] = set()
            preds = []
            if res.boxes is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                for box, score, cls_id in zip(boxes, scores, cls_ids):
                    cat_id = COCO80_TO_91[int(cls_id)]
                    if cat_id not in nav_ids:
                        continue
                    preds.append((float(score), int(cat_id), [float(v) for v in box.tolist()]))
            preds.sort(reverse=True, key=lambda x: x[0])
            prediction_count += len(preds)
            for _score, cat_id, box in preds:
                best_idx = -1
                best_iou = 0.0
                for idx, item in enumerate(gts):
                    if idx in matched or item["category_id"] != cat_id:
                        continue
                    val = iou_xyxy(box, item["bbox"])
                    if val > best_iou:
                        best_iou = val
                        best_idx = idx
                if best_idx >= 0 and best_iou >= args.match_iou:
                    matched.add(best_idx)
                    tp += 1
                else:
                    fp += 1
            fn += max(0, len(gts) - len(matched))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    score = f1 + args.recall_weight * recall - args.fp_weight * (fp / max(1, len(images)))
    return CandidateScore(candidate_id, score, precision, recall, f1, tp, fp, fn, prediction_count, candidate)


def save_lut_bundle(path: Path, candidate: Candidate, score: CandidateScore, args: argparse.Namespace) -> None:
    lut = candidate_to_lut(candidate)
    adapter = lut_uint8_to_adapter(lut.tolist())
    save_adapter_bundle(
        adapter,
        path / "gray_adapter",
        {
            "weights": str(args.weights),
            "imgsz": args.imgsz,
            "input_channels": 1,
            "output_channels": 3,
            "training": "metric_optimized_lut_search",
            "objective": "F1_nav + recall_weight*Recall_nav - fp_weight*FP_per_image",
            "score": asdict(score),
            "search_params": {
                "max_images": args.max_images,
                "train_ratio": args.train_ratio,
                "generations": args.generations,
                "population": args.population,
                "elite": args.elite,
                "conf": args.conf,
                "match_iou": args.match_iou,
                "recall_weight": args.recall_weight,
                "fp_weight": args.fp_weight,
                "knot_count": args.knot_count,
                "knot_range": args.knot_range,
            },
        },
    )


def write_scores(path: Path, rows: list[CandidateScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["candidate_id", "score", "precision", "recall", "f1", "tp", "fp", "fn", "prediction_count", "params"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            d["params"] = json.dumps(d["params"], ensure_ascii=False)
            writer.writerow(d)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    coco = load_coco(args.annotations)
    images = select_images(coco, args.images, args.max_images, args.seed)
    split = max(1, min(len(images) - 1, int(round(len(images) * args.train_ratio))))
    train_images = images[:split]
    holdout_images = images[split:]
    nav_ids = nav_category_ids(coco)
    gt_train = build_gt_index(coco, {int(im["id"]) for im in train_images}, nav_ids)
    gt_holdout = build_gt_index(coco, {int(im["id"]) for im in holdout_images}, nav_ids)
    model = YOLO(str(args.weights))
    rng = random.Random(args.seed)

    population = [identity_candidate(args.knot_count)] + [
        random_candidate(rng, args.knot_count, args.knot_range) for _ in range(args.population - 1)
    ]
    all_scores: list[CandidateScore] = []
    best: CandidateScore | None = None
    best_candidate: Candidate | None = None

    for gen in range(args.generations):
        gen_scores: list[CandidateScore] = []
        scale = max(0.25, 1.0 - gen / max(1, args.generations - 1))
        pbar = tqdm(population, desc=f"metric LUT generation {gen + 1}/{args.generations}")
        for idx, cand in enumerate(pbar):
            candidate_id = gen * args.population + idx
            score = evaluate_candidate(candidate_id, cand, model, train_images, args.images, gt_train, nav_ids, args)
            gen_scores.append(score)
            all_scores.append(score)
            pbar.set_postfix(best=max([s.score for s in gen_scores], default=0.0), f1=score.f1, recall=score.recall)
        gen_scores.sort(key=lambda s: s.score, reverse=True)
        if best is None or gen_scores[0].score > best.score:
            best = gen_scores[0]
            best_candidate = gen_scores[0].params
            save_lut_bundle(args.out_dir / "best_train", best_candidate, best, args)
        if args.save_every_generation:
            save_lut_bundle(args.out_dir / f"generation_{gen + 1:02d}_best", gen_scores[0].params, gen_scores[0], args)
        elites = [s.params for s in gen_scores[: max(1, args.elite)]]
        population = [elites[0]]
        while len(population) < args.population:
            population.append(mutate(rng.choice(elites), rng, scale))
        print(
            f"generation={gen + 1} best_score={gen_scores[0].score:.6f} "
            f"f1={gen_scores[0].f1:.6f} recall={gen_scores[0].recall:.6f} "
            f"tp={gen_scores[0].tp} fp={gen_scores[0].fp} fn={gen_scores[0].fn}",
            flush=True,
        )

    if best is None or best_candidate is None:
        raise RuntimeError("search produced no candidate")
    holdout = evaluate_candidate(999999, best_candidate, model, holdout_images, args.images, gt_holdout, nav_ids, args)
    save_lut_bundle(args.out_dir / "best_holdout_checked", best_candidate, holdout, args)
    write_scores(args.out_dir / "candidate_scores.csv", all_scores)
    summary = {
        "settings": vars(args) | {"images": str(args.images), "annotations": str(args.annotations), "weights": str(args.weights), "out_dir": str(args.out_dir)},
        "input_mode": "Every candidate is evaluated by converting the source image to single-channel grayscale first, then applying the candidate LUT to produce pseudo-RGB. Original RGB pixels are never used for inference.",
        "train_image_count": len(train_images),
        "holdout_image_count": len(holdout_images),
        "best_train": asdict(best),
        "best_holdout": asdict(holdout),
        "artifact_train": str(args.out_dir / "best_train" / "gray_adapter.pt"),
        "artifact_holdout_checked": str(args.out_dir / "best_holdout_checked" / "gray_adapter.pt"),
    }
    (args.out_dir / "search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
