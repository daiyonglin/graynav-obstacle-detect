#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from gray_adapter import save_adapter_bundle, spatial_params_to_adapter
from optimize_lut_adapter_metric import (
    COCO80_TO_91,
    build_gt_index,
    iou_xyxy,
    load_coco,
    nav_category_ids,
    select_images,
)


@dataclass
class SpatialCandidate:
    strength: float
    gamma: list[float]
    contrast: list[float]
    bias: list[float]
    local_gain: list[float]
    edge_gain: list[float]


@dataclass
class SpatialScore:
    candidate_id: int
    score: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    prediction_count: int
    params: SpatialCandidate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric-driven search for a stronger spatial gray adapter.")
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--max-images", type=int, default=240)
    p.add_argument("--train-ratio", type=float, default=0.60)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--population", type=int, default=18)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-nms", type=float, default=0.70)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--recall-weight", type=float, default=0.20)
    p.add_argument("--fp-weight", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20260622)
    return p.parse_args()


def identity_candidate() -> SpatialCandidate:
    return SpatialCandidate(
        strength=0.0,
        gamma=[1.0, 1.0, 1.0],
        contrast=[1.0, 1.0, 1.0],
        bias=[0.0, 0.0, 0.0],
        local_gain=[0.0, 0.0, 0.0],
        edge_gain=[0.0, 0.0, 0.0],
    )


def random_candidate(rng: random.Random) -> SpatialCandidate:
    return SpatialCandidate(
        strength=rng.uniform(0.25, 0.95),
        gamma=[rng.uniform(0.70, 1.35) for _ in range(3)],
        contrast=[rng.uniform(0.80, 1.25) for _ in range(3)],
        bias=[rng.uniform(-0.06, 0.06) for _ in range(3)],
        local_gain=[rng.uniform(-0.38, 0.38) for _ in range(3)],
        edge_gain=[rng.uniform(-0.12, 0.34) for _ in range(3)],
    )


def mutate(parent: SpatialCandidate, rng: random.Random, scale: float) -> SpatialCandidate:
    def clamp(v: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, v))

    return SpatialCandidate(
        strength=clamp(parent.strength + rng.gauss(0.0, 0.16 * scale), 0.0, 1.0),
        gamma=[clamp(v + rng.gauss(0.0, 0.12 * scale), 0.50, 1.85) for v in parent.gamma],
        contrast=[clamp(v + rng.gauss(0.0, 0.09 * scale), 0.55, 1.65) for v in parent.contrast],
        bias=[clamp(v + rng.gauss(0.0, 0.025 * scale), -0.18, 0.18) for v in parent.bias],
        local_gain=[clamp(v + rng.gauss(0.0, 0.09 * scale), -0.65, 0.65) for v in parent.local_gain],
        edge_gain=[clamp(v + rng.gauss(0.0, 0.07 * scale), -0.35, 0.60) for v in parent.edge_gain],
    )


def candidate_to_adapter(candidate: SpatialCandidate) -> torch.nn.Module:
    return spatial_params_to_adapter(asdict(candidate)).eval()


def adapter_image(adapter: torch.nn.Module, image_path: Path) -> Image.Image:
    gray = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
    x = torch.from_numpy(gray.astype(np.float32) / 255.0).view(1, 1, gray.shape[0], gray.shape[1])
    with torch.no_grad():
        y = adapter(x).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    rgb = np.transpose(y, (1, 2, 0))
    return Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), "RGB")


def evaluate_candidate(
    candidate_id: int,
    candidate: SpatialCandidate,
    model: YOLO,
    images: list[dict[str, Any]],
    image_dir: Path,
    gt: dict[int, list[dict[str, Any]]],
    nav_ids: set[int],
    args: argparse.Namespace,
) -> SpatialScore:
    adapter = candidate_to_adapter(candidate)
    tp = fp = fn = prediction_count = 0
    for offset in range(0, len(images), args.batch):
        metas = images[offset : offset + args.batch]
        batch = [adapter_image(adapter, image_dir / im["file_name"]) for im in metas]
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
                    if cat_id in nav_ids:
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
    return SpatialScore(candidate_id, score, precision, recall, f1, tp, fp, fn, prediction_count, candidate)


def save_bundle(path: Path, candidate: SpatialCandidate, score: SpatialScore, args: argparse.Namespace) -> None:
    adapter = candidate_to_adapter(candidate)
    save_adapter_bundle(
        adapter,
        path / "gray_adapter",
        {
            "weights": str(args.weights),
            "imgsz": args.imgsz,
            "input_channels": 1,
            "output_channels": 3,
            "training": "metric_optimized_spatial_search",
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
            },
        },
    )


def write_scores(path: Path, rows: list[SpatialScore]) -> None:
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

    population = [identity_candidate()] + [random_candidate(rng) for _ in range(args.population - 1)]
    all_scores: list[SpatialScore] = []
    best: SpatialScore | None = None
    best_candidate: SpatialCandidate | None = None

    for gen in range(args.generations):
        gen_scores: list[SpatialScore] = []
        scale = max(0.25, 1.0 - gen / max(1, args.generations - 1))
        pbar = tqdm(population, desc=f"spatial adapter generation {gen + 1}/{args.generations}")
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
            save_bundle(args.out_dir / "best_train", best_candidate, best, args)
        elites = [s.params for s in gen_scores[: max(1, args.elite)]]
        population = [elites[0]]
        while len(population) < args.population:
            population.append(mutate(random.choice(elites), rng, scale))
        print(
            f"generation={gen + 1} best_score={gen_scores[0].score:.6f} "
            f"f1={gen_scores[0].f1:.6f} recall={gen_scores[0].recall:.6f} "
            f"tp={gen_scores[0].tp} fp={gen_scores[0].fp} fn={gen_scores[0].fn}",
            flush=True,
        )

    if best is None or best_candidate is None:
        raise RuntimeError("search produced no candidate")
    holdout = evaluate_candidate(999999, best_candidate, model, holdout_images, args.images, gt_holdout, nav_ids, args)
    save_bundle(args.out_dir / "best_holdout_checked", best_candidate, holdout, args)
    write_scores(args.out_dir / "candidate_scores.csv", all_scores)
    summary = {
        "settings": vars(args) | {
            "images": str(args.images),
            "annotations": str(args.annotations),
            "weights": str(args.weights),
            "out_dir": str(args.out_dir),
        },
        "input_mode": "Every candidate is evaluated by converting the source image to single-channel grayscale first, then applying the spatial adapter to produce pseudo-RGB.",
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
