#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from gray_adapter import load_adapter_bundle


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

VARIANT_COLORS = {
    "baseline": (76, 132, 190),
    "lut": (76, 160, 84),
    "conv": (224, 133, 52),
    "spatial": (156, 95, 181),
    "g2rgb": (210, 88, 120),
    "bc_gmfe_dca": (196, 68, 92),
}
GT_COLOR = (0, 210, 220)


@dataclass
class Detection:
    image_id: int
    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    score: float
    category_id: int
    label: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate gray-copy and gray adapter variants against COCO ground truth.")
    p.add_argument("--images", required=True, type=Path, help="COCO val2017 image directory.")
    p.add_argument("--annotations", required=True, type=Path, help="instances_val2017.json.")
    p.add_argument("--weights", required=True, type=Path, help="YOLOv8n .pt weights.")
    p.add_argument("--lut-adapter", type=Path, help="gray_adapter_lut/gray_adapter.pt.")
    p.add_argument("--conv-adapter", type=Path, help="gray_adapter_conv/gray_adapter.pt.")
    p.add_argument("--spatial-adapter", type=Path, help="spatial gray_adapter.pt.")
    p.add_argument("--g2rgb-adapter", type=Path, help="G2RGB residual adapter .pt.")
    p.add_argument("--bc-gmfe-dca-adapter", type=Path, help="BC-GMFE-DCA adapter .pt.")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--conf", type=float, default=0.001, help="Low confidence for COCO mAP. Use 0.001 for standard eval.")
    p.add_argument("--iou", type=float, default=0.70, help="NMS IoU used by YOLO predict.")
    p.add_argument("--match-conf", type=float, default=0.25, help="Confidence threshold for per-image TP/FP/FN summary.")
    p.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for per-image TP matching.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-images", type=int, default=0, help="0 means all available images.")
    p.add_argument("--variants", default="baseline,lut,conv", help="Comma-separated subset: baseline,lut,conv,spatial,g2rgb,bc_gmfe_dca.")
    p.add_argument("--save-visuals", type=int, default=24, help="Number of top true-improvement side-by-side images to save.")
    p.add_argument("--save-input-samples", type=int, default=12, help="Save actual YOLO input images per variant for audit.")
    return p.parse_args()


def require_files(args: argparse.Namespace) -> None:
    missing = [p for p in [args.images, args.annotations, args.weights] if not p.exists()]
    if "lut" in args.variants.split(",") and (not args.lut_adapter or not args.lut_adapter.exists()):
        missing.append(args.lut_adapter or Path("<missing lut adapter>"))
    if "conv" in args.variants.split(",") and (not args.conv_adapter or not args.conv_adapter.exists()):
        missing.append(args.conv_adapter or Path("<missing conv adapter>"))
    if "spatial" in args.variants.split(",") and (not args.spatial_adapter or not args.spatial_adapter.exists()):
        missing.append(args.spatial_adapter or Path("<missing spatial adapter>"))
    if "g2rgb" in args.variants.split(",") and (not args.g2rgb_adapter or not args.g2rgb_adapter.exists()):
        missing.append(args.g2rgb_adapter or Path("<missing g2rgb adapter>"))
    if "bc_gmfe_dca" in args.variants.split(",") and (not args.bc_gmfe_dca_adapter or not args.bc_gmfe_dca_adapter.exists()):
        missing.append(args.bc_gmfe_dca_adapter or Path("<missing bc gmfe dca adapter>"))
    if missing:
        joined = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"missing required input:\n{joined}")


def load_coco_annotations(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_images(coco: dict[str, Any], image_dir: Path, max_images: int) -> list[dict[str, Any]]:
    images = sorted(coco["images"], key=lambda x: x["file_name"])
    selected = [im for im in images if (image_dir / im["file_name"]).exists()]
    if max_images > 0:
        selected = selected[:max_images]
    if not selected:
        raise FileNotFoundError(f"no annotated images found in {image_dir}")
    return selected


def load_gray_array(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def adapter_to_rgb(adapter: torch.nn.Module, gray: np.ndarray) -> Image.Image:
    x = torch.from_numpy(gray.astype(np.float32) / 255.0).view(1, 1, gray.shape[0], gray.shape[1])
    with torch.no_grad():
        y = adapter(x).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    rgb = np.transpose(y, (1, 2, 0))
    rgb = np.round(rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def baseline_to_rgb(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1), "RGB")


def build_variant_image(variant: str, image_path: Path, adapters: dict[str, torch.nn.Module]) -> Image.Image:
    gray = load_gray_array(image_path)
    if variant == "baseline":
        return baseline_to_rgb(gray)
    if variant in adapters:
        return adapter_to_rgb(adapters[variant], gray)
    raise ValueError(f"unsupported variant: {variant}")


def xyxy_to_xywh(box: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def run_variant(
    variant: str,
    model: YOLO,
    images: list[dict[str, Any]],
    image_dir: Path,
    adapters: dict[str, torch.nn.Module],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[int, list[Detection]], dict[str, float]]:
    coco_results: list[dict[str, Any]] = []
    det_by_image: dict[int, list[Detection]] = defaultdict(list)
    speed_rows: list[dict[str, float]] = []
    adapter_ms: list[float] = []
    saved_input_samples = 0

    start_total = time.perf_counter()
    for offset in range(0, len(images), args.batch):
        batch_meta = images[offset : offset + args.batch]
        batch_pil: list[Image.Image] = []
        for im in batch_meta:
            t0 = time.perf_counter()
            prepared = build_variant_image(variant, image_dir / im["file_name"], adapters)
            if saved_input_samples < args.save_input_samples:
                sample_dir = args.out_dir / "input_samples" / variant
                sample_dir.mkdir(parents=True, exist_ok=True)
                prepared.save(sample_dir / im["file_name"])
                saved_input_samples += 1
            batch_pil.append(prepared)
            adapter_ms.append((time.perf_counter() - t0) * 1000.0)

        results = model.predict(
            source=batch_pil,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=len(batch_pil),
            verbose=False,
            save=False,
        )

        for im, res in zip(batch_meta, results):
            speed = getattr(res, "speed", {}) or {}
            speed_rows.append(
                {
                    "preprocess_ms": float(speed.get("preprocess", 0.0)),
                    "inference_ms": float(speed.get("inference", 0.0)),
                    "postprocess_ms": float(speed.get("postprocess", 0.0)),
                }
            )
            if res.boxes is None:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls_id in zip(boxes, scores, cls_ids):
                cat_id = COCO80_TO_91[int(cls_id)]
                xywh = xyxy_to_xywh(box)
                label = str(model.names.get(int(cls_id), cls_id))
                item = {
                    "image_id": int(im["id"]),
                    "category_id": int(cat_id),
                    "bbox": [round(x, 3) for x in xywh],
                    "score": round(float(score), 6),
                }
                coco_results.append(item)
                det_by_image[int(im["id"])].append(
                    Detection(
                        image_id=int(im["id"]),
                        bbox_xyxy=[round(float(x), 3) for x in box.tolist()],
                        bbox_xywh=item["bbox"],
                        score=float(score),
                        category_id=int(cat_id),
                        label=label,
                    )
                )

    wall_ms_per_image = (time.perf_counter() - start_total) * 1000.0 / max(1, len(images))
    perf = {
        "adapter_ms": float(np.mean(adapter_ms)) if adapter_ms else 0.0,
        "yolo_preprocess_ms": float(np.mean([x["preprocess_ms"] for x in speed_rows])) if speed_rows else 0.0,
        "yolo_inference_ms": float(np.mean([x["inference_ms"] for x in speed_rows])) if speed_rows else 0.0,
        "yolo_postprocess_ms": float(np.mean([x["postprocess_ms"] for x in speed_rows])) if speed_rows else 0.0,
        "wall_ms_per_image": wall_ms_per_image,
        "fps_wall": 1000.0 / wall_ms_per_image if wall_ms_per_image > 0 else 0.0,
    }
    return coco_results, det_by_image, perf


def run_coco_eval(
    annotation_path: Path,
    result_path: Path,
    img_ids: list[int],
    cat_ids: list[int] | None = None,
) -> dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(annotation_path))
    if result_path.stat().st_size == 0:
        return {}
    with result_path.open("r", encoding="utf-8") as f:
        results = json.load(f)
    if not results:
        return {}
    coco_dt = coco_gt.loadRes(str(result_path))
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    if cat_ids:
        ev.params.catIds = cat_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats.tolist()
    return {
        "AP": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR1": stats[6],
        "AR10": stats[7],
        "AR100": stats[8],
    }


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


def gt_index(coco: dict[str, Any], nav_cat_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        cat_id = int(ann["category_id"])
        anns_by_image[int(ann["image_id"])].append(
            {
                "category_id": cat_id,
                "bbox_xyxy": [x, y, x + w, y + h],
                "is_nav": cat_id in nav_cat_ids,
            }
        )
    return anns_by_image


def match_image(
    gt_items: list[dict[str, Any]],
    dets: list[Detection],
    conf: float,
    match_iou: float,
    nav_only: bool,
) -> dict[str, float]:
    gts = [g for g in gt_items if (g["is_nav"] or not nav_only)]
    preds = [d for d in dets if d.score >= conf]
    if nav_only:
        # nav_only is enforced through category ids by matching to nav GT categories.
        nav_cat_ids = {g["category_id"] for g in gts}
        preds = [d for d in preds if d.category_id in nav_cat_ids]
    preds = sorted(preds, key=lambda d: d.score, reverse=True)
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    for pred in preds:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gts):
            if idx in matched_gt or gt["category_id"] != pred.category_id:
                continue
            val = iou_xyxy(pred.bbox_xyxy, gt["bbox_xyxy"])
            if val > best_iou:
                best_iou = val
                best_idx = idx
        if best_idx >= 0 and best_iou >= match_iou:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
    fn = len(gts) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "jaccard_acc": jaccard}


def draw_boxes(draw: ImageDraw.ImageDraw, boxes: list[tuple[list[float], str, tuple[int, int, int]]], width: int = 3) -> None:
    for box, text, color in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        tb = draw.textbbox((x1, y1), text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        y0 = max(0, y1 - th - 5)
        draw.rectangle([x1, y0, x1 + tw + 4, y0 + th + 4], fill=color)
        draw.text((x1 + 2, y0 + 2), text, fill=(255, 255, 255))


def save_visuals(
    out_dir: Path,
    images: list[dict[str, Any]],
    image_dir: Path,
    coco: dict[str, Any],
    dets_by_variant: dict[str, dict[int, list[Detection]]],
    rows: list[dict[str, Any]],
    nav_cat_ids: set[int],
    limit: int,
) -> None:
    if limit <= 0:
        return
    cat_names = {int(c["id"]): c["name"] for c in coco["categories"]}
    image_by_id = {int(im["id"]): im for im in images}
    anns = gt_index(coco, nav_cat_ids)
    visual_dir = out_dir / "visual_true_improvements"
    visual_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda r: r.get("best_delta_f1_nav", 0.0), reverse=True)[:limit]
    variants = ["baseline"] + [v for v in dets_by_variant if v != "baseline"]
    for idx, row in enumerate(ranked, start=1):
        im = image_by_id[int(row["image_id"])]
        base = Image.open(image_dir / im["file_name"]).convert("RGB")
        panels = []
        gt_boxes = [
            (g["bbox_xyxy"], f"GT {cat_names.get(g['category_id'], g['category_id'])}", GT_COLOR)
            for g in anns[int(im["id"])]
            if g["is_nav"]
        ]
        for variant in variants:
            panel = base.copy()
            draw = ImageDraw.Draw(panel)
            draw_boxes(draw, gt_boxes, width=2)
            pred_boxes = [
                (d.bbox_xyxy, f"{d.label} {d.score:.2f}", VARIANT_COLORS.get(variant, (255, 0, 0)))
                for d in dets_by_variant[variant][int(im["id"])]
                if d.score >= 0.25
            ]
            draw_boxes(draw, pred_boxes, width=3)
            panel.thumbnail((420, 420))
            canvas = Image.new("RGB", (420, 460), (245, 245, 245))
            canvas.paste(panel, ((420 - panel.width) // 2, 38))
            d = ImageDraw.Draw(canvas)
            d.rectangle([0, 0, 420, 34], fill=VARIANT_COLORS.get(variant, (80, 80, 80)))
            d.text((8, 9), f"{variant}  F1nav={row.get(f'{variant}_f1_nav', 0):.3f}", fill=(255, 255, 255))
            panels.append(canvas)
        merged = Image.new("RGB", (420 * len(panels), 460), "white")
        for j, panel in enumerate(panels):
            merged.paste(panel, (j * 420, 0))
        score = row.get("best_delta_f1_nav", 0.0)
        merged.save(visual_dir / f"{idx:02d}_{row['best_adapter']}_{score:.3f}_{im['file_name']}")


def main() -> None:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.variants = ",".join(variants)
    require_files(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco_annotations(args.annotations)
    images = select_images(coco, args.images, args.max_images)
    cat_name_to_id = {c["name"]: int(c["id"]) for c in coco["categories"]}
    nav_cat_ids = {cat_name_to_id[x] for x in NAV_LABELS if x in cat_name_to_id}

    adapters: dict[str, torch.nn.Module] = {}
    if "lut" in variants:
        adapters["lut"] = load_adapter_bundle(args.lut_adapter, map_location="cpu").eval()
    if "conv" in variants:
        adapters["conv"] = load_adapter_bundle(args.conv_adapter, map_location="cpu").eval()
    if "spatial" in variants:
        adapters["spatial"] = load_adapter_bundle(args.spatial_adapter, map_location="cpu").eval()
    if "g2rgb" in variants:
        adapters["g2rgb"] = load_adapter_bundle(args.g2rgb_adapter, map_location="cpu").eval()
    if "bc_gmfe_dca" in variants:
        adapters["bc_gmfe_dca"] = load_adapter_bundle(args.bc_gmfe_dca_adapter, map_location="cpu").eval()

    model = YOLO(str(args.weights))

    all_metrics: dict[str, Any] = {
        "settings": {
            "images": str(args.images),
            "annotations": str(args.annotations),
            "weights": str(args.weights),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "match_conf": args.match_conf,
            "match_iou": args.match_iou,
            "device": args.device,
            "batch": args.batch,
            "max_images": args.max_images,
            "evaluated_images": len(images),
            "variants": variants,
            "input_mode": "All variants convert the original RGB dataset image to single-channel grayscale first. Baseline then replicates gray to 3 channels; adapters consume only that gray image or gray-replicated board-compatible input and output pseudo-RGB for YOLO.",
            "visualization_note": "visual_true_improvements overlays GT/pred boxes on the original RGB image for human readability only; inference inputs are saved under input_samples/.",
        },
        "variants": {},
    }
    dets_by_variant: dict[str, dict[int, list[Detection]]] = {}

    for variant in variants:
        print(f"evaluating variant={variant} images={len(images)}", flush=True)
        coco_results, dets, perf = run_variant(variant, model, images, args.images, adapters, args)
        dets_by_variant[variant] = dets
        pred_path = args.out_dir / f"{variant}_coco_predictions.json"
        pred_path.write_text(json.dumps(coco_results, ensure_ascii=False), encoding="utf-8")
        print(f"running COCOeval variant={variant}", flush=True)
        selected_img_ids = [int(im["id"]) for im in images]
        all_ap = run_coco_eval(args.annotations, pred_path, selected_img_ids)
        nav_ap = run_coco_eval(args.annotations, pred_path, selected_img_ids, sorted(nav_cat_ids))
        all_metrics["variants"][variant] = {
            "prediction_count": len(coco_results),
            "performance": perf,
            "coco_all": all_ap,
            "coco_navigation_subset": nav_ap,
        }

    anns_by_image = gt_index(coco, nav_cat_ids)
    per_image_rows: list[dict[str, Any]] = []
    image_by_id = {int(im["id"]): im for im in images}
    for im in images:
        image_id = int(im["id"])
        row: dict[str, Any] = {"image_id": image_id, "file_name": im["file_name"]}
        for variant in variants:
            m_all = match_image(anns_by_image[image_id], dets_by_variant[variant][image_id], args.match_conf, args.match_iou, False)
            m_nav = match_image(anns_by_image[image_id], dets_by_variant[variant][image_id], args.match_conf, args.match_iou, True)
            for k, v in m_all.items():
                row[f"{variant}_{k}_all"] = v
            for k, v in m_nav.items():
                row[f"{variant}_{k}_nav"] = v
        if "baseline" in variants:
            best_variant = "baseline"
            best_delta = 0.0
            base_f1 = float(row["baseline_f1_nav"])
            for variant in variants:
                if variant == "baseline":
                    continue
                delta = float(row[f"{variant}_f1_nav"]) - base_f1
                if delta > best_delta:
                    best_delta = delta
                    best_variant = variant
            row["best_adapter"] = best_variant
            row["best_delta_f1_nav"] = best_delta
        per_image_rows.append(row)

    csv_path = args.out_dir / "per_image_truth_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    if "baseline" in variants:
        save_visuals(args.out_dir, images, args.images, coco, dets_by_variant, per_image_rows, nav_cat_ids, args.save_visuals)

    summary_path = args.out_dir / "truth_eval_summary.json"
    summary_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"summary={summary_path}")
    print(f"per_image={csv_path}")


if __name__ == "__main__":
    main()
