#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize gray adapter COCO truth evaluation results.")
    p.add_argument("--eval-dir", required=True, type=Path, help="Directory containing truth_eval_summary.json.")
    p.add_argument("--out", type=Path, help="Markdown report path. Defaults to <eval-dir>/ADAPTER_TRUTH_EVAL_REPORT.md.")
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: Any, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "N/A"


def pct_delta(new: float, base: float) -> str:
    if base == 0:
        return "N/A"
    return f"{(new - base) / base * 100:+.2f}%"


def metric(summary: dict[str, Any], variant: str, group: str, name: str) -> float:
    return float(summary["variants"][variant][group].get(name, 0.0))


def perf(summary: dict[str, Any], variant: str, name: str) -> float:
    return float(summary["variants"][variant]["performance"].get(name, 0.0))


def plot_metrics(eval_dir: Path, summary: dict[str, Any]) -> list[Path]:
    plot_dir = eval_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    variants = list(summary["variants"].keys())
    out_paths: list[Path] = []

    for group, title, fname in [
        ("coco_all", "COCO all categories", "coco_all_metrics.png"),
        ("coco_navigation_subset", "Navigation-relevant subset", "coco_navigation_metrics.png"),
    ]:
        keys = ["AP", "AP50", "AP75", "AR100"]
        x = np.arange(len(keys))
        width = 0.8 / max(1, len(variants))
        plt.figure(figsize=(9, 5.2))
        for i, variant in enumerate(variants):
            vals = [metric(summary, variant, group, k) for k in keys]
            plt.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width=width, label=variant)
        plt.xticks(x, keys)
        plt.ylabel("score")
        plt.title(title)
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        path = plot_dir / fname
        plt.savefig(path, dpi=160)
        plt.close()
        out_paths.append(path)

    keys = ["adapter_ms", "yolo_preprocess_ms", "yolo_inference_ms", "yolo_postprocess_ms", "wall_ms_per_image"]
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(variants))
    plt.figure(figsize=(10, 5.6))
    for i, variant in enumerate(variants):
        vals = [perf(summary, variant, k) for k in keys]
        plt.bar(x + (i - (len(variants) - 1) / 2) * width, vals, width=width, label=variant)
    plt.xticks(x, keys, rotation=20, ha="right")
    plt.ylabel("ms/image")
    plt.title("Runtime breakdown")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = plot_dir / "runtime_breakdown.png"
    plt.savefig(path, dpi=160)
    plt.close()
    out_paths.append(path)

    return out_paths


def aggregate_per_image(rows: list[dict[str, str]], variants: list[str]) -> dict[str, Any]:
    if not rows:
        return {}
    out: dict[str, Any] = {}
    for variant in variants:
        f1_key = f"{variant}_f1_nav"
        recall_key = f"{variant}_recall_nav"
        precision_key = f"{variant}_precision_nav"
        tp_key = f"{variant}_tp_nav"
        fp_key = f"{variant}_fp_nav"
        fn_key = f"{variant}_fn_nav"
        out[variant] = {
            "mean_f1_nav": float(np.mean([float(r[f1_key]) for r in rows])),
            "mean_recall_nav": float(np.mean([float(r[recall_key]) for r in rows])),
            "mean_precision_nav": float(np.mean([float(r[precision_key]) for r in rows])),
            "tp_nav": int(sum(float(r[tp_key]) for r in rows)),
            "fp_nav": int(sum(float(r[fp_key]) for r in rows)),
            "fn_nav": int(sum(float(r[fn_key]) for r in rows)),
        }
    if "baseline" in variants:
        for variant in variants:
            if variant == "baseline":
                continue
            out[variant]["images_f1_improved_vs_baseline"] = sum(
                float(r[f"{variant}_f1_nav"]) > float(r["baseline_f1_nav"]) for r in rows
            )
            out[variant]["images_f1_dropped_vs_baseline"] = sum(
                float(r[f"{variant}_f1_nav"]) < float(r["baseline_f1_nav"]) for r in rows
            )
    return out


def top_rows(rows: list[dict[str, str]], top_k: int) -> list[dict[str, str]]:
    if not rows or "best_delta_f1_nav" not in rows[0]:
        return []
    return sorted(rows, key=lambda r: float(r.get("best_delta_f1_nav", 0.0)), reverse=True)[:top_k]


def write_report(eval_dir: Path, out_path: Path, summary: dict[str, Any], rows: list[dict[str, str]], plots: list[Path], top_k: int) -> None:
    variants = list(summary["variants"].keys())
    settings = summary.get("settings", {})
    per_image = aggregate_per_image(rows, variants)

    lines: list[str] = []
    lines.append("# Adapter Truth Evaluation Report")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    for k in ["images", "annotations", "weights", "imgsz", "conf", "iou", "match_conf", "match_iou", "device", "batch", "max_images", "evaluated_images"]:
        if k in settings:
            lines.append(f"- `{k}`: `{settings[k]}`")
    lines.append(f"- variants: `{', '.join(variants)}`")
    if "input_mode" in settings:
        lines.append(f"- input mode: {settings['input_mode']}")
    if "visualization_note" in settings:
        lines.append(f"- visualization note: {settings['visualization_note']}")
    lines.append("")

    lines.append("## COCO Metrics")
    lines.append("")
    for group, title in [("coco_all", "All COCO categories"), ("coco_navigation_subset", "Navigation-relevant categories")]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Variant | AP | AP50 | AP75 | AR100 | AP delta vs baseline | AP50 delta vs baseline |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        base_ap = metric(summary, "baseline", group, "AP") if "baseline" in variants else 0.0
        base_ap50 = metric(summary, "baseline", group, "AP50") if "baseline" in variants else 0.0
        for variant in variants:
            ap = metric(summary, variant, group, "AP")
            ap50 = metric(summary, variant, group, "AP50")
            ar100 = metric(summary, variant, group, "AR100")
            lines.append(
                f"| `{variant}` | {ap:.5f} | {ap50:.5f} | {metric(summary, variant, group, 'AP75'):.5f} | {ar100:.5f} | {ap - base_ap:+.5f} ({pct_delta(ap, base_ap)}) | {ap50 - base_ap50:+.5f} ({pct_delta(ap50, base_ap50)}) |"
            )
        lines.append("")

    lines.append("## Runtime")
    lines.append("")
    lines.append("| Variant | Adapter/prep ms | YOLO preprocess ms | Inference ms | Postprocess ms | Wall ms/img | Wall FPS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for variant in variants:
        p = summary["variants"][variant]["performance"]
        lines.append(
            f"| `{variant}` | {fnum(p.get('adapter_ms'))} | {fnum(p.get('yolo_preprocess_ms'))} | {fnum(p.get('yolo_inference_ms'))} | {fnum(p.get('yolo_postprocess_ms'))} | {fnum(p.get('wall_ms_per_image'))} | {fnum(p.get('fps_wall'))} |"
        )
    lines.append("")

    if per_image:
        lines.append("## Per-image TP/FP/FN Summary")
        lines.append("")
        lines.append("| Variant | TP(nav) | FP(nav) | FN(nav) | Mean precision | Mean recall | Mean F1 | F1 improved images | F1 dropped images |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for variant in variants:
            p = per_image[variant]
            lines.append(
                f"| `{variant}` | {p['tp_nav']} | {p['fp_nav']} | {p['fn_nav']} | {p['mean_precision_nav']:.4f} | {p['mean_recall_nav']:.4f} | {p['mean_f1_nav']:.4f} | {p.get('images_f1_improved_vs_baseline', 0)} | {p.get('images_f1_dropped_vs_baseline', 0)} |"
            )
        lines.append("")

    if plots:
        lines.append("## Plots")
        lines.append("")
        for path in plots:
            rel = path.relative_to(eval_dir).as_posix()
            lines.append(f"![{path.stem}]({rel})")
            lines.append("")

    selected = top_rows(rows, top_k)
    if selected:
        lines.append("## Top True-improvement Candidates")
        lines.append("")
        metric_cols = [f"{v.upper()} F1" for v in variants]
        lines.append("| File | Best adapter | Delta F1(nav) | " + " | ".join(metric_cols) + " |")
        lines.append("|---|---:|---:|" + "---:|" * len(metric_cols))
        for r in selected:
            f1_values = " | ".join(f"{float(r.get(f'{v}_f1_nav', 0)):.4f}" for v in variants)
            lines.append(
                f"| `{r.get('file_name')}` | `{r.get('best_adapter')}` | {float(r.get('best_delta_f1_nav', 0)):+.4f} | {f1_values} |"
            )
        lines.append("")
        lines.append("Visual comparison images, if generated, are in `visual_true_improvements/`.")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- COCO `AP/mAP` is the rigorous detection metric. Plain classification `acc` is not suitable for object detection with multiple boxes per image.")
    lines.append("- If `--max-images` is not zero, this report is a subset result. Use it for quick trend checks, then rerun full `val2017` before final conclusions.")
    lines.append("- Runtime on PC/cloud is useful for relative comparison, but board runtime must be measured separately after conversion and integration.")
    lines.append("- Audit the actual images sent to YOLO under `input_samples/<variant>/`; these should be gray-copy or pseudo-RGB derived from single-channel grayscale, not original RGB photos.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir
    summary = load_json(eval_dir / "truth_eval_summary.json")
    rows = load_csv(eval_dir / "per_image_truth_metrics.csv")
    plots = plot_metrics(eval_dir, summary)
    out = args.out or (eval_dir / "ADAPTER_TRUTH_EVAL_REPORT.md")
    write_report(eval_dir, out, summary, rows, plots, args.top_k)
    print(f"report={out}")
    for path in plots:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
