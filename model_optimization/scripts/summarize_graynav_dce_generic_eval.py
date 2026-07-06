#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GrayNav-DCE generic evaluation JSON into compact CSV/Markdown evidence tables.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", default="M0_raw_yolov8n_graycopy")
    return parser.parse_args()


def metric(block: dict[str, Any], group: str, key: str) -> float:
    """Read one metric from a model/corruption result block."""
    return float(block.get(group, {}).get(key, 0.0))


def fp(block: dict[str, Any]) -> float:
    """Read empty-scene FP per image."""
    return float(block.get("empty_fp", {}).get("empty_fp_per_image", 0.0))


def fps(block: dict[str, Any]) -> float:
    """Read wall-clock prediction FPS."""
    return float(block.get("performance", {}).get("fps", 0.0))


def parse_key(key: str) -> tuple[str, str]:
    """Split summary model key into model name and corruption name."""
    name, corruption = key.rsplit("/", 1)
    return name, corruption


def rows(summary: dict[str, Any], reference: str) -> list[dict[str, Any]]:
    """Build per-scene delta rows against the reference model."""
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, block in summary["models"].items():
        name, corruption = parse_key(key)
        grouped[corruption][name] = block
    out: list[dict[str, Any]] = []
    for corruption, models in grouped.items():
        ref = models.get(reference)
        if ref is None:
            continue
        for model, block in sorted(models.items()):
            if model == reference:
                continue
            out.append(
                {
                    "corruption": corruption,
                    "model": model,
                    "all_AP50": metric(block, "all", "AP50"),
                    "all_AP50_delta": metric(block, "all", "AP50") - metric(ref, "all", "AP50"),
                    "all_AP": metric(block, "all", "AP"),
                    "all_AP_delta": metric(block, "all", "AP") - metric(ref, "all", "AP"),
                    "all_AR100": metric(block, "all", "AR100"),
                    "all_AR100_delta": metric(block, "all", "AR100") - metric(ref, "all", "AR100"),
                    "overlap_AP50": metric(block, "overlap", "AP50"),
                    "overlap_AP50_delta": metric(block, "overlap", "AP50") - metric(ref, "overlap", "AP50"),
                    "non_overlap_AP50": metric(block, "non_overlap", "AP50"),
                    "non_overlap_AP50_delta": metric(block, "non_overlap", "AP50") - metric(ref, "non_overlap", "AP50"),
                    "empty_fp_per_image": fp(block),
                    "empty_fp_delta": fp(block) - fp(ref),
                    "fps": fps(block),
                    "fps_delta": fps(block) - fps(ref),
                    "prediction_count": int(block.get("prediction_count", 0)),
                }
            )
    return out


def write_csv(path: Path, table: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)


def write_markdown(path: Path, table: list[dict[str, Any]], summary: dict[str, Any], reference: str) -> None:
    """Write a concise evidence report for competition material."""
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table:
        by_model[row["model"]].append(row)
    lines = [
        "# GrayNav-DCE Generic Evaluation Summary",
        "",
        f"Reference model: `{reference}`",
        "",
        "Important interpretation: large gains against M0 include domain-specific training and dataset-native classes. DCE contribution must be judged against a same-data no-DCE ablation when available.",
        "",
        "## Dataset / Evaluation Settings",
        "",
        f"- Classes: {summary.get('settings', {}).get('names', [])}",
        f"- Overlap classes: {summary.get('settings', {}).get('overlap_names', [])}",
        f"- Non-overlap classes: {summary.get('settings', {}).get('non_overlap_names', [])}",
        f"- Corruptions: {summary.get('settings', {}).get('corruptions', [])}",
        "",
        "## Average Deltas vs Reference",
        "",
        "| model | all AP50 | all AP | all AR100 | overlap AP50 | non-overlap AP50 | empty FP/img | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, model_rows in sorted(by_model.items()):
        lines.append(
            "| {model} | {all_ap50:+.4f} | {all_ap:+.4f} | {ar100:+.4f} | {ov:+.4f} | {non:+.4f} | {fp:+.4f} | {fps:+.2f} |".format(
                model=model,
                all_ap50=mean(r["all_AP50_delta"] for r in model_rows),
                all_ap=mean(r["all_AP_delta"] for r in model_rows),
                ar100=mean(r["all_AR100_delta"] for r in model_rows),
                ov=mean(r["overlap_AP50_delta"] for r in model_rows),
                non=mean(r["non_overlap_AP50_delta"] for r in model_rows),
                fp=mean(r["empty_fp_delta"] for r in model_rows),
                fps=mean(r["fps_delta"] for r in model_rows),
            )
        )
    lines.extend(["", "## Per-Corruption Rows", ""])
    lines.append("| corruption | model | all AP50 delta | all AP delta | overlap AP50 delta | non-overlap AP50 delta |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in table:
        lines.append(
            f"| {row['corruption']} | {row['model']} | {row['all_AP50_delta']:+.4f} | {row['all_AP_delta']:+.4f} | {row['overlap_AP50_delta']:+.4f} | {row['non_overlap_AP50_delta']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    table = rows(summary, args.reference)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "eval_delta_table.csv", table)
    write_markdown(args.out_dir / "eval_evidence_summary.md", table, summary, args.reference)
    print(f"csv={args.out_dir / 'eval_delta_table.csv'}")
    print(f"report={args.out_dir / 'eval_evidence_summary.md'}")


if __name__ == "__main__":
    main()
