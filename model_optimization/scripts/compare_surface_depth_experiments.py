#!/usr/bin/env python3
"""Build the immutable E0/E1/E2/E3 metric comparison in JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result", action="append", required=True, metavar="NAME=EVALUATION_JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for item in args.result:
        if "=" not in item:
            raise RuntimeError(f"invalid result specification: {item}")
        name, raw_path = item.split("=", 1)
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append({
            "experiment": name,
            "ground_iou": metrics["iou"]["ground_candidate"],
            "blocked_iou": metrics["iou"]["blocked_surface"],
            "step_precision": metrics["precision"]["step_or_drop"],
            "step_recall": metrics["recall"]["step_or_drop"],
            "step_f1": metrics["f1"]["step_or_drop"],
            "hazard_to_ground": metrics["safety"]["hazard_to_ground_rate"],
            "whole_frame_step": metrics["safety"]["whole_frame_step_prediction_count"],
            "false_whole_frame_step": metrics["safety"].get(
                "false_whole_frame_step_prediction_count"
            ),
            "ade_bottom_false_images": metrics["safety"]["ade_no_step_bottom_false_image_rate"],
            "absrel": metrics["depth"]["absrel"],
            "delta1": metrics["depth"]["delta1"],
            "order": metrics["depth"]["near_far_order_accuracy"],
            "depth_gradient_mae": metrics["depth"]["gradient_mae"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"experiments": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{row[name]:.4f}" if isinstance(row[name], float) else str(row[name])
            for name in columns
        ) + " |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"experiments": len(rows), "output": str(args.output)}, indent=2))
    print("GRAYNAV_EXPERIMENT_COMPARISON_OK")


if __name__ == "__main__":
    main()
