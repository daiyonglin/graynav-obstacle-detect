#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log COCO adapter evaluation metrics to TensorBoard.")
    parser.add_argument("--summary", required=True, type=Path, help="truth_eval_summary.json path.")
    parser.add_argument("--logdir", required=True, type=Path, help="TensorBoard log directory.")
    parser.add_argument("--step", type=int, default=0)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)
    writer = SummaryWriter(log_dir=str(args.logdir))
    try:
        for variant, info in summary.get("variants", {}).items():
            for group in ["coco_all", "coco_navigation_subset"]:
                for key, value in info.get(group, {}).items():
                    writer.add_scalar(f"eval/{variant}/{group}/{key}", float(value), args.step)
            for key, value in info.get("performance", {}).items():
                writer.add_scalar(f"eval/{variant}/performance/{key}", float(value), args.step)

        variants = summary.get("variants", {})
        if "baseline" in variants:
            base = variants["baseline"]
            for variant, info in variants.items():
                if variant == "baseline":
                    continue
                for group in ["coco_all", "coco_navigation_subset"]:
                    for key in ["AP", "AP50", "AP75", "AR100"]:
                        if key in base.get(group, {}) and key in info.get(group, {}):
                            delta = float(info[group][key]) - float(base[group][key])
                            writer.add_scalar(f"eval_delta/{variant}/{group}/{key}", delta, args.step)
    finally:
        writer.flush()
        writer.close()
    print(f"tensorboard_eval_summary={args.summary}")
    print(f"tensorboard_logdir={args.logdir}")


if __name__ == "__main__":
    main()
