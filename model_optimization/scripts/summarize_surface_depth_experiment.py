#!/usr/bin/env python3
"""Summarize one gated experiment without silently promoting a failed model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("e1", "e2", "e3"), required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.train_dir / "history.json").read_text(encoding="utf-8"))
    if not rows:
        raise RuntimeError("empty experiment history")
    checkpoints = {}
    for name in ("best_overall.pt", "best_seg.pt", "best_step.pt", "best_depth.pt", "last.pt"):
        path = args.train_dir / name
        checkpoints[name] = {
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
    best_rows = {
        name: max(rows, key=lambda row: float(row["checkpoint_scores"][name]))["epoch"]
        for name in ("seg", "step", "depth")
    }
    passing = [row for row in rows if row.get("gates", {}).get("passed", False)]
    best_overall = (
        max(passing, key=lambda row: float(row["checkpoint_scores"]["overall"]))
        if passing else None
    )
    report = {
        "experiment": args.experiment,
        "epochs": len(rows),
        "last_epoch": rows[-1]["epoch"],
        "last_metrics": rows[-1]["metrics"],
        "last_gates": rows[-1]["gates"],
        "best_epochs": {
            **best_rows,
            "overall": None if best_overall is None else best_overall["epoch"],
        },
        "gate_ever_passed": bool(passing),
        "checkpoints": checkpoints,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment": args.experiment,
        "gate_ever_passed": bool(passing),
        "best_epochs": report["best_epochs"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    print("GRAYNAV_EXPERIMENT_SUMMARY_OK")


if __name__ == "__main__":
    main()
