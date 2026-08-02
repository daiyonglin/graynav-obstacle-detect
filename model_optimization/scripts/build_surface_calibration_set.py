#!/usr/bin/env python3
"""Select a deterministic, class-aware 200-image grayscale INT8 calibration set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    records = [json.loads(line) for line in (args.data / "manifest_val.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rare = [record for record in records if record.get("rare")]
    normal = [record for record in records if not record.get("rare")]
    rare_n = min(len(rare), args.count // 2)
    chosen = rare[:rare_n] + normal[: args.count - rare_n]
    if len(chosen) < args.count:
        remaining = [record for record in rare[rare_n:] if record not in chosen]
        chosen += remaining[: args.count - len(chosen)]
    if len(chosen) < args.count:
        raise RuntimeError(f"validation split has only {len(chosen)} usable images")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, record in enumerate(chosen):
        source = args.data / record["image"]
        target = args.output / f"{index:04d}_{source.name}"
        gray = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"cannot read calibration image: {source}")
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        if not cv2.imwrite(str(target), gray):
            raise RuntimeError(f"cannot write calibration image: {target}")
        manifest.append({"image": target.name, "source": record["image"], "rare": record.get("rare", False)})
    (args.output / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(manifest), "rare": sum(int(item["rare"]) for item in manifest)}, indent=2))


if __name__ == "__main__":
    main()
