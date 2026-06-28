#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--script", type=Path, default=Path(__file__).with_name("prepare_public_coco_obstacle8_gray.py"))
    return parser.parse_args()


def run_source(script: Path, output_root: Path, src: Dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        str(script),
        "--images",
        str(src["images"]),
        "--annotations",
        str(src["annotations"]),
        "--out",
        str(output_root),
        "--split",
        str(src["split"]),
        "--prefix",
        str(src.get("prefix", f"{src['name']}_")),
    ]
    limit = int(src.get("limit", 0) or 0)
    if limit > 0:
        cmd += ["--limit", str(limit)]
    empty_keep_prob = src.get("empty_keep_prob")
    if empty_keep_prob is not None:
        cmd += ["--empty-keep-prob", str(empty_keep_prob)]
    if bool(src.get("copy_rgb", False)):
        cmd.append("--copy-rgb")
    if src.get("unknown_policy"):
        cmd += ["--unknown-policy", str(src["unknown_policy"])]

    print("=" * 80)
    print("source:", src.get("name", "unnamed"))
    print("cmd:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    output_root = Path(manifest["output_root"]).expanduser().resolve()
    sources = manifest.get("sources", [])
    if not sources:
        raise SystemExit("manifest contains no sources")
    for src in sources:
        run_source(args.script.resolve(), output_root, src)
    print("=" * 80)
    print("dataset ready:", output_root)
    print("yaml:", output_root / "obstacle8-gray.yaml")


if __name__ == "__main__":
    main()
