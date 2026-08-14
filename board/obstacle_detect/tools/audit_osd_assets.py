#!/usr/bin/env python3
"""Audit the fixed-budget two-line Aurora HUD resource contract."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


LABELS = (
    "PERSON", "CHAIR", "TABLE", "BAG", "COUCH", "BENCH",
    "STAIR", "STEP_CHECK", "BLOCKED", "PATH", "UNKNOWN", "AI_FAIL",
)
RANGES = ("NEAR", "MID", "FAR", "UNKNOWN")
DIRECTIONS = ("LEFT", "FRONT", "RIGHT")


def expected_names() -> set[str]:
    return {
        f"INFO_{label}_{range_name}_{direction}.ssbmp"
        for label in LABELS
        for range_name in RANGES
        for direction in DIRECTIONS
    }


def audit(root: Path) -> None:
    expected = expected_names()
    present = {path.name for path in root.glob("INFO_*.ssbmp")}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(f"missing INFO assets: {missing[:8]} (total={len(missing)})")

    total_bytes = 0
    for name in sorted(expected):
        path = root / name
        raw = path.read_bytes()
        if len(raw) < 16:
            raise RuntimeError(f"truncated SSBMP: {path}")
        magic, width, height, colors = struct.unpack("<IIII", raw[:16])
        if magic != 0x5353424D or colors != 32:
            raise RuntimeError(f"invalid SSBMP header: {path}")
        if len(raw) != 16 + width * height:
            raise RuntimeError(f"invalid SSBMP payload length: {path}")
        if width > 420 or height > 120:
            raise RuntimeError(f"HUD asset exceeds display budget: {path} {width}x{height}")
        total_bytes += len(raw)

    print(f"info_assets={len(expected)}")
    print(f"info_asset_bytes={total_bytes}")
    print("GRAYNAV_OSD_ASSET_AUDIT_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "app_assets" / "osd",
    )
    args = parser.parse_args()
    audit(args.root)


if __name__ == "__main__":
    main()
