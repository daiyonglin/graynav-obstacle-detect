#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot a saved GrayChannelAdapter LUT JSON.")
    p.add_argument("--adapter-json", required=True, type=Path, help="Path to gray_adapter.json.")
    p.add_argument("--out-dir", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.adapter_json.read_text(encoding="utf-8"))
    if data.get("deploy_mode") != "lut_3x256_uint8" or "lut" not in data:
        raise SystemExit(f"{args.adapter_json} is not a deployable LUT adapter JSON")
    lut = np.asarray(data["lut"], dtype=np.float32)
    if lut.shape != (3, 256):
        raise SystemExit(f"expected 3x256 LUT, got {lut.shape}")
    x = np.arange(256, dtype=np.float32)

    stats = []
    for idx, ch in enumerate(lut):
        delta = ch - x
        stats.append(
            {
                "channel": idx,
                "mean_delta": float(np.mean(delta)),
                "mae_delta": float(np.mean(np.abs(delta))),
                "max_abs_delta": float(np.max(np.abs(delta))),
                "monotonic_violations": int(np.sum(np.diff(ch) < 0)),
                "min": int(ch.min()),
                "max": int(ch.max()),
            }
        )

    plt.figure(figsize=(9, 5.2))
    for idx, ch in enumerate(lut):
        plt.plot(x, ch, label=f"ch{idx}", linewidth=1.8)
    plt.plot(x, x, "--", color="black", linewidth=1.0, label="identity")
    plt.xlabel("input gray")
    plt.ylabel("output value")
    plt.title("GrayChannelAdapter LUT mapping")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "lut_mapping.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5.2))
    for idx, ch in enumerate(lut):
        plt.plot(x, ch - x, label=f"ch{idx}", linewidth=1.8)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.xlabel("input gray")
    plt.ylabel("output - input")
    plt.title("LUT delta from gray-copy baseline")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "lut_delta.png", dpi=160)
    plt.close()

    (args.out_dir / "lut_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"plots={args.out_dir}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
