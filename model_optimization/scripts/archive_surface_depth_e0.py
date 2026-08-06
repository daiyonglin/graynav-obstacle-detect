#!/usr/bin/env python3
"""Create an immutable E0 checkpoint archive without touching source artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite E0 archive: {args.output}")
    required = ("best.pt", "last.pt", "history.json")
    missing = [name for name in required if not (args.train_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"E0 source is incomplete: {missing}")
    args.output.mkdir(parents=True)
    artifacts = {}
    for name in required:
        source = args.train_dir / name
        destination = args.output / name
        shutil.copy2(source, destination)
        artifacts[name] = {"bytes": destination.stat().st_size, "sha256": sha256(destination)}
    manifest = {
        "experiment": "E0",
        "immutable_archive": True,
        "source": str(args.train_dir),
        "artifacts": artifacts,
    }
    (args.output / "archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("GRAYNAV_E0_ARCHIVE_OK")


if __name__ == "__main__":
    main()
