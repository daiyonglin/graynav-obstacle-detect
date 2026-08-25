#!/usr/bin/env python3
"""Create the exact three-file official A1 conversion submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import onnx


EXPECTED = {
    "cls_p3": [1, 8, 48, 48],
    "reg_p3": [1, 64, 48, 48],
    "cls_p4": [1, 8, 24, 24],
    "reg_p4": [1, 64, 24, 24],
    "cls_p5": [1, 8, 12, 12],
    "reg_p5": [1, 64, 12, 12],
    "scene_logits": [1, 21, 48, 48],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape(value: onnx.ValueInfoProto) -> list[int]:
    return [int(item.dim_value) for item in value.type.tensor_type.shape.dim]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--datasets-contract", type=Path, required=True)
    parser.add_argument("--consistency-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    graph = onnx.load(str(args.onnx))
    onnx.checker.check_model(graph)
    inputs = {item.name: shape(item) for item in graph.graph.input}
    outputs = {item.name: shape(item) for item in graph.graph.output}
    if inputs != {"images": [1, 1, 384, 384]} or outputs != EXPECTED:
        raise RuntimeError(f"deployment contract mismatch: {inputs=} {outputs=}")
    datasets_contract = json.loads(
        args.datasets_contract.read_text(encoding="utf-8")
    )
    if (
        datasets_contract["shape"] != [1, 1, 384, 384]
        or datasets_contract["calibrate"] != 160
        or datasets_contract["evaluate"] != 40
        or datasets_contract["calibration_evaluation_overlap"] != 0
    ):
        raise RuntimeError("calibration dataset contract is invalid")
    consistency = json.loads(args.consistency_report.read_text(encoding="utf-8"))
    if not consistency.get("passed"):
        raise RuntimeError("PyTorch/ONNX consistency did not pass")

    upload = args.output / "upload"
    evidence = args.output / "evidence"
    upload.mkdir(parents=True)
    evidence.mkdir(parents=True)
    onnx_target = upload / args.onnx.name
    datasets_target = upload / "datasets.zip"
    shutil.copy2(args.onnx, onnx_target)
    shutil.copy2(args.datasets, datasets_target)
    config = upload / "config.toml"
    config.write_text(
        "# GrayNav unified true-mono A1 calibration contract.\n"
        "[calibrate.inputs.images]\n"
        "mean = [0]\n"
        "std = [1]\n",
        encoding="utf-8",
    )
    shutil.copy2(args.datasets_contract, evidence / "datasets_contract.json")
    shutil.copy2(args.consistency_report, evidence / "onnx_consistency.json")

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (onnx_target, datasets_target, config)
    }
    manifest = {
        "model": "graynav_unified_indoor8_scene21_gray1",
        "input": inputs,
        "outputs": outputs,
        "calibration": datasets_contract,
        "pytorch_onnx_consistency": consistency,
        "files": files,
        "formal_conversion_policy": "convert the selected trained model once",
    }
    manifest_path = evidence / "CONVERSION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = args.output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(upload.iterdir()):
            handle.write(path, path.name)
    with zipfile.ZipFile(archive) as handle:
        names = sorted(handle.namelist())
    expected_names = sorted([args.onnx.name, "datasets.zip", "config.toml"])
    if names != expected_names:
        raise RuntimeError(f"submission members mismatch: {names}")
    print(json.dumps({
        "submission_zip": str(archive),
        "submission_sha256": sha256(archive),
        "members": names,
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))
    print("GRAYNAV_UNIFIED_A1_SUBMISSION_READY")


if __name__ == "__main__":
    main()
