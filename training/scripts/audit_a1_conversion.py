#!/usr/bin/env python3
"""Audit an official A1 conversion archive for the unified seven-output model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np


EXPECTED_OUTPUTS = (
    "cls_p3",
    "reg_p3",
    "cls_p4",
    "reg_p4",
    "cls_p5",
    "reg_p5",
    "scene_logits",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    one = left.astype(np.float64, copy=False).reshape(-1)
    two = right.astype(np.float64, copy=False).reshape(-1)
    denominator = math.sqrt(float(one @ one) * float(two @ two))
    if denominator <= 1e-20:
        return 1.0 if np.array_equal(one, two) else 0.0
    return float((one @ two) / denominator)


def parse_tensor_order(text: str, prefix: str) -> list[dict[str, object]]:
    pattern = re.compile(
        rf"{prefix}Tensor\s+name:\s*(\S+)\s+Scale:\s*"
        r"([-+0-9.eE]+)\s+OrderIn\s+M1MODEL:\s*(\d+)"
    )
    return [
        {"name": name, "scale": float(scale), "order": int(order)}
        for name, scale, order in pattern.findall(text)
    ]


def audit(
    names: list[str],
    read: Callable[[str], bytes],
    threshold: float,
) -> dict[str, object]:
    files = [name for name in names if not name.endswith("/")]
    models = [name for name in files if name.endswith(".m1model")]
    reports = [name for name in files if name.endswith("_evaluate_report.json")]
    input_scales = [name for name in files if name.endswith("_InputOrderScale.txt")]
    output_scales = [name for name in files if name.endswith("_OutputOrderScale.txt")]
    errors: list[str] = []
    if len(models) != 1:
        errors.append(f"expected exactly one m1model, found {len(models)}")
    if len(reports) != 1:
        errors.append(f"expected exactly one evaluation report, found {len(reports)}")
    if len(input_scales) != 1 or len(output_scales) != 1:
        errors.append(
            "expected exactly one InputOrderScale and one OutputOrderScale file"
        )
    if errors:
        return {"passed": False, "errors": errors}

    official = json.loads(read(reports[0]).decode("utf-8-sig"))
    if official.get("method") != "cosine_similarity":
        errors.append(f"unexpected evaluation method: {official.get('method')}")
    aggregate = {
        str(item["output_name"]): float(item["similarity"])
        for item in official.get("similarity", [])
    }
    if set(aggregate) != set(EXPECTED_OUTPUTS):
        errors.append(
            f"evaluation outputs mismatch: expected={list(EXPECTED_OUTPUTS)} "
            f"actual={sorted(aggregate)}"
        )

    detail = official.get("detail", {})
    detail_values: dict[str, list[float]] = {
        name: [] for name in EXPECTED_OUTPUTS
    }
    if not isinstance(detail, dict) or not detail:
        errors.append("evaluation detail is empty")
    else:
        for sample, values in detail.items():
            if set(values) != set(EXPECTED_OUTPUTS):
                errors.append(f"sample {sample} outputs mismatch")
                continue
            for name in EXPECTED_OUTPUTS:
                detail_values[name].append(float(values[name]))

    input_order = parse_tensor_order(
        read(input_scales[0]).decode("utf-8-sig"), "Input"
    )
    output_order = parse_tensor_order(
        read(output_scales[0]).decode("utf-8-sig"), "Output"
    )
    if len(input_order) != 1 or input_order[0]["name"] != "images":
        errors.append(f"input order contract mismatch: {input_order}")
    ordered_outputs = [
        item["name"] for item in sorted(output_order, key=lambda item: item["order"])
    ]
    if len(output_order) != 7 or set(ordered_outputs) != set(EXPECTED_OUTPUTS):
        errors.append(f"output order contract mismatch: {output_order}")
    if sorted(int(item["order"]) for item in output_order) != list(range(7)):
        errors.append("output order indices must be exactly 0..6")
    if any(float(item["scale"]) <= 0.0 for item in input_order + output_order):
        errors.append("all input/output quantization scales must be positive")

    independent: dict[str, list[float]] = {
        name: [] for name in EXPECTED_OUTPUTS
    }
    independent_mismatches: list[str] = []
    by_name = set(files)
    for ori_name in files:
        if not ori_name.endswith(".ori.npy"):
            continue
        sim_name = ori_name[:-8] + ".sim.npy"
        if sim_name not in by_name:
            errors.append(f"missing simulated tensor pair for {ori_name}")
            continue
        output_name = PurePosixPath(ori_name).name[:-8]
        if output_name not in independent:
            errors.append(f"unexpected output tensor evidence: {output_name}")
            continue
        original = np.load(io.BytesIO(read(ori_name)), allow_pickle=False)
        simulated = np.load(io.BytesIO(read(sim_name)), allow_pickle=False)
        if original.shape != simulated.shape:
            errors.append(
                f"shape mismatch for {ori_name}: {original.shape} vs {simulated.shape}"
            )
            continue
        value = cosine(original, simulated)
        independent[output_name].append(value)
        sample = PurePosixPath(ori_name).parent.name.removesuffix(".d")
        reported = detail.get(sample, {}).get(output_name)
        if reported is None or abs(value - float(reported)) > 1e-5:
            independent_mismatches.append(
                f"{sample}/{output_name}: recomputed={value} reported={reported}"
            )
    if independent_mismatches:
        errors.append(
            f"{len(independent_mismatches)} independently recomputed values "
            "do not match the official report"
        )
    if any(not values for values in independent.values()):
        missing = [name for name, values in independent.items() if not values]
        errors.append(f"missing ori/sim evidence for outputs: {missing}")

    aggregate_min = min(aggregate.values(), default=0.0)
    detail_min = min(
        (value for values in detail_values.values() for value in values),
        default=0.0,
    )
    independent_min = min(
        (value for values in independent.values() for value in values),
        default=0.0,
    )
    for label, value in (
        ("official aggregate", aggregate_min),
        ("official per-sample", detail_min),
        ("independent per-sample", independent_min),
    ):
        if value < threshold:
            errors.append(f"{label} cosine {value:.6f} is below {threshold:.6f}")

    def stats(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        return {
            name: {
                "min": min(items),
                "mean": float(np.mean(items)),
                "max": max(items),
            }
            for name, items in values.items()
            if items
        }

    return {
        "passed": not errors,
        "threshold": threshold,
        "m1model": {
            "member": models[0],
            "bytes": len(read(models[0])),
            "sha256": sha256_bytes(read(models[0])),
        },
        "official_report": reports[0],
        "official_samples": int(official.get("num", len(detail))),
        "official_aggregate": aggregate,
        "official_per_output": stats(detail_values),
        "independent_per_output": stats(independent),
        "global_min": {
            "official_aggregate": aggregate_min,
            "official_per_sample": detail_min,
            "independent_per_sample": independent_min,
        },
        "input_order_scale": input_order,
        "output_order_scale": output_order,
        "independent_report_mismatches": independent_mismatches,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if args.report.exists():
        raise FileExistsError(args.report)

    if args.package.is_file() and zipfile.is_zipfile(args.package):
        with zipfile.ZipFile(args.package) as archive:
            names = archive.namelist()
            result = audit(names, archive.read, args.threshold)
    elif args.package.is_dir():
        paths = {
            path.relative_to(args.package).as_posix(): path
            for path in args.package.rglob("*")
            if path.is_file()
        }
        result = audit(list(paths), lambda name: paths[name].read_bytes(), args.threshold)
    else:
        raise ValueError("--package must be an official conversion ZIP or directory")
    result["package"] = str(args.package)
    result["package_sha256"] = sha256(args.package) if args.package.is_file() else None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)
    print("GRAYNAV_UNIFIED_A1_CONVERSION_AUDIT_OK")


if __name__ == "__main__":
    main()
