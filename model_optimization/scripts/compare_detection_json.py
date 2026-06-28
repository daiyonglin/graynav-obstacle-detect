#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="golden_predictions.json")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate json/jsonl predictions")
    parser.add_argument("--box-threshold", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=1e-3)
    return parser.parse_args()


def load_reference(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, List[Dict[str, Any]]] = {}
    for case in data.get("cases", []):
        key = Path(case.get("source", case.get("preprocessed_image", ""))).stem
        out[key] = case.get("detections", [])
    return out


def load_candidate(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            packet = json.loads(line)
            key = Path(packet.get("source", str(packet.get("frame", "")))).stem
            dets = []
            for obj in packet.get("objects", []):
                dets.append({
                    "box": obj.get("box", []),
                    "score": obj.get("conf", obj.get("score", 0.0)),
                    "class_id": obj.get("raw_cls", obj.get("class_id", -1)),
                    "label": obj.get("raw_label", obj.get("label", "")),
                })
            out[key] = dets
        return out

    data = json.loads(path.read_text(encoding="utf-8"))
    for case in data.get("cases", []):
        key = Path(case.get("source", case.get("preprocessed_image", ""))).stem
        out[key] = case.get("detections", [])
    return out


def mean_box_error(a: List[float], b: List[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return float("inf")
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / 4.0


def main() -> None:
    args = parse_args()
    ref = load_reference(args.reference)
    cand = load_candidate(args.candidate)
    failures = []
    compared = 0
    for key, ref_dets in ref.items():
        cand_dets = cand.get(key, [])
        if len(ref_dets) != len(cand_dets):
            failures.append(f"{key}: detection count ref={len(ref_dets)} cand={len(cand_dets)}")
            continue
        for idx, (r, c) in enumerate(zip(ref_dets, cand_dets)):
            compared += 1
            if int(r.get("class_id", -1)) != int(c.get("class_id", -2)):
                failures.append(f"{key}[{idx}]: class ref={r.get('class_id')} cand={c.get('class_id')}")
            box_err = mean_box_error(r.get("box", []), c.get("box", []))
            if box_err > args.box_threshold:
                failures.append(f"{key}[{idx}]: box_mean_abs_error={box_err:.4f}")
            score_err = abs(float(r.get("score", 0.0)) - float(c.get("score", 0.0)))
            if score_err > args.score_threshold:
                failures.append(f"{key}[{idx}]: score_abs_error={score_err:.6f}")

    summary = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "cases": len(ref),
        "detections_compared": compared,
        "failures": failures,
        "passed": len(failures) == 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
