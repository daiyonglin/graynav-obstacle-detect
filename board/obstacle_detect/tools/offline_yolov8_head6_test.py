#!/usr/bin/env python3
"""Validate GrayNav 1-channel/25-class head6 decode with board dual ROIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


ROD25_NAMES = [
    "bike", "building", "car", "person", "stairs", "traffic_sign",
    "electrical_pole", "road", "motorcycle", "dustbin", "dog", "manhole",
    "tree", "guard_rail", "pedestrian_crosswalk", "truck", "bus", "bench",
    "traffic_cone", "fire_hydrant", "teraffic_barrel", "plant_pot",
    "electrical_box", "chair", "bicycle_rack",
]
SEMANTIC_NAMES = [
    "vehicle/bicycle", "generic_obstacle", "vehicle/bicycle", "person",
    "generic_obstacle", "generic_obstacle", "generic_obstacle", "ignored",
    "vehicle/bicycle", "generic_obstacle", "generic_obstacle", "generic_obstacle",
    "generic_obstacle", "generic_obstacle", "generic_obstacle", "vehicle/bicycle",
    "vehicle/bicycle", "chair/seat", "generic_obstacle", "generic_obstacle",
    "generic_obstacle", "small_object", "generic_obstacle", "chair/seat",
    "generic_obstacle",
]
REG_MAX = 16
REG_CHANNELS = 64
MODEL_SIZE = 384


def list_images(paths: Sequence[str], limit: int) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for candidate in candidates:
            if candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                found.append(candidate)
                if limit and len(found) >= limit:
                    return found
    return found


def to_chw(array: np.ndarray, channels: int) -> np.ndarray:
    """Normalizes an ONNX output to C,H,W using the expected channel count."""
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise RuntimeError(f"expected 3D output, got {value.shape}")
    if value.shape[0] == channels:
        return value
    if value.shape[-1] == channels:
        return np.transpose(value, (2, 0, 1))
    raise RuntimeError(f"cannot locate {channels} channels in output {value.shape}")


def infer_class_count(outputs: Sequence[np.ndarray]) -> int:
    candidates: List[int] = []
    for output in outputs:
        shape = np.asarray(output).shape
        for value in shape:
            if value not in {1, 12, 24, 48, 64, 144, 576, 2304} and 1 < value <= 200:
                candidates.append(int(value))
    for preferred in (25, 80, 8):
        if preferred in candidates:
            return preferred
    raise RuntimeError(f"cannot infer class count from {[np.asarray(o).shape for o in outputs]}")


def group_heads(outputs: Sequence[np.ndarray], classes: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    class_heads = [to_chw(out, classes) for out in outputs if classes in np.asarray(out).shape]
    reg_heads = [to_chw(out, REG_CHANNELS) for out in outputs if REG_CHANNELS in np.asarray(out).shape]
    class_heads.sort(key=lambda value: value.shape[1] * value.shape[2], reverse=True)
    regs = {(value.shape[1], value.shape[2]): value for value in reg_heads}
    pairs = [(head, regs[(head.shape[1], head.shape[2])]) for head in class_heads]
    if len(pairs) != 3:
        raise RuntimeError(f"expected 3 head pairs, got {len(pairs)}")
    return pairs


def threshold(raw_class: int, classes: int) -> float:
    if classes == 25:
        if raw_class == 3:
            return 0.14
        if raw_class in {17, 23}:
            return 0.18
        if raw_class in {0, 2, 8, 15, 16}:
            return 0.28
        if raw_class == 21:
            return 0.22
        return 0.24
    return 0.20


def preprocess(gray: np.ndarray, roi: Tuple[int, int, int, int], channels: int):
    x1, y1, x2, y2 = roi
    crop = gray[y1:y2, x1:x2]
    scale = min(MODEL_SIZE / crop.shape[1], MODEL_SIZE / crop.shape[0])
    rw, rh = round(crop.shape[1] * scale), round(crop.shape[0] * scale)
    px, py = (MODEL_SIZE - rw) // 2, (MODEL_SIZE - rh) // 2
    letter = np.full((MODEL_SIZE, MODEL_SIZE), 114, dtype=np.uint8)
    letter[py:py + rh, px:px + rw] = cv2.resize(crop, (rw, rh))
    if channels == 1:
        tensor = letter[None, None].astype(np.float32) / 255.0
    else:
        tensor = np.repeat(letter[None, None], channels, axis=1).astype(np.float32) / 255.0
    return tensor, {"roi": roi, "scale": scale, "pad_x": px, "pad_y": py}


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def dfl(reg: np.ndarray, side: int, y: int, x: int) -> float:
    logits = reg[side * REG_MAX:(side + 1) * REG_MAX, y, x]
    weights = np.exp(logits - logits.max())
    return float((weights * np.arange(REG_MAX)).sum() / weights.sum())


def map_box(box: np.ndarray, meta: Dict) -> np.ndarray:
    x1, y1, _, _ = meta["roi"]
    box[[0, 2]] = (box[[0, 2]] - meta["pad_x"]) / meta["scale"] + x1
    box[[1, 3]] = (box[[1, 3]] - meta["pad_y"]) / meta["scale"] + y1
    return box


def decode(outputs: Sequence[np.ndarray], meta: Dict, classes: int) -> List[Dict]:
    items: List[Dict] = []
    for cls, reg in group_heads(outputs, classes):
        _, height, width = cls.shape
        stride = MODEL_SIZE // width
        for y in range(height):
            for x in range(width):
                raw_class = int(np.argmax(cls[:, y, x]))
                if classes == 25 and raw_class == 7:
                    continue
                score = sigmoid(float(cls[raw_class, y, x]))
                if score < threshold(raw_class, classes):
                    continue
                left, top, right, bottom = [dfl(reg, side, y, x) for side in range(4)]
                ax, ay = x + 0.5, y + 0.5
                box = np.array([(ax - left) * stride, (ay - top) * stride,
                                (ax + right) * stride, (ay + bottom) * stride], np.float32)
                box = map_box(box, meta)
                if box[2] - box[0] < 8 or box[3] - box[1] < 10:
                    continue
                label = ROD25_NAMES[raw_class] if classes == 25 else str(raw_class)
                semantic = SEMANTIC_NAMES[raw_class] if classes == 25 else label
                items.append({"box": box, "raw_cls": raw_class, "raw_label": label,
                              "semantic": semantic, "score": score})
    return items


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) +
                (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / union)


def nms(items: List[Dict], top_k: int, keep: int) -> List[Dict]:
    ordered = sorted(items, key=lambda item: item["score"], reverse=True)[:top_k]
    result: List[Dict] = []
    for item in ordered:
        duplicate = False
        for previous in result:
            overlap = iou(item["box"], previous["box"])
            if item["raw_cls"] == previous["raw_cls"] and overlap > 0.60:
                duplicate = True
            elif overlap > 0.88:
                duplicate = True
            if duplicate:
                break
        if not duplicate:
            result.append(item)
            if len(result) >= keep:
                break
    return result


def run(args: argparse.Namespace) -> None:
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    channels = int(input_meta.shape[1]) if isinstance(input_meta.shape[1], int) else args.channels
    output_names = [output.name for output in session.get_outputs()]
    paths = list_images(args.inputs, args.limit)
    if not paths:
        raise RuntimeError("no input images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "detections.jsonl").open("w", encoding="utf-8") as stream:
        for frame, path in enumerate(paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (args.width, args.height))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rois = [(0, 0, args.width, args.width),
                    (0, args.height - args.width, args.width, args.height)]
            combined: List[Dict] = []
            classes = 0
            for view, roi in enumerate(rois):
                tensor, meta = preprocess(gray, roi, channels)
                outputs = session.run(output_names, {input_name: tensor})
                if classes == 0:
                    classes = infer_class_count(outputs)
                    print(f"model={args.model} input_channels={channels} classes={classes}")
                combined.extend(decode(outputs, meta, classes))
                if args.dump_inputs:
                    cv2.imwrite(str(args.output_dir / f"{frame:04d}_view{view}_input.png"),
                                (tensor[0, 0] * 255).astype(np.uint8))
            objects = nms(combined, args.top_k, args.keep_top_k)
            packet = {"frame": frame, "source": str(path), "objects": []}
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for item in objects:
                box = np.rint(item["box"]).astype(int)
                cv2.rectangle(canvas, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
                text = f'{item["raw_label"]} {item["score"]:.2f}'
                cv2.putText(canvas, text, (box[0], max(18, box[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                packet["objects"].append({**{k: v for k, v in item.items() if k != "box"},
                                           "box": box.tolist()})
            cv2.imwrite(str(args.output_dir / f"{frame:04d}_{path.stem}.jpg"), canvas)
            stream.write(json.dumps(packet, ensure_ascii=False) + "\n")
            print(f"frame={frame} objects={len(objects)} source={path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/offline_head6"))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--keep-top-k", type=int, default=40)
    parser.add_argument("--dump-inputs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
