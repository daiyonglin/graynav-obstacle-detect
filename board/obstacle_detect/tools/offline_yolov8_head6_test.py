#!/usr/bin/env python3
"""
Offline obstacle_detect validator.

This script runs yolov8n_head6.onnx on local images and uses a Python port of
the board-side C++ postprocess in src/yolov8_gray.cpp. It cannot validate the
board NPU .m1model runtime, but it can validate the most failure-prone parts
before flashing: gray preprocessing assumptions, head6 decode, NMS, class
folding, OSD boxes, distance estimation, and JSON packet shape.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

NUM_CLASSES = 80
REG_MAX = 16
REG_CHANNELS = 64
MODEL_W = 384
MODEL_H = 384

SEMANTIC_NAMES = [
    "person",
    "chair/seat",
    "table/desk",
    "sofa/bed",
    "bag/suitcase",
    "small_object",
    "vehicle/bicycle",
    "generic_obstacle",
]


def semantic_class_from_raw(raw_cls: int) -> int:
    if raw_cls == 0:
        return 0
    if raw_cls in {13, 56}:
        return 1
    if raw_cls in {60}:
        return 2
    if raw_cls in {57, 59}:
        return 3
    if raw_cls in {24, 26, 28}:
        return 4
    if raw_cls in {39, 41, 63, 65, 66, 67, 73}:
        return 5
    if raw_cls in {1, 2, 3}:
        return 6
    return 7


def is_furniture_semantic(sem: int) -> bool:
    return sem in {1, 2, 3}


def candidate_threshold(raw_cls: int) -> float:
    sem = semantic_class_from_raw(raw_cls)
    if sem == 0 or is_furniture_semantic(sem):
        return 0.16
    if sem in {4, 5}:
        return 0.18
    if sem == 6:
        return 0.20
    return 0.22


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def list_images(paths: Sequence[Path], limit: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    result: List[Path] = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in exts:
            result.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in exts:
                    result.append(child)
                    if limit > 0 and len(result) >= limit:
                        return result
    return result[:limit] if limit > 0 else result


def preprocess_image(
    image_path: Path,
    board_w: int,
    board_h: int,
    simulate_board_shape: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if simulate_board_shape:
        board_gray = cv2.resize(gray, (board_w, board_h), interpolation=cv2.INTER_LINEAR)
    else:
        board_gray = gray
        board_h, board_w = gray.shape[:2]

    scale = min(MODEL_W / float(board_w), MODEL_H / float(board_h))
    resize_w = int(round(board_w * scale))
    resize_h = int(round(board_h * scale))
    pad_x = (MODEL_W - resize_w) // 2
    pad_y = (MODEL_H - resize_h) // 2

    resized = cv2.resize(board_gray, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    letter = np.full((MODEL_H, MODEL_W), 114, dtype=np.uint8)
    letter[pad_y:pad_y + resize_h, pad_x:pad_x + resize_w] = resized

    bgr = cv2.cvtColor(letter, cv2.COLOR_GRAY2BGR)
    chw = bgr.astype(np.float32) / 255.0
    chw = np.transpose(chw, (2, 0, 1))[None, ...]

    info = {
        "board_w": float(board_w),
        "board_h": float(board_h),
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "resize_w": float(resize_w),
        "resize_h": float(resize_h),
    }
    board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
    return chw, board_bgr, info


def normalize_output(out: np.ndarray) -> np.ndarray:
    arr = np.asarray(out)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"unexpected output ndim={arr.ndim}, shape={arr.shape}")
    return arr.astype(np.float32, copy=False)


def group_head6_outputs(outputs: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    cls: List[np.ndarray] = []
    reg: List[np.ndarray] = []
    for out in outputs:
        arr = normalize_output(out)
        if arr.shape[0] == NUM_CLASSES:
            cls.append(arr)
        elif arr.shape[0] == REG_CHANNELS:
            reg.append(arr)

    cls.sort(key=lambda a: a.shape[1] * a.shape[2], reverse=True)
    reg_by_hw = {(a.shape[1], a.shape[2]): a for a in reg}
    paired_reg = []
    for c in cls:
        key = (c.shape[1], c.shape[2])
        if key not in reg_by_hw:
            raise RuntimeError(f"missing regression branch for cls hw={key}")
        paired_reg.append(reg_by_hw[key])

    if len(cls) != 3 or len(paired_reg) != 3:
        raise RuntimeError(f"expected 3 cls + 3 reg outputs, got {len(cls)} cls + {len(paired_reg)} reg")
    return cls, paired_reg


def map_box_to_board(box: np.ndarray, lb: Dict[str, float]) -> np.ndarray:
    out = box.copy()
    out[[0, 2]] = (out[[0, 2]] - lb["pad_x"]) / lb["scale"]
    out[[1, 3]] = (out[[1, 3]] - lb["pad_y"]) / lb["scale"]
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, lb["board_w"] - 1.0)
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, lb["board_h"] - 1.0)
    return out


def sector_from_box(box: np.ndarray, board_w: int) -> str:
    cx = 0.5 * (box[0] + box[2])
    if cx < 0.35 * board_w:
        return "left"
    if cx > 0.65 * board_w:
        return "right"
    return "center"


def estimate_ground_distance_m(box: np.ndarray, board_h: int) -> float:
    fov_v_deg = 78.9
    camera_height_m = 0.85
    camera_pitch_down_deg = 15.0
    min_distance_m = 0.2
    max_distance_m = 8.0

    foot_y = float(np.clip(box[3], 0.0, board_h - 1.0))
    normalized_y = (foot_y - 0.5 * board_h) / max(1.0, float(board_h))
    pixel_angle_deg = normalized_y * fov_v_deg
    ray_down_deg = camera_pitch_down_deg + pixel_angle_deg
    if ray_down_deg <= 0.5:
        return -1.0

    distance = camera_height_m / math.tan(math.radians(ray_down_deg))
    if distance < min_distance_m or distance > max_distance_m:
        return -1.0
    return float(distance)


def size_prior(raw_cls: int):
    priors = {
        0: (1.70, True),
        39: (0.25, True),
        41: (0.12, True),
        56: (0.85, True),
        57: (0.80, True),
        60: (0.75, True),
        62: (0.80, False),
        63: (0.35, False),
        65: (0.18, False),
        66: (0.45, False),
        67: (0.15, False),
        73: (0.24, False),
    }
    return priors.get(raw_cls)


def estimate_size_distance_m(box: np.ndarray, board_w: int, board_h: int, raw_cls: int) -> float:
    prior = size_prior(raw_cls)
    if prior is None:
        return -1.0
    physical_size_m, use_height = prior
    fov_h_deg = 49.7
    fov_v_deg = 78.9
    min_distance_m = 0.2
    max_distance_m = 8.0
    fx = (0.5 * board_w) / math.tan(math.radians(0.5 * fov_h_deg))
    fy = (0.5 * board_h) / math.tan(math.radians(0.5 * fov_v_deg))
    pixel_size = max(1.0, float(box[3] - box[1] if use_height else box[2] - box[0]))
    distance = (fy if use_height else fx) * physical_size_m / pixel_size
    if distance < min_distance_m or distance > max_distance_m:
        return -1.0
    return float(distance)


def fuse_distance(ground_m: float, size_m: float):
    has_ground = ground_m >= 0.0
    has_size = size_m >= 0.0
    if has_ground and has_size:
        return min(ground_m, size_m), "fused"
    if has_ground:
        return ground_m, "ground"
    if has_size:
        return size_m, "size"
    return -1.0, "unknown"


def risk_from_distance(distance_m: float) -> str:
    if distance_m < 0.0:
        return "unknown"
    if distance_m < 1.0:
        return "near"
    if distance_m < 2.0:
        return "warning"
    return "far"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def nms(items: List[Dict], iou_threshold: float, top_k: int) -> List[Dict]:
    items = sorted(items, key=lambda x: x["conf"], reverse=True)[:top_k]
    suppressed = [False] * len(items)
    kept: List[Dict] = []
    for i, cur in enumerate(items):
        if suppressed[i]:
            continue
        kept.append(cur)
        for j in range(i + 1, len(items)):
            if suppressed[j]:
                continue
            if cur["raw_cls"] != items[j]["raw_cls"]:
                continue
            if iou(cur["box_np"], items[j]["box_np"]) > iou_threshold:
                suppressed[j] = True
    return kept


def decode_outputs(
    outputs: Sequence[np.ndarray],
    lb: Dict[str, float],
    conf_threshold: float,
    iou_threshold: float,
    top_k: int,
    keep_top_k: int,
) -> List[Dict]:
    cls_branches, reg_branches = group_head6_outputs(outputs)

    cls_scores = []
    reg_raw = []
    anchor_x = []
    anchor_y = []
    strides = []

    for cls, reg in zip(cls_branches, reg_branches):
        _, h, w = cls.shape
        stride = MODEL_W // w
        hw = h * w
        cls_scores.append(sigmoid(cls.reshape(NUM_CLASSES, hw)))
        reg_raw.append(reg.reshape(REG_CHANNELS, hw))

        grid_y, grid_x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        anchor_x.append((grid_x.reshape(-1) + 0.5).astype(np.float32))
        anchor_y.append((grid_y.reshape(-1) + 0.5).astype(np.float32))
        strides.append(np.full(hw, float(stride), dtype=np.float32))

    cls_all = np.concatenate(cls_scores, axis=1)
    reg_all = np.concatenate(reg_raw, axis=1)
    ax = np.concatenate(anchor_x)
    ay = np.concatenate(anchor_y)
    stride_vec = np.concatenate(strides)
    total_points = cls_all.shape[1]

    dist_logits = reg_all.reshape(4, REG_MAX, total_points)
    bins = np.arange(REG_MAX, dtype=np.float32).reshape(1, REG_MAX, 1)
    dist = np.sum(softmax(dist_logits, axis=1) * bins, axis=1)

    items: List[Dict] = []
    board_w = int(lb["board_w"])
    board_h = int(lb["board_h"])

    for j in range(total_points):
        best_by_semantic = [(-1, 0.0) for _ in SEMANTIC_NAMES]
        for raw_cls in range(NUM_CLASSES):
            score = float(cls_all[raw_cls, j])
            sem = semantic_class_from_raw(raw_cls)
            if score > best_by_semantic[sem][1]:
                best_by_semantic[sem] = (raw_cls, score)
        candidates = [
            (raw_cls, score)
            for raw_cls, score in best_by_semantic
            if raw_cls >= 0 and score >= candidate_threshold(raw_cls)
        ]
        if not candidates:
            continue

        l, t, r, b = dist[:, j]
        x1_fm = ax[j] - l
        y1_fm = ay[j] - t
        x2_fm = ax[j] + r
        y2_fm = ay[j] + b

        cx = (x1_fm + x2_fm) * 0.5 * stride_vec[j]
        cy = (y1_fm + y2_fm) * 0.5 * stride_vec[j]
        bw = (x2_fm - x1_fm) * stride_vec[j]
        bh = (y2_fm - y1_fm) * stride_vec[j]
        box = np.array([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], dtype=np.float32)
        box[[0, 2]] = np.clip(box[[0, 2]], 0.0, MODEL_W - 1.0)
        box[[1, 3]] = np.clip(box[[1, 3]], 0.0, MODEL_H - 1.0)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        box = map_box_to_board(box, lb)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue

        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        if width < 12.0 or height < 12.0:
            continue

        area_ratio = width * height / max(1.0, float(board_w * board_h))
        width_ratio = width / max(1.0, float(board_w))
        height_ratio = height / max(1.0, float(board_h))
        center_y = 0.5 * float(box[1] + box[3]) / max(1.0, float(board_h))
        touch = int(box[0] <= 2.0) + int(box[1] <= 2.0) + int(box[2] >= board_w - 3.0) + int(box[3] >= board_h - 3.0)

        if area_ratio > 0.98 or (width_ratio > 0.98 and height_ratio > 0.98):
            continue
        if raw_cls == 0 and touch >= 4:
            continue
        if raw_cls != 0 and center_y < 0.08 and score < 0.30:
            continue

        for raw_cls, score in candidates:
            raw_label = COCO_NAMES[raw_cls] if 0 <= raw_cls < len(COCO_NAMES) else "unknown"
            semantic_id = semantic_class_from_raw(raw_cls)
            label = SEMANTIC_NAMES[semantic_id]
            ground_m = estimate_ground_distance_m(box, board_h)
            size_m = estimate_size_distance_m(box, board_w, board_h, raw_cls)
            dist_m, dist_src = fuse_distance(ground_m, size_m)

            items.append({
                "dir": sector_from_box(box, board_w),
                "label": label,
                "semantic_class": label,
                "raw_label": raw_label,
                "raw_cls": raw_cls,
                "conf": round(score, 4),
                "dist_m": round(dist_m, 2) if dist_m >= 0 else -1,
                "dist_src": dist_src,
                "risk": risk_from_distance(dist_m),
                "box": [int(round(float(v))) for v in box.tolist()],
                "box_np": box,
            })

    kept = nms(items, iou_threshold, top_k)
    kept = sorted(kept, key=lambda x: x["conf"], reverse=True)[:keep_top_k]
    for item in kept:
        item.pop("box_np", None)
    return kept


def nearest_object(objects: Sequence[Dict]) -> Dict | None:
    valid = [o for o in objects if float(o["dist_m"]) >= 0.0]
    if not valid:
        return None
    obj = min(valid, key=lambda o: float(o["dist_m"]))
    return {
        "dir": obj["dir"],
        "label": obj["label"],
        "semantic_class": obj.get("semantic_class", obj["label"]),
        "dist_m": obj["dist_m"],
        "risk": obj["risk"],
    }


def build_zones_and_nav(objects: Sequence[Dict]):
    zones = {
        "left": {"occupied": False, "dist_m": -1, "risk": "unknown", "label": "", "semantic_class": ""},
        "center": {"occupied": False, "dist_m": -1, "risk": "unknown", "label": "", "semantic_class": ""},
        "right": {"occupied": False, "dist_m": -1, "risk": "unknown", "label": "", "semantic_class": ""},
    }
    for obj in objects:
        d = obj["dir"]
        if d not in zones:
            continue
        cur = zones[d]
        if (not cur["occupied"] or
                (obj["dist_m"] >= 0 and (cur["dist_m"] < 0 or obj["dist_m"] < cur["dist_m"])) or
                (cur["dist_m"] < 0 and obj["conf"] > 0)):
            zones[d] = {
                "occupied": True,
                "label": obj["label"],
                "semantic_class": obj.get("semantic_class", obj["label"]),
                "dist_m": obj["dist_m"],
                "risk": "urgent" if d == "center" and 0 <= obj["dist_m"] < 0.8 else obj["risk"],
            }

    center = zones["center"]
    if center["occupied"] and center["dist_m"] >= 0 and center["dist_m"] < 0.8:
        nav = {"action": "stop", "prompt": f"stop center obstacle {center['dist_m']:.1f}m"}
    elif center["occupied"] and center["risk"] in {"near", "warning"}:
        left_blocked = zones["left"]["occupied"] and 0 <= zones["left"]["dist_m"] < 1.2
        right_blocked = zones["right"]["occupied"] and 0 <= zones["right"]["dist_m"] < 1.2
        if right_blocked and not left_blocked:
            nav = {"action": "turn_left", "prompt": f"turn left center obstacle {center['dist_m']:.1f}m"}
        elif left_blocked and not right_blocked:
            nav = {"action": "turn_right", "prompt": f"turn right center obstacle {center['dist_m']:.1f}m"}
        else:
            nav = {"action": "slow", "prompt": f"slow center obstacle {center['dist_m']:.1f}m"}
    else:
        near = nearest_object(objects)
        if near and near["dist_m"] >= 0 and near["dist_m"] < 1.0:
            nav = {"action": "slow", "prompt": f"slow {near['dir']} obstacle {near['dist_m']:.1f}m"}
        else:
            nav = {"action": "clear", "prompt": "clear"}
    nav["nearest_track"] = -1
    return zones, nav


def draw_objects(img: np.ndarray, objects: Sequence[Dict]) -> np.ndarray:
    out = img.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        color = (0, 220, 0) if obj["label"] == "person" else (0, 180, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        text = f'{obj["label"]} {obj["conf"]:.2f} {obj["dir"]} {obj["dist_m"]}m {obj["risk"]}'
        y = max(24, y1 - 8)
        cv2.putText(out, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return out


def run(args: argparse.Namespace) -> int:
    image_paths = list_images([Path(p) for p in args.inputs], args.limit)
    if not image_paths:
        raise RuntimeError("no input images found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "detections.jsonl"

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"[offline] model={args.model}")
    print(f"[offline] input={input_name}")
    print(f"[offline] outputs={len(output_names)}")
    for o in session.get_outputs():
        print(f"[offline] output {o.name}: shape={o.shape}, type={o.type}")

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for frame, image_path in enumerate(image_paths):
            tensor, board_img, lb = preprocess_image(
                image_path,
                board_w=args.board_width,
                board_h=args.board_height,
                simulate_board_shape=args.simulate_board_shape,
            )
            outputs = session.run(output_names, {input_name: tensor})
            objects = decode_outputs(
                outputs,
                lb,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                top_k=args.top_k,
                keep_top_k=args.keep_top_k,
            )
            packet = {
                "type": "obstacle",
                "frame": frame,
                "source": str(image_path),
                "objects": objects,
                "nearest": nearest_object(objects),
            }
            packet["zones"], packet["nav"] = build_zones_and_nav(objects)
            jf.write(json.dumps(packet, ensure_ascii=False, separators=(",", ":")) + "\n")

            annotated = draw_objects(board_img, objects)
            out_name = f"{frame:04d}_{image_path.stem}.jpg"
            cv2.imwrite(str(output_dir / out_name), annotated)
            print(f"[offline] frame={frame} objects={len(objects)} image={image_path.name} out={out_name}")

    print(f"[offline] wrote {jsonl_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline YOLOv8 head6 obstacle_detect postprocess validation.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("E:/jichuang/gray-test/yolov8n_head6.onnx"),
        help="Path to yolov8n_head6.onnx.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["E:/jichuang/gray-test/coco_val2017_subset"],
        help="Image files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="E:/jichuang/gray-test/runs/offline_obstacle_detect",
        help="Directory for annotated images and detections.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of images to process. 0 means all.")
    parser.add_argument("--conf", type=float, default=0.20, help="Compatibility option; candidate thresholds are class-aware like board code.")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold, same default as board code.")
    parser.add_argument("--top-k", type=int, default=800, help="Pre-NMS top_k, same default as board code.")
    parser.add_argument("--keep-top-k", type=int, default=80, help="Post-NMS keep_top_k, same default as board code.")
    parser.add_argument("--board-width", type=int, default=720, help="Board preview width used by current demo.")
    parser.add_argument("--board-height", type=int, default=1280, help="Board preview height used by current demo.")
    parser.add_argument(
        "--simulate-board-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resize still images to 720x1280 before 384x384 letterbox, matching board-side assumptions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
