#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnxruntime as ort
from onnx.utils import extract_model
from ultralytics import YOLO

from graynav_dce import register_ultralytics_dce


DEFAULT_HEAD6_OUTPUTS = [
    "/model.24/cv3.0/cv3.0.2/Conv_output_0",
    "/model.24/cv3.1/cv3.1.2/Conv_output_0",
    "/model.24/cv3.2/cv3.2.2/Conv_output_0",
    "/model.24/cv2.0/cv2.0.2/Conv_output_0",
    "/model.24/cv2.1/cv2.1.2/Conv_output_0",
    "/model.24/cv2.2/cv2.2.2/Conv_output_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GrayNav-DCE-YOLOv8n full and head6 ONNX models.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=25)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--output-names", nargs="*", default=DEFAULT_HEAD6_OUTPUTS)
    return parser.parse_args()


def inspect_head6(path: Path, num_classes: int) -> None:
    """Validate that the extracted ONNX contains 3 cls heads and 3 DFL heads."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("Inputs:")
    for x in session.get_inputs():
        print(f"  {x.name}: {x.shape} {x.type}")
    print("Outputs:")
    cls = 0
    reg = 0
    for y in session.get_outputs():
        shape = y.shape
        print(f"  {y.name}: {shape} {y.type}")
        channels = shape[1] if len(shape) == 4 else None
        if channels == num_classes:
            cls += 1
        elif channels == 64:
            reg += 1
    if cls != 3 or reg != 3:
        raise RuntimeError(f"Unexpected head6 outputs: cls={cls}, reg={reg}, expected 3/3")


def main() -> None:
    args = parse_args()
    register_ultralytics_dce()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False, nms=False))
    full_onnx = args.out_dir / "graynav_dce_yolov8n.onnx"
    if exported.resolve() != full_onnx.resolve():
        shutil.copy2(exported, full_onnx)
    head6_onnx = args.out_dir / "graynav_dce_yolov8n_head6.onnx"
    extract_model(str(full_onnx), str(head6_onnx), input_names=[args.input_name], output_names=list(args.output_names))
    inspect_head6(head6_onnx, args.num_classes)
    print(f"full onnx : {full_onnx}")
    print(f"head6 onnx: {head6_onnx}")


if __name__ == "__main__":
    main()
