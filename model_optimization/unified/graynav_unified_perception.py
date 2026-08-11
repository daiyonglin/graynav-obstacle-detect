"""True-mono YOLOv8n with raw detection, surface, and depth outputs.

The wrapper deliberately stops before Ultralytics' DFL/decode/NMS path.  It
uses the three PAN/FPN features consumed by the Detect module, exposes six raw
detection tensors, and adds one A1-safe 48x48 road/depth branch.  All nonlinear
decoding and temporal logic remain board-side CPU work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from segmentation.graynav_fast_scnn import ConvBNReLU, SeparableConvBNReLU


SURFACE_CLASSES = (
    "ground_candidate",
    "blocked_surface",
    "step_or_drop",
    "unknown_other",
)
DEPTH_BINS = 16
OUTPUT_NAMES = (
    "cls_p3",
    "reg_p3",
    "cls_p4",
    "reg_p4",
    "cls_p5",
    "reg_p5",
    "seg_logits",
    "depth_logits",
)


def _first_conv2d(module: nn.Module) -> nn.Conv2d:
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    raise TypeError(f"no Conv2d found in {type(module).__name__}")


def _fold_detector_stem_to_gray(detector: nn.Module) -> tuple[int, ...]:
    """Replace an RGB detector stem with an exactly equivalent gray stem."""

    wrapper = detector.model[0]
    source = _first_conv2d(wrapper)
    if source.in_channels == 1:
        return tuple(source.weight.shape)
    if source.in_channels != 3 or source.groups != 1:
        raise ValueError(
            "expected a conventional RGB YOLO stem, got "
            f"shape={tuple(source.weight.shape)} groups={source.groups}"
        )
    replacement = nn.Conv2d(
        1,
        source.out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=1,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(source.weight.sum(dim=1, keepdim=True))
        if source.bias is not None:
            replacement.bias.copy_(source.bias)
    # Ultralytics Conv stores its torch Conv2d in the public ``conv`` field.
    if not hasattr(wrapper, "conv"):
        raise TypeError("unexpected Ultralytics stem wrapper without .conv")
    wrapper.conv = replacement
    return tuple(replacement.weight.shape)


def _use_static_c2f_splits(detector: nn.Module) -> int:
    """Avoid exporter-generated Shape/Gather/Slice for channel chunking."""

    changed = 0
    for module in detector.modules():
        if module.__class__.__name__ == "C2f" and hasattr(module, "forward_split"):
            module.forward = module.forward_split
            changed += 1
    return changed


def _copy_last_conv(source: nn.Module, target: nn.Module) -> int:
    source_convs = [item for item in source.modules() if isinstance(item, nn.Conv2d)]
    target_convs = [item for item in target.modules() if isinstance(item, nn.Conv2d)]
    if not source_convs or not target_convs:
        return 0
    old, new = source_convs[-1], target_convs[-1]
    if tuple(old.weight.shape) != tuple(new.weight.shape):
        return 0
    with torch.no_grad():
        new.weight.copy_(old.weight)
        copied = 1
        if old.bias is not None and new.bias is not None:
            new.bias.copy_(old.bias)
            copied += 1
    return copied


def _a1_safe_branch(in_channels: int, hidden: int, out_channels: int) -> nn.Sequential:
    """Two depthwise-separable blocks plus the raw 1x1 output projection."""

    return nn.Sequential(
        SeparableConvBNReLU(in_channels, hidden),
        SeparableConvBNReLU(hidden, hidden),
        nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
    )


def _replace_detect_heads_a1_safe(detector: nn.Module) -> int:
    """Remove P5 3x3x256 convolutions while preserving final COCO tensors."""

    detect = detector.model[-1]
    old_reg = list(detect.cv2)
    old_cls = list(detect.cv3)
    channels = [_first_conv2d(branch).in_channels for branch in old_reg]
    reg_hidden = 64
    cls_hidden = 80
    new_reg = nn.ModuleList(
        _a1_safe_branch(value, reg_hidden, 4 * int(detect.reg_max))
        for value in channels
    )
    new_cls = nn.ModuleList(
        _a1_safe_branch(value, cls_hidden, int(detect.nc))
        for value in channels
    )
    copied = sum(_copy_last_conv(old, new) for old, new in zip(old_reg, new_reg))
    copied += sum(_copy_last_conv(old, new) for old, new in zip(old_cls, new_cls))
    detect.cv2 = new_reg
    detect.cv3 = new_cls
    return copied


class GrayNavUnifiedPerception(nn.Module):
    """One SSNE model contract with eight static NCHW raw outputs."""

    def __init__(
        self,
        detector: nn.Module,
        fusion_channels: int = 64,
        num_surface_classes: int = len(SURFACE_CLASSES),
        depth_bins: int = DEPTH_BINS,
    ) -> None:
        super().__init__()
        if fusion_channels != 64:
            raise ValueError("the first A1 deployment contract fixes fusion_channels=64")
        if num_surface_classes != 4 or depth_bins != 16:
            raise ValueError("deployment contract requires 4 surface classes and 16 depth bins")
        if not hasattr(detector, "model") or len(detector.model) < 2:
            raise TypeError("expected an Ultralytics DetectionModel")
        detect = detector.model[-1]
        required = ("f", "cv2", "cv3", "nc", "reg_max")
        if any(not hasattr(detect, name) for name in required):
            raise TypeError("the final detector module does not expose raw YOLO heads")
        if int(detect.nc) != 80 or int(detect.reg_max) != 16:
            raise ValueError(
                f"expected COCO80/reg_max16, got nc={detect.nc} reg_max={detect.reg_max}"
            )
        if len(detect.f) != 3 or len(detect.cv2) != 3 or len(detect.cv3) != 3:
            raise ValueError("expected three P3/P4/P5 detection scales")
        if self._stem(detector).in_channels != 1:
            raise ValueError("unified deployment input must be true single-channel")

        self.detector = detector
        p3_channels = _first_conv2d(detect.cv2[0]).in_channels
        p4_channels = _first_conv2d(detect.cv2[1]).in_channels
        self.detail_projection = ConvBNReLU(
            p3_channels, fusion_channels, kernel_size=1, padding=0
        )
        self.semantic_projection = ConvBNReLU(
            p4_channels, fusion_channels, kernel_size=1, padding=0, relu=False
        )
        self.semantic_upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.detail_refinement = SeparableConvBNReLU(fusion_channels, fusion_channels)
        self.seg_head = nn.Conv2d(fusion_channels, num_surface_classes, 1, bias=True)
        self.depth_head = nn.Conv2d(fusion_channels, depth_bins, 1, bias=True)

    @staticmethod
    def _stem(detector: nn.Module) -> nn.Conv2d:
        return _first_conv2d(detector.model[0])

    @property
    def first_conv(self) -> nn.Conv2d:
        return self._stem(self.detector)

    @property
    def detect_head(self) -> nn.Module:
        return self.detector.model[-1]

    def _neck_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Run the Ultralytics graph up to the Detect input feature list."""

        saved: list[torch.Tensor | None] = []
        value: torch.Tensor | list[torch.Tensor] = images
        for module in self.detector.model[:-1]:
            if module.f != -1:
                if isinstance(module.f, int):
                    value = saved[module.f]
                else:
                    value = [
                        value if index == -1 else saved[index]
                        for index in module.f
                    ]
            value = module(value)
            saved.append(value if module.i in self.detector.save else None)

        routes = self.detect_head.f
        features = [value if index == -1 else saved[index] for index in routes]
        if any(not isinstance(item, torch.Tensor) for item in features):
            raise RuntimeError("failed to collect P3/P4/P5 tensors")
        return features  # type: ignore[return-value]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self._neck_features(images)
        detect = self.detect_head
        regressions = [detect.cv2[index](features[index]) for index in range(3)]
        classes = [detect.cv3[index](features[index]) for index in range(3)]

        semantic = self.semantic_upsample(self.semantic_projection(features[1]))
        road = F.relu(self.detail_projection(features[0]) + semantic, inplace=False)
        road = self.detail_refinement(road)
        seg_logits = self.seg_head(road)
        depth_logits = self.depth_head(road)

        return (
            classes[0], regressions[0],
            classes[1], regressions[1],
            classes[2], regressions[2],
            seg_logits, depth_logits,
        )

    def import_surface_e3_heads(self, checkpoint: Path | str) -> dict[str, int]:
        """Import only E3 layers whose shapes and semantics remain compatible."""

        payload = torch.load(checkpoint, map_location="cpu")
        source = payload.get("model", payload.get("model_state", payload))
        if not isinstance(source, dict):
            raise TypeError("SurfaceDepth checkpoint does not contain a state dictionary")
        target = self.state_dict()
        prefixes = (
            "semantic_projection.",
            "detail_refinement.",
            "seg_head.",
            "depth_head.",
        )
        imported: dict[str, torch.Tensor] = {}
        for key, tensor in source.items():
            if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
                continue
            local_key = key[7:] if key.startswith("module.") else key
            if local_key.startswith(prefixes) and local_key in target:
                if tuple(tensor.shape) == tuple(target[local_key].shape):
                    imported[local_key] = tensor
        missing, unexpected = self.load_state_dict(imported, strict=False)
        del missing, unexpected
        return {
            "imported_tensors": len(imported),
            "candidate_tensors": sum(
                1 for key in target if key.startswith(prefixes)
            ),
        }


def build_random_unified_yolov8n() -> GrayNavUnifiedPerception:
    """Build the random graph used before any expensive joint training."""

    from ultralytics.nn.tasks import DetectionModel

    detector = DetectionModel("yolov8n.yaml", ch=1, nc=80, verbose=False)
    _use_static_c2f_splits(detector)
    _replace_detect_heads_a1_safe(detector)
    return GrayNavUnifiedPerception(detector)


def build_unified_from_yolo_weights(
    weights: Path | str,
) -> tuple[GrayNavUnifiedPerception, dict[str, Any]]:
    """Load official COCO weights and fold the detector stem to one channel."""

    from ultralytics import YOLO

    source = YOLO(str(weights)).model
    before = tuple(_first_conv2d(source.model[0]).weight.shape)
    after = _fold_detector_stem_to_gray(source)
    static_c2f_count = _use_static_c2f_splits(source)
    preserved_head_tensors = _replace_detect_heads_a1_safe(source)
    model = GrayNavUnifiedPerception(source)
    return model, {
        "weights": str(weights),
        "first_conv_before": before,
        "first_conv_after": after,
        "static_c2f_count": static_c2f_count,
        "preserved_final_head_tensors": preserved_head_tensors,
        "a1_safe_depthwise_detection_heads": True,
        "one_channel_first_conv_initialized": True,
        "rgb_input_used": False,
    }
