"""True-mono indoor YOLOv8n with packed scene/depth/stair output.

The wrapper deliberately stops before Ultralytics' DFL/decode/NMS path.  It
uses the three PAN/FPN features consumed by the Detect module, exposes six raw
detection tensors, and adds one A1-safe 48x48 scene branch.  All nonlinear
decoding and temporal logic remain board-side CPU work.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .graynav_fast_scnn import ConvBNReLU, SeparableConvBNReLU
except ImportError:  # Direct execution from the training/ working directory.
    from models.graynav_fast_scnn import ConvBNReLU, SeparableConvBNReLU


SURFACE_CLASSES = (
    "ground_candidate",
    "blocked_surface",
    "step_or_drop",
    "unknown_other",
)
DEPTH_BINS = 16
INDOOR_CLASS_NAMES = (
    "person",
    "chair",
    "dining_table",
    "backpack",
    "handbag",
    "suitcase",
    "couch",
    "bench",
)
# COCO80 indices in exactly the order exposed by the indoor deployment head.
INDOOR_COCO_CLASS_IDS = (0, 56, 60, 24, 26, 28, 57, 13)
SCENE_CHANNELS = len(SURFACE_CLASSES) + DEPTH_BINS + 1
STAIR_EDGE_CHANNEL = SCENE_CHANNELS - 1
OUTPUT_NAMES = (
    "cls_p3",
    "reg_p3",
    "cls_p4",
    "reg_p4",
    "cls_p5",
    "reg_p5",
    "scene_logits",
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


@torch.no_grad()
def _initialize_passthrough(module: nn.Module) -> None:
    """Give new A1-safe adapters a deterministic identity-like start."""

    for item in module.modules():
        if isinstance(item, nn.Conv2d):
            item.weight.zero_()
            if item.kernel_size == (1, 1):
                diagonal = min(item.in_channels, item.out_channels)
                for index in range(diagonal):
                    item.weight[index, index, 0, 0] = 1.0
            elif item.groups == item.in_channels and item.in_channels == item.out_channels:
                cy = item.kernel_size[0] // 2
                cx = item.kernel_size[1] // 2
                item.weight[:, 0, cy, cx] = 1.0
            else:
                nn.init.dirac_(item.weight)
            if item.bias is not None:
                item.bias.zero_()
        elif isinstance(item, nn.BatchNorm2d):
            item.weight.fill_(1.0)
            item.bias.zero_()
            item.running_mean.zero_()
            item.running_var.fill_(1.0)


def _adapt_p5_from_p4(p5_channels: int, p4_branch: nn.Sequential) -> nn.Sequential:
    """Reuse a pretrained P4 head behind one A1-safe P5 channel adapter.

    The original YOLOv8n P5 head starts with a 3x3 convolution whose
    ``3*3*Cin`` value is 2304 and therefore exceeds the A1 limit of 2048.
    P3 and P4 are already safe and must not be replaced.  Only P5 is adapted
    from 256 to the P4 input width (128), after which the complete pretrained
    P4 branch can be reused.
    """

    p4_channels = _first_conv2d(p4_branch).in_channels
    adapter = ConvBNReLU(
        p5_channels,
        p4_channels,
        kernel_size=1,
        stride=1,
        padding=0,
    )
    _initialize_passthrough(adapter)
    return nn.Sequential(adapter, *deepcopy(list(p4_branch.children())))


def _select_classifier_rows(
    branch: nn.Sequential,
    class_ids: tuple[int, ...],
) -> int:
    """Replace only the last classifier projection with selected COCO rows."""

    source = branch[-1]
    if not isinstance(source, nn.Conv2d):
        raise TypeError("expected YOLO classifier branch to end in Conv2d")
    indices = torch.tensor(class_ids, dtype=torch.long, device=source.weight.device)
    target = nn.Conv2d(
        source.in_channels,
        len(class_ids),
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
    ).to(device=source.weight.device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight.index_select(0, indices))
        copied = 1
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias.index_select(0, indices))
            copied += 1
    branch[-1] = target
    return copied


def _replace_detect_heads_a1_safe(
    detector: nn.Module,
    class_ids: tuple[int, ...] | None = None,
) -> int:
    """Preserve safe pretrained heads and adapt only the unsafe P5 input."""

    detect = detector.model[-1]
    old_reg = list(detect.cv2)
    old_cls = list(detect.cv3)
    output_classes = len(class_ids) if class_ids is not None else int(detect.nc)
    if not all(isinstance(item, nn.Sequential) for item in old_reg + old_cls):
        raise TypeError("expected YOLOv8 detection branches to be Sequential")

    p5_channels = _first_conv2d(old_reg[2]).in_channels
    new_reg = nn.ModuleList((
        old_reg[0],
        old_reg[1],
        _adapt_p5_from_p4(p5_channels, old_reg[1]),
    ))
    new_cls = nn.ModuleList((
        old_cls[0],
        old_cls[1],
        _adapt_p5_from_p4(p5_channels, old_cls[1]),
    ))

    copied = 0
    if class_ids is not None:
        copied += sum(
            _select_classifier_rows(branch, class_ids)
            for branch in new_cls
        )
    detect.cv2 = new_reg
    detect.cv3 = new_cls
    detect.nc = output_classes
    if hasattr(detect, "no"):
        detect.no = output_classes + int(detect.reg_max) * 4
    return copied


class GrayNavUnifiedPerception(nn.Module):
    """One SSNE model contract with seven static NCHW raw outputs."""

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
        if int(detect.nc) != len(INDOOR_CLASS_NAMES) or int(detect.reg_max) != 16:
            raise ValueError(
                f"expected indoor8/reg_max16, got nc={detect.nc} reg_max={detect.reg_max}"
            )
        if len(detect.f) != 3 or len(detect.cv2) != 3 or len(detect.cv3) != 3:
            raise ValueError("expected three P3/P4/P5 detection scales")
        if self._stem(detector).in_channels != 1:
            raise ValueError("unified deployment input must be true single-channel")

        self.detector = detector
        p3_channels = _first_conv2d(detect.cv2[0]).in_channels
        p4_channels = _first_conv2d(detect.cv2[1]).in_channels
        # Compatibility adapters keep the proven E3 detail64 tensor shapes.
        # Their identity-like initialization avoids inserting an unconstrained
        # random bottleneck between the pretrained YOLO neck and E3 heads.
        self.p3_compat = ConvBNReLU(
            p3_channels, 48, kernel_size=1, padding=0
        )
        self.p4_compat = ConvBNReLU(
            p4_channels, 128, kernel_size=1, padding=0
        )
        self.detail_projection = ConvBNReLU(
            48, fusion_channels, kernel_size=1, padding=0
        )
        self.semantic_projection = ConvBNReLU(
            128, fusion_channels, kernel_size=1, padding=0, relu=False
        )
        self.semantic_upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        self.detail_refinement = SeparableConvBNReLU(fusion_channels, fusion_channels)
        self.seg_head = nn.Conv2d(fusion_channels, num_surface_classes, 1, bias=True)
        self.depth_head = nn.Conv2d(fusion_channels, depth_bins, 1, bias=True)
        self.stair_edge_head = nn.Conv2d(fusion_channels, 1, 1, bias=True)
        for module in (
            self.p3_compat,
            self.p4_compat,
            self.detail_projection,
            self.semantic_projection,
            self.detail_refinement,
            self.seg_head,
            self.depth_head,
            self.stair_edge_head,
        ):
            _initialize_passthrough(module)

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
        detection = self._detection_outputs(features)
        scene_logits = self._scene_output(features)
        return (*detection, scene_logits)

    def _detection_outputs(
        self, features: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        detect = self.detect_head
        regressions = [detect.cv2[index](features[index]) for index in range(3)]
        classes = [detect.cv3[index](features[index]) for index in range(3)]
        return (
            classes[0], regressions[0],
            classes[1], regressions[1],
            classes[2], regressions[2],
        )

    def _scene_output(self, features: list[torch.Tensor]) -> torch.Tensor:
        semantic = self.semantic_upsample(
            self.semantic_projection(self.p4_compat(features[1]))
        )
        road = F.relu(
            self.detail_projection(self.p3_compat(features[0])) + semantic,
            inplace=False,
        )
        road = self.detail_refinement(road)
        scene_logits = torch.cat(
            (self.seg_head(road), self.depth_head(road), self.stair_edge_head(road)),
            dim=1,
        )
        return scene_logits

    def forward_detection(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Training-only fast path; deployment still exports ``forward``."""

        return self._detection_outputs(self._neck_features(images))

    def forward_scene(self, images: torch.Tensor) -> torch.Tensor:
        """Training-only fast path; deployment still exports ``forward``."""

        return self._scene_output(self._neck_features(images))

    def import_surface_e3_heads(self, checkpoint: Path | str) -> dict[str, int]:
        """Import only E3 layers whose shapes and semantics remain compatible."""

        payload = torch.load(checkpoint, map_location="cpu")
        source = payload.get("model", payload.get("model_state", payload))
        if not isinstance(source, dict):
            raise TypeError("SurfaceDepth checkpoint does not contain a state dictionary")
        target = self.state_dict()
        prefixes = (
            "detail_projection.",
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
        required = {
            "seg_head.weight",
            "seg_head.bias",
            "depth_head.weight",
            "depth_head.bias",
        }
        missing_required = sorted(required.difference(imported))
        if missing_required:
            raise RuntimeError(
                "SurfaceDepth E3 import is incomplete; refusing random scene heads: "
                + ", ".join(missing_required)
            )
        missing, unexpected = self.load_state_dict(imported, strict=False)
        del missing, unexpected
        edge_initialized = False
        if "seg_head.weight" in imported and "seg_head.bias" in imported:
            with torch.no_grad():
                self.stair_edge_head.weight.copy_(self.seg_head.weight[2:3])
                self.stair_edge_head.bias.copy_(self.seg_head.bias[2:3])
            edge_initialized = True
        return {
            "imported_tensors": len(imported),
            "candidate_tensors": sum(
                1 for key in target if key.startswith(prefixes)
            ),
            "stair_edge_initialized_from_step": edge_initialized,
        }


def build_random_unified_yolov8n() -> GrayNavUnifiedPerception:
    """Build the random graph used before any expensive joint training."""

    from ultralytics.nn.tasks import DetectionModel

    detector = DetectionModel(
        "yolov8n.yaml", ch=1, nc=len(INDOOR_CLASS_NAMES), verbose=False
    )
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
    preserved_head_tensors = _replace_detect_heads_a1_safe(
        source, INDOOR_COCO_CLASS_IDS
    )
    model = GrayNavUnifiedPerception(source)
    return model, {
        "weights": str(weights),
        "first_conv_before": before,
        "first_conv_after": after,
        "static_c2f_count": static_c2f_count,
        "preserved_classifier_tensors": preserved_head_tensors,
        "indoor_class_names": list(INDOOR_CLASS_NAMES),
        "source_coco_class_ids": list(INDOOR_COCO_CLASS_IDS),
        "pretrained_p3_p4_heads_preserved": True,
        "a1_safe_p5_channel_adapter": True,
        "one_channel_first_conv_initialized": True,
        "rgb_input_used": False,
    }
