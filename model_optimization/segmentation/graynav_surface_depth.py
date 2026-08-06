"""A1-safe grayscale Fast-SCNN with segmentation and ordinal-depth heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .graynav_fast_scnn import (
    ConvBNReLU,
    FixedPyramidPooling,
    SeparableConvBNReLU,
    _make_bottlenecks,
    _scaled_channels,
)


NUM_SURFACE_CLASSES = 4
NUM_DEPTH_BINS = 16
SURFACE_CLASS_NAMES = (
    "ground_candidate",
    "blocked_surface",
    "step_or_drop",
    "unknown_other",
)
DEPTH_MIN_M = 0.30
DEPTH_MAX_M = 8.0


class GrayNavSurfaceDepth(nn.Module):
    """True-mono shared encoder with raw 64x64 segmentation/depth logits."""

    def __init__(
        self,
        in_channels: int = 1,
        width_mult: float = 1.0,
        detail64: bool = False,
        num_surface_classes: int = NUM_SURFACE_CLASSES,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("deployment contract requires one grayscale input channel")
        if width_mult not in (1.0, 0.75):
            raise ValueError("supported width multipliers are 1.0 and 0.75")
        c32, c48, c64, c96, c128 = (
            _scaled_channels(value, width_mult) for value in (32, 48, 64, 96, 128)
        )
        self.width_mult = width_mult
        self.detail64 = detail64
        self.num_surface_classes = num_surface_classes
        self.learning_to_downsample = nn.Sequential(
            ConvBNReLU(1, c32, kernel_size=3, stride=2, padding=1),
            SeparableConvBNReLU(c32, c48, stride=2),
            SeparableConvBNReLU(c48, c64, stride=2),
        )
        self.bottleneck1 = _make_bottlenecks(c64, c64, 3, expansion=6, stride=2)
        self.bottleneck2 = _make_bottlenecks(c64, c96, 3, expansion=6, stride=2)
        self.bottleneck3 = _make_bottlenecks(c96, c128, 3, expansion=6, stride=1)
        self.pyramid_pooling = FixedPyramidPooling(c128, c128)
        self.low_depthwise = ConvBNReLU(c128, c128, groups=c128)
        self.low_pointwise = ConvBNReLU(c128, c128, kernel_size=1, padding=0, relu=False)
        self.high_pointwise = ConvBNReLU(c64, c128, kernel_size=1, padding=0, relu=False)
        self.shared_decoder = nn.Sequential(
            SeparableConvBNReLU(c128, c128),
            SeparableConvBNReLU(c128, c128),
        )
        head_channels = c128
        if detail64:
            c_detail = _scaled_channels(64, width_mult)
            self.detail_projection = ConvBNReLU(c48, c_detail, kernel_size=1, padding=0)
            self.semantic_projection = ConvBNReLU(
                c128, c_detail, kernel_size=1, padding=0, relu=False
            )
            self.detail_refinement = SeparableConvBNReLU(c_detail, c_detail)
            head_channels = c_detail
        self.seg_head = nn.Conv2d(
            head_channels, num_surface_classes, kernel_size=1, bias=True
        )
        self.depth_head = nn.Conv2d(head_channels, NUM_DEPTH_BINS, kernel_size=1, bias=True)

    @property
    def first_conv(self) -> nn.Conv2d:
        conv = self.learning_to_downsample[0][0]
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("unexpected grayscale stem")
        return conv

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.learning_to_downsample[0](images)       # 128x128
        detail = self.learning_to_downsample[1](stem)       # 64x64
        high = self.learning_to_downsample[2](detail)       # 32x32
        low = self.bottleneck1(high)
        low = self.bottleneck2(low)                 # 8x8
        low = self.bottleneck3(low)
        low = self.pyramid_pooling(low)
        low = F.interpolate(low, scale_factor=4.0, mode="nearest")
        fused = F.relu(
            self.high_pointwise(high) + self.low_pointwise(self.low_depthwise(low)),
            inplace=False,
        )
        shared = self.shared_decoder(fused)
        if self.detail64:
            semantic = self.semantic_projection(shared)
            semantic = F.interpolate(semantic, scale_factor=2.0, mode="nearest")
            shared = F.relu(
                self.detail_projection(detail) + semantic,
                inplace=False,
            )
            shared = self.detail_refinement(shared)
        seg = self.seg_head(shared)
        depth = self.depth_head(shared)
        # Keep the deployment output coarse but retain thin stair boundaries better
        # than the previous 32x32 contract. Both Resize sizes are compile-time static.
        # A constant scale vector exports directly to Resize.  Passing
        # size=(64,64) makes PyTorch synthesize Shape/Slice nodes to preserve N/C,
        # which the A1 compiler contract forbids even though H/W are fixed.
        if not self.detail64:
            seg = F.interpolate(seg, scale_factor=2.0, mode="nearest")
            depth = F.interpolate(depth, scale_factor=2.0, mode="nearest")
        return seg, depth


@torch.no_grad()
def fold_rgb_first_conv_to_gray(model: GrayNavSurfaceDepth, rgb_weight: torch.Tensor) -> None:
    """Preserve repeated-grayscale RGB inference by summing OIHW input kernels."""

    if rgb_weight.ndim != 4 or rgb_weight.shape[1] != 3:
        raise ValueError(f"expected [out,3,k,k], got {tuple(rgb_weight.shape)}")
    folded = rgb_weight.sum(dim=1, keepdim=True)
    if folded.shape != model.first_conv.weight.shape:
        raise ValueError(
            f"folded weight {tuple(folded.shape)} does not match "
            f"{tuple(model.first_conv.weight.shape)}"
        )
    model.first_conv.weight.copy_(
        folded.to(model.first_conv.weight.device, model.first_conv.weight.dtype)
    )


def depth_bin_centers(device: torch.device | None = None) -> torch.Tensor:
    """Return the 16 log-spaced metric centers used by training and CPU decode."""

    edges = torch.linspace(
        torch.log(torch.tensor(DEPTH_MIN_M)),
        torch.log(torch.tensor(DEPTH_MAX_M)),
        NUM_DEPTH_BINS + 1,
        device=device,
    )
    return torch.exp((edges[:-1] + edges[1:]) * 0.5)
