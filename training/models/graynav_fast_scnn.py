"""A1-safe single-channel Fast-SCNN for GrayNav.

Architecture reference:
  Poudel et al., Fast-SCNN (arXiv:1902.04502)
  PaddleSeg release/2.10, paddleseg/models/fast_scnn.py (Apache-2.0)

The deployment graph deliberately returns raw 1/8-resolution logits.  Loss
upsampling, Softmax, ArgMax, temporal voting and corridor decisions stay
outside the exported network.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


NUM_SURFACE_CLASSES = 4
SURFACE_CLASS_NAMES = (
    "ground_candidate",
    "blocked_surface",
    "step_or_drop",
    "pothole",
)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        relu: bool = True,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=False))
        super().__init__(*layers)


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            ConvBNReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0),
        )


class InvertedBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int, stride: int) -> None:
        super().__init__()
        expanded = in_channels * expansion
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, expanded, kernel_size=1, padding=0),
            ConvBNReLU(
                expanded,
                expanded,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=expanded,
            ),
            ConvBNReLU(expanded, out_channels, kernel_size=1, padding=0, relu=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_shortcut else out


class FixedPyramidPooling(nn.Module):
    """Static 8x8 pyramid pooling whose largest pool kernel is exactly 8."""

    def __init__(self, in_channels: int = 128, out_channels: int = 128) -> None:
        super().__init__()
        branch_channels = in_channels // 4
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.proj8 = ConvBNReLU(in_channels, branch_channels, kernel_size=1, padding=0)
        self.proj4 = ConvBNReLU(in_channels, branch_channels, kernel_size=1, padding=0)
        self.proj2 = ConvBNReLU(in_channels, branch_channels, kernel_size=1, padding=0)
        self.proj1 = ConvBNReLU(in_channels, branch_channels, kernel_size=1, padding=0)
        self.fuse = ConvBNReLU(
            in_channels + 4 * branch_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (
            x,
            F.interpolate(self.proj8(self.pool8(x)), scale_factor=8.0, mode="nearest"),
            F.interpolate(self.proj4(self.pool4(x)), scale_factor=4.0, mode="nearest"),
            F.interpolate(self.proj2(self.pool2(x)), scale_factor=2.0, mode="nearest"),
            self.proj1(self.pool1(x)),
        )
        return self.fuse(torch.cat(branches, dim=1))


def _make_bottlenecks(
    in_channels: int,
    out_channels: int,
    blocks: int,
    expansion: int,
    stride: int,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        InvertedBottleneck(in_channels, out_channels, expansion, stride)
    ]
    layers.extend(
        InvertedBottleneck(out_channels, out_channels, expansion, 1)
        for _ in range(1, blocks)
    )
    return nn.Sequential(*layers)


def _scaled_channels(channels: int, width_mult: float) -> int:
    return max(8, int(round(channels * width_mult / 8.0)) * 8)


class GrayNavFastSCNN(nn.Module):
    """Fast-SCNN with a true grayscale input and raw 32x32 logits output."""

    def __init__(
        self,
        num_classes: int = NUM_SURFACE_CLASSES,
        in_channels: int = 1,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("GrayNav deployment contract requires exactly one input channel")
        if width_mult not in (1.0, 0.75):
            raise ValueError("supported deployment profiles are width_mult=1.0 or 0.75")
        c32, c48, c64, c96, c128 = (
            _scaled_channels(value, width_mult) for value in (32, 48, 64, 96, 128)
        )
        self.width_mult = width_mult
        self.learning_to_downsample = nn.Sequential(
            ConvBNReLU(in_channels, c32, kernel_size=3, stride=2, padding=1),
            SeparableConvBNReLU(c32, c48, stride=2),
            SeparableConvBNReLU(c48, c64, stride=2),
        )
        self.bottleneck1 = _make_bottlenecks(c64, c64, 3, expansion=6, stride=2)
        self.bottleneck2 = _make_bottlenecks(c64, c96, 3, expansion=6, stride=2)
        self.bottleneck3 = _make_bottlenecks(c96, c128, 3, expansion=6, stride=1)
        self.pyramid_pooling = FixedPyramidPooling(c128, c128)

        self.low_depthwise = ConvBNReLU(
            c128, c128, kernel_size=3, padding=1, groups=c128
        )
        self.low_pointwise = ConvBNReLU(c128, c128, kernel_size=1, padding=0, relu=False)
        self.high_pointwise = ConvBNReLU(c64, c128, kernel_size=1, padding=0, relu=False)
        self.classifier = nn.Sequential(
            SeparableConvBNReLU(c128, c128),
            SeparableConvBNReLU(c128, c128),
            nn.Conv2d(c128, num_classes, kernel_size=1, bias=True),
        )

    @property
    def first_conv(self) -> nn.Conv2d:
        conv = self.learning_to_downsample[0][0]
        if not isinstance(conv, nn.Conv2d):
            raise TypeError("unexpected first convolution layout")
        return conv

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        high = self.learning_to_downsample(images)  # 32x32
        low = self.bottleneck1(high)                # 16x16
        low = self.bottleneck2(low)                 # 8x8
        low = self.bottleneck3(low)
        low = self.pyramid_pooling(low)
        low = F.interpolate(low, scale_factor=4.0, mode="nearest")
        fused = F.relu(
            self.high_pointwise(high)
            + self.low_pointwise(self.low_depthwise(low)),
            inplace=False,
        )
        return self.classifier(fused)


@torch.no_grad()
def fold_rgb_first_conv_to_gray(
    model: GrayNavFastSCNN,
    rgb_weight: torch.Tensor,
    rgb_bias: torch.Tensor | None = None,
) -> None:
    """Initialize the true one-channel stem by summing RGB kernels."""

    target = model.first_conv
    if rgb_weight.ndim != 4 or rgb_weight.shape[1] != 3:
        raise ValueError(f"expected RGB OIHW weight, got {tuple(rgb_weight.shape)}")
    folded = rgb_weight.sum(dim=1, keepdim=True)
    if folded.shape != target.weight.shape:
        raise ValueError(
            f"folded shape {tuple(folded.shape)} does not match {tuple(target.weight.shape)}"
        )
    target.weight.copy_(folded.to(target.weight.device, target.weight.dtype))
    if target.bias is not None and rgb_bias is not None:
        target.bias.copy_(rgb_bias.to(target.bias.device, target.bias.dtype))
