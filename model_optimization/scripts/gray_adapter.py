#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F


class FixedReplicateAdapter(nn.Module):
    """Baseline adapter: gray -> [gray, gray, gray]."""

    adapter_type = "ggg"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1)

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters(), torch.zeros(1)).device)

    def metadata(self) -> Dict[str, Any]:
        return {"adapter_type": self.adapter_type, "deploy_mode": "replicate"}


class LUTGrayChannelAdapter(nn.Module):
    """Learnable 1->3 gray adapter that can be exported as 3x256 LUT tables."""

    adapter_type = "lut"

    def __init__(self) -> None:
        super().__init__()
        base = torch.linspace(0.0, 1.0, 256, dtype=torch.float32)
        self.lut = nn.Parameter(torch.stack([base, base, base], dim=0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        scaled = x * 255.0
        lo = torch.floor(scaled).long().clamp(0, 255)
        hi = (lo + 1).clamp(0, 255)
        alpha = scaled - lo.float()

        table = self.lut.clamp(0.0, 1.0)
        outs = []
        for c in range(3):
            t = table[c].view(1, 256, 1, 1).expand(x.shape[0], -1, x.shape[2], x.shape[3])
            lo_v = torch.gather(t, 1, lo)
            hi_v = torch.gather(t, 1, hi)
            outs.append(lo_v * (1.0 - alpha) + hi_v * alpha)
        return torch.cat(outs, dim=1).clamp(0.0, 1.0)

    def regularization_loss(self) -> torch.Tensor:
        table = self.lut.clamp(0.0, 1.0)
        smooth = (table[:, 2:] - 2.0 * table[:, 1:-1] + table[:, :-2]).abs().mean()
        monotonic = F.relu(table[:, :-1] - table[:, 1:]).mean()
        base = torch.linspace(0.0, 1.0, 256, dtype=table.dtype, device=table.device)
        identity = (table - base.view(1, 256)).abs().mean()
        return smooth * 0.50 + monotonic * 0.25 + identity * 0.05

    def metadata(self) -> Dict[str, Any]:
        table = self.lut.detach().clamp(0.0, 1.0).cpu()
        return {
            "adapter_type": self.adapter_type,
            "deploy_mode": "lut_3x256_uint8",
            "lut": [[int(round(float(v) * 255.0)) for v in row] for row in table],
        }


class ConvGrayChannelAdapter(nn.Module):
    """Small conv adapter for experiments when the conversion toolchain accepts it."""

    adapter_type = "conv"

    def __init__(self, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            first = self.net[0]
            second = self.net[2]
            nn.init.zeros_(first.weight)
            nn.init.zeros_(first.bias)
            center = first.weight.shape[-1] // 2
            for i in range(min(hidden, 3)):
                first.weight[i, 0, center, center] = 1.0
            nn.init.zeros_(second.weight)
            nn.init.zeros_(second.bias)
            for c in range(3):
                second.weight[c, c % hidden, 0, 0] = 4.0
                second.bias[c] = -2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.clamp(0.0, 1.0))

    def regularization_loss(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def metadata(self) -> Dict[str, Any]:
        return {"adapter_type": self.adapter_type, "deploy_mode": "onnx_conv_experimental"}


class SpatialGrayAdapter(nn.Module):
    """Parametric gray adapter with local contrast and edge channels.

    This is stronger than a LUT because the output depends on local image
    structure, not only on the current pixel value. It is intended as a CPU
    preprocessing candidate first; deployment should use the exported scalar
    parameters and simple blur/Sobel kernels rather than a new NPU graph.
    """

    adapter_type = "spatial"

    def __init__(self) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(3))
        self.contrast = nn.Parameter(torch.ones(3))
        self.bias = nn.Parameter(torch.zeros(3))
        self.local_gain = nn.Parameter(torch.tensor([0.00, 0.22, -0.18], dtype=torch.float32))
        self.edge_gain = nn.Parameter(torch.tensor([0.00, 0.10, 0.18], dtype=torch.float32))
        self.strength = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        blur = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        local = (x - blur).clamp(-0.5, 0.5)
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6).clamp(0.0, 0.5)
        base = torch.pow(x + 1e-6, self.gamma.clamp(0.45, 1.85).view(1, 3, 1, 1))
        base = (base - 0.5) * self.contrast.clamp(0.55, 1.65).view(1, 3, 1, 1) + 0.5
        base = base + self.bias.clamp(-0.18, 0.18).view(1, 3, 1, 1)
        adapted = base
        adapted = adapted + self.local_gain.clamp(-0.65, 0.65).view(1, 3, 1, 1) * local
        adapted = adapted + self.edge_gain.clamp(-0.35, 0.60).view(1, 3, 1, 1) * edge
        strength = self.strength.clamp(0.0, 1.0)
        replicated = x.repeat(1, 3, 1, 1)
        return ((1.0 - strength) * replicated + strength * adapted).clamp(0.0, 1.0)

    def regularization_loss(self) -> torch.Tensor:
        identity = (self.gamma - 1.0).abs().mean() + (self.contrast - 1.0).abs().mean() + self.bias.abs().mean()
        spatial = self.local_gain.abs().mean() * 0.25 + self.edge_gain.abs().mean() * 0.20
        return identity * 0.05 + spatial

    def metadata(self) -> Dict[str, Any]:
        def vals(t: torch.Tensor) -> list[float]:
            return [round(float(v), 6) for v in t.detach().cpu().flatten()]

        return {
            "adapter_type": self.adapter_type,
            "deploy_mode": "cpu_spatial_parametric_gray_to_3ch",
            "params": {
                "gamma": vals(self.gamma.clamp(0.45, 1.85)),
                "contrast": vals(self.contrast.clamp(0.55, 1.65)),
                "bias": vals(self.bias.clamp(-0.18, 0.18)),
                "local_gain": vals(self.local_gain.clamp(-0.65, 0.65)),
                "edge_gain": vals(self.edge_gain.clamp(-0.35, 0.60)),
                "strength": round(float(self.strength.detach().cpu().clamp(0.0, 1.0)), 6),
                "blur_kernel": 7,
                "edge_kernel": "sobel_3x3",
            },
        }


class ResidualBlock(nn.Module):
    """Small Conv-BN-ReLU residual block used by the G2RGB adapter."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class G2RGBResidualAdapter(nn.Module):
    """Residual gray-to-pseudo-RGB adapter for frozen RGB-pretrained YOLOv8n.

    The module starts as the current GGG baseline because the final projection
    is zero-initialized. Training only learns a bounded residual correction:
    output = repeat(gray) + alpha * residual(gray).
    """

    adapter_type = "g2rgb_residual"

    def __init__(
        self,
        hidden: int = 16,
        blocks: int = 2,
        alpha: float = 0.1,
        use_5x5_branch: bool = False,
    ) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if blocks < 0:
            raise ValueError("blocks must be non-negative")
        self.hidden = int(hidden)
        self.blocks = int(blocks)
        self.alpha = float(alpha)
        self.use_5x5_branch = bool(use_5x5_branch)

        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks_net = nn.Sequential(*(ResidualBlock(hidden) for _ in range(blocks)))

        branch_channels = max(4, hidden // 2)
        self.branch_1x1 = nn.Conv2d(hidden, branch_channels, kernel_size=1, bias=False)
        self.branch_3x3 = nn.Conv2d(hidden, branch_channels, kernel_size=3, padding=1, bias=False)
        if self.use_5x5_branch:
            self.branch_5x5 = nn.Conv2d(hidden, branch_channels, kernel_size=5, padding=2, bias=False)
            fused_channels = branch_channels * 3
        else:
            self.branch_5x5 = None
            fused_channels = branch_channels * 2

        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(branch_channels, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.clamp(0.0, 1.0)
        base = gray.repeat(1, 3, 1, 1)
        return (base + self.alpha * self.residual(gray)).clamp(0.0, 1.0)

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unscaled 3-channel residual for regularization."""
        gray = x.clamp(0.0, 1.0)
        feat = self.blocks_net(self.stem(gray))
        branches = [self.branch_1x1(feat), self.branch_3x3(feat)]
        if self.branch_5x5 is not None:
            branches.append(self.branch_5x5(feat))
        return self.out_conv(self.fuse(torch.cat(branches, dim=1)))

    def luminance_loss(self, gray: torch.Tensor) -> torch.Tensor:
        """Constrain adapted pseudo-RGB luminance to preserve input geometry."""
        adapted = self.forward(gray)
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=adapted.dtype, device=adapted.device).view(1, 3, 1, 1)
        luminance = (adapted * weights).sum(dim=1, keepdim=True)
        return F.l1_loss(luminance, gray.clamp(0.0, 1.0))

    def regularization_loss(self, gray: torch.Tensor | None = None) -> torch.Tensor:
        if gray is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.residual(gray).abs().mean()

    def metadata(self) -> Dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "deploy_mode": "onnx_conv_residual",
            "hidden": self.hidden,
            "blocks": self.blocks,
            "alpha": self.alpha,
            "use_5x5_branch": self.use_5x5_branch,
            "initial_state": "equivalent_to_gray_replicate",
            "operators": ["Conv", "BatchNormalization", "Relu", "Add", "Mul", "Concat"],
        }

    def init_config(self) -> Dict[str, Any]:
        return {
            "hidden": self.hidden,
            "blocks": self.blocks,
            "alpha": self.alpha,
            "use_5x5_branch": self.use_5x5_branch,
        }


class FixedGrayProjection(nn.Module):
    """Board-compatible fixed Conv1x1 projection from 3-channel input to gray."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight.fill_(1.0 / 3.0)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x
        if x.shape[1] != 3:
            raise ValueError(f"expected 1 or 3 input channels, got {x.shape[1]}")
        return self.proj(x)


class GMFEFixedEncoder(nn.Module):
    """Fixed gray multi-domain encoder: G, blurred B, edge energy S, texture T."""

    def __init__(self) -> None:
        super().__init__()
        self.blur = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        kernel = torch.tensor(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=torch.float32,
        ) / 256.0
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32) / 8.0
        sobel_y = sobel_x.t()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.blur.weight.copy_(kernel.view(1, 1, 5, 5))
            self.sobel_x.weight.copy_(sobel_x.view(1, 1, 3, 3))
            self.sobel_y.weight.copy_(sobel_y.view(1, 1, 3, 3))
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        b = self.blur(gray)
        dx = self.sobel_x(b)
        dy = self.sobel_y(b)
        s = dx * dx + dy * dy
        gray2 = gray * gray
        m = F.avg_pool2d(gray, kernel_size=7, stride=1, padding=3)
        q = F.avg_pool2d(gray2, kernel_size=7, stride=1, padding=3)
        m2 = m * m
        t = q + (-1.0 * m2)
        return torch.cat([gray, b, s, t], dim=1)


class BCGMFEDCAAdapter(nn.Module):
    """Board-compatible GMFE-DCA adapter with 3-channel input and pseudo-RGB output.

    The deployment graph accepts the current board-compatible 3-channel input,
    projects it back to gray internally, extracts fixed multi-domain GMFE
    features, then trains a tiny DCA head to produce a residual pseudo-RGB
    correction for frozen YOLOv8n.
    """

    adapter_type = "bc_gmfe_dca"

    def __init__(self, hidden: int = 16, alpha: float = 0.1) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = int(hidden)
        self.alpha = float(alpha)
        self.gray_projection = FixedGrayProjection()
        self.gmfe = GMFEFixedEncoder()
        self.dca = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )
        final = self.dca[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def gray(self, x: torch.Tensor) -> torch.Tensor:
        return self.gray_projection(x)

    def gmfe_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.gmfe(self.gray(x))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        return self.dca(self.gmfe_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = self.gray(x)
        base = gray.repeat(1, 3, 1, 1)
        return base + self.alpha * self.residual(x)

    def luminance_loss(self, x: torch.Tensor, gray_target: torch.Tensor | None = None) -> torch.Tensor:
        """Constrain pseudo-RGB luminance to preserve the recovered gray image."""
        gray = self.gray(x) if gray_target is None else gray_target
        adapted = self.forward(x)
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=adapted.dtype, device=adapted.device).view(1, 3, 1, 1)
        luminance = (adapted * weights).sum(dim=1, keepdim=True)
        return F.l1_loss(luminance, gray)

    def regularization_loss(self, x: torch.Tensor | None = None) -> torch.Tensor:
        if x is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return self.residual(x).abs().mean()

    def init_config(self) -> Dict[str, Any]:
        return {"hidden": self.hidden, "alpha": self.alpha}

    def metadata(self) -> Dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "deploy_mode": "board_compatible_3ch_onnx",
            "input_channels": 3,
            "output_channels": 3,
            "hidden": self.hidden,
            "alpha": self.alpha,
            "gray_projection": "fixed_conv1x1_3_to_1_mean",
            "gmfe_domains": ["G", "B_gaussian5x5", "S_sobel_energy", "T_avgpool7_variance_energy"],
            "operators": ["Conv", "AveragePool", "BatchNormalization", "Relu", "Mul", "Add", "Concat"],
            "forbidden_ops": ["Sqrt", "Sub", "Div", "Softmax", "Clip", "NMS"],
            "initial_state": "equivalent_to_gray_replicate",
        }


class AdaptedYolo(nn.Module):
    def __init__(self, adapter: nn.Module, yolo_model: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.yolo_model = yolo_model

    def forward(self, gray: torch.Tensor):
        return self.yolo_model(self.adapter(gray))


def build_adapter(adapter_type: str, **kwargs: Any) -> nn.Module:
    if adapter_type == "ggg":
        return FixedReplicateAdapter()
    if adapter_type == "lut":
        return LUTGrayChannelAdapter()
    if adapter_type == "conv":
        return ConvGrayChannelAdapter()
    if adapter_type == "spatial":
        return SpatialGrayAdapter()
    if adapter_type == "g2rgb_residual":
        return G2RGBResidualAdapter(**kwargs)
    if adapter_type == "bc_gmfe_dca":
        return BCGMFEDCAAdapter(**kwargs)
    raise ValueError(f"unknown adapter type: {adapter_type}")


def lut_uint8_to_adapter(lut: list[list[int]] | torch.Tensor) -> LUTGrayChannelAdapter:
    """Build a deployable LUT adapter from 3x256 uint8 tables."""
    if torch.is_tensor(lut):
        table = lut.detach().float()
    else:
        table = torch.tensor(lut, dtype=torch.float32)
    if tuple(table.shape) != (3, 256):
        raise ValueError(f"expected LUT shape 3x256, got {tuple(table.shape)}")
    adapter = LUTGrayChannelAdapter()
    with torch.no_grad():
        adapter.lut.copy_(table.clamp(0, 255) / 255.0)
    return adapter


def spatial_params_to_adapter(params: Dict[str, Any]) -> SpatialGrayAdapter:
    adapter = SpatialGrayAdapter()

    def copy_param(name: str) -> None:
        if name in params:
            value = torch.tensor(params[name], dtype=torch.float32)
            target = getattr(adapter, name)
            if value.numel() != target.numel():
                raise ValueError(f"parameter {name} expected {target.numel()} values, got {value.numel()}")
            with torch.no_grad():
                target.copy_(value.view_as(target))

    for key in ["gamma", "contrast", "bias", "local_gain", "edge_gain"]:
        copy_param(key)
    if "strength" in params:
        with torch.no_grad():
            adapter.strength.copy_(torch.tensor(float(params["strength"]), dtype=torch.float32))
    return adapter


def save_adapter_bundle(adapter: nn.Module, path: Path, extra_meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_path = path.with_suffix(".pt")
    meta_path = path.with_suffix(".json")
    ckpt: Dict[str, Any] = {"adapter_type": getattr(adapter, "adapter_type", "unknown"), "state_dict": adapter.state_dict()}
    if hasattr(adapter, "init_config"):
        ckpt["init_config"] = adapter.init_config()
    torch.save(ckpt, state_path)
    meta = dict(extra_meta)
    if hasattr(adapter, "metadata"):
        meta.update(adapter.metadata())
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_adapter_bundle(path: Path, map_location: str | torch.device = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=map_location)
    adapter = build_adapter(str(ckpt["adapter_type"]), **dict(ckpt.get("init_config", {})))
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter
