#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MonoSimConfig:
    """Parameter ranges for online mono-sensor simulation."""

    wr_min: float = 0.20
    wr_max: float = 0.40
    wg_min: float = 0.45
    wg_max: float = 0.70
    wb_min: float = 0.05
    wb_max: float = 0.20
    scale_min: float = 0.60
    scale_max: float = 1.50
    bias_min: float = -0.08
    bias_max: float = 0.08
    gamma_min: float = 0.60
    gamma_max: float = 1.80
    contrast_min: float = 0.50
    contrast_max: float = 1.80
    gaussian_noise_max: float = 0.04
    shot_noise_max: float = 0.03
    p_blur: float = 0.25
    p_motion_blur: float = 0.15
    p_over: float = 0.10
    p_under: float = 0.10
    over_threshold_min: float = 0.85
    over_threshold_max: float = 0.95
    under_scale_min: float = 0.35
    under_scale_max: float = 0.70


def _uniform(batch: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty(batch, 1, 1, 1, device=device).uniform_(low, high)


def _apply_masked(x: torch.Tensor, prob: float, value: torch.Tensor) -> torch.Tensor:
    """Apply per-image augmented values with a Bernoulli mask."""
    if prob <= 0:
        return x
    mask = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) < prob).to(dtype=x.dtype)
    return x * (1.0 - mask) + value * mask


class MonoSim(nn.Module):
    """Generate single-channel grayscale images from RGB tensors during training.

    Input and output are float tensors in [0, 1]. In eval mode the module uses
    deterministic luma conversion so validation is repeatable.
    """

    def __init__(self, config: MonoSimConfig | None = None, strong: bool = False) -> None:
        super().__init__()
        self.config = config or MonoSimConfig()
        self.strong = bool(strong)
        motion_h = torch.ones(1, 1, 1, 5, dtype=torch.float32) / 5.0
        motion_v = torch.ones(1, 1, 5, 1, dtype=torch.float32) / 5.0
        self.register_buffer("motion_h", motion_h)
        self.register_buffer("motion_v", motion_v)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.clamp(0.0, 1.0)
        if not self.training:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
            return (rgb * weights).sum(dim=1, keepdim=True)

        cfg = self.config
        batch = rgb.shape[0]
        wr = _uniform(batch, cfg.wr_min, cfg.wr_max, rgb.device)
        wg = _uniform(batch, cfg.wg_min, cfg.wg_max, rgb.device)
        wb = _uniform(batch, cfg.wb_min, cfg.wb_max, rgb.device)
        norm = (wr + wg + wb).clamp_min(1e-6)
        weights = torch.cat([wr / norm, wg / norm, wb / norm], dim=1)
        gray = (rgb * weights).sum(dim=1, keepdim=True)

        gray = gray * _uniform(batch, cfg.scale_min, cfg.scale_max, rgb.device)
        gray = gray + _uniform(batch, cfg.bias_min, cfg.bias_max, rgb.device)
        gray = gray.clamp(0.0, 1.0)

        gamma = _uniform(batch, cfg.gamma_min, cfg.gamma_max, rgb.device)
        gray = torch.pow(gray.clamp_min(1e-6), gamma)

        mean = gray.mean(dim=(2, 3), keepdim=True)
        contrast = _uniform(batch, cfg.contrast_min, cfg.contrast_max, rgb.device)
        gray = (mean + contrast * (gray - mean)).clamp(0.0, 1.0)

        noise_scale = cfg.gaussian_noise_max * (1.5 if self.strong else 1.0)
        shot_scale = cfg.shot_noise_max * (1.5 if self.strong else 1.0)
        gray = gray + torch.randn_like(gray) * _uniform(batch, 0.0, noise_scale, rgb.device)
        gray = gray + torch.sqrt(gray.clamp_min(0.0)) * torch.randn_like(gray) * _uniform(batch, 0.0, shot_scale, rgb.device)
        gray = gray.clamp(0.0, 1.0)

        blurred = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        gray = _apply_masked(gray, cfg.p_blur * (1.3 if self.strong else 1.0), blurred)

        motion_kernel = self.motion_h if torch.rand((), device=rgb.device).item() < 0.5 else self.motion_v
        motion = F.conv2d(gray, motion_kernel, padding=(motion_kernel.shape[2] // 2, motion_kernel.shape[3] // 2))
        gray = _apply_masked(gray, cfg.p_motion_blur * (1.3 if self.strong else 1.0), motion)

        threshold = _uniform(batch, cfg.over_threshold_min, cfg.over_threshold_max, rgb.device)
        over = torch.where(gray > threshold, torch.ones_like(gray), gray)
        gray = _apply_masked(gray, cfg.p_over * (1.3 if self.strong else 1.0), over)

        under = gray * _uniform(batch, cfg.under_scale_min, cfg.under_scale_max, rgb.device)
        gray = _apply_masked(gray, cfg.p_under * (1.3 if self.strong else 1.0), under)
        return gray.clamp(0.0, 1.0)
