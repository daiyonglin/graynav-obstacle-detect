#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class GMFEMeta:
    """Deployment-relevant constants for deterministic GMFE preprocessing."""

    version: str = "gmfe_v2_cpu_preprocess"
    channel_order_rgb: tuple[str, str, str] = ("B_blur_luma", "S_sobel_energy", "T_local_variance")
    gaussian_kernel: int = 5
    variance_kernel: int = 7
    percentile: float = 99.5
    s_scale: float = 0.08
    t_scale: float = 0.05
    clip_min: float = 0.0
    clip_max: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_path(cls, path: Path) -> "GMFEMeta":
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def read_gray(path: Path) -> np.ndarray:
    """Read an image as uint8 grayscale."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"failed to read image: {path}")
    return gray


def gray_to_rgb_copy(gray: np.ndarray) -> np.ndarray:
    """Convert single-channel gray to RGB gray-copy image."""
    return np.stack([gray, gray, gray], axis=2)


def gmfe_float_channels(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute unnormalized GMFE channels from a uint8 gray image."""
    y = gray.astype(np.float32) / 255.0
    b = cv2.GaussianBlur(y, (5, 5), sigmaX=0.0, sigmaY=0.0, borderType=cv2.BORDER_REPLICATE)
    dx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REPLICATE)
    s = dx * dx + dy * dy
    mean = cv2.blur(y, (7, 7), borderType=cv2.BORDER_REPLICATE)
    mean2 = cv2.blur(y * y, (7, 7), borderType=cv2.BORDER_REPLICATE)
    t = np.maximum(mean2 - mean * mean, 0.0)
    return b, s, t


def estimate_gmfe_meta(gray_images: Iterable[Path], percentile: float = 99.5, limit: int = 512) -> GMFEMeta:
    """Estimate robust normalization constants from sampled training images."""
    s_values: list[float] = []
    t_values: list[float] = []
    for idx, path in enumerate(gray_images):
        if idx >= limit:
            break
        gray = read_gray(path)
        _, s, t = gmfe_float_channels(gray)
        s_values.append(float(np.percentile(s, percentile)))
        t_values.append(float(np.percentile(t, percentile)))
    if not s_values or not t_values:
        return GMFEMeta(percentile=percentile)
    s_scale = max(float(np.percentile(s_values, 90.0)), 1e-6)
    t_scale = max(float(np.percentile(t_values, 90.0)), 1e-6)
    return GMFEMeta(percentile=percentile, s_scale=s_scale, t_scale=t_scale)


def gmfe_rgb(gray: np.ndarray, meta: GMFEMeta) -> np.ndarray:
    """Encode gray as RGB [B, S, T] with fixed clipping and scaling."""
    b, s, t = gmfe_float_channels(gray)
    b_u8 = np.round(np.clip(b, meta.clip_min, meta.clip_max) * 255.0).astype(np.uint8)
    s_u8 = np.round(np.clip(s / max(meta.s_scale, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    t_u8 = np.round(np.clip(t / max(meta.t_scale, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([b_u8, s_u8, t_u8], axis=2)


def write_rgb_image(path: Path, rgb: np.ndarray) -> None:
    """Write an RGB image with OpenCV while preserving channel intent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def channel_correlation(rgb: np.ndarray) -> list[list[float]]:
    """Return a 3x3 Pearson correlation matrix for an RGB image."""
    flat = rgb.reshape(-1, 3).astype(np.float32)
    if flat.shape[0] < 2:
        return np.eye(3).tolist()
    corr = np.corrcoef(flat, rowvar=False)
    corr = np.nan_to_num(corr, nan=1.0, posinf=1.0, neginf=-1.0)
    return [[float(v) for v in row] for row in corr]


def save_gmfe_audit(path: Path, gray: np.ndarray, gmfe: np.ndarray) -> None:
    """Save a compact visual audit panel: gray, B, S, T, and GMFE preview."""
    g3 = gray_to_rgb_copy(gray)
    panels = [g3, gray_to_rgb_copy(gmfe[:, :, 0]), gray_to_rgb_copy(gmfe[:, :, 1]), gray_to_rgb_copy(gmfe[:, :, 2]), gmfe]
    h = min(p.shape[0] for p in panels)
    resized = []
    for p in panels:
        scale = h / float(p.shape[0])
        w = int(round(p.shape[1] * scale))
        resized.append(cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA))
    merged = np.concatenate(resized, axis=1)
    write_rgb_image(path, merged)
