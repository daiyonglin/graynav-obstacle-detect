#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Iterable, List

import torch
from torch import nn
import torch.nn.functional as F


def flatten_float_tensors(value: Any) -> List[torch.Tensor]:
    """Collect floating tensors from nested model outputs."""
    tensors: List[torch.Tensor] = []
    if torch.is_tensor(value):
        if value.dtype.is_floating_point:
            tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_float_tensors(item))
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_float_tensors(item))
    return tensors


def matched_tensor_distill(student_out: Any, teacher_out: Any, loss_type: str = "smooth_l1") -> torch.Tensor:
    """Distill matching-shape tensors from two nested outputs."""
    student = flatten_float_tensors(student_out)
    teacher = flatten_float_tensors(teacher_out)
    pairs = [(s, t) for s, t in zip(student, teacher) if tuple(s.shape) == tuple(t.shape)]
    if not pairs:
        raise RuntimeError("no matching floating tensors found for head/output distillation")
    loss = torch.zeros((), device=pairs[0][0].device)
    for s, t in pairs:
        if loss_type == "mse":
            loss = loss + F.mse_loss(s.float(), t.detach().float())
        elif loss_type == "smooth_l1":
            loss = loss + F.smooth_l1_loss(s.float(), t.detach().float())
        else:
            raise ValueError(f"unknown loss_type: {loss_type}")
    return loss / len(pairs)


class FeatureHookSet:
    """Capture selected intermediate module outputs during YOLO forward passes."""

    def __init__(self, model: nn.Module, layer_indices: Iterable[int]) -> None:
        self.model = model
        self.layer_indices = [int(i) for i in layer_indices]
        self.features: List[torch.Tensor] = []
        self.handles: List[Any] = []
        layers = getattr(model, "model", None)
        if layers is None:
            raise ValueError("expected an Ultralytics model with a .model module list")
        for idx in self.layer_indices:
            if idx < 0 or idx >= len(layers):
                raise ValueError(f"feature layer index {idx} out of range 0..{len(layers) - 1}")
            self.handles.append(layers[idx].register_forward_hook(self._capture))

    def _capture(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        tensors = flatten_float_tensors(output)
        if tensors:
            self.features.append(tensors[0])

    def clear(self) -> None:
        self.features = []

    def snapshot(self, detach: bool) -> List[torch.Tensor]:
        if detach:
            return [f.detach() for f in self.features]
        return list(self.features)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def feature_distill_loss(student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]) -> torch.Tensor:
    """SmoothL1 distillation for same-shape feature-map pairs."""
    pairs = [(s, t) for s, t in zip(student_features, teacher_features) if tuple(s.shape) == tuple(t.shape)]
    if not pairs:
        raise RuntimeError("no matching feature tensors found for feature distillation")
    loss = torch.zeros((), device=pairs[0][0].device)
    for s, t in pairs:
        loss = loss + F.smooth_l1_loss(s.float(), t.detach().float())
    return loss / len(pairs)
