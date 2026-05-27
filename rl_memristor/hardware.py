from __future__ import annotations

import torch
from torch import Tensor


def conductance_to_weight(
    conductance: Tensor,
    *,
    g_min: float = 0.0,
    g_max: float = 1.0,
    w_max: float = 1.0,
) -> Tensor:
    normalized = (conductance - g_min) / (g_max - g_min)
    return w_max * normalized.clamp(0.0, 1.0)


def weight_to_conductance(
    weight: Tensor,
    *,
    g_min: float = 0.0,
    g_max: float = 1.0,
    w_max: float = 1.0,
) -> Tensor:
    normalized = torch.clamp(weight / w_max, 0.0, 1.0)
    return g_min + (g_max - g_min) * normalized
