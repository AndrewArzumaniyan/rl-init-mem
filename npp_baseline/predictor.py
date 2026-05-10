from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class NormalizationStats:
    g_offset: float
    g_scale: float
    delta_g_scale: float
    t_offset: float
    t_scale: float

    def normalize_inputs(self, g_start: Tensor, delta_g: Tensor) -> Tensor:
        g_start_norm = (g_start - self.g_offset) / self.g_scale
        delta_g_norm = delta_g / self.delta_g_scale
        return torch.stack((g_start_norm, delta_g_norm), dim=-1)

    def normalize_target(self, t_pulse: Tensor) -> Tensor:
        return (t_pulse - self.t_offset) / self.t_scale

    def denormalize_output(self, t_pulse_norm: Tensor) -> Tensor:
        return t_pulse_norm * self.t_scale + self.t_offset


class NeuralPulsePredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
