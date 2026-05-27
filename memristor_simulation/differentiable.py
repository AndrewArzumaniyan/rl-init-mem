from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class MemristorPhysicsConfig:
    g_min: float = 0.0
    g_max: float = 1.0
    alpha: float = 4.0
    beta: float = 0.02
    tau: float = 1.0
    action_limit: float = 0.2
    stable_zero_action: bool = True


class DifferentiableMemristorPhysics(nn.Module):
    """Torch implementation of the differentiable memristor transition."""

    def __init__(self, config: MemristorPhysicsConfig | None = None) -> None:
        super().__init__()
        self.config = config or MemristorPhysicsConfig()

    def forward(
        self,
        conductance: Tensor,
        pulse_time: Tensor,
        noise: Tensor | None = None,
    ) -> Tensor:
        config = self.config
        if noise is None:
            noise = torch.randn_like(conductance)

        bounded_pulse = pulse_time.clamp(-config.action_limit, config.action_limit)
        if config.stable_zero_action:
            return self._stable_transition(conductance, bounded_pulse, noise)

        return self._raw_pdf_transition(conductance, bounded_pulse, noise)

    def _raw_pdf_transition(
        self,
        conductance: Tensor,
        bounded_pulse: Tensor,
        noise: Tensor,
    ) -> Tensor:
        config = self.config
        saturation = (config.g_max - conductance) * (conductance - config.g_min)
        driven = (
            conductance
            + config.alpha * bounded_pulse * saturation
            + config.beta * bounded_pulse.abs() * noise
        )
        normalized = (driven - config.g_min) / config.tau
        updated = config.g_min + (config.g_max - config.g_min) * torch.sigmoid(normalized)
        return updated.clamp(config.g_min, config.g_max)

    def _stable_transition(
        self,
        conductance: Tensor,
        bounded_pulse: Tensor,
        noise: Tensor,
    ) -> Tensor:
        config = self.config
        span = config.g_max - config.g_min
        normalized_g = ((conductance - config.g_min) / span).clamp(1e-6, 1.0 - 1e-6)
        logit_g = torch.logit(normalized_g)
        saturation = normalized_g * (1.0 - normalized_g)
        delta = (
            config.alpha * bounded_pulse * saturation
            + config.beta * bounded_pulse.abs() * noise
        ) / config.tau
        updated = config.g_min + span * torch.sigmoid(logit_g + delta)
        return updated.clamp(config.g_min, config.g_max)
