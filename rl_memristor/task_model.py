from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .hardware import conductance_to_weight


@dataclass(slots=True)
class TaskModelCache:
    fc1_input: Tensor
    fc1_weight: Tensor
    fc1_pre_activation: Tensor
    fc1_post_activation: Tensor
    # FC2 and FC3 fields — populated when hardware-controlled via conductance
    fc2_input: Tensor | None = None
    fc2_weight: Tensor | None = None
    fc2_pre_activation: Tensor | None = None
    fc2_post_activation: Tensor | None = None
    fc3_input: Tensor | None = None
    fc3_weight: Tensor | None = None
    fc3_pre_activation: Tensor | None = None
    fc3_post_activation: Tensor | None = None


class HardwareNoisyFeedForward(nn.Module):
    """Small NoisyFeedForward-style network with hardware-controlled fc1."""

    def __init__(
        self,
        input_shape: tuple[int, ...] = (1, 28, 28),
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 10,
        w_max: float = 1.0,
        g_min: float = 0.0,
        g_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.input_features = int(reduce(mul, input_shape, 1))
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        self.w_max = w_max
        self.g_min = g_min
        self.g_max = g_max

        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1 * 2, hidden2, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2 * 2, num_classes, bias=False)
        self.ln = nn.LayerNorm(num_classes)
        self.apply(self._init_positive_weights)

    @property
    def fc1_conductance_shape(self) -> tuple[int, int]:
        return (self.hidden1, self.input_features * 2)

    @property
    def fc2_conductance_shape(self) -> tuple[int, int]:
        return (self.hidden2, self.hidden1 * 2)

    @property
    def fc3_conductance_shape(self) -> tuple[int, int]:
        return (self.num_classes, self.hidden2 * 2)

    def _init_positive_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.abs_()
                module.weight.clamp_(0.0, self.w_max)

    @staticmethod
    def negative_concat(values: Tensor, dim: int) -> Tensor:
        return torch.cat((values, -values), dim=dim)

    def _conductance_to_w(self, conductance: Tensor) -> Tensor:
        return conductance_to_weight(conductance, g_min=self.g_min, g_max=self.g_max, w_max=self.w_max)

    def fc1_weight_from_conductance(self, conductance: Tensor) -> Tensor:
        return self._conductance_to_w(conductance)

    def fc2_weight_from_conductance(self, conductance: Tensor) -> Tensor:
        return self._conductance_to_w(conductance)

    def fc3_weight_from_conductance(self, conductance: Tensor) -> Tensor:
        return self._conductance_to_w(conductance)

    def forward_with_conductance(
        self,
        inputs: Tensor,
        fc1_conductance: Tensor,
        fc2_conductance: Tensor | None = None,
        fc3_conductance: Tensor | None = None,
    ) -> tuple[Tensor, TaskModelCache]:
        flattened = inputs.view(inputs.shape[0], self.input_features)
        fc1_input = self.negative_concat(flattened, dim=-1)
        fc1_weight = self._conductance_to_w(fc1_conductance)

        fc1_pre_activation = F.linear(fc1_input, fc1_weight, bias=None)
        fc1_post_activation = torch.tanh(self.bn1(fc1_pre_activation))
        if torch.is_grad_enabled() and not fc1_post_activation.requires_grad:
            fc1_post_activation = fc1_post_activation.detach().requires_grad_(True)

        fc2_input = self.negative_concat(fc1_post_activation, dim=-1)
        if fc2_conductance is not None:
            fc2_weight = self._conductance_to_w(fc2_conductance)
            fc2_pre_activation = F.linear(fc2_input, fc2_weight, bias=None)
        else:
            fc2_weight = self.fc2.weight
            fc2_pre_activation = self.fc2(fc2_input)
        fc2_post_activation = torch.tanh(self.bn2(fc2_pre_activation))
        if torch.is_grad_enabled() and (fc2_conductance is not None) and not fc2_post_activation.requires_grad:
            fc2_post_activation = fc2_post_activation.detach().requires_grad_(True)

        fc3_input = self.negative_concat(fc2_post_activation, dim=-1)
        if fc3_conductance is not None:
            fc3_weight = self._conductance_to_w(fc3_conductance)
            fc3_pre_activation = F.linear(fc3_input, fc3_weight, bias=None)
        else:
            fc3_weight = self.fc3.weight
            fc3_pre_activation = self.fc3(fc3_input)
        fc3_post_activation = self.ln(fc3_pre_activation)
        if torch.is_grad_enabled() and (fc3_conductance is not None) and not fc3_post_activation.requires_grad:
            fc3_post_activation = fc3_post_activation.detach().requires_grad_(True)

        logits = fc3_post_activation
        return logits, TaskModelCache(
            fc1_input=fc1_input,
            fc1_weight=fc1_weight,
            fc1_pre_activation=fc1_pre_activation,
            fc1_post_activation=fc1_post_activation,
            fc2_input=fc2_input,
            fc2_weight=fc2_weight,
            fc2_pre_activation=fc2_pre_activation,
            fc2_post_activation=fc2_post_activation,
            fc3_input=fc3_input,
            fc3_weight=fc3_weight,
            fc3_pre_activation=fc3_pre_activation,
            fc3_post_activation=fc3_post_activation,
        )
