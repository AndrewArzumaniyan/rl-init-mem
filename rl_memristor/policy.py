from __future__ import annotations

import torch
from torch import Tensor, nn


class LinearGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_limit: float = 1.0,
        initial_log_sigma: float = -2.0,
    ) -> None:
        super().__init__()
        self.action_limit = action_limit
        self.mean = nn.Linear(state_dim, 1)
        self.log_sigma = nn.Parameter(torch.tensor(float(initial_log_sigma)))

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    def forward(self, state: Tensor) -> Tensor:
        return self.mean(state)

    def sample(
        self,
        state: Tensor,
        *,
        output_shape: tuple[int, ...],
        eps: Tensor | None = None,
    ) -> Tensor:
        mean = self.forward(state)
        if eps is None:
            eps = torch.randn_like(mean)
        action = mean + self.sigma * eps
        return action.clamp(-self.action_limit, self.action_limit).view(output_shape)


class MLPGaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        action_limit: float = 1.0,
        initial_log_sigma: float = -2.0,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")
        self.action_limit = action_limit
        layers: list[nn.Module] = []
        input_dim = state_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mean = nn.Sequential(*layers)
        self.log_sigma = nn.Parameter(torch.tensor(float(initial_log_sigma)))

    @property
    def sigma(self) -> Tensor:
        return self.log_sigma.exp()

    def forward(self, state: Tensor) -> Tensor:
        return self.mean(state)

    def sample(
        self,
        state: Tensor,
        *,
        output_shape: tuple[int, ...],
        eps: Tensor | None = None,
    ) -> Tensor:
        mean = self.forward(state)
        if eps is None:
            eps = torch.randn_like(mean)
        action = mean + self.sigma * eps
        return action.clamp(-self.action_limit, self.action_limit).view(output_shape)
