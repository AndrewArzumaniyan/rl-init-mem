from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from memristor_simulation.differentiable import DifferentiableMemristorPhysics

from .policy import LinearGaussianPolicy
from .rollout import run_programming_rollout
from .task_model import HardwareNoisyFeedForward


def accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def train_policy_step(
    *,
    model: HardwareNoisyFeedForward,
    policy: LinearGaussianPolicy,
    physics: DifferentiableMemristorPhysics,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    initial_conductance: Tensor,
    steps: int,
    noise_scale: float = 1.0,
    policy_noise_scale: float = 1.0,
    programmable_mask: Tensor | None = None,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    runtime_criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    optimizer.zero_grad()
    result = run_programming_rollout(
        model=model,
        policy=policy,
        physics=physics,
        inputs=inputs,
        targets=targets,
        initial_conductance=initial_conductance,
        steps=steps,
        criterion=runtime_criterion,
        noise_scale=noise_scale,
        policy_noise_scale=policy_noise_scale,
        programmable_mask=programmable_mask,
    )
    result.loss.backward()
    optimizer.step()

    if programmable_mask is None:
        action_values = torch.cat([action.detach().flatten() for action in result.actions])
    else:
        mask = programmable_mask.to(device=result.actions[0].device, dtype=torch.bool)
        action_values = torch.cat([action.detach()[mask].flatten() for action in result.actions])
    return {
        "loss": float(result.loss.detach().item()),
        "accuracy": accuracy_from_logits(result.logits.detach(), targets),
        "mean_action": float(action_values.mean().item()),
        "std_action": float(action_values.std(unbiased=False).item()),
        "mean_conductance": float(result.final_conductance.detach().mean().item()),
        "std_conductance": float(result.final_conductance.detach().std(unbiased=False).item()),
    }
