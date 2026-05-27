from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .task_model import HardwareNoisyFeedForward
from .train import accuracy_from_logits


def train_task_model_step(
    *,
    model: HardwareNoisyFeedForward,
    fc1_conductance: Tensor,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    runtime_criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    logits, _ = model.forward_with_conductance(inputs, fc1_conductance)
    loss = runtime_criterion(logits, targets)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        fc1_conductance.clamp_(model.g_min, model.g_max)
    return {
        "loss": float(loss.detach().item()),
        "accuracy": accuracy_from_logits(logits.detach(), targets),
    }


def train_full_hardware_task_model_step(
    *,
    model: HardwareNoisyFeedForward,
    fc1_conductance: Tensor,
    fc2_conductance: Tensor,
    fc3_conductance: Tensor,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    runtime_criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    logits, _ = model.forward_with_conductance(
        inputs,
        fc1_conductance,
        fc2_conductance,
        fc3_conductance,
    )
    loss = runtime_criterion(logits, targets)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        fc1_conductance.clamp_(model.g_min, model.g_max)
        fc2_conductance.clamp_(model.g_min, model.g_max)
        fc3_conductance.clamp_(model.g_min, model.g_max)
    return {
        "loss": float(loss.detach().item()),
        "accuracy": accuracy_from_logits(logits.detach(), targets),
    }


@torch.no_grad()
def evaluate_task_model(
    *,
    model: HardwareNoisyFeedForward,
    fc1_conductance: Tensor,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    runtime_criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    logits, _ = model.forward_with_conductance(inputs, fc1_conductance)
    loss = runtime_criterion(logits, targets)
    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_from_logits(logits, targets),
    }


@torch.no_grad()
def evaluate_full_hardware_task_model(
    *,
    model: HardwareNoisyFeedForward,
    fc1_conductance: Tensor,
    fc2_conductance: Tensor,
    fc3_conductance: Tensor,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    runtime_criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    logits, _ = model.forward_with_conductance(
        inputs,
        fc1_conductance,
        fc2_conductance,
        fc3_conductance,
    )
    loss = runtime_criterion(logits, targets)
    return {
        "loss": float(loss.item()),
        "accuracy": accuracy_from_logits(logits, targets),
    }
