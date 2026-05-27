from __future__ import annotations

import torch
from torch import Tensor

from .task_model import TaskModelCache


def compute_layer_ce_gradient(
    *,
    logits: Tensor,
    targets: Tensor,
    post_activation: Tensor,
    criterion,
) -> Tensor:
    loss = criterion(logits, targets)
    gradient = torch.autograd.grad(
        loss,
        post_activation,
        retain_graph=True,
        create_graph=False,
    )[0]
    return gradient.mean(dim=0).detach()


def build_layer_state(
    *,
    conductance: Tensor,
    previous_action: Tensor,
    layer_input: Tensor,
    layer_weight: Tensor,
    layer_pre_activation: Tensor,
    layer_post_activation: Tensor,
    post_activation_ce_gradient: Tensor,
    step_index: int,
    total_steps: int,
) -> Tensor:
    mean_hin = layer_input.mean(dim=0)
    mean_pre_activation = layer_pre_activation.mean(dim=0)
    mean_post_activation = layer_post_activation.mean(dim=0)
    local_contribution = layer_weight * mean_hin.unsqueeze(0)
    t_norm = 0.0 if total_steps <= 1 else step_index / (total_steps - 1)
    t_feature = torch.full_like(conductance, float(t_norm))
    return torch.stack(
        (
            conductance,
            mean_hin.unsqueeze(0).expand_as(conductance),
            mean_pre_activation.unsqueeze(1).expand_as(conductance),
            mean_post_activation.unsqueeze(1).expand_as(conductance),
            local_contribution,
            post_activation_ce_gradient.unsqueeze(1).expand_as(conductance),
            previous_action,
            t_feature,
        ),
        dim=-1,
    ).view(-1, 8)


# Backward-compatible wrappers used by existing code and tests

def compute_post_activation_ce_gradient(
    *,
    logits: Tensor,
    targets: Tensor,
    cache: TaskModelCache,
    criterion,
) -> Tensor:
    return compute_layer_ce_gradient(
        logits=logits,
        targets=targets,
        post_activation=cache.fc1_post_activation,
        criterion=criterion,
    )


def build_fc1_state(
    *,
    conductance: Tensor,
    previous_conductance: Tensor,
    previous_action: Tensor,
    cache: TaskModelCache,
    post_activation_ce_gradient: Tensor,
    step_index: int,
    total_steps: int,
) -> Tensor:
    return build_layer_state(
        conductance=conductance,
        previous_action=previous_action,
        layer_input=cache.fc1_input,
        layer_weight=cache.fc1_weight,
        layer_pre_activation=cache.fc1_pre_activation,
        layer_post_activation=cache.fc1_post_activation,
        post_activation_ce_gradient=post_activation_ce_gradient,
        step_index=step_index,
        total_steps=total_steps,
    )
