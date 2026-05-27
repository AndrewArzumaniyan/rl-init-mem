from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from memristor_simulation.differentiable import DifferentiableMemristorPhysics

from .policy import LinearGaussianPolicy
from .state import build_layer_state, build_fc1_state, compute_layer_ce_gradient, compute_post_activation_ce_gradient
from .task_model import HardwareNoisyFeedForward, TaskModelCache


@dataclass(slots=True)
class RolloutResult:
    logits: Tensor
    loss: Tensor
    final_conductance: Tensor
    actions: list[Tensor]


def _assemble_conductances(
    conductance: Tensor,
    active_layer: int,
    fc1_context: Tensor | None,
    fc2_context: Tensor | None,
    fc3_context: Tensor | None,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """Returns (fc1_cond, fc2_cond, fc3_cond) for the model forward call."""
    if active_layer == 1:
        return conductance, fc2_context, fc3_context
    elif active_layer == 2:
        if fc1_context is None:
            raise ValueError("fc1_context is required when active_layer=2.")
        return fc1_context, conductance, fc3_context
    elif active_layer == 3:
        if fc1_context is None or fc2_context is None:
            raise ValueError("fc1_context and fc2_context are required when active_layer=3.")
        return fc1_context, fc2_context, conductance
    raise ValueError(f"active_layer must be 1, 2, or 3, got {active_layer}.")


def _get_layer_cache(cache: TaskModelCache, active_layer: int):
    """Returns (layer_input, layer_weight, layer_pre_activation, layer_post_activation)."""
    if active_layer == 1:
        return cache.fc1_input, cache.fc1_weight, cache.fc1_pre_activation, cache.fc1_post_activation
    elif active_layer == 2:
        return cache.fc2_input, cache.fc2_weight, cache.fc2_pre_activation, cache.fc2_post_activation
    elif active_layer == 3:
        return cache.fc3_input, cache.fc3_weight, cache.fc3_pre_activation, cache.fc3_post_activation
    raise ValueError(f"active_layer must be 1, 2, or 3, got {active_layer}.")


def _layer_to_index(layer: str | int) -> int:
    if isinstance(layer, int):
        return layer
    normalized = layer.lower()
    if normalized == "fc1":
        return 1
    if normalized == "fc2":
        return 2
    if normalized == "fc3":
        return 3
    raise ValueError(f"layer must be 'fc1', 'fc2', or 'fc3', got {layer!r}.")


def run_programming_rollout(
    *,
    model: HardwareNoisyFeedForward,
    policy: LinearGaussianPolicy,
    physics: DifferentiableMemristorPhysics,
    inputs: Tensor,
    targets: Tensor,
    initial_conductance: Tensor,
    steps: int,
    criterion: Callable[[Tensor, Tensor], Tensor] | nn.Module,
    noise_scale: float = 1.0,
    policy_noise_scale: float = 1.0,
    programmable_mask: Tensor | None = None,
    # Multi-layer programming params (active_layer=1 gives original FC1-only behavior)
    active_layer: int = 1,
    fc1_context: Tensor | None = None,
    fc2_context: Tensor | None = None,
    fc3_context: Tensor | None = None,
) -> RolloutResult:
    with torch.enable_grad():
        conductance = initial_conductance
        previous_conductance = initial_conductance
        previous_action = torch.zeros_like(initial_conductance)
        actions: list[Tensor] = []
        flat_mask = None
        selected_count = conductance.numel()
        if programmable_mask is not None:
            programmable_mask = programmable_mask.to(device=conductance.device, dtype=torch.bool)
            flat_mask = programmable_mask.view(-1)
            selected_count = int(flat_mask.sum().item())
            if selected_count == 0:
                raise ValueError("programmable_mask must select at least one conductance.")

        for step_index in range(steps):
            fc1_c, fc2_c, fc3_c = _assemble_conductances(
                conductance, active_layer, fc1_context, fc2_context, fc3_context
            )
            logits, cache = model.forward_with_conductance(inputs, fc1_c, fc2_c, fc3_c)
            layer_input, layer_weight, layer_pre_act, layer_post_act = _get_layer_cache(cache, active_layer)
            post_activation_ce_gradient = compute_layer_ce_gradient(
                logits=logits,
                targets=targets,
                post_activation=layer_post_act,
                criterion=criterion,
            )
            state = build_layer_state(
                conductance=conductance,
                previous_action=previous_action,
                layer_input=layer_input,
                layer_weight=layer_weight,
                layer_pre_activation=layer_pre_act,
                layer_post_activation=layer_post_act,
                post_activation_ce_gradient=post_activation_ce_gradient,
                step_index=step_index,
                total_steps=steps,
            )
            policy_state = state if flat_mask is None else state[flat_mask]
            eps = None
            if policy_noise_scale == 0.0:
                eps = torch.zeros(policy_state.shape[0], 1, device=policy_state.device, dtype=policy_state.dtype)
            elif policy_noise_scale != 1.0:
                eps = (
                    torch.randn(policy_state.shape[0], 1, device=policy_state.device, dtype=policy_state.dtype)
                    * policy_noise_scale
                )
            selected_action = policy.sample(policy_state, output_shape=(selected_count,), eps=eps)
            if flat_mask is None:
                action = selected_action.view_as(conductance)
            else:
                action = torch.zeros_like(conductance)
                action = (
                    action.view(-1)
                    .scatter(0, flat_mask.nonzero(as_tuple=False).view(-1), selected_action)
                    .view_as(conductance)
                )
            noise = torch.randn_like(conductance) * noise_scale
            next_conductance = physics(conductance, action, noise=noise)

            previous_conductance = conductance
            previous_action = action
            conductance = next_conductance
            actions.append(action)

        fc1_c, fc2_c, fc3_c = _assemble_conductances(
            conductance, active_layer, fc1_context, fc2_context, fc3_context
        )
        logits, _ = model.forward_with_conductance(inputs, fc1_c, fc2_c, fc3_c)
        loss = criterion(logits, targets)
        return RolloutResult(
            logits=logits,
            loss=loss,
            final_conductance=conductance,
            actions=actions,
        )


def run_layer_rollout(
    *,
    layer: str | int,
    model: HardwareNoisyFeedForward,
    policy: LinearGaussianPolicy,
    physics: DifferentiableMemristorPhysics,
    inputs: Tensor,
    targets: Tensor,
    initial_conductance: Tensor,
    steps: int,
    criterion: Callable[[Tensor, Tensor], Tensor] | nn.Module,
    noise_scale: float = 1.0,
    policy_noise_scale: float = 1.0,
    programmable_mask: Tensor | None = None,
    fc1_context: Tensor | None = None,
    fc2_context: Tensor | None = None,
    fc3_context: Tensor | None = None,
) -> RolloutResult:
    return run_programming_rollout(
        model=model,
        policy=policy,
        physics=physics,
        inputs=inputs,
        targets=targets,
        initial_conductance=initial_conductance,
        steps=steps,
        criterion=criterion,
        noise_scale=noise_scale,
        policy_noise_scale=policy_noise_scale,
        programmable_mask=programmable_mask,
        active_layer=_layer_to_index(layer),
        fc1_context=fc1_context,
        fc2_context=fc2_context,
        fc3_context=fc3_context,
    )
