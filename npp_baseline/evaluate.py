from __future__ import annotations

import torch
from torch import Tensor


def relative_percentage_difference(output: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    denominator = torch.clamp(target.abs(), min=eps)
    return ((output - target).abs() / denominator).mean()


def map_pulse_time_to_conductance(
    pulse_time_ns: float,
    history_times_ns: list[float],
    history_conductances_us: list[float],
) -> float:
    if pulse_time_ns <= history_times_ns[0]:
        return history_conductances_us[0]
    if pulse_time_ns >= history_times_ns[-1]:
        return history_conductances_us[-1]

    for left, right in zip(range(len(history_times_ns) - 1), range(1, len(history_times_ns))):
        t_left = history_times_ns[left]
        t_right = history_times_ns[right]
        if t_left <= pulse_time_ns <= t_right:
            alpha = (pulse_time_ns - t_left) / (t_right - t_left)
            g_left = history_conductances_us[left]
            g_right = history_conductances_us[right]
            return g_left + alpha * (g_right - g_left)

    return history_conductances_us[-1]


def success_rate(rpds: Tensor, threshold: float = 0.5) -> float:
    return float((rpds <= threshold).float().mean().item())
