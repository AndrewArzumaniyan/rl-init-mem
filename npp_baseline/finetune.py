from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def moving_average(values: Tensor, kernel_size: int) -> Tensor:
    if kernel_size <= 1:
        return values
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, device=values.device, dtype=values.dtype) / kernel_size
    source = values.view(1, 1, -1)
    smoothed = torch.nn.functional.conv1d(source, kernel, padding=padding)
    return smoothed.view(-1)[: values.shape[0]]


@dataclass(slots=True)
class FineTuneKernelSchedule:
    kernel_sizes: tuple[int, ...] = (1001, 101, 11, 1)
    epochs_per_kernel: int = 50


class HistoryMappedLoss(nn.Module):
    def forward(
        self,
        predicted_pulse_times_ns: Tensor,
        target_conductances_us: Tensor,
        history_times_ns: Tensor,
        history_conductances_us: Tensor,
    ) -> Tensor:
        mapped = torch.empty_like(target_conductances_us)
        for index in range(predicted_pulse_times_ns.shape[0]):
            mapped[index] = self._map_single(
                pulse_time_ns=predicted_pulse_times_ns[index],
                history_times_ns=history_times_ns[index],
                history_conductances_us=history_conductances_us[index],
            )
        return torch.nn.functional.mse_loss(mapped, target_conductances_us)

    @staticmethod
    def _map_single(
        pulse_time_ns: Tensor,
        history_times_ns: Tensor,
        history_conductances_us: Tensor,
    ) -> Tensor:
        if pulse_time_ns <= history_times_ns[0]:
            return history_conductances_us[0]
        if pulse_time_ns >= history_times_ns[-1]:
            return history_conductances_us[-1]

        for index in range(history_times_ns.shape[0] - 1):
            t_left = history_times_ns[index]
            t_right = history_times_ns[index + 1]
            if t_left <= pulse_time_ns <= t_right:
                alpha = (pulse_time_ns - t_left) / (t_right - t_left)
                g_left = history_conductances_us[index]
                g_right = history_conductances_us[index + 1]
                return g_left + alpha * (g_right - g_left)

        return history_conductances_us[-1]
