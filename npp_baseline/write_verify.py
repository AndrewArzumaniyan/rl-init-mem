from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from memristor_simulation import MemristorSimulator


@dataclass(slots=True)
class WriteVerifyStep:
    step_index: int
    g_before_us: float
    g_after_us: float
    t_pulse_ns: float


@dataclass(slots=True)
class WriteVerifyResult:
    steps: list[WriteVerifyStep]
    converged: bool


def run_write_and_verify(
    simulator: MemristorSimulator,
    predictor: nn.Module,
    g_target_us: float,
    max_steps: int,
    tolerance_us: float,
    make_input: callable,
) -> WriteVerifyResult:
    steps: list[WriteVerifyStep] = []
    converged = False

    for step_index in range(max_steps):
        g_before_us = simulator.read()
        if abs(g_before_us - g_target_us) <= tolerance_us:
            converged = True
            break

        with torch.no_grad():
            model_input = make_input(g_before_us, g_target_us).unsqueeze(0)
            t_pulse_ns = float(predictor(model_input).squeeze().item())

        raise NotImplementedError("Write-and-verify will be wired after the simulator pulse API is finalized.")

    return WriteVerifyResult(steps=steps, converged=converged)
