from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from memristor_simulation import MemristorSimulator, PulseTrace


@dataclass(slots=True)
class NppSample:
    g_start_us: float
    g_target_us: float
    delta_g_us: float
    t_pulse_ns: float
    g_end_us: float
    trace: PulseTrace


class NppDatasetGenerator:
    def __init__(
        self,
        simulator: MemristorSimulator,
        v_set: float,
        v_reset: float,
        time_step_ns: float = 10.0,
    ) -> None:
        self.simulator = simulator
        self.v_set = v_set
        self.v_reset = v_reset
        self.time_step_ns = time_step_ns

    def voltage_for(self, g_start_us: float, g_target_us: float) -> float:
        return self.v_set if g_target_us >= g_start_us else self.v_reset

    def generate_sample(self, g_start_us: float, g_target_us: float) -> NppSample:
        raise NotImplementedError("Dataset generation will follow the exact NPP procedure from the paper.")

    def generate(self, targets: Iterable[tuple[float, float]]) -> list[NppSample]:
        return [self.generate_sample(g_start_us, g_target_us) for g_start_us, g_target_us in targets]

    def sample_targets(
        self,
        count: int,
        conductance_range_us: tuple[float, float],
        seed: int | None = None,
    ) -> list[tuple[float, float]]:
        rng = np.random.default_rng(seed)
        low, high = conductance_range_us
        values = rng.uniform(low=low, high=high, size=(count, 2))
        return [(float(g_start), float(g_target)) for g_start, g_target in values]

    @staticmethod
    def split(
        samples: Sequence[NppSample],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> tuple[list[NppSample], list[NppSample], list[NppSample]]:
        train_end = int(len(samples) * train_ratio)
        val_end = train_end + int(len(samples) * val_ratio)
        return list(samples[:train_end]), list(samples[train_end:val_end]), list(samples[val_end:])
