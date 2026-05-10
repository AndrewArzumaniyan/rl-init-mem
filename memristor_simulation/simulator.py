from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_aihwkit() -> Any:
    try:
        import aihwkit  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("aihwkit is not installed in the current environment.") from exc
    return aihwkit


@dataclass(slots=True)
class DeviceState:
    conductance_us: float


@dataclass(slots=True)
class PulseTrace:
    times_ns: list[float]
    conductances_us: list[float]


class MemristorSimulator:
    def __init__(self, device_config: Any | None = None) -> None:
        self.device_config = device_config
        self.state = DeviceState(conductance_us=0.0)

    @property
    def backend(self) -> Any:
        return require_aihwkit()

    def reset(self, conductance_us: float) -> DeviceState:
        self.state = DeviceState(conductance_us=conductance_us)
        return self.state

    def read(self) -> float:
        return self.state.conductance_us

    def simulate_constant_voltage(
        self,
        voltage: float,
        duration_ns: float,
        sample_every_ns: float = 10.0,
    ) -> PulseTrace:
        raise NotImplementedError("Concrete aihwkit-backed simulation will be added next.")
