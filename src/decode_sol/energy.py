"""GPU energy from the NVML total-energy counter.

This covers the GPU only: the GB10 exposes no CPU or memory power sensors, so every
figure from here is "GPU energy". The counter advances in steps every ~100 ms, so a
window's energy is uncertain by up to one step at each end; keep windows to seconds.
"""

import argparse
import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import pynvml

from decode_sol.guard import collect_run_metadata, ensure_gpu_is_free

logger = logging.getLogger(__name__)

IDLE_DURATION_S = 30.0
NANOSECONDS_PER_SECOND = 1e9
MILLIJOULES_PER_JOULE = 1000.0


@dataclass(slots=True)
class GpuEnergyWindow:
    """Counter and clock readings at both ends of a timed region. Filled on exit."""

    start_energy_mj: int = 0
    end_energy_mj: int = 0
    start_time_ns: int = 0
    end_time_ns: int = 0

    @property
    def energy_j(self) -> float:
        return (self.end_energy_mj - self.start_energy_mj) / MILLIJOULES_PER_JOULE

    @property
    def duration_s(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / NANOSECONDS_PER_SECOND

    @property
    def mean_power_w(self) -> float:
        return self.energy_j / self.duration_s


@contextmanager
def open_nvml_device(device_index: int = 0) -> Iterator[object]:
    pynvml.nvmlInit()
    try:
        yield pynvml.nvmlDeviceGetHandleByIndex(device_index)
    finally:
        pynvml.nvmlShutdown()


def read_gpu_energy_mj(device: object) -> int:
    energy_mj: int = pynvml.nvmlDeviceGetTotalEnergyConsumption(device)
    return energy_mj


@contextmanager
def measure_gpu_energy(device: object) -> Iterator[GpuEnergyWindow]:
    """Read the counter and the clock around the block.

    The caller must make sure GPU work inside the block has finished before it exits;
    this does not synchronize.
    """
    window = GpuEnergyWindow()
    window.start_energy_mj = read_gpu_energy_mj(device)
    window.start_time_ns = time.perf_counter_ns()
    yield window
    window.end_energy_mj = read_gpu_energy_mj(device)
    window.end_time_ns = time.perf_counter_ns()


def measure_idle_gpu_energy(device: object, *, duration_s: float) -> GpuEnergyWindow:
    with measure_gpu_energy(device) as window:
        time.sleep(duration_s)
    return window


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure idle GPU power on the GB10.")
    parser.add_argument("--duration-s", type=float, default=IDLE_DURATION_S)
    parser.add_argument("--output", type=Path, default=Path("results/raw/idle_power.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ensure_gpu_is_free()
    record = collect_run_metadata()
    with open_nvml_device() as device:
        window = measure_idle_gpu_energy(device, duration_s=args.duration_s)
    record["idle_gpu_energy_window"] = asdict(window)
    record["idle_gpu_energy_j"] = window.energy_j
    record["idle_gpu_power_w"] = window.mean_power_w

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output_file:
        output_file.write(json.dumps(record) + "\n")
    logger.info(
        "Idle GPU power: power_w=%.2f duration_s=%.1f", window.mean_power_w, window.duration_s
    )


if __name__ == "__main__":
    main()
