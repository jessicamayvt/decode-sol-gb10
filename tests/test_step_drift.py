import pytest

from decode_sol.grid import LONG_OUTPUT_LEN, SHORT_OUTPUT_LEN
from decode_sol.step_drift import compute_long_generation_power_w, compute_step_times_ms

NUM_EXTRA_STEPS = LONG_OUTPUT_LEN - SHORT_OUTPUT_LEN
NS_PER_MS = 1_000_000


def make_raw_pair(*, short_ms: int, long_ms: int, long_energy_mj: int) -> dict[str, dict[str, int]]:
    return {
        "short": {
            "start_energy_mj": 0,
            "end_energy_mj": 1_000,
            "start_time_ns": 0,
            "end_time_ns": short_ms * NS_PER_MS,
        },
        "long": {
            "start_energy_mj": 5_000,
            "end_energy_mj": 5_000 + long_energy_mj,
            "start_time_ns": 0,
            "end_time_ns": long_ms * NS_PER_MS,
        },
    }


def test_step_time_is_the_long_minus_short_duration_per_extra_step() -> None:
    raw_pairs = [
        make_raw_pair(short_ms=1_000, long_ms=1_000 + 75 * NUM_EXTRA_STEPS, long_energy_mj=1),
        make_raw_pair(short_ms=1_000, long_ms=1_000 + 76 * NUM_EXTRA_STEPS, long_energy_mj=1),
    ]

    step_times_ms = compute_step_times_ms(raw_pairs)

    assert step_times_ms == pytest.approx([75.0, 76.0])


def test_long_generation_power_is_its_energy_over_its_duration() -> None:
    raw_pairs = [make_raw_pair(short_ms=1_000, long_ms=10_000, long_energy_mj=800_000)]

    power_w = compute_long_generation_power_w(raw_pairs)

    assert power_w == pytest.approx([80.0])
