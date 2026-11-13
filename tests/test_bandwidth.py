import pytest
import torch

from decode_sol.bandwidth import compute_bandwidth_gb_per_s, measure_read_times_ms


def test_compute_bandwidth_gb_per_s_converts_bytes_and_milliseconds() -> None:
    # 2e9 bytes in 10 ms is 200 GB/s.
    assert compute_bandwidth_gb_per_s(2_000_000_000, 10.0) == pytest.approx(200.0)


@pytest.mark.gpu
def test_measure_read_times_ms_returns_one_positive_time_per_iteration() -> None:
    read_buffer = torch.ones(2**24, dtype=torch.bfloat16, device="cuda")  # [num_elements], 32 MiB

    times_ms = measure_read_times_ms(read_buffer, num_warmup=2, num_timed=5)

    assert len(times_ms) == 5
    assert min(times_ms) > 0.0
