import pytest

from decode_sol.energy import (
    GpuEnergyWindow,
    measure_idle_gpu_energy,
    open_nvml_device,
)


def test_energy_window_converts_millijoules_and_nanoseconds() -> None:
    window = GpuEnergyWindow(
        start_energy_mj=1_000,
        end_energy_mj=11_000,
        start_time_ns=0,
        end_time_ns=2_000_000_000,
    )

    assert window.energy_j == pytest.approx(10.0)
    assert window.duration_s == pytest.approx(2.0)
    assert window.mean_power_w == pytest.approx(5.0)


@pytest.mark.gpu
def test_idle_gpu_power_is_plausible() -> None:
    with open_nvml_device() as device:
        window = measure_idle_gpu_energy(device, duration_s=1.0)

    assert window.duration_s == pytest.approx(1.0, abs=0.05)
    assert 1.0 < window.mean_power_w < 150.0
