import re

import pytest

from decode_sol.guard import (
    MAX_GPU_MEMORY_UTILIZATION,
    GpuMemoryCapError,
    GpuProcess,
    collect_run_metadata,
    find_gpu_busy_reason,
    parse_compute_apps_csv,
    parse_gpu_memory_utilization,
)


@pytest.mark.parametrize("gpu_memory_utilization", [0.1, MAX_GPU_MEMORY_UTILIZATION])
def test_parse_gpu_memory_utilization_accepts_values_up_to_the_cap(
    gpu_memory_utilization: float,
) -> None:
    assert parse_gpu_memory_utilization(gpu_memory_utilization) == gpu_memory_utilization


@pytest.mark.parametrize("gpu_memory_utilization", [0.2501, 0.9, 0.0, -0.1])
def test_parse_gpu_memory_utilization_rejects_values_outside_the_cap(
    gpu_memory_utilization: float,
) -> None:
    with pytest.raises(GpuMemoryCapError, match="gpu_memory_utilization"):
        parse_gpu_memory_utilization(gpu_memory_utilization)


def test_parse_compute_apps_csv_with_no_processes_returns_empty_list() -> None:
    assert parse_compute_apps_csv("") == []


def test_parse_compute_apps_csv_reads_pid_and_memory() -> None:
    csv_text = "1234, 30000\n5678, [N/A]\n"

    processes = parse_compute_apps_csv(csv_text)

    assert processes == [
        GpuProcess(pid=1234, used_memory_mib=30000),
        GpuProcess(pid=5678, used_memory_mib=None),
    ]


def test_find_gpu_busy_reason_is_none_when_idle() -> None:
    assert find_gpu_busy_reason([], gpu_utilization_percent=0, own_pid=1) is None


def test_find_gpu_busy_reason_ignores_own_process() -> None:
    own_process = GpuProcess(pid=42, used_memory_mib=100)

    assert find_gpu_busy_reason([own_process], gpu_utilization_percent=0, own_pid=42) is None


def test_find_gpu_busy_reason_reports_other_process() -> None:
    other_process = GpuProcess(pid=1234, used_memory_mib=30000)

    reason = find_gpu_busy_reason([other_process], gpu_utilization_percent=0, own_pid=42)

    assert reason is not None
    assert "1234" in reason


def test_find_gpu_busy_reason_reports_high_utilization_without_visible_process() -> None:
    reason = find_gpu_busy_reason([], gpu_utilization_percent=80, own_pid=42)

    assert reason is not None
    assert "80%" in reason


@pytest.mark.gpu
def test_collect_run_metadata_records_commit_and_versions() -> None:
    metadata = collect_run_metadata()

    assert re.fullmatch(r"[0-9a-f]{40}", str(metadata["git_commit"]))
    assert metadata["gpu_name"] == "NVIDIA GB10"
    assert isinstance(metadata["vllm_version"], str)
