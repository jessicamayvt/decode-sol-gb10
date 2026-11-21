"""Shared-machine checks and the metadata every raw record carries.

The GPU is shared with other students, so every measurement starts by checking that
nobody else is using it, and vLLM's memory share is capped by department policy.
"""

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

logger = logging.getLogger(__name__)

# Department policy for the shared GB10: vLLM may claim at most a quarter of the memory.
MAX_GPU_MEMORY_UTILIZATION = 0.25
# Idle reads 0%. Anything above this with no visible process means a user NVML hides from us.
BUSY_GPU_UTILIZATION_PERCENT = 10
NVIDIA_SMI_TIMEOUT_S = 30
KIB_PER_GIB = 2**20
# Columns of the cpu line in /proc/stat after the label: user nice system idle iowait ...
PROC_STAT_IDLE_FIELD = 3
PROC_STAT_IOWAIT_FIELD = 4


class GpuMemoryCapError(ValueError):
    """A requested gpu_memory_utilization breaks the shared-machine cap."""


class GpuBusyError(RuntimeError):
    """Another user is running GPU work, so a measurement must not start."""


@dataclass(frozen=True, slots=True)
class GpuProcess:
    pid: int
    used_memory_mib: int | None


@dataclass(frozen=True, slots=True)
class GpuStatus:
    name: str
    driver_version: str
    utilization_percent: int


@dataclass(frozen=True, slots=True)
class NodeLoad:
    """The node's CPU and memory state. The CPU shares the GPU's memory bus on the GB10.

    The jiffy counters are cumulative since boot; diff two readings with
    compute_cpu_busy_fraction.
    """

    load_average_1min: float
    mem_available_gib: float
    cpu_busy_jiffies: int
    cpu_total_jiffies: int


def parse_gpu_memory_utilization(gpu_memory_utilization: float) -> float:
    if not 0.0 < gpu_memory_utilization <= MAX_GPU_MEMORY_UTILIZATION:
        raise GpuMemoryCapError(
            f"gpu_memory_utilization must be in (0, {MAX_GPU_MEMORY_UTILIZATION}], "
            f"got {gpu_memory_utilization}"
        )
    return gpu_memory_utilization


def parse_compute_apps_csv(csv_text: str) -> list[GpuProcess]:
    """Parse `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`."""
    processes: list[GpuProcess] = []
    for line in csv_text.splitlines():
        if not line.strip():
            continue
        pid_text, memory_text = line.split(",")
        memory_text = memory_text.strip()
        used_memory_mib = int(memory_text) if memory_text.isdigit() else None
        processes.append(GpuProcess(pid=int(pid_text), used_memory_mib=used_memory_mib))
    return processes


def find_gpu_busy_reason(
    processes: list[GpuProcess], *, gpu_utilization_percent: int, own_pid: int
) -> str | None:
    other_processes: list[GpuProcess] = []
    for process in processes:
        if process.pid != own_pid:
            other_processes.append(process)

    if other_processes:
        return f"other GPU processes are running: {other_processes}"
    if gpu_utilization_percent > BUSY_GPU_UTILIZATION_PERCENT:
        return f"GPU utilization is {gpu_utilization_percent}% with no visible process"
    return None


def _query_nvidia_smi(*args: str) -> str:
    completed = subprocess.run(
        ["nvidia-smi", *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=NVIDIA_SMI_TIMEOUT_S,
    )
    return completed.stdout


def load_gpu_processes() -> list[GpuProcess]:
    csv_text = _query_nvidia_smi(
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
    )
    return parse_compute_apps_csv(csv_text)


def load_gpu_status() -> GpuStatus:
    csv_text = _query_nvidia_smi(
        "--query-gpu=name,driver_version,utilization.gpu", "--format=csv,noheader,nounits"
    )
    name, driver_version, utilization_text = csv_text.strip().split(", ")
    return GpuStatus(
        name=name, driver_version=driver_version, utilization_percent=int(utilization_text)
    )


def parse_mem_available_gib(meminfo_text: str) -> float:
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / KIB_PER_GIB
    raise ValueError("no MemAvailable line in /proc/meminfo")


def parse_cpu_jiffies(proc_stat_text: str) -> tuple[int, int]:
    """(busy, total) jiffies over all CPUs, from the first line of /proc/stat.

    Idle and iowait count as not busy.
    """
    fields = proc_stat_text.splitlines()[0].split()
    if fields[0] != "cpu":
        raise ValueError(f"first line of /proc/stat is not the cpu total: {fields[0]!r}")
    jiffies = [int(field) for field in fields[1:]]
    idle_jiffies = jiffies[PROC_STAT_IDLE_FIELD] + jiffies[PROC_STAT_IOWAIT_FIELD]
    total_jiffies = sum(jiffies)
    return total_jiffies - idle_jiffies, total_jiffies


def load_node_load() -> NodeLoad:
    cpu_busy_jiffies, cpu_total_jiffies = parse_cpu_jiffies(Path("/proc/stat").read_text())
    return NodeLoad(
        load_average_1min=os.getloadavg()[0],
        mem_available_gib=parse_mem_available_gib(Path("/proc/meminfo").read_text()),
        cpu_busy_jiffies=cpu_busy_jiffies,
        cpu_total_jiffies=cpu_total_jiffies,
    )


def compute_cpu_busy_fraction(before: NodeLoad, after: NodeLoad) -> float:
    """Fraction of all the node's CPU time between two readings that was busy, ours included."""
    busy_jiffies = after.cpu_busy_jiffies - before.cpu_busy_jiffies
    total_jiffies = after.cpu_total_jiffies - before.cpu_total_jiffies
    return busy_jiffies / total_jiffies


def ensure_gpu_is_free() -> None:
    """Raise GpuBusyError if anyone else is using the GPU."""
    processes = load_gpu_processes()
    status = load_gpu_status()
    reason = find_gpu_busy_reason(
        processes, gpu_utilization_percent=status.utilization_percent, own_pid=os.getpid()
    )
    if reason is not None:
        raise GpuBusyError(reason)
    logger.info("GPU is free: utilization_percent=%d", status.utilization_percent)


def _load_git_commit() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    porcelain = subprocess.run(
        # Only code counts: results/ is appended to by every run.
        ["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return commit, bool(porcelain.strip())


def collect_run_metadata() -> dict[str, object]:
    """Everything needed to reproduce a measurement, for the start of every raw record."""
    import torch  # Deferred so importing this module stays cheap and never initializes CUDA.

    git_commit, has_uncommitted_changes = _load_git_commit()
    status = load_gpu_status()
    other_processes: list[dict[str, object]] = []
    for process in load_gpu_processes():
        if process.pid != os.getpid():
            other_processes.append(asdict(process))

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "has_uncommitted_changes": has_uncommitted_changes,
        "hostname": platform.node(),
        "gpu_name": status.name,
        "driver_version": status.driver_version,
        "gpu_utilization_percent_at_start": status.utilization_percent,
        "other_gpu_processes_at_start": other_processes,
        "node_load_at_start": asdict(load_node_load()),
        "cuda_version": torch.version.cuda,
        "torch_version": version("torch"),
        "triton_version": version("triton"),
        "vllm_version": version("vllm"),
        "python_version": platform.python_version(),
    }
