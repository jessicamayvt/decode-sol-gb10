"""Top kernels, GPU idle time and read bandwidths per decode step, from Nsight Systems profiles.

Reads the SQLite exports of the reports scripts/run_profile.sh --nsys writes
(`nsys export --type sqlite <report>.nsys-rep`) and writes results/profile_summary.json.
vLLM marks each engine step with an NVTX range on the CPU. A kernel belongs to the step whose
range contains the CUDA API call that launched it; for a CUDA graph that is the graph launch.
"""

import argparse
import itertools
import json
import logging
import re
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from decode_sol.grid import KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE
from decode_sol.roofline import (
    BF16_BYTES,
    ModelDims,
    compute_kv_bytes,
    compute_num_weight_params_read,
    parse_model_dims,
)

logger = logging.getLogger(__name__)

# vLLM names a step's range execute_context_<num_prefills>(<num_prefill_tokens>)_generation_...
DECODE_STEP_RANGE_PATTERN = "execute_context_0(0)_generation_%"
NUM_TOP_KERNELS = 10
NS_PER_MS = 1e6
MS_PER_S = 1e3
BYTES_PER_GB = 1e9
TEMPLATE_ARGS_PATTERN = re.compile(r"<[^<>]*>")
# Named like ProfileConfig.trace_prefix in profile_decode.py, optionally followed by a label
# such as the host, to keep a second profile of the same point.
PROFILE_NAME_PATTERN = re.compile(
    r"(?P<model_name>.+)_(?P<kv_cache_dtype>auto|fp8)_b(?P<batch_size>\d+)_c(?P<context_len>\d+)"
    r"(?:_[^_]+)?"
)
# The kernels that read the weights and the KV cache in this project's profiles. The split-KV
# combine kernel is not one of them: it reads the partial outputs, not the KV cache.
MATMUL_KERNEL_FAMILIES = frozenset({"internal::gemvx::kernel", "cutlass::Kernel2"})
ATTENTION_KERNEL_FAMILIES = frozenset({"flash::flash_fwd_splitkv_kernel", "kernel_mha"})


@dataclass(frozen=True, slots=True)
class StepRange:
    start_ns: int
    end_ns: int


@dataclass(frozen=True, slots=True)
class KernelLaunch:
    name: str
    launch_ns: int  # Start of the CUDA API call on the CPU.
    start_ns: int
    end_ns: int


@dataclass(frozen=True, slots=True)
class DecodeStep:
    start_ns: int
    end_ns: int
    kernels: tuple[KernelLaunch, ...]


@dataclass(frozen=True, slots=True)
class KernelShare:
    name: str
    num_launches_per_step: float
    gpu_time_ms_per_step: float
    fraction_of_gpu_busy_time: float


@dataclass(frozen=True, slots=True)
class ReadBandwidth:
    """Bytes one decode step must read, over the GPU time of the kernels that read them."""

    num_bytes_per_step: int
    gpu_time_ms_per_step: float
    bandwidth_gb_per_s: float


@dataclass(frozen=True, slots=True)
class ProfiledPoint:
    model_name: str  # The part after the organization, such as Qwen3-4B.
    kv_cache_dtype: str
    batch_size: int
    context_len: int


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    """Medians over the decode steps of one profile. All kernels run on one CUDA stream.

    gpu_step_time_ms runs from a step's first kernel to the next step's first kernel and
    compares with the measured step time. gpu_busy_ms is the kernel time in that window and
    gpu_idle_ms the rest, when the GPU waits for the CPU. The medians need not add up.
    weight_read covers the matmul kernels and kv_cache_read the attention kernels, as means
    over the decode steps like top_kernels.
    """

    profile: str
    num_decode_steps: int
    gpu_step_time_ms: float
    gpu_busy_ms: float
    gpu_idle_ms: float
    cpu_step_range_ms: float
    top_kernels: tuple[KernelShare, ...]
    weight_read: ReadBandwidth
    kv_cache_read: ReadBandwidth


def to_kernel_family(demangled_name: str) -> str:
    """Drop template arguments, parameters and return type: the kernel's qualified name."""
    name = demangled_name.replace("<unnamed>::", "")
    while TEMPLATE_ARGS_PATTERN.search(name):
        name = TEMPLATE_ARGS_PATTERN.sub("", name)
    name_without_params = name.split("(")[0]
    return name_without_params.split(" ")[-1]


def split_into_decode_steps(
    kernels: list[KernelLaunch], step_ranges: list[StepRange]
) -> list[DecodeStep]:
    """Each step runs on the GPU clock from its first kernel to the next step's first kernel.

    A step's first kernel is the first one launched inside its NVTX range. The window cannot
    end with the range, because the LM head and sampling launch after the range closes. The
    last range only ends the step before it, so there is one step fewer than ranges.
    """
    step_start_ns: list[int] = []
    for step_range in step_ranges:
        launched_in_range: list[KernelLaunch] = []
        for kernel in kernels:
            if step_range.start_ns <= kernel.launch_ns < step_range.end_ns:
                launched_in_range.append(kernel)
        if not launched_in_range:
            raise ValueError(f"no kernels launched in step range {step_range}")
        step_start_ns.append(min(kernel.start_ns for kernel in launched_in_range))

    steps: list[DecodeStep] = []
    for start_ns, end_ns in itertools.pairwise(step_start_ns):
        step_kernels: list[KernelLaunch] = []
        for kernel in kernels:
            if start_ns <= kernel.start_ns < end_ns:
                step_kernels.append(kernel)
        steps.append(DecodeStep(start_ns=start_ns, end_ns=end_ns, kernels=tuple(step_kernels)))
    return steps


def compute_top_kernels(steps: list[DecodeStep], *, num_kernels: int) -> tuple[KernelShare, ...]:
    gpu_time_ns_by_name: dict[str, int] = defaultdict(int)
    num_launches_by_name: dict[str, int] = defaultdict(int)
    for step in steps:
        for kernel in step.kernels:
            gpu_time_ns_by_name[kernel.name] += kernel.end_ns - kernel.start_ns
            num_launches_by_name[kernel.name] += 1

    total_gpu_time_ns = sum(gpu_time_ns_by_name.values())
    names_by_gpu_time = sorted(gpu_time_ns_by_name, key=gpu_time_ns_by_name.__getitem__)
    top_kernels: list[KernelShare] = []
    for name in reversed(names_by_gpu_time[-num_kernels:]):
        gpu_time_ns = gpu_time_ns_by_name[name]
        top_kernels.append(
            KernelShare(
                name=name,
                num_launches_per_step=num_launches_by_name[name] / len(steps),
                gpu_time_ms_per_step=gpu_time_ns / len(steps) / NS_PER_MS,
                fraction_of_gpu_busy_time=gpu_time_ns / total_gpu_time_ns,
            )
        )
    return tuple(top_kernels)


def compute_read_bandwidth(
    steps: list[DecodeStep], *, kernel_families: frozenset[str], num_bytes_per_step: int
) -> ReadBandwidth:
    """Raises ValueError if no kernel of kernel_families ran, rather than divide by zero."""
    gpu_time_ns = 0
    for step in steps:
        for kernel in step.kernels:
            if kernel.name in kernel_families:
                gpu_time_ns += kernel.end_ns - kernel.start_ns
    if gpu_time_ns == 0:
        raise ValueError(f"no kernel of {sorted(kernel_families)} ran in the decode steps")

    gpu_time_ms_per_step = gpu_time_ns / len(steps) / NS_PER_MS
    return ReadBandwidth(
        num_bytes_per_step=num_bytes_per_step,
        gpu_time_ms_per_step=gpu_time_ms_per_step,
        bandwidth_gb_per_s=num_bytes_per_step / BYTES_PER_GB / (gpu_time_ms_per_step / MS_PER_S),
    )


def parse_profiled_point(profile: str) -> ProfiledPoint:
    match = PROFILE_NAME_PATTERN.fullmatch(profile)
    if match is None:
        raise ValueError(f"profile name {profile!r} is not <model>_<kv dtype>_b<n>_c<n>[_label]")
    return ProfiledPoint(
        model_name=match["model_name"],
        kv_cache_dtype=match["kv_cache_dtype"],
        batch_size=int(match["batch_size"]),
        context_len=int(match["context_len"]),
    )


def compute_bytes_per_decode_step(point: ProfiledPoint, dims: ModelDims) -> tuple[int, int]:
    """Weight bytes and KV bytes one decode step reads at the profiled point.

    The KV bytes use the prompt length. The profiled steps follow the prefill, so each holds
    at most NUM_PROFILED_DECODE_STEPS more tokens: under 4% more KV at context 512, where
    attention takes under 2% of the step, and under 0.1% at context 32768.
    """
    weight_bytes = compute_num_weight_params_read(dims) * BF16_BYTES
    kv_bytes = compute_kv_bytes(
        dims,
        batch_size=point.batch_size,
        context_len=point.context_len,
        kv_element_bytes=KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE[point.kv_cache_dtype],
    )
    return weight_bytes, kv_bytes


def summarize_decode_steps(
    profile: str,
    kernels: list[KernelLaunch],
    step_ranges: list[StepRange],
    *,
    weight_bytes: int,
    kv_bytes: int,
) -> ProfileSummary:
    if len(step_ranges) < 2:
        raise ValueError(f"{profile}: need at least 2 decode step ranges, got {len(step_ranges)}")
    steps = split_into_decode_steps(kernels, step_ranges)

    step_time_ms: list[float] = []
    busy_ms: list[float] = []
    idle_ms: list[float] = []
    for step in steps:
        step_busy_ns = sum(kernel.end_ns - kernel.start_ns for kernel in step.kernels)
        step_time_ms.append((step.end_ns - step.start_ns) / NS_PER_MS)
        busy_ms.append(step_busy_ns / NS_PER_MS)
        idle_ms.append((step.end_ns - step.start_ns - step_busy_ns) / NS_PER_MS)

    range_ms: list[float] = []
    for step_range in step_ranges:
        range_ms.append((step_range.end_ns - step_range.start_ns) / NS_PER_MS)

    return ProfileSummary(
        profile=profile,
        num_decode_steps=len(steps),
        gpu_step_time_ms=statistics.median(step_time_ms),
        gpu_busy_ms=statistics.median(busy_ms),
        gpu_idle_ms=statistics.median(idle_ms),
        cpu_step_range_ms=statistics.median(range_ms),
        top_kernels=compute_top_kernels(steps, num_kernels=NUM_TOP_KERNELS),
        weight_read=compute_read_bandwidth(
            steps, kernel_families=MATMUL_KERNEL_FAMILIES, num_bytes_per_step=weight_bytes
        ),
        kv_cache_read=compute_read_bandwidth(
            steps, kernel_families=ATTENTION_KERNEL_FAMILIES, num_bytes_per_step=kv_bytes
        ),
    )


def load_decode_step_ranges(connection: sqlite3.Connection) -> list[StepRange]:
    rows = connection.execute(
        """
        SELECT nvtx.start, nvtx.end FROM NVTX_EVENTS AS nvtx
        LEFT JOIN StringIds AS text ON text.id = nvtx.textId
        WHERE COALESCE(nvtx.text, text.value) LIKE ? ORDER BY nvtx.start
        """,
        (DECODE_STEP_RANGE_PATTERN,),
    )
    step_ranges: list[StepRange] = []
    for start_ns, end_ns in rows:
        step_ranges.append(StepRange(start_ns=start_ns, end_ns=end_ns))
    return step_ranges


def load_kernel_launches(connection: sqlite3.Connection) -> list[KernelLaunch]:
    rows = connection.execute(
        """
        SELECT name.value, api.start, kernel.start, kernel.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS api USING (correlationId)
        JOIN StringIds AS name ON name.id = kernel.demangledName
        """
    )
    kernels: list[KernelLaunch] = []
    for demangled_name, launch_ns, start_ns, end_ns in rows:
        kernels.append(
            KernelLaunch(
                name=to_kernel_family(demangled_name),
                launch_ns=launch_ns,
                start_ns=start_ns,
                end_ns=end_ns,
            )
        )
    return kernels


def load_model_dims_by_name(measurements_path: Path) -> dict[str, ModelDims]:
    """Each model's dimensions from the config.json its sweep records stored, keyed like
    ProfiledPoint.model_name, so the bytes match the measured revision without the network."""
    dims_by_model_name: dict[str, ModelDims] = {}
    with measurements_path.open() as measurements_file:
        for line in measurements_file:
            record = json.loads(line)
            model_name = record["model"].split("/")[-1]
            if model_name not in dims_by_model_name:
                dims_by_model_name[model_name] = parse_model_dims(record["model_config"])
    return dims_by_model_name


def load_profile_summary(
    sqlite_path: Path, dims_by_model_name: dict[str, ModelDims]
) -> ProfileSummary:
    point = parse_profiled_point(sqlite_path.stem)
    weight_bytes, kv_bytes = compute_bytes_per_decode_step(
        point, dims_by_model_name[point.model_name]
    )
    with sqlite3.connect(sqlite_path) as connection:
        step_ranges = load_decode_step_ranges(connection)
        kernels = load_kernel_launches(connection)
    return summarize_decode_steps(
        sqlite_path.stem, kernels, step_ranges, weight_bytes=weight_bytes, kv_bytes=kv_bytes
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize decode steps in nsys profiles.")
    parser.add_argument("--profiles-dir", type=Path, default=Path("profiles/nsys"))
    parser.add_argument("--measurements", type=Path, default=Path("results/raw/decode.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/profile_summary.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dims_by_model_name = load_model_dims_by_name(args.measurements)
    summaries: list[ProfileSummary] = []
    for sqlite_path in sorted(args.profiles_dir.glob("*.sqlite")):
        summary = load_profile_summary(sqlite_path, dims_by_model_name)
        summaries.append(summary)
        logger.info(
            "%s: step %.3f ms, GPU idle %.3f ms, weights at %.1f GB/s, KV cache at %.1f GB/s",
            summary.profile,
            summary.gpu_step_time_ms,
            summary.gpu_idle_ms,
            summary.weight_read.bandwidth_gb_per_s,
            summary.kv_cache_read.bandwidth_gb_per_s,
        )
    serialized = [asdict(summary) for summary in summaries]
    args.output.write_text(json.dumps(serialized, indent=2) + "\n")


if __name__ == "__main__":
    main()
