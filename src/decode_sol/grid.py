"""The sweep grid, which of its points fit, and how long a sweep should take. No GPU.

A sweep runs one model and one KV cache dtype per process (scripts/run_sweep.sh), so the
model is loaded once for every point. `python -m decode_sol.grid` prints the plan that
the brief asks to approve before any sweep: the feasible points and the GPU minutes.
"""

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from decode_sol.analyze import load_achievable_bandwidth_gb_per_s
from decode_sol.bandwidth import BYTES_PER_GB
from decode_sol.energy import IDLE_DURATION_S
from decode_sol.roofline import (
    BF16_BYTES,
    ModelDims,
    compute_kv_bytes,
    compute_num_weight_params_read,
    parse_model_dims,
    predict_decode_step,
)

logger = logging.getLogger(__name__)

# One pair is the same batch generated at the short and then the long output length.
SHORT_OUTPUT_LEN = 16
LONG_OUTPUT_LEN = 144
NUM_WARMUP_PAIRS = 2
NUM_TIMED_PAIRS = 10

# Most tokens vLLM may prefill in one step. Sequences of a batch start decoding as their
# prefill finishes, so a batch's prefill must take at most SHORT_OUTPUT_LEN steps, or the
# earliest sequence finishes its short generation before the last one starts decoding and
# the two generations of a pair no longer share the same prefill phase (docs/MODEL.md).
MAX_NUM_BATCHED_TOKENS = 32768

SWEEP_MODELS = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B")
KV_CACHE_DTYPES = ("auto", "fp8")
KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE = {"auto": 2, "fp8": 1}
VLLM_BLOCK_SIZE = 16

# For the estimate only. The real capacity is whatever vLLM reports at startup.
GPU_MEMORY_BYTES = int(121.6 * 2**30)  # visible to PyTorch on the GB10 (docs/PROJECT_BRIEF.md)
GPU_MEMORY_UTILIZATION = 0.25
# vLLM's activation profile and CUDA graphs took 2.6-4.3 GiB in M0-M2 with 8192-token
# prefill steps; 32768-token steps need more.
ESTIMATED_NON_KV_RESERVE_BYTES = 6 * 2**30
# M2 reached 0.65 of the achievable ceiling. Its 512-token prefill of Qwen3-1.7B implies
# about 80 TFLOP/s; 50 leaves margin for long-context attention.
ASSUMED_DECODE_EFFICIENCY = 0.65
ASSUMED_PREFILL_FLOPS_PER_S = 50e12
ESTIMATED_PROCESS_STARTUP_S = 180.0  # model load, torch.compile and CUDA graph capture
SECONDS_PER_MINUTE = 60.0
NUM_REPEATS = 3


@dataclass(frozen=True, slots=True)
class GridPoint:
    batch_size: int
    context_len: int


@dataclass(frozen=True, slots=True)
class Grid:
    batch_sizes: tuple[int, ...]
    context_lens: tuple[int, ...]

    @property
    def points(self) -> list[GridPoint]:
        points: list[GridPoint] = []
        for context_len in self.context_lens:
            for batch_size in self.batch_sizes:
                points.append(GridPoint(batch_size=batch_size, context_len=context_len))
        return points

    @property
    def max_batch_size(self) -> int:
        return max(self.batch_sizes)

    @property
    def max_context_len(self) -> int:
        return max(self.context_lens)

    @property
    def max_model_len(self) -> int:
        return self.max_context_len + LONG_OUTPUT_LEN

    @property
    def max_num_batched_tokens(self) -> int:
        """vLLM's warmup overruns its request buffers if a prefill step could hold more
        tokens than max_num_seqs sequences of max_model_len, so the budget is capped there."""
        return min(MAX_NUM_BATCHED_TOKENS, self.max_batch_size * self.max_model_len)


GRID_BY_NAME = {
    "smoke": Grid(batch_sizes=(1,), context_lens=(512,)),
    "full": Grid(batch_sizes=(1, 2, 4, 8, 16, 32, 64), context_lens=(512, 2048, 8192, 32768)),
}


def compute_num_blocks_needed(point: GridPoint, *, block_size: int) -> int:
    """KV cache blocks the batch holds at the end of the long generation."""
    num_blocks_per_sequence = math.ceil((point.context_len + LONG_OUTPUT_LEN) / block_size)
    return point.batch_size * num_blocks_per_sequence


def compute_num_prefill_steps(point: GridPoint, *, max_num_batched_tokens: int) -> int:
    return math.ceil(point.batch_size * point.context_len / max_num_batched_tokens)


def find_skip_reason(
    point: GridPoint, *, num_gpu_blocks: int, block_size: int, max_num_batched_tokens: int
) -> str | None:
    """Why the point cannot be measured as a single decode batch, or None if it can."""
    num_blocks_needed = compute_num_blocks_needed(point, block_size=block_size)
    if num_blocks_needed > num_gpu_blocks:
        return f"needs {num_blocks_needed} KV cache blocks, {num_gpu_blocks} available"
    num_prefill_steps = compute_num_prefill_steps(
        point, max_num_batched_tokens=max_num_batched_tokens
    )
    if num_prefill_steps > SHORT_OUTPUT_LEN:
        return (
            f"prefill takes {num_prefill_steps} steps, more than the {SHORT_OUTPUT_LEN}-token"
            " short generation"
        )
    return None


def estimate_num_gpu_blocks(dims: ModelDims, *, kv_element_bytes: int, block_size: int) -> int:
    """KV cache blocks vLLM should have left under the memory cap after weights and reserve."""
    kv_budget_bytes = (
        GPU_MEMORY_BYTES * GPU_MEMORY_UTILIZATION
        - compute_num_weight_params_read(dims) * BF16_BYTES
        - ESTIMATED_NON_KV_RESERVE_BYTES
    )
    kv_bytes_per_block = compute_kv_bytes(
        dims, batch_size=1, context_len=block_size, kv_element_bytes=kv_element_bytes
    )
    return max(0, int(kv_budget_bytes // kv_bytes_per_block))


def estimate_prefill_s(point: GridPoint, dims: ModelDims) -> float:
    """Dense FLOPs of the linear layers plus causal attention, at the assumed rate."""
    num_prompt_tokens = point.batch_size * point.context_len
    linear_flops = 2 * compute_num_weight_params_read(dims) * num_prompt_tokens
    q_size = dims.num_attention_heads * dims.head_dim
    attention_flops = 2 * dims.num_layers * q_size * point.batch_size * point.context_len**2
    return (linear_flops + attention_flops) / ASSUMED_PREFILL_FLOPS_PER_S


def estimate_decode_step_s(
    point: GridPoint, dims: ModelDims, *, kv_element_bytes: int, bandwidth_bytes_per_s: float
) -> float:
    effective_context_len = point.context_len + (SHORT_OUTPUT_LEN + LONG_OUTPUT_LEN) // 2
    prediction = predict_decode_step(
        dims,
        batch_size=point.batch_size,
        context_len=effective_context_len,
        kv_element_bytes=kv_element_bytes,
        bandwidth_bytes_per_s=bandwidth_bytes_per_s,
    )
    return prediction.step_time_s / ASSUMED_DECODE_EFFICIENCY


def estimate_point_duration_s(
    point: GridPoint, dims: ModelDims, *, kv_element_bytes: int, bandwidth_bytes_per_s: float
) -> float:
    """Wall time of every generation at the point: warmup and timed, short and long."""
    prefill_s = estimate_prefill_s(point, dims)
    step_s = estimate_decode_step_s(
        point, dims, kv_element_bytes=kv_element_bytes, bandwidth_bytes_per_s=bandwidth_bytes_per_s
    )
    pair_s = 2 * prefill_s + (SHORT_OUTPUT_LEN + LONG_OUTPUT_LEN) * step_s
    return (NUM_WARMUP_PAIRS + NUM_TIMED_PAIRS) * pair_s


@dataclass(frozen=True, slots=True)
class SweepPlan:
    """One process: a model at one KV cache dtype over a grid."""

    model: str
    kv_cache_dtype: str
    estimated_num_gpu_blocks: int
    skip_reason_by_point: dict[GridPoint, str | None]
    duration_s_by_point: dict[GridPoint, float]

    @property
    def feasible_points(self) -> list[GridPoint]:
        feasible: list[GridPoint] = []
        for point, skip_reason in self.skip_reason_by_point.items():
            if skip_reason is None:
                feasible.append(point)
        return feasible

    @property
    def duration_s(self) -> float:
        total_s = ESTIMATED_PROCESS_STARTUP_S + IDLE_DURATION_S
        for point in self.feasible_points:
            total_s += self.duration_s_by_point[point]
        return total_s


def plan_sweep(
    grid: Grid, dims: ModelDims, *, model: str, kv_cache_dtype: str, bandwidth_bytes_per_s: float
) -> SweepPlan:
    kv_element_bytes = KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE[kv_cache_dtype]
    num_gpu_blocks = estimate_num_gpu_blocks(
        dims, kv_element_bytes=kv_element_bytes, block_size=VLLM_BLOCK_SIZE
    )
    skip_reason_by_point: dict[GridPoint, str | None] = {}
    duration_s_by_point: dict[GridPoint, float] = {}
    for point in grid.points:
        skip_reason_by_point[point] = find_skip_reason(
            point,
            num_gpu_blocks=num_gpu_blocks,
            block_size=VLLM_BLOCK_SIZE,
            max_num_batched_tokens=grid.max_num_batched_tokens,
        )
        duration_s_by_point[point] = estimate_point_duration_s(
            point,
            dims,
            kv_element_bytes=kv_element_bytes,
            bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        )
    return SweepPlan(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        estimated_num_gpu_blocks=num_gpu_blocks,
        skip_reason_by_point=skip_reason_by_point,
        duration_s_by_point=duration_s_by_point,
    )


def load_model_config(model: str, revision: str | None) -> dict[str, object]:
    config_path = hf_hub_download(model, "config.json", revision=revision)
    config: dict[str, object] = json.loads(Path(config_path).read_text())
    return config


def log_sweep_plan(plan: SweepPlan) -> None:
    logger.info(
        "%s kv_cache_dtype=%s: estimated %d KV cache blocks (%d tokens)",
        plan.model,
        plan.kv_cache_dtype,
        plan.estimated_num_gpu_blocks,
        plan.estimated_num_gpu_blocks * VLLM_BLOCK_SIZE,
    )
    logger.info("  %5s  %7s  %9s  %s", "batch", "context", "est. min", "skip reason")
    for point, skip_reason in plan.skip_reason_by_point.items():
        minutes = plan.duration_s_by_point[point] / SECONDS_PER_MINUTE
        if skip_reason is None:
            logger.info("  %5d  %7d  %9.1f", point.batch_size, point.context_len, minutes)
        else:
            logger.info(
                "  %5d  %7d  %9s  %s", point.batch_size, point.context_len, "-", skip_reason
            )
    logger.info(
        "  %d of %d points fit, est. %.0f min including %.0f s startup and idle baseline",
        len(plan.feasible_points),
        len(plan.skip_reason_by_point),
        plan.duration_s / SECONDS_PER_MINUTE,
        ESTIMATED_PROCESS_STARTUP_S + IDLE_DURATION_S,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the feasible grid and GPU-minute estimate.")
    parser.add_argument("--grid", choices=sorted(GRID_BY_NAME), default="full")
    parser.add_argument("--bandwidth", type=Path, default=Path("results/raw/bandwidth.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    grid = GRID_BY_NAME[args.grid]
    bandwidth_bytes_per_s = load_achievable_bandwidth_gb_per_s(args.bandwidth) * BYTES_PER_GB
    plans: list[SweepPlan] = []
    for model in SWEEP_MODELS:
        dims = parse_model_dims(load_model_config(model, revision=None))
        for kv_cache_dtype in KV_CACHE_DTYPES:
            plan = plan_sweep(
                grid,
                dims,
                model=model,
                kv_cache_dtype=kv_cache_dtype,
                bandwidth_bytes_per_s=bandwidth_bytes_per_s,
            )
            log_sweep_plan(plan)
            plans.append(plan)

    num_feasible_points = 0
    total_s = 0.0
    for plan in plans:
        num_feasible_points += len(plan.feasible_points)
        total_s += plan.duration_s
    logger.info(
        "Grid %r: %d sweeps, %d feasible points, est. %.0f GPU minutes per repeat, "
        "%.0f for %d repeats. Downloads are extra.",
        args.grid,
        len(plans),
        num_feasible_points,
        total_s / SECONDS_PER_MINUTE,
        NUM_REPEATS * total_s / SECONDS_PER_MINUTE,
        NUM_REPEATS,
    )


if __name__ == "__main__":
    main()
