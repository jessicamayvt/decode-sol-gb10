"""Profile vLLM decode steps at one grid point, with the engine sized exactly as in the sweep.

Uses the GPU: start it only through scripts/run_profile.sh. The torch profiler writes a
trace and a table of kernels by CUDA time under --output-dir. The cuda profiler only marks
the profiled steps, for `nsys profile --capture-range=cudaProfilerApi` to record.

With --soak-minutes, the point is first measured over and over as in a sweep, so the
profile shows a process that has run that long.
"""

import argparse
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from huggingface_hub import HfApi
from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

from decode_sol.energy import open_nvml_device
from decode_sol.grid import GridPoint
from decode_sol.guard import ensure_gpu_is_free, parse_gpu_memory_utilization
from decode_sol.measure import (
    PROMPT_SEED,
    SweepConfig,
    load_llm,
    make_random_prompts,
    measure_point,
)
from decode_sol.step_drift import S_PER_MIN, compute_step_times_ms

logger = logging.getLogger(__name__)

ProfilerKind = Literal["torch", "cuda"]
SWEEP_GRID_NAME = "full"  # Same max_model_len, max_num_seqs and CUDA graph sizes as the sweep.
NUM_WARMUP_GENERATIONS = 2
NUM_PREFILL_ITERATIONS = 1  # Every profiled point's prompts fit in one prefill chunk.
NUM_PROFILED_DECODE_STEPS = 20
NUM_UNPROFILED_TAIL_STEPS = 4  # Keeps the last profiled step away from request teardown.


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    model: str
    kv_cache_dtype: str
    batch_size: int
    context_len: int
    profiler: ProfilerKind
    output_dir: Path
    gpu_memory_utilization: float
    soak_minutes: float

    @property
    def trace_prefix(self) -> str:
        model_name = self.model.split("/")[-1]
        return f"{model_name}_{self.kv_cache_dtype}_b{self.batch_size}_c{self.context_len}"


def parse_profile_config() -> ProfileConfig:
    parser = argparse.ArgumentParser(description="Profile vLLM decode at one grid point.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--profiler", choices=["torch", "cuda"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("profiles"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument(
        "--soak-minutes", type=float, default=0.0, help="measure the point this long first"
    )
    args = parser.parse_args()
    return ProfileConfig(
        model=args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        batch_size=args.batch_size,
        context_len=args.context_len,
        profiler=args.profiler,
        output_dir=args.output_dir.resolve(),
        gpu_memory_utilization=parse_gpu_memory_utilization(args.gpu_memory_utilization),
        soak_minutes=args.soak_minutes,
    )


def to_vllm_profiler_config(config: ProfileConfig) -> ProfilerConfig:
    """Stack capture off: it slows the CPU side, which is part of what is being measured."""
    match config.profiler:
        case "torch":
            return ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=str(config.output_dir / "torch" / config.trace_prefix),
                torch_profiler_with_stack=False,
                ignore_frontend=True,
            )
        case "cuda":
            return ProfilerConfig(profiler="cuda")
        case _:
            assert_never(config.profiler)


def soak_point(llm: LLM, config: ProfileConfig, *, num_token_ids: int) -> None:
    """Measure the point the way measure.py does, over and over, for config.soak_minutes."""
    point = GridPoint(batch_size=config.batch_size, context_len=config.context_len)
    start_s = time.monotonic()
    with open_nvml_device() as device:
        while (time.monotonic() - start_s) / S_PER_MIN < config.soak_minutes:
            raw_pairs = measure_point(llm, point, num_token_ids=num_token_ids, device=device)
            logger.info(
                "Soak: elapsed_min=%.1f median_step_ms=%.2f",
                (time.monotonic() - start_s) / S_PER_MIN,
                statistics.median(compute_step_times_ms(raw_pairs)),
            )


def main() -> None:
    config = parse_profile_config()
    logging.basicConfig(level=logging.INFO)

    ensure_gpu_is_free()
    revision = HfApi().model_info(config.model).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face returned no revision for {config.model!r}")
    sweep_config = SweepConfig(
        model=config.model,
        grid_name=SWEEP_GRID_NAME,
        kv_cache_dtype=config.kv_cache_dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        output=config.output_dir,  # load_llm never writes it.
        points=(),  # Nor measures any.
    )
    llm = load_llm(sweep_config, revision=revision, profiler_config=to_vllm_profiler_config(config))
    num_token_ids = llm.get_tokenizer().vocab_size
    soak_point(llm, config, num_token_ids=num_token_ids)
    prompts = make_random_prompts(
        batch_size=config.batch_size,
        context_len=config.context_len,
        num_token_ids=num_token_ids,
        seed=PROMPT_SEED,
    )
    num_output_tokens = NUM_PREFILL_ITERATIONS + NUM_PROFILED_DECODE_STEPS
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=num_output_tokens + NUM_UNPROFILED_TAIL_STEPS,
        ignore_eos=True,
    )
    for _ in range(NUM_WARMUP_GENERATIONS):
        llm.generate(prompts, sampling_params, use_tqdm=False)

    llm.start_profile(
        config.trace_prefix,
        delay_iterations=NUM_PREFILL_ITERATIONS,
        max_iterations=NUM_PROFILED_DECODE_STEPS,
    )
    llm.generate(prompts, sampling_params, use_tqdm=False)
    llm.stop_profile()
    logger.info(
        # vLLM 0.33 starts the profile at the prefill despite delay_iterations, so the traces
        # hold the prefill and one decode step fewer; profile_summary.py keeps only decode.
        "Profiled %d engine steps of %s with the %s profiler",
        NUM_PROFILED_DECODE_STEPS,
        config.trace_prefix,
        config.profiler,
    )


if __name__ == "__main__":
    main()
