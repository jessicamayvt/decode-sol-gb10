"""Measure one grid point in a fresh process, to tell a slow point from a slow process.

A sweep measures its last points over an hour into the process. Each mode loads the model
the way measure.py does, then:

  fresh     measures the point once.
  scramble  first runs one short generation at every earlier feasible grid point, in sweep
            order, so the KV cache blocks are handed out as scattered as in a sweep, then
            measures the point.
  soak      measures the point over and over for --minutes.

Each measurement appends one record with the step time of every timed pair, the GPU power
of the long generations, the GPU temperature, SM clock and clock-event reasons before and
after, and the node's load: other users' CPU jobs share the GPU's memory bus. A run takes
15-25 minutes for fresh and scramble, longer for soak, so wrap it in `timeout` with room
to spare (see docs/REPORT.md for the runs behind
results/raw/fp8_gap_check.jsonl).
"""

import argparse
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pynvml
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams

from decode_sol.analyze import compute_decode_step_time_s, to_paired_generation
from decode_sol.energy import open_nvml_device
from decode_sol.grid import (
    LONG_OUTPUT_LEN,
    NUM_WARMUP_PAIRS,
    SHORT_OUTPUT_LEN,
    GridPoint,
    find_skip_reason,
)
from decode_sol.guard import (
    collect_run_metadata,
    compute_cpu_busy_fraction,
    ensure_gpu_is_free,
    load_node_load,
    parse_gpu_memory_utilization,
)
from decode_sol.measure import (
    PROMPT_SEED,
    SweepConfig,
    append_record,
    load_llm,
    make_random_prompts,
    measure_point,
)

logger = logging.getLogger(__name__)

MS_PER_S = 1e3
S_PER_MIN = 60.0


class DriftMode(Enum):
    FRESH = "fresh"
    SCRAMBLE = "scramble"
    SOAK = "soak"


@dataclass(frozen=True, slots=True)
class GpuState:
    temp_c: int
    sm_clock_mhz: int
    throttle_reasons: int  # NVML clock-event reasons bitmask; 0 means none


@dataclass(frozen=True, slots=True)
class DriftConfig:
    sweep: SweepConfig
    point: GridPoint
    mode: DriftMode
    soak_minutes: float


def load_gpu_state(device: object) -> GpuState:
    return GpuState(
        temp_c=pynvml.nvmlDeviceGetTemperature(device, pynvml.NVML_TEMPERATURE_GPU),
        sm_clock_mhz=pynvml.nvmlDeviceGetClockInfo(device, pynvml.NVML_CLOCK_SM),
        throttle_reasons=pynvml.nvmlDeviceGetCurrentClocksEventReasons(device),
    )


def compute_step_times_ms(raw_pairs: list[dict[str, dict[str, int]]]) -> list[float]:
    step_times_ms: list[float] = []
    for raw_pair in raw_pairs:
        pair = to_paired_generation(raw_pair)
        step_time_s = compute_decode_step_time_s(
            pair, short_output_len=SHORT_OUTPUT_LEN, long_output_len=LONG_OUTPUT_LEN
        )
        step_times_ms.append(MS_PER_S * step_time_s)
    return step_times_ms


def compute_long_generation_power_w(raw_pairs: list[dict[str, dict[str, int]]]) -> list[float]:
    power_w: list[float] = []
    for raw_pair in raw_pairs:
        pair = to_paired_generation(raw_pair)
        power_w.append(pair.long_energy_j / pair.long_duration_s)
    return power_w


def find_earlier_feasible_points(llm: LLM, config: DriftConfig) -> list[GridPoint]:
    """The feasible grid points a sweep measures before config.point, in sweep order."""
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    if num_gpu_blocks is None:
        raise RuntimeError("vLLM did not report its KV cache size")
    earlier_points: list[GridPoint] = []
    for point in config.sweep.grid.points:
        if point == config.point:
            return earlier_points
        skip_reason = find_skip_reason(
            point,
            num_gpu_blocks=num_gpu_blocks,
            block_size=cache_config.block_size,
            max_num_batched_tokens=config.sweep.grid.max_num_batched_tokens,
        )
        if skip_reason is None:
            earlier_points.append(point)
    raise ValueError(f"{config.point} is not in grid {config.sweep.grid_name!r}")


def scramble_kv_cache_blocks(llm: LLM, config: DriftConfig, *, num_token_ids: int) -> None:
    """Run one short generation at every feasible grid point before config.point."""
    sampling_params = SamplingParams(temperature=0.0, max_tokens=SHORT_OUTPUT_LEN, ignore_eos=True)
    for point in find_earlier_feasible_points(llm, config):
        prompts = make_random_prompts(
            batch_size=point.batch_size,
            context_len=point.context_len,
            num_token_ids=num_token_ids,
            seed=PROMPT_SEED,
        )
        llm.generate(prompts, sampling_params, use_tqdm=False)
        logger.info("Scrambled with %s", point)


def measure_and_append(
    llm: LLM,
    config: DriftConfig,
    *,
    device: object,
    num_token_ids: int,
    process_metadata: dict[str, object],
    process_start_s: float,
) -> None:
    gpu_state_before = load_gpu_state(device)
    node_load_before = load_node_load()
    raw_pairs = measure_point(llm, config.point, num_token_ids=num_token_ids, device=device)
    gpu_state_after = load_gpu_state(device)
    node_load_after = load_node_load()
    cpu_busy_fraction = compute_cpu_busy_fraction(node_load_before, node_load_after)
    step_times_ms = compute_step_times_ms(raw_pairs)
    median_step_ms = statistics.median(step_times_ms)
    elapsed_min = (time.monotonic() - process_start_s) / S_PER_MIN
    append_record(
        {
            **process_metadata,
            "mode": config.mode.value,
            "elapsed_min": elapsed_min,
            "median_step_ms": median_step_ms,
            "step_ms": step_times_ms,
            "median_long_power_w": statistics.median(compute_long_generation_power_w(raw_pairs)),
            "gpu_state_before": asdict(gpu_state_before),
            "gpu_state_after": asdict(gpu_state_after),
            "node_load_before": asdict(node_load_before),
            "node_load_after": asdict(node_load_after),
            "node_cpu_busy_fraction": cpu_busy_fraction,
        },
        config.sweep.output,
    )
    logger.info(
        "mode=%s elapsed_min=%.1f median_step_ms=%.2f temp_c=%d sm_clock_mhz=%d"
        " node_cpu_busy_fraction=%.3f",
        config.mode.value,
        elapsed_min,
        median_step_ms,
        gpu_state_after.temp_c,
        gpu_state_after.sm_clock_mhz,
        cpu_busy_fraction,
    )


def parse_drift_config() -> DriftConfig:
    parser = argparse.ArgumentParser(
        description="Measure one grid point fresh, scrambled or soaked."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in DriftMode], required=True)
    parser.add_argument("--minutes", type=float, default=60.0, help="soak duration")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("results/raw/fp8_gap_check.jsonl"))
    args = parser.parse_args()
    point = GridPoint(batch_size=args.batch_size, context_len=args.context_len)
    sweep = SweepConfig(
        model=args.model,
        grid_name="full",
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=parse_gpu_memory_utilization(args.gpu_memory_utilization),
        output=args.output,
        points=(point,),
    )
    if point not in sweep.grid.points:
        parser.error(f"{point} is not in the full grid")
    return DriftConfig(
        sweep=sweep, point=point, mode=DriftMode(args.mode), soak_minutes=args.minutes
    )


def collect_process_metadata(config: DriftConfig) -> dict[str, object]:
    return {
        **collect_run_metadata(),
        "model": config.sweep.model,
        "kv_cache_dtype": config.sweep.kv_cache_dtype,
        "grid": config.sweep.grid_name,
        "gpu_memory_utilization": config.sweep.gpu_memory_utilization,
        "batch_size": config.point.batch_size,
        "context_len": config.point.context_len,
        "short_output_len": SHORT_OUTPUT_LEN,
        "long_output_len": LONG_OUTPUT_LEN,
        "num_warmup_pairs": NUM_WARMUP_PAIRS,
        "measured_with": "decode_sol.step_drift",
        "process_start_local": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main() -> None:
    config = parse_drift_config()
    logging.basicConfig(level=logging.INFO)

    ensure_gpu_is_free()
    process_metadata = collect_process_metadata(config)
    revision = HfApi().model_info(config.sweep.model).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face returned no revision for {config.sweep.model!r}")
    llm = load_llm(config.sweep, revision=revision)
    num_token_ids = llm.get_tokenizer().vocab_size  # ordinary tokens only, no special tokens
    process_start_s = time.monotonic()

    with open_nvml_device() as device:
        if config.mode is DriftMode.SCRAMBLE:
            scramble_kv_cache_blocks(llm, config, num_token_ids=num_token_ids)
        while True:
            measure_and_append(
                llm,
                config,
                device=device,
                num_token_ids=num_token_ids,
                process_metadata=process_metadata,
                process_start_s=process_start_s,
            )
            elapsed_min = (time.monotonic() - process_start_s) / S_PER_MIN
            if config.mode is not DriftMode.SOAK or elapsed_min >= config.soak_minutes:
                return


if __name__ == "__main__":
    main()
