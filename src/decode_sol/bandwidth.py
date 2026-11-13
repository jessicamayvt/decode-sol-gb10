"""Achievable read bandwidth: the honest ceiling for decode.

Decode streams weights and KV cache from memory and writes almost nothing, so the
benchmark is a read-only reduction over a buffer far larger than L2.
"""

import argparse
import json
import logging
import statistics
from pathlib import Path

import torch

from decode_sol.guard import collect_run_metadata, ensure_gpu_is_free

logger = logging.getLogger(__name__)

READ_BUFFER_BYTES = 4 * 2**30
READ_BUFFER_SEED = 0
NUM_WARMUP_ITERS = 10
NUM_TIMED_ITERS = 50
NUM_REPEATS = 3
SPEC_PEAK_BANDWIDTH_GB_PER_S = 273.0
BYTES_PER_GB = 1e9
MILLISECONDS_PER_SECOND = 1e3


def compute_bandwidth_gb_per_s(num_bytes: int, time_ms: float) -> float:
    return num_bytes / BYTES_PER_GB / (time_ms / MILLISECONDS_PER_SECOND)


def measure_read_times_ms(
    read_buffer: torch.Tensor, *, num_warmup: int, num_timed: int
) -> list[float]:
    """Time `read_buffer.sum()` with CUDA events, one pair per iteration.

    read_buffer: [num_elements], bf16, on the GPU. sum accumulates in fp32 internally.
    Synchronizes once, after the last timed iteration.
    """
    for _ in range(num_warmup):
        read_buffer.sum()

    start_events: list[torch.cuda.Event] = []
    end_events: list[torch.cuda.Event] = []
    for _ in range(num_timed):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        read_buffer.sum()
        end_event.record()
        start_events.append(start_event)
        end_events.append(end_event)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for start_event, end_event in zip(start_events, end_events, strict=True):
        times_ms.append(start_event.elapsed_time(end_event))
    return times_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure achievable GPU read bandwidth.")
    parser.add_argument("--output", type=Path, default=Path("results/raw/bandwidth.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ensure_gpu_is_free()
    record = collect_run_metadata()

    num_elements = READ_BUFFER_BYTES // torch.bfloat16.itemsize
    with torch.inference_mode():
        # [num_elements], bf16. Random values, so memory compression cannot shrink the bytes
        # read, and written, not just allocated, so every unified-memory page is backed.
        torch.manual_seed(READ_BUFFER_SEED)
        read_buffer = torch.randn(num_elements, dtype=torch.bfloat16, device="cuda")
        times_ms_by_repeat: list[list[float]] = []
        for _ in range(NUM_REPEATS):
            times_ms = measure_read_times_ms(
                read_buffer, num_warmup=NUM_WARMUP_ITERS, num_timed=NUM_TIMED_ITERS
            )
            times_ms_by_repeat.append(times_ms)
        del read_buffer

    median_bandwidth_gb_per_s_by_repeat: list[float] = []
    for times_ms in times_ms_by_repeat:
        median_time_ms = statistics.median(times_ms)
        median_bandwidth_gb_per_s_by_repeat.append(
            compute_bandwidth_gb_per_s(READ_BUFFER_BYTES, median_time_ms)
        )

    record["read_buffer_bytes"] = READ_BUFFER_BYTES
    record["read_buffer_seed"] = READ_BUFFER_SEED
    record["num_warmup_iters"] = NUM_WARMUP_ITERS
    record["times_ms_by_repeat"] = times_ms_by_repeat
    record["median_bandwidth_gb_per_s_by_repeat"] = median_bandwidth_gb_per_s_by_repeat
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output_file:
        output_file.write(json.dumps(record) + "\n")

    for repeat_idx, bandwidth_gb_per_s in enumerate(median_bandwidth_gb_per_s_by_repeat):
        logger.info(
            "Repeat %d: median read bandwidth %.1f GB/s (%.1f%% of %.0f GB/s spec)",
            repeat_idx,
            bandwidth_gb_per_s,
            100 * bandwidth_gb_per_s / SPEC_PEAK_BANDWIDTH_GB_PER_S,
            SPEC_PEAK_BANDWIDTH_GB_PER_S,
        )


if __name__ == "__main__":
    main()
