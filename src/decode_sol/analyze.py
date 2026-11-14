"""Decode-only step time and GPU energy from paired generations, compared with the ceiling.

Each measurement times the same batch at a short and a long output length. Both pay the
same prefill, so their difference is pure decode (docs/PROJECT_BRIEF.md, section 2).
"""

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decode_sol.bandwidth import BYTES_PER_GB, SPEC_PEAK_BANDWIDTH_GB_PER_S
from decode_sol.energy import GpuEnergyWindow
from decode_sol.roofline import parse_model_dims, predict_decode_step

logger = logging.getLogger(__name__)

NUM_BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LOW_PERCENTILE = 2.5
CI_HIGH_PERCENTILE = 97.5
KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE = {"auto": 2, "fp8": 1}


@dataclass(frozen=True, slots=True)
class PairedGeneration:
    """One timed generation at the short output length and one at the long, same prompts."""

    short_duration_s: float
    long_duration_s: float
    short_energy_j: float
    long_energy_j: float


@dataclass(frozen=True, slots=True)
class MedianWithCi:
    median: float
    ci_low: float
    ci_high: float


def compute_effective_context_len(
    *, context_len: int, short_output_len: int, long_output_len: int
) -> int:
    """Mean number of tokens in each sequence's KV cache over the differenced decode steps."""
    return context_len + (short_output_len + long_output_len) // 2


def compute_decode_step_time_s(
    pair: PairedGeneration, *, short_output_len: int, long_output_len: int
) -> float:
    num_extra_steps = long_output_len - short_output_len
    return (pair.long_duration_s - pair.short_duration_s) / num_extra_steps


def compute_decode_energy_per_token_j(
    pair: PairedGeneration,
    *,
    batch_size: int,
    short_output_len: int,
    long_output_len: int,
    idle_power_w: float,
) -> float:
    """GPU energy per decoded token. Pass idle_power_w=0 to keep the idle baseline in."""
    extra_energy_j = pair.long_energy_j - pair.short_energy_j
    extra_duration_s = pair.long_duration_s - pair.short_duration_s
    active_energy_j = extra_energy_j - idle_power_w * extra_duration_s
    num_extra_tokens = batch_size * (long_output_len - short_output_len)
    return active_energy_j / num_extra_tokens


def compute_median_and_bootstrap_ci(samples: list[float], *, seed: int) -> MedianWithCi:
    """Median and a percentile-bootstrap 95% confidence interval of the median."""
    rng = np.random.default_rng(seed)
    sample_array = np.asarray(samples)  # [num_samples], fp64
    resample_idxs = rng.integers(
        0, len(samples), size=(NUM_BOOTSTRAP_RESAMPLES, len(samples))
    )  # [num_resamples, num_samples]
    resampled_medians = np.median(sample_array[resample_idxs], axis=1)  # [num_resamples]
    return MedianWithCi(
        median=statistics.median(samples),
        ci_low=float(np.percentile(resampled_medians, CI_LOW_PERCENTILE)),
        ci_high=float(np.percentile(resampled_medians, CI_HIGH_PERCENTILE)),
    )


def to_paired_generation(raw_pair: dict[str, dict[str, int]]) -> PairedGeneration:
    short_window = GpuEnergyWindow(**raw_pair["short"])
    long_window = GpuEnergyWindow(**raw_pair["long"])
    return PairedGeneration(
        short_duration_s=short_window.duration_s,
        long_duration_s=long_window.duration_s,
        short_energy_j=short_window.energy_j,
        long_energy_j=long_window.energy_j,
    )


def load_achievable_bandwidth_gb_per_s(bandwidth_path: Path) -> float:
    """Median over repeats of the most recent bandwidth benchmark."""
    lines = bandwidth_path.read_text().splitlines()
    latest_record = json.loads(lines[-1])
    median_bandwidth_gb_per_s: float = statistics.median(
        latest_record["median_bandwidth_gb_per_s_by_repeat"]
    )
    return median_bandwidth_gb_per_s


@dataclass(frozen=True, slots=True)
class DecodeSamples:
    """One value per paired generation."""

    step_times_s: list[float]
    energy_per_token_j_with_idle: list[float]
    energy_per_token_j_above_idle: list[float]


def compute_decode_samples(record: dict[str, object]) -> DecodeSamples:
    batch_size = int(str(record["batch_size"]))
    short_output_len = int(str(record["short_output_len"]))
    long_output_len = int(str(record["long_output_len"]))
    idle_power_w = float(str(record["idle_gpu_power_w"]))
    raw_pairs = record["paired_generations"]
    assert isinstance(raw_pairs, list)

    samples = DecodeSamples([], [], [])
    for raw_pair in raw_pairs:
        pair = to_paired_generation(raw_pair)
        samples.step_times_s.append(
            compute_decode_step_time_s(
                pair, short_output_len=short_output_len, long_output_len=long_output_len
            )
        )
        samples.energy_per_token_j_with_idle.append(
            compute_decode_energy_per_token_j(
                pair,
                batch_size=batch_size,
                short_output_len=short_output_len,
                long_output_len=long_output_len,
                idle_power_w=0.0,
            )
        )
        samples.energy_per_token_j_above_idle.append(
            compute_decode_energy_per_token_j(
                pair,
                batch_size=batch_size,
                short_output_len=short_output_len,
                long_output_len=long_output_len,
                idle_power_w=idle_power_w,
            )
        )
    return samples


def log_measurement_summary(record: dict[str, object], achievable_gb_per_s: float) -> None:
    batch_size = int(str(record["batch_size"]))
    model_config = record["model_config"]
    assert isinstance(model_config, dict)
    effective_context_len = compute_effective_context_len(
        context_len=int(str(record["context_len"])),
        short_output_len=int(str(record["short_output_len"])),
        long_output_len=int(str(record["long_output_len"])),
    )
    samples = compute_decode_samples(record)
    step_time = compute_median_and_bootstrap_ci(samples.step_times_s, seed=BOOTSTRAP_SEED)

    logger.info(
        "%s batch_size=%d context_len=%s kv_cache_dtype=%s effective_context_len=%d n=%d",
        record["model"],
        batch_size,
        record["context_len"],
        record["kv_cache_dtype"],
        effective_context_len,
        len(samples.step_times_s),
    )
    logger.info(
        "measured step_time_ms=%.3f (95%% CI %.3f-%.3f) tokens_per_s=%.1f",
        1e3 * step_time.median,
        1e3 * step_time.ci_low,
        1e3 * step_time.ci_high,
        batch_size / step_time.median,
    )
    bandwidth_gb_per_s_by_label = {
        "spec": SPEC_PEAK_BANDWIDTH_GB_PER_S,
        "achievable": achievable_gb_per_s,
    }
    for label, bandwidth_gb_per_s in bandwidth_gb_per_s_by_label.items():
        prediction = predict_decode_step(
            parse_model_dims(model_config),
            batch_size=batch_size,
            context_len=effective_context_len,
            kv_element_bytes=KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE[str(record["kv_cache_dtype"])],
            bandwidth_bytes_per_s=bandwidth_gb_per_s * BYTES_PER_GB,
        )
        logger.info(
            "ceiling (%s, %.1f GB/s) step_time_ms=%.3f tokens_per_s=%.1f efficiency=%.3f",
            label,
            bandwidth_gb_per_s,
            1e3 * prediction.step_time_s,
            prediction.tokens_per_s,
            prediction.step_time_s / step_time.median,
        )
    energy_samples_by_label = {
        "with idle": samples.energy_per_token_j_with_idle,
        "above idle": samples.energy_per_token_j_above_idle,
    }
    for label, energies_j in energy_samples_by_label.items():
        energy = compute_median_and_bootstrap_ci(energies_j, seed=BOOTSTRAP_SEED)
        logger.info(
            "GPU energy per token (%s) mj=%.2f (95%% CI %.2f-%.2f)",
            label,
            1e3 * energy.median,
            1e3 * energy.ci_low,
            1e3 * energy.ci_high,
        )
    logger.info("idle GPU power_w=%.2f", record["idle_gpu_power_w"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize raw decode measurements.")
    parser.add_argument("--measurements", type=Path, default=Path("results/raw/decode.jsonl"))
    parser.add_argument("--bandwidth", type=Path, default=Path("results/raw/bandwidth.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    achievable_gb_per_s = load_achievable_bandwidth_gb_per_s(args.bandwidth)
    for line in args.measurements.read_text().splitlines():
        log_measurement_summary(json.loads(line), achievable_gb_per_s)


if __name__ == "__main__":
    main()
