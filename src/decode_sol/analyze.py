"""Decode-only step time and GPU energy from paired generations, compared with the ceiling.

Each measurement times the same batch at a short and a long output length. Both pay the
same prefill, so their difference is pure decode (docs/PROJECT_BRIEF.md, section 2).
"""

import argparse
import csv
import json
import logging
import statistics
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from decode_sol.bandwidth import BYTES_PER_GB, SPEC_PEAK_BANDWIDTH_GB_PER_S
from decode_sol.energy import GpuEnergyWindow
from decode_sol.roofline import (
    DecodeStepPrediction,
    compute_num_weight_params_read,
    parse_model_dims,
    predict_decode_step,
)

logger = logging.getLogger(__name__)

NUM_BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LOW_PERCENTILE = 2.5
CI_HIGH_PERCENTILE = 97.5
KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE = {"auto": 2, "fp8": 1}
SWEEP_GRID_NAME = "full"
MS_PER_S = 1e3


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


def compute_median_and_hierarchical_bootstrap_ci(
    samples_by_repeat: list[list[float]], *, seed: int
) -> MedianWithCi:
    """Median of the pooled samples, with a 95% CI that resamples repeats, then samples in each.

    Resampling whole repeats carries run-to-run variation (KV cache allocation, clocks) into
    the CI; a bootstrap over the pooled samples would hide it. With one repeat this is a plain
    percentile bootstrap. Raises ValueError unless every repeat has the same number of samples.
    """
    num_samples_by_repeat = [len(samples) for samples in samples_by_repeat]
    if len(set(num_samples_by_repeat)) != 1:
        raise ValueError(
            f"every repeat needs the same number of samples, got {num_samples_by_repeat}"
        )
    sample_array = np.asarray(samples_by_repeat)  # [num_repeats, num_samples], fp64
    num_repeats, num_samples = sample_array.shape

    rng = np.random.default_rng(seed)
    repeat_idxs = rng.integers(
        0, num_repeats, size=(NUM_BOOTSTRAP_RESAMPLES, num_repeats, 1)
    )  # [num_resamples, num_repeats, 1]
    sample_idxs = rng.integers(
        0, num_samples, size=(NUM_BOOTSTRAP_RESAMPLES, num_repeats, num_samples)
    )  # [num_resamples, num_repeats, num_samples]
    resampled = sample_array[repeat_idxs, sample_idxs]  # [num_resamples, num_repeats, num_samples]
    resampled_medians = np.median(
        resampled.reshape(NUM_BOOTSTRAP_RESAMPLES, -1), axis=1
    )  # [num_resamples]
    return MedianWithCi(
        median=float(np.median(sample_array)),
        ci_low=float(np.percentile(resampled_medians, CI_LOW_PERCENTILE)),
        ci_high=float(np.percentile(resampled_medians, CI_HIGH_PERCENTILE)),
    )


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """A grid point of one sweep. Each repeat of the sweep adds one record to it."""

    model: str
    kv_cache_dtype: str
    batch_size: int
    context_len: int


@dataclass(frozen=True, slots=True)
class PointSummary:
    """One row of results/summary.csv: all repeats of a grid point, pooled.

    Efficiency is the predicted step time at achievable bandwidth over the measured one;
    spec_efficiency uses the spec-peak bandwidth instead. CIs are 95% hierarchical bootstraps.
    """

    model: str
    kv_cache_dtype: str
    batch_size: int
    context_len: int
    effective_context_len: int
    num_repeats: int
    num_paired_generations: int
    num_weight_params: int
    weight_bytes: int
    kv_bytes: int
    predicted_step_time_ms: float
    predicted_tokens_per_s: float
    step_time_ms: float
    step_time_ms_ci_low: float
    step_time_ms_ci_high: float
    step_time_ms_min_repeat: float
    step_time_ms_max_repeat: float
    tokens_per_s: float
    efficiency: float
    efficiency_ci_low: float
    efficiency_ci_high: float
    spec_efficiency: float
    gpu_energy_per_token_j_with_idle: float
    gpu_energy_per_token_j_with_idle_ci_low: float
    gpu_energy_per_token_j_with_idle_ci_high: float
    gpu_energy_per_token_j_above_idle: float
    gpu_energy_per_token_j_above_idle_ci_low: float
    gpu_energy_per_token_j_above_idle_ci_high: float


def group_sweep_records_by_point(
    records: list[dict[str, object]],
) -> dict[SweepPoint, list[dict[str, object]]]:
    """Records of the full sweep grid, one per repeat, grouped by grid point.

    Smoke-grid records and the M2 record (which predates the grid field) are left out.
    """
    records_by_point: dict[SweepPoint, list[dict[str, object]]] = {}
    for record in records:
        if record.get("grid") != SWEEP_GRID_NAME:
            continue
        point = SweepPoint(
            model=str(record["model"]),
            kv_cache_dtype=str(record["kv_cache_dtype"]),
            batch_size=int(str(record["batch_size"])),
            context_len=int(str(record["context_len"])),
        )
        records_by_point.setdefault(point, []).append(record)
    return records_by_point


def predict_record_decode_step(
    record: dict[str, object], *, bandwidth_gb_per_s: float
) -> DecodeStepPrediction:
    model_config = record["model_config"]
    assert isinstance(model_config, dict)
    effective_context_len = compute_effective_context_len(
        context_len=int(str(record["context_len"])),
        short_output_len=int(str(record["short_output_len"])),
        long_output_len=int(str(record["long_output_len"])),
    )
    return predict_decode_step(
        parse_model_dims(model_config),
        batch_size=int(str(record["batch_size"])),
        context_len=effective_context_len,
        kv_element_bytes=KV_ELEMENT_BYTES_BY_KV_CACHE_DTYPE[str(record["kv_cache_dtype"])],
        bandwidth_bytes_per_s=bandwidth_gb_per_s * BYTES_PER_GB,
    )


def compute_point_summary(
    point: SweepPoint, records: list[dict[str, object]], *, achievable_gb_per_s: float
) -> PointSummary:
    """Pools every repeat of one grid point. All records must share the point's settings."""
    first_record = records[0]
    model_config = first_record["model_config"]
    assert isinstance(model_config, dict)
    predicted = predict_record_decode_step(first_record, bandwidth_gb_per_s=achievable_gb_per_s)
    predicted_at_spec = predict_record_decode_step(
        first_record, bandwidth_gb_per_s=SPEC_PEAK_BANDWIDTH_GB_PER_S
    )

    samples_by_repeat = [compute_decode_samples(record) for record in records]
    step_time_s = compute_median_and_hierarchical_bootstrap_ci(
        [samples.step_times_s for samples in samples_by_repeat], seed=BOOTSTRAP_SEED
    )
    energy_with_idle_j = compute_median_and_hierarchical_bootstrap_ci(
        [samples.energy_per_token_j_with_idle for samples in samples_by_repeat],
        seed=BOOTSTRAP_SEED,
    )
    energy_above_idle_j = compute_median_and_hierarchical_bootstrap_ci(
        [samples.energy_per_token_j_above_idle for samples in samples_by_repeat],
        seed=BOOTSTRAP_SEED,
    )
    repeat_median_step_times_s = [
        statistics.median(samples.step_times_s) for samples in samples_by_repeat
    ]

    return PointSummary(
        model=point.model,
        kv_cache_dtype=point.kv_cache_dtype,
        batch_size=point.batch_size,
        context_len=point.context_len,
        effective_context_len=compute_effective_context_len(
            context_len=point.context_len,
            short_output_len=int(str(first_record["short_output_len"])),
            long_output_len=int(str(first_record["long_output_len"])),
        ),
        num_repeats=len(records),
        num_paired_generations=sum(len(samples.step_times_s) for samples in samples_by_repeat),
        num_weight_params=compute_num_weight_params_read(parse_model_dims(model_config)),
        weight_bytes=predicted.weight_bytes,
        kv_bytes=predicted.kv_bytes,
        predicted_step_time_ms=MS_PER_S * predicted.step_time_s,
        predicted_tokens_per_s=predicted.tokens_per_s,
        step_time_ms=MS_PER_S * step_time_s.median,
        step_time_ms_ci_low=MS_PER_S * step_time_s.ci_low,
        step_time_ms_ci_high=MS_PER_S * step_time_s.ci_high,
        step_time_ms_min_repeat=MS_PER_S * min(repeat_median_step_times_s),
        step_time_ms_max_repeat=MS_PER_S * max(repeat_median_step_times_s),
        tokens_per_s=point.batch_size / step_time_s.median,
        efficiency=predicted.step_time_s / step_time_s.median,
        efficiency_ci_low=predicted.step_time_s / step_time_s.ci_high,
        efficiency_ci_high=predicted.step_time_s / step_time_s.ci_low,
        spec_efficiency=predicted_at_spec.step_time_s / step_time_s.median,
        gpu_energy_per_token_j_with_idle=energy_with_idle_j.median,
        gpu_energy_per_token_j_with_idle_ci_low=energy_with_idle_j.ci_low,
        gpu_energy_per_token_j_with_idle_ci_high=energy_with_idle_j.ci_high,
        gpu_energy_per_token_j_above_idle=energy_above_idle_j.median,
        gpu_energy_per_token_j_above_idle_ci_low=energy_above_idle_j.ci_low,
        gpu_energy_per_token_j_above_idle_ci_high=energy_above_idle_j.ci_high,
    )


def write_point_summaries(summaries: list[PointSummary], summary_path: Path) -> None:
    field_names = [field.name for field in fields(PointSummary)]
    with summary_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def load_point_summaries(summary_path: Path) -> list[PointSummary]:
    summaries: list[PointSummary] = []
    with summary_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            values: dict[str, Any] = {}
            for field in fields(PointSummary):
                parse_value = field.type  # str, int or float
                assert callable(parse_value)
                values[field.name] = parse_value(row[field.name])
            summaries.append(PointSummary(**values))
    return summaries


def load_measurement_records(measurements_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in measurements_path.read_text().splitlines():
        records.append(json.loads(line))
    return records


def compute_point_summaries(
    records: list[dict[str, object]], *, achievable_gb_per_s: float
) -> list[PointSummary]:
    """One summary per grid point, ordered by model size, KV cache dtype, batch and context."""
    summaries: list[PointSummary] = []
    for point, point_records in group_sweep_records_by_point(records).items():
        summaries.append(
            compute_point_summary(point, point_records, achievable_gb_per_s=achievable_gb_per_s)
        )
    summaries.sort(
        key=lambda summary: (
            summary.num_weight_params,
            summary.kv_cache_dtype,
            summary.batch_size,
            summary.context_len,
        )
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pool the sweep's repeats into results/summary.csv."
    )
    parser.add_argument("--measurements", type=Path, default=Path("results/raw/decode.jsonl"))
    parser.add_argument("--bandwidth", type=Path, default=Path("results/raw/bandwidth.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    achievable_gb_per_s = load_achievable_bandwidth_gb_per_s(args.bandwidth)
    records = load_measurement_records(args.measurements)
    summaries = compute_point_summaries(records, achievable_gb_per_s=achievable_gb_per_s)
    write_point_summaries(summaries, args.summary)
    logger.info("Wrote %d grid points to %s", len(summaries), args.summary)


if __name__ == "__main__":
    main()
