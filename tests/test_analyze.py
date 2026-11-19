import json
from pathlib import Path

import pytest

from decode_sol.analyze import (
    SWEEP_GRID_NAME,
    PairedGeneration,
    SweepPoint,
    compute_decode_energy_per_token_j,
    compute_decode_step_time_s,
    compute_effective_context_len,
    compute_median_and_hierarchical_bootstrap_ci,
    compute_point_summary,
    group_sweep_records_by_point,
    load_point_summaries,
    write_point_summaries,
)

# A batch of 2, generating 16 then 144 tokens per sequence: 128 extra decode steps, 256 extra
# tokens. The long run takes 1.28 s more and uses 12.8 J more, so a step is 10 ms and the
# GPU averaged 10 W over the extra time.
HAND_PAIR = PairedGeneration(
    short_duration_s=0.5,
    long_duration_s=1.78,
    short_energy_j=4.0,
    long_energy_j=16.8,
)


def test_decode_step_time_is_extra_time_over_extra_steps() -> None:
    step_time_s = compute_decode_step_time_s(HAND_PAIR, short_output_len=16, long_output_len=144)

    assert step_time_s == pytest.approx(0.01)


def test_decode_energy_per_token_divides_by_every_sequence_in_the_batch() -> None:
    energy_per_token_j = compute_decode_energy_per_token_j(
        HAND_PAIR, batch_size=2, short_output_len=16, long_output_len=144, idle_power_w=0.0
    )

    assert energy_per_token_j == pytest.approx(12.8 / 256)


def test_decode_energy_per_token_subtracts_idle_power_over_the_extra_time() -> None:
    # Idle 5 W over the extra 1.28 s is 6.4 J, leaving 6.4 J for 256 tokens.
    energy_per_token_j = compute_decode_energy_per_token_j(
        HAND_PAIR, batch_size=2, short_output_len=16, long_output_len=144, idle_power_w=5.0
    )

    assert energy_per_token_j == pytest.approx(6.4 / 256)


def test_effective_context_is_prompt_plus_mean_of_output_lengths() -> None:
    effective_context_len = compute_effective_context_len(
        context_len=512, short_output_len=16, long_output_len=144
    )

    assert effective_context_len == 592


def test_median_and_ci_of_identical_samples_collapse_to_that_value() -> None:
    summary = compute_median_and_hierarchical_bootstrap_ci(
        [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], seed=0
    )

    assert summary.median == 3.0
    assert summary.ci_low == 3.0
    assert summary.ci_high == 3.0


def test_bootstrap_ci_of_one_repeat_brackets_the_median_and_stays_inside_the_samples() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 10.0]

    summary = compute_median_and_hierarchical_bootstrap_ci([samples], seed=0)

    assert summary.median == 3.0
    assert 1.0 <= summary.ci_low <= 3.0 <= summary.ci_high <= 10.0
    assert summary.ci_low < summary.ci_high


def test_bootstrap_ci_is_deterministic_for_a_seed() -> None:
    samples_by_repeat = [[1.0, 2.0, 3.0], [4.0, 10.0, 2.0]]

    first = compute_median_and_hierarchical_bootstrap_ci(samples_by_repeat, seed=7)
    second = compute_median_and_hierarchical_bootstrap_ci(samples_by_repeat, seed=7)

    assert first == second


def test_hierarchical_ci_includes_the_spread_between_repeats() -> None:
    # Each repeat is internally constant, so all the uncertainty is between repeats. A
    # bootstrap over the 30 pooled samples would give a CI of [2, 2].
    samples_by_repeat = [[1.0] * 10, [2.0] * 10, [3.0] * 10]

    summary = compute_median_and_hierarchical_bootstrap_ci(samples_by_repeat, seed=0)

    assert summary.median == 2.0
    assert summary.ci_low == 1.0
    assert summary.ci_high == 3.0


def test_hierarchical_ci_rejects_repeats_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="same number of samples"):
        compute_median_and_hierarchical_bootstrap_ci([[1.0, 2.0], [1.0]], seed=0)


QWEN3_06B_CONFIG_PATH = Path(__file__).parent / "fixtures" / "qwen3-0.6b-config.json"
ACHIEVABLE_GB_PER_S = 265.0
NS_PER_S = 1_000_000_000
MJ_PER_J = 1_000


def make_raw_pair(pair: PairedGeneration) -> dict[str, dict[str, int]]:
    """The NVML windows of one paired generation, as measure.py writes them."""
    short_end_ns = round(pair.short_duration_s * NS_PER_S)
    short_end_mj = round(pair.short_energy_j * MJ_PER_J)
    return {
        "short": {
            "start_energy_mj": 0,
            "end_energy_mj": short_end_mj,
            "start_time_ns": 0,
            "end_time_ns": short_end_ns,
        },
        "long": {
            "start_energy_mj": short_end_mj,
            "end_energy_mj": short_end_mj + round(pair.long_energy_j * MJ_PER_J),
            "start_time_ns": short_end_ns,
            "end_time_ns": short_end_ns + round(pair.long_duration_s * NS_PER_S),
        },
    }


def make_record(*, grid: str | None = SWEEP_GRID_NAME, num_pairs: int = 4) -> dict[str, object]:
    """A batch-2, context-512 record of HAND_PAIR repeated, with 5 W idle power."""
    record: dict[str, object] = {
        "model": "Qwen/Qwen3-0.6B",
        "model_config": json.loads(QWEN3_06B_CONFIG_PATH.read_text()),
        "kv_cache_dtype": "auto",
        "batch_size": 2,
        "context_len": 512,
        "short_output_len": 16,
        "long_output_len": 144,
        "idle_gpu_power_w": 5.0,
        "paired_generations": [make_raw_pair(HAND_PAIR)] * num_pairs,
    }
    if grid is not None:
        record["grid"] = grid
    return record


def test_grouping_keeps_only_full_sweep_records_and_pools_repeats_of_a_point() -> None:
    records = [make_record(), make_record(), make_record(grid="smoke"), make_record(grid=None)]

    records_by_point = group_sweep_records_by_point(records)

    point = SweepPoint(
        model="Qwen/Qwen3-0.6B", kv_cache_dtype="auto", batch_size=2, context_len=512
    )
    assert list(records_by_point) == [point]
    assert len(records_by_point[point]) == 2


def test_point_summary_divides_predicted_by_measured_step_time() -> None:
    records = [make_record(), make_record()]
    point = SweepPoint(
        model="Qwen/Qwen3-0.6B", kv_cache_dtype="auto", batch_size=2, context_len=512
    )

    summary = compute_point_summary(point, records, achievable_gb_per_s=ACHIEVABLE_GB_PER_S)

    assert summary.num_repeats == 2
    assert summary.num_paired_generations == 8
    assert summary.step_time_ms == pytest.approx(10.0)
    assert summary.tokens_per_s == pytest.approx(200.0)
    assert summary.efficiency == pytest.approx(summary.predicted_step_time_ms / 10.0)
    assert summary.spec_efficiency < summary.efficiency
    assert summary.gpu_energy_per_token_j_with_idle == pytest.approx(12.8 / 256)
    assert summary.gpu_energy_per_token_j_above_idle == pytest.approx(6.4 / 256)


def test_point_summaries_survive_a_csv_round_trip(tmp_path: Path) -> None:
    point = SweepPoint(
        model="Qwen/Qwen3-0.6B", kv_cache_dtype="auto", batch_size=2, context_len=512
    )
    summary = compute_point_summary(point, [make_record()], achievable_gb_per_s=ACHIEVABLE_GB_PER_S)
    summary_path = tmp_path / "summary.csv"

    write_point_summaries([summary], summary_path)

    assert load_point_summaries(summary_path) == [summary]
