from pathlib import Path

import pytest

from decode_sol.energy import IDLE_DURATION_S
from decode_sol.grid import (
    ASSUMED_DECODE_EFFICIENCY,
    ASSUMED_PREFILL_FLOPS_PER_S,
    ESTIMATED_PROCESS_STARTUP_S,
    GRID_BY_NAME,
    LONG_OUTPUT_LEN,
    NUM_TIMED_PAIRS,
    NUM_WARMUP_PAIRS,
    SHORT_OUTPUT_LEN,
    GridPoint,
    estimate_num_gpu_blocks,
    estimate_point_duration_s,
    find_skip_reason,
    plan_sweep,
)
from decode_sol.roofline import ModelDims, load_model_dims

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Every dimension 1: 13 parameters read per step, 2 KV elements per token per layer.
UNIT_DIMS = ModelDims(
    num_layers=1,
    hidden_size=1,
    num_attention_heads=1,
    num_kv_heads=1,
    head_dim=1,
    intermediate_size=1,
    vocab_size=1,
)


def test_full_grid_has_every_batch_size_at_every_context_len() -> None:
    points = GRID_BY_NAME["full"].points

    assert len(points) == 7 * 4
    assert len(set(points)) == len(points)
    assert points[0] == GridPoint(batch_size=1, context_len=512)
    assert points[-1] == GridPoint(batch_size=64, context_len=32768)


def test_smoke_grid_is_the_m2_point() -> None:
    assert GRID_BY_NAME["smoke"].points == [GridPoint(batch_size=1, context_len=512)]


def test_point_fits_when_its_blocks_exactly_match_the_cache() -> None:
    # Each sequence holds 512 + 144 tokens = 41 blocks of 16.
    point = GridPoint(batch_size=2, context_len=512)

    skip_reason = find_skip_reason(
        point, num_gpu_blocks=82, block_size=16, max_num_batched_tokens=32768
    )

    assert skip_reason is None


def test_point_is_skipped_when_one_block_short() -> None:
    point = GridPoint(batch_size=2, context_len=512)

    skip_reason = find_skip_reason(
        point, num_gpu_blocks=81, block_size=16, max_num_batched_tokens=32768
    )

    assert skip_reason == "needs 82 KV cache blocks, 81 available"


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "expected_skip_reason"),
    [
        (8192, "prefill takes 64 steps, more than the 16-token short generation"),
        (16384, "prefill takes 32 steps, more than the 16-token short generation"),
        (32768, None),
    ],
)
def test_point_is_skipped_only_when_prefill_outlasts_the_short_generation(
    max_num_batched_tokens: int, expected_skip_reason: str | None
) -> None:
    point = GridPoint(batch_size=64, context_len=8192)

    skip_reason = find_skip_reason(
        point,
        num_gpu_blocks=10**6,
        block_size=16,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    assert skip_reason == expected_skip_reason


def test_fp8_kv_cache_doubles_the_estimated_blocks() -> None:
    dims = load_model_dims(FIXTURES_DIR / "qwen3-0.6b-config.json")

    bf16_blocks = estimate_num_gpu_blocks(dims, kv_element_bytes=2, block_size=16)
    fp8_blocks = estimate_num_gpu_blocks(dims, kv_element_bytes=1, block_size=16)

    assert bf16_blocks > 0
    assert fp8_blocks in (2 * bf16_blocks, 2 * bf16_blocks + 1)


def test_estimated_blocks_never_go_negative_for_a_model_too_large_for_the_cap() -> None:
    too_large = ModelDims(
        num_layers=1000,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=32768,
        vocab_size=151936,
    )

    assert estimate_num_gpu_blocks(too_large, kv_element_bytes=2, block_size=16) == 0


def test_point_duration_matches_hand_computation_for_unit_dims() -> None:
    point = GridPoint(batch_size=1, context_len=1)
    bandwidth_bytes_per_s = 1.0

    duration_s = estimate_point_duration_s(
        point, UNIT_DIMS, kv_element_bytes=2, bandwidth_bytes_per_s=bandwidth_bytes_per_s
    )

    prefill_flops = 2 * 13 * 1 + 2 * 1 * 1 * 1 * 1**2
    prefill_s = prefill_flops / ASSUMED_PREFILL_FLOPS_PER_S
    effective_context_len = 1 + (SHORT_OUTPUT_LEN + LONG_OUTPUT_LEN) // 2
    ceiling_step_s = (13 * 2 + effective_context_len * 2 * 2) / bandwidth_bytes_per_s
    step_s = ceiling_step_s / ASSUMED_DECODE_EFFICIENCY
    pair_s = 2 * prefill_s + (SHORT_OUTPUT_LEN + LONG_OUTPUT_LEN) * step_s
    assert duration_s == pytest.approx((NUM_WARMUP_PAIRS + NUM_TIMED_PAIRS) * pair_s)


def test_point_duration_grows_with_batch_size_and_context_len() -> None:
    dims = load_model_dims(FIXTURES_DIR / "qwen3-0.6b-config.json")
    bandwidth_bytes_per_s = 265e9

    def duration_s(batch_size: int, context_len: int) -> float:
        point = GridPoint(batch_size=batch_size, context_len=context_len)
        return estimate_point_duration_s(
            point, dims, kv_element_bytes=2, bandwidth_bytes_per_s=bandwidth_bytes_per_s
        )

    assert duration_s(1, 512) < duration_s(2, 512) < duration_s(2, 2048)


def test_sweep_plan_skips_the_largest_point_and_sums_only_feasible_durations() -> None:
    dims = load_model_dims(FIXTURES_DIR / "qwen3-0.6b-config.json")

    plan = plan_sweep(
        GRID_BY_NAME["full"],
        dims,
        model="Qwen/Qwen3-0.6B",
        kv_cache_dtype="auto",
        bandwidth_bytes_per_s=265e9,
    )

    largest = GridPoint(batch_size=64, context_len=32768)
    assert plan.skip_reason_by_point[largest] is not None
    assert plan.skip_reason_by_point[GridPoint(batch_size=1, context_len=512)] is None
    expected_s = ESTIMATED_PROCESS_STARTUP_S + IDLE_DURATION_S
    for point in plan.feasible_points:
        expected_s += plan.duration_s_by_point[point]
    assert plan.duration_s == pytest.approx(expected_s)
    assert 0 < len(plan.feasible_points) < len(plan.skip_reason_by_point)
