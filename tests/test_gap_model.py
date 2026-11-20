import itertools
from dataclasses import fields
from typing import Any

import numpy as np
import pytest

from decode_sol.analyze import PointSummary
from decode_sol.gap_model import (
    compute_leave_one_out_step_times_s,
    compute_stratified_resample_idxs,
    fit_gap_model,
    fit_relative_least_squares,
)

ACHIEVABLE_BANDWIDTH_BYTES_PER_S = 265e9
WEIGHT_BYTES_BY_MODEL = {"small": 2_000_000_000, "large": 8_000_000_000}
TRUE_ESTIMATES = {
    "batch_1_matmul_bandwidth[large]": 220.0,
    "batch_1_matmul_bandwidth[small]": 180.0,
    "batch_2_plus_matmul_bandwidth": 240.0,
    "bf16_attention_bandwidth": 250.0,
    "fp8_attention_bandwidth": 225.0,
    "fixed_time": 1.0,
    "time_per_sequence_per_weight_gb": 0.004,
}


def make_summary(
    *, model: str, batch_size: int, kv_bytes: int, kv_cache_dtype: str, step_time_ms: float
) -> PointSummary:
    """A summary with only the fields the gap model reads set to meaningful values."""
    weight_bytes = WEIGHT_BYTES_BY_MODEL[model]
    placeholder_by_type: dict[object, object] = {str: "", int: 0, float: 0.0}
    values: dict[str, Any] = {}
    for field in fields(PointSummary):
        values[field.name] = placeholder_by_type[field.type]
    values.update(
        model=model,
        batch_size=batch_size,
        kv_cache_dtype=kv_cache_dtype,
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        step_time_ms=step_time_ms,
        predicted_step_time_ms=(weight_bytes + kv_bytes) / ACHIEVABLE_BANDWIDTH_BYTES_PER_S * 1e3,
    )
    return PointSummary(**values)


def compute_true_step_time_ms(
    *, model: str, batch_size: int, kv_bytes: int, kv_cache_dtype: str
) -> float:
    weight_bytes = WEIGHT_BYTES_BY_MODEL[model]
    if batch_size == 1:
        matmul_bandwidth_gb_per_s = TRUE_ESTIMATES[f"batch_1_matmul_bandwidth[{model}]"]
    else:
        matmul_bandwidth_gb_per_s = TRUE_ESTIMATES["batch_2_plus_matmul_bandwidth"]
    attention_bandwidth_gb_per_s = TRUE_ESTIMATES[f"{kv_cache_dtype}_attention_bandwidth"]
    return (
        weight_bytes / matmul_bandwidth_gb_per_s / 1e6
        + kv_bytes / attention_bandwidth_gb_per_s / 1e6
        + TRUE_ESTIMATES["fixed_time"]
        + batch_size * weight_bytes / 1e9 * TRUE_ESTIMATES["time_per_sequence_per_weight_gb"]
    )


def make_exact_summaries() -> list[PointSummary]:
    summaries: list[PointSummary] = []
    for model, batch_size, kv_bytes, kv_cache_dtype in itertools.product(
        ["small", "large"],
        [1, 2, 8, 64],
        [100_000_000, 1_000_000_000, 10_000_000_000],
        ["bf16", "fp8"],
    ):
        step_time_ms = compute_true_step_time_ms(
            model=model, batch_size=batch_size, kv_bytes=kv_bytes, kv_cache_dtype=kv_cache_dtype
        )
        summaries.append(
            make_summary(
                model=model,
                batch_size=batch_size,
                kv_bytes=kv_bytes,
                kv_cache_dtype=kv_cache_dtype,
                step_time_ms=step_time_ms,
            )
        )
    return summaries


def test_gap_model_recovers_the_bandwidths_and_times_of_exact_step_times() -> None:
    summaries = make_exact_summaries()

    fit = fit_gap_model(summaries, seed=0)

    assert fit.num_points == 48
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.leave_one_out_r_squared == pytest.approx(1.0)
    assert [estimate.name for estimate in fit.estimates] == list(TRUE_ESTIMATES)
    for estimate in fit.estimates:
        true_estimate = TRUE_ESTIMATES[estimate.name]
        assert estimate.estimate == pytest.approx(true_estimate, rel=1e-6)
        assert estimate.ci_low == pytest.approx(true_estimate, rel=1e-6)
        assert estimate.ci_high == pytest.approx(true_estimate, rel=1e-6)


def test_gap_model_raises_when_a_model_has_no_batch_1_point() -> None:
    summaries: list[PointSummary] = []
    for summary in make_exact_summaries():
        if summary.model == "large" and summary.batch_size == 1:
            continue
        summaries.append(summary)

    with pytest.raises(ValueError, match="can't fit all 7 terms"):
        fit_gap_model(summaries, seed=0)


def test_stratified_resamples_keep_every_group_at_its_size() -> None:
    groups = [("a", True), ("a", False), ("a", False), ("b", True), ("b", False)]

    resample_idxs = compute_stratified_resample_idxs(groups, seed=0)

    for point_idxs in resample_idxs:
        resampled_groups = sorted(groups[point_idx] for point_idx in point_idxs)
        assert resampled_groups == sorted(groups)
    assert len(np.unique(resample_idxs[:, 1:3])) == 2  # Points 1 and 2 are drawn from both.


def test_leave_one_out_step_times_match_refitting_without_each_point() -> None:
    rng = np.random.default_rng(0)
    features = rng.uniform(1.0, 2.0, size=(12, 3))  # [num_points, num_terms]
    step_times_s = features @ np.array([0.5, 1.0, 2.0]) * rng.uniform(0.9, 1.1, size=12)
    coefficients = fit_relative_least_squares(features, step_times_s)

    leave_one_out_step_times_s = compute_leave_one_out_step_times_s(
        features, step_times_s, coefficients
    )

    for point_idx in range(len(step_times_s)):
        kept = np.arange(len(step_times_s)) != point_idx
        refit_coefficients = fit_relative_least_squares(features[kept], step_times_s[kept])
        expected = features[point_idx] @ refit_coefficients
        assert leave_one_out_step_times_s[point_idx] == pytest.approx(expected, rel=1e-9)
