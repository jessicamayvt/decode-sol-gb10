import itertools
import math
from dataclasses import fields
from typing import Any

import pytest

from decode_sol.analyze import PointSummary
from decode_sol.gap_model import fit_gap_model


def make_summary(
    *,
    batch_size: int,
    context_len: int,
    num_weight_params: int,
    kv_cache_dtype: str,
    efficiency: float,
) -> PointSummary:
    """A summary with only the fields the gap model reads set to meaningful values."""
    placeholder_by_type: dict[object, object] = {str: "", int: 0, float: 0.0}
    values: dict[str, Any] = {}
    for field in fields(PointSummary):
        values[field.name] = placeholder_by_type[field.type]
    values.update(
        batch_size=batch_size,
        context_len=context_len,
        num_weight_params=num_weight_params,
        kv_cache_dtype=kv_cache_dtype,
        efficiency=efficiency,
    )
    return PointSummary(**values)


def test_gap_model_recovers_the_coefficients_of_an_exactly_linear_efficiency() -> None:
    true_estimates = [0.6, 0.05, -0.02, 0.03, 0.01]
    summaries: list[PointSummary] = []
    for batch_size, context_len, num_weight_params, kv_cache_dtype in itertools.product(
        [1, 2, 4, 8],
        [512, 2048, 8192],
        [1_000_000_000, 2_000_000_000, 4_000_000_000],
        ["auto", "fp8"],
    ):
        efficiency = (
            0.6
            + 0.05 * math.log2(batch_size)
            - 0.02 * math.log2(context_len / 512)
            + 0.03 * math.log2(num_weight_params / 1e9)
            + 0.01 * (kv_cache_dtype == "fp8")
        )
        summaries.append(
            make_summary(
                batch_size=batch_size,
                context_len=context_len,
                num_weight_params=num_weight_params,
                kv_cache_dtype=kv_cache_dtype,
                efficiency=efficiency,
            )
        )

    fit = fit_gap_model(summaries, seed=0)

    assert fit.num_points == 72
    assert fit.r_squared == pytest.approx(1.0)
    for coefficient, true_estimate in zip(fit.coefficients, true_estimates, strict=True):
        assert coefficient.estimate == pytest.approx(true_estimate, abs=1e-9)
        assert coefficient.ci_low == pytest.approx(true_estimate, abs=1e-9)
        assert coefficient.ci_high == pytest.approx(true_estimate, abs=1e-9)
