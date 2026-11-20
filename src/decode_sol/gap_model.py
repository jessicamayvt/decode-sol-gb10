"""A physical model of the gap, fitted to the grid points in results/summary.csv.

The roofline reads every byte at the achievable bandwidth. This model fits decode step time as
the bytes of each kind of read over that read's own bandwidth, plus time that bytes don't
explain:

    step time = weight bytes / matmul bandwidth + KV bytes / attention bandwidth
                + fixed time + batch size * weight GB * time per sequence per weight GB

The profiles (results/profile_summary.json) show why the reads differ. At batch 1 the matmuls
run as a matrix-vector kernel whose bandwidth depends on the matrix shapes, so each model gets
its own batch-1 bandwidth; batch >= 2 runs a different kernel, with one bandwidth for all
models. Attention gets one bandwidth per KV cache dtype. The fixed time covers small kernels,
launches and GPU idle. The last term grows with the matmul arithmetic, batch size * weights.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, assert_never

import numpy as np

from decode_sol.analyze import (
    BOOTSTRAP_SEED,
    CI_HIGH_PERCENTILE,
    CI_LOW_PERCENTILE,
    NUM_BOOTSTRAP_RESAMPLES,
    PointSummary,
    load_point_summaries,
)

logger = logging.getLogger(__name__)

TermUnit = Literal["GB/s", "ms", "ms per sequence per weight GB"]
BYTES_PER_GB = 1e9
MS_PER_S = 1e3


@dataclass(frozen=True, slots=True)
class GapModelTerm:
    name: str
    unit: TermUnit


@dataclass(frozen=True, slots=True)
class GapModelEstimate:
    name: str
    unit: TermUnit
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class GapModelFit:
    """Least-squares fit of step time, minimizing relative error, one row per grid point.

    r_squared is for efficiency, the predicted step time at the achievable bandwidth over the
    fitted step time, so it compares with a model of efficiency. leave_one_out_r_squared
    predicts each point from a fit to the others. CIs are 95% percentile bootstraps that
    resample grid points within each model and batch-1-or-not group, so every term is fitted
    in every resample.
    """

    num_points: int
    r_squared: float
    leave_one_out_r_squared: float
    efficiency_residual_std: float
    estimates: tuple[GapModelEstimate, ...]


def compute_gap_model_terms(models: list[str]) -> list[GapModelTerm]:
    terms: list[GapModelTerm] = []
    for model in models:
        terms.append(GapModelTerm(name=f"batch_1_matmul_bandwidth[{model}]", unit="GB/s"))
    terms.append(GapModelTerm(name="batch_2_plus_matmul_bandwidth", unit="GB/s"))
    terms.append(GapModelTerm(name="bf16_attention_bandwidth", unit="GB/s"))
    terms.append(GapModelTerm(name="fp8_attention_bandwidth", unit="GB/s"))
    terms.append(GapModelTerm(name="fixed_time", unit="ms"))
    terms.append(
        GapModelTerm(name="time_per_sequence_per_weight_gb", unit="ms per sequence per weight GB")
    )
    return terms


def compute_step_time_features(summary: PointSummary, models: list[str]) -> list[float]:
    """One value per term of compute_gap_model_terms(models), in that order.

    Each coefficient times its feature is seconds: a bandwidth term's coefficient is seconds per
    byte, so its feature is bytes.
    """
    is_batch_1 = summary.batch_size == 1
    is_fp8 = summary.kv_cache_dtype == "fp8"
    features: list[float] = []
    for model in models:
        is_batch_1_of_model = is_batch_1 and summary.model == model
        features.append(summary.weight_bytes if is_batch_1_of_model else 0.0)
    features.append(0.0 if is_batch_1 else summary.weight_bytes)
    features.append(summary.kv_bytes if not is_fp8 else 0.0)
    features.append(summary.kv_bytes if is_fp8 else 0.0)
    features.append(1.0)
    features.append(summary.batch_size * summary.weight_bytes / BYTES_PER_GB)
    return features


def to_reported_values(coefficients: np.ndarray, unit: TermUnit) -> np.ndarray:
    match unit:
        case "GB/s":
            return 1.0 / coefficients / BYTES_PER_GB
        case "ms" | "ms per sequence per weight GB":
            return coefficients * MS_PER_S
        case _:
            assert_never(unit)


def fit_relative_least_squares(features: np.ndarray, step_times_s: np.ndarray) -> np.ndarray:
    """Coefficients minimizing the sum of squared (fitted - measured) / measured step time."""
    relative_features = features / step_times_s[:, None]  # [num_points, num_terms]
    coefficients, _, _, _ = np.linalg.lstsq(
        relative_features, np.ones_like(step_times_s), rcond=None
    )
    return np.asarray(coefficients, dtype=np.float64)  # [num_terms]


def compute_r_squared(efficiencies: np.ndarray, fitted_efficiencies: np.ndarray) -> float:
    residuals = efficiencies - fitted_efficiencies  # [num_points]
    total_sum_of_squares = float(np.sum((efficiencies - efficiencies.mean()) ** 2))
    return 1.0 - float(np.sum(residuals**2)) / total_sum_of_squares


def compute_leave_one_out_step_times_s(
    features: np.ndarray, step_times_s: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    """Each point's step time as fitted without it, from the hat matrix of the relative fit."""
    relative_features = features / step_times_s[:, None]  # [num_points, num_terms]
    hat_matrix = relative_features @ np.linalg.pinv(relative_features)  # [num_points, num_points]
    hat_diagonal = np.diagonal(hat_matrix)  # [num_points]
    relative_residuals = 1.0 - relative_features @ coefficients  # [num_points]
    leave_one_out_relative_residuals = relative_residuals / (1.0 - hat_diagonal)  # [num_points]
    leave_one_out_step_times_s = step_times_s * (1.0 - leave_one_out_relative_residuals)
    return np.asarray(leave_one_out_step_times_s, dtype=np.float64)  # [num_points]


def compute_stratified_resample_idxs(
    group_by_point: list[tuple[str, bool]], *, seed: int
) -> np.ndarray:
    """Resamples of point indices, each drawn with replacement within its own group."""
    point_idxs_by_group: dict[tuple[str, bool], list[int]] = {}
    for point_idx, group in enumerate(group_by_point):
        point_idxs_by_group.setdefault(group, []).append(point_idx)

    rng = np.random.default_rng(seed)
    resample_idx_blocks: list[np.ndarray] = []
    for point_idxs in point_idxs_by_group.values():
        drawn = rng.integers(
            0, len(point_idxs), size=(NUM_BOOTSTRAP_RESAMPLES, len(point_idxs))
        )  # [num_resamples, num_points_in_group]
        resample_idx_blocks.append(np.asarray(point_idxs)[drawn])
    return np.concatenate(resample_idx_blocks, axis=1)  # [num_resamples, num_points]


def fit_gap_model(summaries: list[PointSummary], *, seed: int) -> GapModelFit:
    """Raises ValueError if the points can't fit every term, such as a model with no batch 1."""
    models = sorted({summary.model for summary in summaries})
    terms = compute_gap_model_terms(models)
    feature_rows = [compute_step_time_features(summary, models) for summary in summaries]
    features = np.asarray(feature_rows, dtype=np.float64)  # [num_points, num_terms]
    if np.linalg.matrix_rank(features) < len(terms):
        raise ValueError(f"the {len(summaries)} points can't fit all {len(terms)} terms")
    step_times_s = np.asarray([summary.step_time_ms / MS_PER_S for summary in summaries])
    predicted_step_times_s = np.asarray(
        [summary.predicted_step_time_ms / MS_PER_S for summary in summaries]
    )  # [num_points]
    efficiencies = predicted_step_times_s / step_times_s  # [num_points]

    coefficients = fit_relative_least_squares(features, step_times_s)  # [num_terms]
    fitted_efficiencies = predicted_step_times_s / (features @ coefficients)  # [num_points]
    leave_one_out_step_times_s = compute_leave_one_out_step_times_s(
        features, step_times_s, coefficients
    )  # [num_points]

    groups = [(summary.model, summary.batch_size == 1) for summary in summaries]
    resample_idxs = compute_stratified_resample_idxs(groups, seed=seed)
    bootstrap_coefficients = np.empty((NUM_BOOTSTRAP_RESAMPLES, len(terms)))
    for resample_idx, point_idxs in enumerate(resample_idxs):
        bootstrap_coefficients[resample_idx] = fit_relative_least_squares(
            features[point_idxs], step_times_s[point_idxs]
        )

    estimates: list[GapModelEstimate] = []
    for term_idx, term in enumerate(terms):
        bootstrap_values = to_reported_values(
            bootstrap_coefficients[:, term_idx], term.unit
        )  # [num_resamples]
        estimate_value = to_reported_values(coefficients[term_idx : term_idx + 1], term.unit)
        estimates.append(
            GapModelEstimate(
                name=term.name,
                unit=term.unit,
                estimate=float(estimate_value[0]),
                ci_low=float(np.percentile(bootstrap_values, CI_LOW_PERCENTILE)),
                ci_high=float(np.percentile(bootstrap_values, CI_HIGH_PERCENTILE)),
            )
        )
    return GapModelFit(
        num_points=len(summaries),
        r_squared=compute_r_squared(efficiencies, fitted_efficiencies),
        leave_one_out_r_squared=compute_r_squared(
            efficiencies, predicted_step_times_s / leave_one_out_step_times_s
        ),
        efficiency_residual_std=float(np.std(efficiencies - fitted_efficiencies, ddof=len(terms))),
        estimates=tuple(estimates),
    )


def log_gap_model_fit(fit: GapModelFit) -> None:
    logger.info(
        "Gap model: num_points=%d r_squared=%.3f leave_one_out_r_squared=%.3f "
        "efficiency_residual_std=%.4f",
        fit.num_points,
        fit.r_squared,
        fit.leave_one_out_r_squared,
        fit.efficiency_residual_std,
    )
    for estimate in fit.estimates:
        logger.info(
            "  %-45s %8.4g %s (95%% CI %.4g to %.4g)",
            estimate.name,
            estimate.estimate,
            estimate.unit,
            estimate.ci_low,
            estimate.ci_high,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the gap model to results/summary.csv.")
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--gap-model", type=Path, default=Path("results/gap_model.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    fit = fit_gap_model(load_point_summaries(args.summary), seed=BOOTSTRAP_SEED)
    args.gap_model.write_text(json.dumps(asdict(fit), indent=2) + "\n")
    log_gap_model_fit(fit)


if __name__ == "__main__":
    main()
