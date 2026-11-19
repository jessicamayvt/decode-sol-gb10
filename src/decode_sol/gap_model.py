"""A linear model of decode efficiency, fitted to the grid points in results/summary.csv."""

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

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

GAP_MODEL_REFERENCE_CONTEXT_LEN = 512
GAP_MODEL_REFERENCE_NUM_WEIGHT_PARAMS = 1e9
GAP_MODEL_FEATURE_NAMES = (
    "intercept",
    "log2_batch_size",
    "log2_context_len_over_512",
    "log2_num_weight_params_over_1e9",
    "is_fp8_kv_cache",
)


@dataclass(frozen=True, slots=True)
class GapModelCoefficient:
    feature: str
    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class GapModelFit:
    """Least-squares fit of efficiency on the GAP_MODEL_FEATURE_NAMES, one row per grid point.

    The intercept is the predicted efficiency at batch 1, context 512, 1e9 weight parameters
    and bf16 KV cache. CIs are 95% percentile bootstraps over grid points.
    """

    num_points: int
    r_squared: float
    residual_std: float
    coefficients: tuple[GapModelCoefficient, ...]


def compute_gap_model_features(summary: PointSummary) -> list[float]:
    """One value per name in GAP_MODEL_FEATURE_NAMES, in that order."""
    return [
        1.0,
        math.log2(summary.batch_size),
        math.log2(summary.context_len / GAP_MODEL_REFERENCE_CONTEXT_LEN),
        math.log2(summary.num_weight_params / GAP_MODEL_REFERENCE_NUM_WEIGHT_PARAMS),
        float(summary.kv_cache_dtype == "fp8"),
    ]


def _fit_least_squares(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(features, targets, rcond=None)
    return np.asarray(coefficients, dtype=np.float64)  # [num_features]


def fit_gap_model(summaries: list[PointSummary], *, seed: int) -> GapModelFit:
    feature_rows = [compute_gap_model_features(summary) for summary in summaries]
    features = np.asarray(feature_rows)  # [num_points, num_features], fp64
    efficiencies = np.asarray([summary.efficiency for summary in summaries])  # [num_points]
    num_points, num_features = features.shape

    estimates = _fit_least_squares(features, efficiencies)  # [num_features]
    residuals = efficiencies - features @ estimates  # [num_points]
    total_sum_of_squares = float(np.sum((efficiencies - efficiencies.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / total_sum_of_squares

    rng = np.random.default_rng(seed)
    resample_idxs = rng.integers(
        0, num_points, size=(NUM_BOOTSTRAP_RESAMPLES, num_points)
    )  # [num_resamples, num_points]
    bootstrap_estimates = np.empty((NUM_BOOTSTRAP_RESAMPLES, num_features))
    for resample_idx, point_idxs in enumerate(resample_idxs):
        bootstrap_estimates[resample_idx] = _fit_least_squares(
            features[point_idxs], efficiencies[point_idxs]
        )

    coefficients: list[GapModelCoefficient] = []
    for feature_idx, feature in enumerate(GAP_MODEL_FEATURE_NAMES):
        feature_estimates = bootstrap_estimates[:, feature_idx]  # [num_resamples]
        coefficients.append(
            GapModelCoefficient(
                feature=feature,
                estimate=float(estimates[feature_idx]),
                ci_low=float(np.percentile(feature_estimates, CI_LOW_PERCENTILE)),
                ci_high=float(np.percentile(feature_estimates, CI_HIGH_PERCENTILE)),
            )
        )
    return GapModelFit(
        num_points=num_points,
        r_squared=r_squared,
        residual_std=float(np.std(residuals, ddof=num_features)),
        coefficients=tuple(coefficients),
    )


def log_gap_model_fit(fit: GapModelFit) -> None:
    logger.info(
        "Gap model: num_points=%d r_squared=%.3f residual_std=%.4f",
        fit.num_points,
        fit.r_squared,
        fit.residual_std,
    )
    for coefficient in fit.coefficients:
        logger.info(
            "  %-32s %+.4f (95%% CI %+.4f to %+.4f)",
            coefficient.feature,
            coefficient.estimate,
            coefficient.ci_low,
            coefficient.ci_high,
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
