"""Compare the odds-rate ESS posterior with the independent R quadrature fixture."""

from __future__ import annotations

import csv
import resource
import sys
import time
from pathlib import Path

import numpy as np

from mdanderson_stats._log_odds_rate import log_odds_rate_components
from mdanderson_stats.tte_family_bayesian_gof import log_odds_rate_bayesian_gof

_TIMES = np.array([0.45, 0.8, 1.3, 2.1, 3.4])
_EVENTS = {
    "complete": np.ones(5, dtype=bool),
    "right_censored": np.array([True, False, True, True, False]),
}
_PRIOR_MEAN = np.log([1.6, 1.2, 0.7])
_PRIOR_COVARIANCE = np.array([[0.13, 0.035, -0.018], [0.035, 0.19, 0.04], [-0.018, 0.04, 0.16]])
_INITIAL_OFFSETS = np.array(
    [[-0.35, -0.35, 0.25], [-0.25, 0.35, -0.25], [0.35, -0.25, -0.3], [0.25, 0.3, 0.3]]
)


def _batch_means_mcse(values: np.ndarray) -> float:
    chains, draws = values.shape
    batch_size = max(2, int(np.sqrt(draws)))
    batches = draws // batch_size
    means = values[:, : batches * batch_size].reshape(chains, batches, batch_size).mean(axis=2)
    return float(np.std(means.reshape(-1), ddof=1) / np.sqrt(chains * batches))


def _metric_series(parameters: np.ndarray, relative: np.ndarray) -> dict[str, np.ndarray]:
    log_shape = parameters[..., 0]
    log_scale = parameters[..., 1]
    log_c = parameters[..., 2]
    means = np.stack((log_shape.mean(), log_scale.mean(), log_c.mean()))
    centered = (log_shape - means[0], log_scale - means[1], log_c - means[2])
    series: dict[str, np.ndarray] = {
        "mean_log_shape": log_shape,
        "mean_log_scale": log_scale,
        "mean_log_c": log_c,
        "var_log_shape": centered[0] ** 2,
        "var_log_scale": centered[1] ** 2,
        "var_log_c": centered[2] ** 2,
        "cov_log_shape_scale": centered[0] * centered[1],
        "cov_log_shape_c": centered[0] * centered[2],
        "cov_log_scale_c": centered[1] * centered[2],
        "mean_shape": np.exp(log_shape),
        "mean_scale": np.exp(log_scale),
        "mean_c": np.exp(log_c),
    }
    cdf = np.empty((*log_shape.shape, relative.size), dtype=float)
    for chain in range(parameters.shape[0]):
        for draw in range(parameters.shape[1]):
            _, _, cdf[chain, draw] = log_odds_rate_components(
                relative,
                float(log_shape[chain, draw]),
                float(log_scale[chain, draw]),
                float(log_c[chain, draw]),
            )
    for index in range(relative.size):
        series[f"posterior_mean_cdf_{index + 1}"] = cdf[..., index]
    return series


def _reference(case: str) -> dict[str, float]:
    path = Path(__file__).parents[1] / "tests" / "fixtures" / "log-odds-rate-reference.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        return {
            row["metric"]: float(row["value"])
            for row in rows
            if row["case"] == case and int(row["nquad"]) == 45
        }


def main() -> None:
    began = time.perf_counter()
    relative = np.log(_TIMES) - np.log(_TIMES[0])
    max_ratio = 0.0
    max_abs_error = 0.0
    max_rhat = 0.0
    worst_ratio: tuple[float, str, str, float, float, float, float] | None = None
    worst_absolute: tuple[float, str, str, float, float] | None = None
    offset = float(np.log(_TIMES[0]))
    absolute_prior_mean = _PRIOR_MEAN.copy()
    absolute_prior_mean[1] += offset
    for case_index, (case, events) in enumerate(_EVENTS.items()):
        fit = log_odds_rate_bayesian_gof(
            _TIMES,
            event=events,
            prior_mean=absolute_prior_mean,
            prior_covariance=_PRIOR_COVARIANCE,
            initial=absolute_prior_mean + _INITIAL_OFFSETS,
            draws=16000,
            warmup=1000,
            chains=4,
            rng=np.random.default_rng(20261003 + case_index),
        )
        series = _metric_series(fit.parameters, relative)
        reference = _reference(case)
        for name, values in series.items():
            estimate = float(values.mean())
            error = abs(estimate - reference[name])
            mcse = _batch_means_mcse(values)
            ratio = error / max(mcse, np.finfo(float).tiny)
            max_ratio = max(max_ratio, ratio)
            max_abs_error = max(max_abs_error, error)
            if worst_ratio is None or ratio > worst_ratio[0]:
                worst_ratio = (ratio, case, name, error, mcse, estimate, reference[name])
            if worst_absolute is None or error > worst_absolute[0]:
                worst_absolute = (error, case, name, estimate, reference[name])
        max_rhat = max(max_rhat, float(np.max(fit.parameter_summary.split_rhat)))
        if (fit.diagnostic is None) != (case != "complete"):
            raise AssertionError(f"unexpected Johnson diagnostic contract in {case}")
    elapsed = time.perf_counter() - began
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    gate = "pass" if max_ratio <= 5.0 and max_rhat <= 1.05 else "review_required"
    print(
        f"reference_gate={gate} cases=2 draws_per_chain=16000 warmup=1000 "
        f"seed=20261003/20261004 "
        f"max_error_over_batch_means_mcse={max_ratio:.4f} "
        f"max_absolute_metric_error={max_abs_error:.8g} max_split_rhat={max_rhat:.6f} "
        f"worst_ratio_case_metric={worst_ratio[1]}:{worst_ratio[2]} "
        f"error={worst_ratio[3]:.8g} mcse={worst_ratio[4]:.8g} "
        f"estimate={worst_ratio[5]:.8g} reference={worst_ratio[6]:.8g} "
        f"worst_absolute_case_metric={worst_absolute[1]}:{worst_absolute[2]} "
        f"absolute_error={worst_absolute[0]:.8g} estimate={worst_absolute[3]:.8g} "
        f"reference={worst_absolute[4]:.8g} elapsed_seconds={elapsed:.3f} "
        f"peak_rss_mib={peak_mib:.2f}"
    )


if __name__ == "__main__":
    main()
