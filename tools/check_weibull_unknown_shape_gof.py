"""Compare joint Weibull posterior draws with independent R quadrature."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np

from mdanderson_stats.hierarchical_binomial import summarize_chains
from mdanderson_stats.weibull_unknown_shape_gof import weibull_unknown_shape_bayesian_gof

root = Path(__file__).resolve().parents[1]
with (root / "tests/fixtures/weibull-unknown-shape-inputs.csv").open() as stream:
    inputs = list(csv.DictReader(stream))
with (root / "tests/fixtures/weibull-unknown-shape-reference.csv").open() as stream:
    references = list(csv.DictReader(stream))
start = time.monotonic()
checks = []
for index, name in enumerate(dict.fromkeys(row["case"] for row in inputs)):
    rows = [row for row in inputs if row["case"] == name]
    reference = {row["metric"]: float(row["value"]) for row in references if row["case"] == name}
    times = np.array([float(row["time"]) for row in rows])
    mean = np.array([float(rows[0]["prior_log_shape"]), float(rows[0]["prior_log_scale"])])
    sd = np.array([float(rows[0]["prior_shape_sd"]), float(rows[0]["prior_scale_sd"])])
    rho = float(rows[0]["prior_correlation"])
    covariance = np.outer(sd, sd) * [[1, rho], [rho, 1]]
    initial = mean + np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]) * sd
    fitted = weibull_unknown_shape_bayesian_gof(
        times,
        prior_mean=mean,
        prior_covariance=covariance,
        initial=initial,
        draws=2500,
        warmup=500,
        chains=4,
        bins=3,
        rng=np.random.default_rng(660940 + index),
    )
    log_shape = fitted.parameters[:, :, 0]
    log_scale = fitted.parameters[:, :, 1] + fitted.log_scale_offset
    shape, scale = np.exp(log_shape), np.exp(log_scale)
    cdf = -np.expm1(
        -np.exp(shape[:, :, None] * (np.log(times)[None, None, :] - log_scale[:, :, None]))
    )
    shape_centered = log_shape - reference["mean_log_shape"]
    scale_centered = log_scale - reference["mean_log_scale"]
    values = np.concatenate(
        (
            np.stack(
                (
                    log_shape,
                    log_scale,
                    shape_centered**2,
                    scale_centered**2,
                    shape_centered * scale_centered,
                    shape,
                    scale,
                ),
                axis=-1,
            ),
            cdf,
        ),
        axis=-1,
    )
    expected = np.array(
        [
            reference["mean_log_shape"],
            reference["mean_log_scale"],
            reference["variance_log_shape"],
            reference["variance_log_scale"],
            reference["covariance"],
            reference["mean_shape"],
            reference["mean_scale"],
            *[reference[f"mean_cdf.{j}"] for j in range(times.size)],
        ]
    )
    summary = summarize_chains(values)
    error = np.abs(summary.mean - expected)
    assert np.all(error <= 5 * summary.batch_mean_mcse + 2e-8), (
        name,
        error,
        summary.batch_mean_mcse,
    )
    assert np.nanmax(summary.split_rhat) < 1.05, (name, summary.split_rhat)
    checks.append(
        dict(
            case=name,
            metrics=len(expected),
            maximum_error_mcse=float(np.max(error / summary.batch_mean_mcse)),
            maximum_split_rhat=float(np.nanmax(summary.split_rhat)),
            likelihood_evaluations=fitted.likelihood_evaluations,
            likelihood_work_units=fitted.likelihood_work_units,
        )
    )
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            checks=checks,
            metrics=sum(row["metrics"] for row in checks),
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
