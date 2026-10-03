"""One fixed-seed Gibbs check against risk's direct R posterior quadrature."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from mdanderson_stats.lognormal_censored_bayesian import (
    lognormal_right_censored_bayesian_fit,
)

_PRIOR = dict(
    prior_location=0.30,
    prior_location_precision=2.0,
    prior_variance_shape=4.5,
    prior_variance_scale=1.1,
)
_CASES = {
    "mixed": (
        np.array([0.4, 1.2, 1.2, 2.6, 5.0, 0.0, 3.5]),
        np.array([True, False, False, True, False, False, False]),
        20261033,
    ),
    "all_censored": (
        np.array([0.7, 1.5, 2.0, 4.0, 4.0]),
        np.zeros(5, dtype=bool),
        20261034,
    ),
}
_DRAWS = 12_000
_WARMUP = 3_000
_CHAINS = 4


def _batch_mcse(values: np.ndarray) -> float:
    chains, draws = values.shape[:2]
    batch_size = max(2, int(np.sqrt(draws)))
    batch_count = draws // batch_size
    reduced = values[:, : batch_count * batch_size]
    means = reduced.reshape((chains * batch_count, batch_size, *values.shape[2:])).mean(axis=1)
    flat = means.reshape((means.shape[0], -1))
    return float(np.max(np.std(flat, axis=0, ddof=1) / np.sqrt(flat.shape[0])))


def _reference(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["resolution"] == "grid151":
                result.setdefault(row["case"], {})[row["metric"]] = float(row["value"])
    if set(result) != set(_CASES):
        raise ValueError("reference fixture must contain grid151 rows for both fixed cases")
    return result


def main() -> None:
    reference = _reference(Path("tests/fixtures/lognormal-censored-reference.csv"))
    all_errors: list[float] = []
    for case_name, (times, event, seed) in _CASES.items():
        fit = lognormal_right_censored_bayesian_fit(
            times,
            event=event,
            **_PRIOR,
            draws=_DRAWS,
            warmup=_WARMUP,
            chains=_CHAINS,
            rng=np.random.default_rng(seed),
        )
        mu = fit.centered_location_samples
        log_variance = fit.log_variance_samples
        ref = reference[case_name]
        estimates: dict[str, tuple[float, float]] = {
            "mean_location_centered": (float(mu.mean()), _batch_mcse(mu)),
            "mean_location_absolute": (
                float((mu + fit.location_offset).mean()),
                _batch_mcse(mu),
            ),
            "mean_log_variance": (float(log_variance.mean()), _batch_mcse(log_variance)),
        }
        centered_mu = mu - mu.mean()
        centered_logv = log_variance - log_variance.mean()
        var_mu = centered_mu**2
        var_logv = centered_logv**2
        cov = centered_mu * centered_logv
        estimates.update(
            {
                "var_location_centered": (
                    float(np.var(mu, ddof=1)),
                    _batch_mcse(var_mu),
                ),
                "var_log_variance": (
                    float(np.var(log_variance, ddof=1)),
                    _batch_mcse(var_logv),
                ),
                "cov_location_log_variance": (
                    float(np.mean(cov)),
                    _batch_mcse(cov),
                ),
            }
        )
        positive_rows = np.flatnonzero(times > 0)
        centered_y = np.log(times[positive_rows]) - fit.location_offset
        for position, row in enumerate(positive_rows, start=1):
            cdf = ndtr((centered_y[position - 1] - mu) / np.exp(0.5 * log_variance))
            estimates[f"posterior_mean_cdf_{row + 1}"] = (
                float(cdf.mean()),
                _batch_mcse(cdf),
            )
        case_errors = []
        for metric, (estimate, mcse) in estimates.items():
            error = abs(estimate - ref[metric])
            scaled = error / mcse if mcse > 0 else (0.0 if error == 0 else np.inf)
            case_errors.append(scaled)
            all_errors.append(scaled)
        print(
            f"{case_name}: max_mcse={max(case_errors):.4f}, "
            f"split_rhat={np.max(fit.parameter_summary.split_rhat):.6f}, "
            f"proposals={fit.truncated_normal_proposals}, "
            f"work={fit.total_work_units}"
        )
    if max(all_errors) > 5.0:
        raise SystemExit(f"posterior comparison exceeded 5 MCSE: {max(all_errors):.4f}")
    print(f"all reference metrics within 5 MCSE; max={max(all_errors):.4f}")


if __name__ == "__main__":
    main()
