"""Check exact Weibull posterior and Johnson diagnostic references from base R."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.weibull_bayesian_gof import weibull_fixed_shape_bayesian_gof

root = Path(__file__).resolve().parents[1]
with (root / "tests/fixtures/weibull-bayesian-inputs.csv").open() as stream:
    inputs = list(csv.DictReader(stream))
with (root / "tests/fixtures/weibull-bayesian-reference.csv").open() as stream:
    references = list(csv.DictReader(stream))
start = time.monotonic()
checked = 0
worst = 0.0
cases = []
for index, name in enumerate(dict.fromkeys(row["case"] for row in inputs)):
    rows = [row for row in inputs if row["case"] == name]
    reference = {row["metric"]: float(row["value"]) for row in references if row["case"] == name}
    times = np.array([float(row["time"]) for row in rows])
    beta = float(rows[0]["weibull_shape"])
    fitted = weibull_fixed_shape_bayesian_gof(
        times,
        weibull_shape=beta,
        prior_shape=float(rows[0]["prior_shape"]),
        prior_rate=float(rows[0]["prior_rate"]),
        samples=16000,
        bins=int(rows[0]["bins"]),
        rng=66071 + index,
    )
    assert fitted.posterior_shape == reference["posterior_shape"]
    assert_allclose(
        fitted.log_posterior_rate, reference["log_posterior_rate"], rtol=2e-15, atol=2e-13
    )
    assert_allclose(
        fitted.posterior_rate_log_scale, reference["log_rate_scale"], rtol=2e-15, atol=2e-13
    )
    assert_allclose(
        np.log(fitted.posterior_rate_scaled_sum), reference["log_rate_factor"], rtol=0, atol=2e-13
    )
    centered = fitted.centered_log_rate_samples
    # Evaluate the dimensionless conditional CDF directly, without forming t**beta.
    relative_power = (times / times.max()) ** beta
    cdf = -np.expm1(-np.exp(centered[:, None]) * relative_power[None, :])
    d = fitted.diagnostic
    values = np.column_stack(
        (
            centered,
            (centered - reference["mean_centered_log_rate"]) ** 2,
            d.statistic,
            1 - d.reference_tail_probability,
            d.reference_tail_probability,
            d.statistic > d.critical_value,
            cdf,
        )
    )
    expected = np.array(
        [
            reference["mean_centered_log_rate"],
            reference["var_log_rate"],
            reference["mean_statistic"],
            reference["area_against_reference"],
            reference["mean_reference_tail"],
            reference["critical_exceedance_fraction"],
            *[reference[f"mean_cdf.{j}"] for j in range(times.size)],
        ]
    )
    mcse = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    error = np.abs(values.mean(axis=0) - expected)
    assert np.all(error <= 5 * mcse + 2e-12), (name, error, mcse)
    variable = mcse > 1e-10
    maximum = float(np.max(error[variable] / mcse[variable]))
    worst = max(worst, maximum)
    checked += len(expected)
    cases.append(dict(case=name, metrics=len(expected), maximum_error_mcse=maximum))
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            cases=cases,
            metrics=checked,
            maximum_error_mcse=worst,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
