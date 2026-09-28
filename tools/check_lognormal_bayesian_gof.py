"""Validate joint lognormal posteriors and diagnostics against independent R."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import ndtr

from mdanderson_stats.lognormal_bayesian_gof import lognormal_complete_data_bayesian_gof

root = Path(__file__).resolve().parents[1]
with (root / "tests/fixtures/lognormal-bayesian-inputs.csv").open() as stream:
    inputs = list(csv.DictReader(stream))
with (root / "tests/fixtures/lognormal-bayesian-reference.csv").open() as stream:
    references = list(csv.DictReader(stream))
start = time.monotonic()
checked = 0
worst = 0.0
cases = []
for index, name in enumerate(dict.fromkeys(row["case"] for row in inputs)):
    rows = [row for row in inputs if row["case"] == name]
    reference = {row["metric"]: float(row["value"]) for row in references if row["case"] == name}
    times = np.array([float(row["time"]) for row in rows])
    fit = lognormal_complete_data_bayesian_gof(
        times,
        **{
            key: float(rows[0][key])
            for key in (
                "prior_location",
                "prior_location_precision",
                "prior_variance_shape",
                "prior_variance_scale",
            )
        },
        samples=16000,
        bins=int(rows[0]["bins"]),
        rng=6609028 + index,
    )
    for field in (
        "posterior_location",
        "posterior_location_precision",
        "posterior_variance_shape",
        "posterior_log_variance_scale",
    ):
        assert_allclose(getattr(fit, field), reference[field], rtol=3e-14, atol=3e-13)
    sigma = np.exp(0.5 * fit.log_variance_samples)
    centered_y = np.log(times / times[0])
    cdf = ndtr((centered_y[None, :] - fit.centered_location_samples[:, None]) / sigma[:, None])
    d = fit.diagnostic
    standardized_location = (
        (fit.centered_location_samples - fit.posterior_centered_location)
        * np.sqrt(fit.posterior_location_precision)
        / sigma
    )
    precision = np.exp(fit.posterior_log_variance_scale - fit.log_variance_samples)
    values = np.column_stack(
        (
            fit.centered_location_samples,
            (fit.centered_location_samples - reference["mean_centered_location"]) ** 2,
            fit.log_variance_samples,
            (fit.log_variance_samples - reference["mean_log_variance"]) ** 2,
            standardized_location,
            standardized_location**2,
            precision * standardized_location**2,
            d.statistic,
            d.reference_tail_probability,
            d.statistic > d.critical_value,
            cdf,
        )
    )
    expected = np.array(
        [
            reference["mean_centered_location"],
            reference["var_location"],
            reference["mean_log_variance"],
            reference["var_log_variance"],
            0,
            1,
            reference["posterior_variance_shape"],
            reference["mean_statistic"],
            reference["mean_reference_tail"],
            reference["critical_exceedance_fraction"],
            *[reference[f"mean_cdf.{j}"] for j in range(times.size)],
        ]
    )
    mcse = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    error = np.abs(values.mean(axis=0) - expected)
    assert np.all(error <= 5 * mcse + 3e-9), (name, error, mcse)
    variable = mcse > 1e-10
    maximum = float(np.max(error[variable] / mcse[variable]))
    checked += len(expected)
    worst = max(worst, maximum)
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
