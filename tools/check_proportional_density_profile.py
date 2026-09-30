"""Bounded comparison with independent R GLM/profile-likelihood references."""

import csv
import json
import resource
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats.proportional_density import proportional_density
from mdanderson_stats.proportional_density_profile import proportional_density_profile

root = Path(__file__).resolve().parents[1]
fixtures = root / "tests/fixtures"


def read(name):
    with (fixtures / name).open() as stream:
        return list(csv.DictReader(stream))


inputs = read("proportional-density-profile-input.csv")
summaries = read("proportional-density-profile.csv")
intervals = read("proportional-density-profile-intervals.csv")
start = time.monotonic()
maximum_error = 0.0
maximum_cutoff_error = 0.0
comparisons = 0
for row in summaries:
    selected = [x for x in inputs if x["case_id"] == row["case_id"]]
    t, d, z = (np.array([float(x[key]) for x in selected]) for key in ("time", "event", "arm"))
    beta0 = float(row["beta_null"])
    for ci in (x for x in intervals if x["case_id"] == row["case_id"]):
        fit = proportional_density_profile(
            t, d, z, beta_null=beta0, equal_censoring=True, confidence=float(ci["confidence"])
        )
        assert fit.beta_interval is not None
        actual = np.r_[
            fit.beta_estimate,
            fit.alpha_at_null,
            fit.likelihood_ratio,
            fit.pvalue,
            fit.beta_interval,
        ]
        expected = np.array(
            [float(row[k]) for k in ("beta_hat", "alpha_at_null", "lr_at_null", "p_at_null")]
            + [float(ci[k]) for k in ("lower", "upper")]
        )
        assert_allclose(actual, expected, atol=2e-9, rtol=0)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        comparisons += actual.size
        for endpoint in fit.beta_interval:
            tested = proportional_density_profile(
                t, d, z, beta_null=endpoint, equal_censoring=True, interval=False
            )
            error = abs(tested.likelihood_ratio - float(ci["lr_cutoff"]))
            assert error < 2e-10
            maximum_cutoff_error = max(maximum_cutoff_error, error)
    zero = proportional_density_profile(t, d, z, beta_null=0, equal_censoring=True, interval=False)
    original = proportional_density(t, d, z, equal_censoring=True)
    assert_allclose(zero.likelihood_ratio, original.likelihood_ratio, atol=2e-12, rtol=0)

selected = [x for x in inputs if x["case_id"] == "moderate_effect"]
t, d, z = (np.array([float(x[key]) for x in selected]) for key in ("time", "event", "arm"))
base = proportional_density_profile(t, d, z, beta_null=0.2, equal_censoring=True)
for scale in (1e-150, 1e150):
    fit = proportional_density_profile(t * scale, d, z, beta_null=0.2 / scale, equal_censoring=True)
    assert_allclose(np.array(fit.beta_interval) * scale, base.beta_interval, atol=2e-12, rtol=0)
    assert_allclose(fit.pvalue, base.pvalue, atol=2e-12, rtol=0)
swapped = proportional_density_profile(t, d, 1 - z, beta_null=-0.2, equal_censoring=True)
assert_allclose(swapped.beta_interval, -np.array(base.beta_interval)[::-1], atol=2e-12, rtol=0)
assert_allclose(swapped.pvalue, base.pvalue, atol=2e-12, rtol=0)
extreme = proportional_density_profile(
    t, d, z, beta_null=0.2, equal_censoring=True, confidence=np.nextafter(1.0, 0.0)
)
assert np.isfinite(extreme.beta_interval).all()
usage = resource.getrusage(resource.RUSAGE_SELF)
print(
    json.dumps(
        dict(
            cases=len(summaries),
            intervals=len(intervals),
            reference_values=comparisons,
            max_reference_error=maximum_error,
            max_endpoint_lr_error=maximum_cutoff_error,
            elapsed_seconds=time.monotonic() - start,
            peak_mib=usage.ru_maxrss / 1024**2,
            swaps=usage.ru_nswap,
        ),
        indent=2,
    )
)
