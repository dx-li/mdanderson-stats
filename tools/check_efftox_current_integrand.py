"""Compare Python likelihood/prior and joint cells with executed native integrands.

Usage: python check_efftox_current_integrand.py current-integrand.json
This can run against an installed wheel outside the repository.
"""

import json
import math
import sys

import numpy as np

from mdanderson_stats.efftox_model import efftox_log_likelihood, efftox_predict

rows = json.load(open(sys.argv[1]))
maximum = 0.0
for row in rows:
    native = np.asarray(row["native_counts"]).reshape(-1, 4)
    counts = np.empty((len(native), 2, 2), dtype=np.int64)
    counts[:, 0, 0] = native[:, 1]
    counts[:, 0, 1] = native[:, 0]
    counts[:, 1, 0] = native[:, 3]
    counts[:, 1, 1] = native[:, 2]
    parameters = np.asarray(row["parameters"])
    means, sds = np.asarray(row["prior_mean"]), np.asarray(row["prior_sd"])
    log_prior = np.sum(-0.5 * ((parameters - means) / sds) ** 2 - np.log(sds * np.sqrt(2 * np.pi)))
    log_weight = float(efftox_log_likelihood(row["dose_codes"], counts, parameters) + log_prior)
    actual = math.exp(log_weight)
    expected = row["posterior_integrand"]
    maximum = max(maximum, abs(actual - expected) / expected)
    assert math.isclose(actual, expected, rel_tol=1e-12)
    joint = efftox_predict(row["dose_codes"], parameters)
    np.testing.assert_allclose(joint.sum(axis=-1)[:, 1], row["raw_efficacy"], rtol=1e-13)
    np.testing.assert_allclose(joint.sum(axis=-2)[:, 1], row["raw_toxicity"], rtol=1e-13)
print(
    json.dumps(
        {
            "cases": len(rows),
            "max_relative_weight_error": maximum,
            "negative_slope_likelihood_uses_raw_toxicity": True,
            "untruncated_normal_integrand_matches": True,
        },
        indent=2,
    )
)
