"""Independent base-R Dirichlet cross-moment and tail references."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.uboin import uboin_posterior


def test_joint_models_against_base_r():
    cases = {
        "binary": ([[2, 1], [5, 2]], [[0.1, 0.2], [0.3, 0.4]], [[30, 0], [100, 50]], 1, 1),
        "categorical": (
            [[2, 1, 0], [3, 2, 1], [5, 1, 2]],
            np.arange(1, 10).reshape(3, 3) / 45,
            [[30, 15, 0], [50, 30, 0], [100, 45, 15]],
            2,
            2,
        ),
        "prior_only": (
            np.zeros((2, 3)),
            [[0.2, 0.1, 0.3], [0.4, 0.5, 0.6]],
            [[30, 15, 0], [100, 70, 50]],
            2,
            1,
        ),
    }
    with (Path(__file__).parent / "fixtures/uboin-posterior.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        counts, prior, utilities, dlt, response = cases[row["case"]]
        result = uboin_posterior(
            [counts], prior=prior, utilities=utilities, dlt_level=dlt, response_level=response
        )
        for field, column in (
            ("mean_utility", "mean"),
            ("utility_variance", "variance"),
            ("overdose_probability", "overdose"),
            ("low_efficacy_probability", "loweff"),
        ):
            np.testing.assert_allclose(getattr(result, field)[0], float(row[column]), rtol=2e-12)


def test_nearly_constant_utilities_preserve_small_positive_variance():
    high = 50.0 + 1e-7
    low = 50.0 - 1e-7
    posterior = uboin_posterior(
        np.zeros((1, 2, 2)),
        prior=np.full((2, 2), 0.25),
        utilities=[[low, high], [low, high]],
    )
    expected = ((high - low) / 2) ** 2 / 2
    assert posterior.utility_variance[0] > 0
    np.testing.assert_allclose(posterior.utility_variance[0], expected, rtol=1e-12)
