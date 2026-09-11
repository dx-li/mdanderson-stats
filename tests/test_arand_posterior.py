import csv
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    arand_best_probability,
    arand_binary_posterior,
    arand_survival_posterior,
)


def test_multiarm_best_probabilities_against_independent_r_and_exact_integrals():
    with (Path(__file__).parent / "fixtures/arand-posterior-r.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for case in dict.fromkeys(row["case"] for row in rows):
        for maximize in (True, False):
            selected = [
                r for r in rows if r["case"] == case and r["maximize"] == str(maximize).upper()
            ]
            parameters = [[float(r["shape"]), float(r["second"])] for r in selected]
            result = arand_best_probability(
                parameters, family=selected[0]["family"], maximize=maximize
            )
            expected = np.array([float(r["probability"]) for r in selected])
            reference_error = np.array([float(r["error"]) for r in selected])
            assert np.all(
                abs(result.probability - expected) <= result.absolute_error + reference_error
            )
            assert_allclose(result.probability.sum(), 1, atol=1e-9, rtol=0)
    # With two Uniform competitors, the integrands are X^2 and (1-X)^2.
    assert_allclose(
        arand_best_probability([[2, 1], [1, 1], [1, 1]]).probability,
        [0.5, 0.25, 0.25],
        atol=1e-9,
        rtol=0,
    )
    assert_allclose(
        arand_best_probability([[2, 1], [1, 1], [1, 1]], maximize=False).probability,
        [1 / 6, 5 / 12, 5 / 12],
        atol=1e-9,
        rtol=0,
    )
    equal = arand_best_probability([[0.5, 1]] * 10, family="inverse_gamma")
    assert_allclose(equal.probability, 0.1, atol=0, rtol=0)
    assert not equal.probability.flags.writeable


def test_binary_allocation_censored_exposure_and_time_unit_invariance():
    binary = arand_binary_posterior([1, 0, 0], [0, 0, 0], prior=np.ones((3, 2)), threshold=0.5)
    assert_allclose(binary.parameters, [[2, 1], [1, 1], [1, 1]])
    assert_allclose(binary.exceedance_probability, [0.75, 0.5, 0.5])
    assert_allclose(
        binary.allocation_probability,
        [np.sqrt(2) / (2 + np.sqrt(2)), 1 / (2 + np.sqrt(2)), 1 / (2 + np.sqrt(2))],
    )
    prior = np.array([[2.0, 3.0], [3.0, 2.0], [1.5, 1.0]])
    events, exposure = [1, 0, 2], np.array([4.0, 8.0, 6.0])
    mean = arand_survival_posterior(events, exposure, prior=prior, threshold=5)
    assert_allclose(mean.parameters, [[3, 7], [3, 10], [3.5, 7]])
    # An arm with no events still updates its scale through censored exposure.
    for scale in (np.log(2), 1e200, 1e-200):
        scaled_prior = prior * [1, scale]
        if scale == np.log(2):
            other = arand_survival_posterior(
                events, exposure, prior=scaled_prior, parameter="median", threshold=5 * scale
            )
        else:
            other = arand_survival_posterior(
                events, exposure * scale, prior=scaled_prior, threshold=5 * scale
            )
        assert_allclose(other.best.probability, mean.best.probability, atol=1e-10, rtol=0)
        assert_allclose(
            other.exceedance_probability, mean.exceedance_probability, atol=1e-12, rtol=0
        )
    equal = arand_survival_posterior(events, exposure, prior=prior, tuning=0)
    assert_allclose(equal.allocation_probability, 1 / 3)
    # A sub-float winning tail cannot safely be raised to a fractional tuning power.
    with pytest.raises(ArithmeticError, match="tail unresolved"):
        arand_survival_posterior([0, 0], [0, 0], prior=[[1, 1e300], [1, 1e-300]])
