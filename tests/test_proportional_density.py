import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import proportional_density, proportional_density_pepe


@pytest.fixture(scope="module")
def native():
    return json.loads(
        (Path(__file__).parent / "fixtures/proportional-density-native.json").read_text()
    )["cases"]


def test_archived_estimator_and_independent_logistic_fit(native):
    for case in native:
        fit = proportional_density(case["time"], case["event"], case["treatment"])
        parameters = np.r_[
            fit.alpha,
            fit.beta,
            fit.alpha_star,
            fit.incidence,
            fit.incidence_standard_error,
            fit.likelihood_ratio,
            fit.incidence_pvalue,
            fit.goodness_of_fit,
        ]
        assert_allclose(parameters, case["parameters"], atol=4e-6, rtol=0)
        assert_allclose([fit.alpha, fit.beta], case["glm"], atol=5e-9, rtol=0)
        for actual, key in (
            (fit.observed_mass, "observed_mass"),
            (fit.disease_mass, "disease_mass"),
            (fit.disease_survival[:, 0], "control_survival"),
        ):
            assert_allclose(actual, case[key], atol=4e-7, rtol=0)
        assert_allclose(
            fit.nonparametric_survival[:, 0], case["control_nonparametric"], atol=1e-14, rtol=0
        )
        assert_allclose(fit.censoring_survival, case["censoring_survival"], atol=1e-14, rtol=0)
        assert_allclose(fit.incidence_standard_error, case["parameters"][5:7], atol=1e-14, rtol=0)
        assert_allclose(fit.observed_mass.sum(axis=0), 1, atol=1e-11, rtol=0)
        assert_allclose(fit.disease_mass.sum(axis=0), 1, atol=1e-14, rtol=0)
        assert fit.score_error <= 1e-12
        assert fit.likelihood_pvalue is None
        with pytest.raises(ValueError):
            fit.disease_mass[0, 0] = 0


def test_time_units_arm_exchange_and_tied_null(native):
    case = native[0]
    t, d, z = (np.array(case[key]) for key in ("time", "event", "treatment"))
    baseline = proportional_density(t, d, z)
    for scale in (1e-150, 1e150):
        fit = proportional_density(t * scale, d, z)
        assert_allclose(fit.beta * scale, baseline.beta, atol=1e-13, rtol=0)
        assert_allclose(fit.disease_mass, baseline.disease_mass, atol=1e-13, rtol=0)
        assert_allclose(fit.goodness_of_fit / scale, baseline.goodness_of_fit, atol=1e-13, rtol=0)
    exchanged = proportional_density(t, d, 1 - z)
    assert_allclose(
        [exchanged.beta, exchanged.alpha_star],
        [-baseline.beta, -baseline.alpha_star],
        atol=1e-13,
        rtol=0,
    )
    assert_allclose(exchanged.disease_mass, baseline.disease_mass[:, ::-1], atol=1e-13, rtol=0)
    # Identical arms, tied events and a shared last censor: exact nonparametric null.
    t, d, z = np.tile([1, 2, 4, 5], 2), np.tile([1, 1, 1, 0], 2), np.repeat([0, 1], 4)
    fit = proportional_density(t, d, z, equal_censoring=True)
    assert_allclose([fit.alpha, fit.beta, fit.alpha_star, fit.goodness_of_fit], 0, atol=1e-30)
    assert_allclose(fit.disease_mass, 1 / 3, atol=1e-15)
    assert_allclose(fit.incidence, 0.75)
    assert fit.likelihood_pvalue == fit.incidence_pvalue == 1
    shuffled = proportional_density(t[::-1], d[::-1], z[::-1], equal_censoring=True)
    assert_allclose(shuffled.disease_mass, fit.disease_mass, atol=1e-15)


def test_unidentified_fits_and_uncalibrated_boundary():
    with pytest.raises(ValueError, match="separated"):
        proportional_density([1, 2, 3, 4], [1] * 4, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="vary"):
        proportional_density([1] * 4, [1] * 4, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="unsupported"):
        proportional_density([1, 2, 3, 4, 5, 6], [1, 1, 1, 0, 1, 0], [0, 1, 0, 0, 1, 1])
    with pytest.raises(ValueError, match="observed failures"):
        proportional_density([1, 2, 3, 4], [0, 0, 1, 1], [0, 0, 1, 1])
    fit = proportional_density([1, 3, 2, 4], [1] * 4, [0, 0, 1, 1])
    assert fit.incidence_pvalue is None  # All observed fail: Greenwood SE=0.
    assert proportional_density_pepe([2], [0.5], [0.25]) == 0.125
    with pytest.raises(ValueError, match="unique"):
        proportional_density_pepe([1, 1], [0.5, 0], [0.5, 0])
