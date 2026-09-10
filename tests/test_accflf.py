import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from mdanderson_stats import (
    accflf_logf,
    accflf_loglikelihood,
    accflf_shape,
    accflf_survival,
    fit_accflf,
)
from mdanderson_stats.accflf_model import _likelihood

FIXTURE = json.loads((Path(__file__).parent / "fixtures/accflf-native.json").read_text())


def test_native_logf_tails_and_derivatives():
    rows = np.array(FIXTURE["cases"])
    for n, d in np.unique(rows[:, :2], axis=0):
        cases = rows[(rows[:, 0] == n) & (rows[:, 1] == d)]
        result = accflf_logf(cases[:, 2], n, d)
        for i, name in enumerate(FIXTURE["columns"][3:], 3):
            assert_allclose(getattr(result, name), cases[:, i], rtol=1e-8, atol=1e-9)
        if n == 8 and d == 1e10:
            assert_allclose(result.log_survival, cases[:, 5], rtol=1e-12, atol=1e-12)
        assert_allclose(np.logaddexp(result.log_cdf, result.log_survival), 0, atol=1e-14)
        reflected = accflf_logf(-cases[:, 2], d, n)
        assert_allclose(result.log_cdf, reflected.log_survival, rtol=1e-12, atol=1e-10)
    integral = quad(lambda w: np.exp(accflf_logf(w, 3, 8).log_density), -100, 100)[0]
    assert integral == pytest.approx(1, abs=1e-10)
    assert accflf_shape(0, 0).numerator_df == 1e10
    assert accflf_shape(0, 1.7e-5).numerator_df == 1e10
    assert accflf_shape(0, 1).numerator_df == 2
    assert accflf_shape(1, 0).denominator_df == 2
    assert accflf_shape(1e100, 1e100).lower_clipped
    shape = accflf_shape(0.7, 0.4)
    a, b = shape.numerator_df / 2, shape.denominator_df / 2
    assert 2 / (a + b) == pytest.approx(0.7)
    assert (1 / a - 1 / b) / np.sqrt(1 / a + 1 / b) == pytest.approx(0.4)


def test_fixed_shape_fit_and_likelihood_geometry():
    t = np.array([1, 2, 3, 4, 6, 8, 9, 12, 15, 19, 23, 30.0])
    e = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0])
    x = np.tile([-1.0, 0, 1], 4)[:, None]
    weights = np.array([1, 2, 1, 1, 3, 1, 2, 1, 1, 2, 1, 1])
    fit = fit_accflf(t, e, covariates=x, weights=weights, p=0.7, q=0.4)
    assert_allclose(
        np.r_[np.log(fit.sigma), fit.coefficients, fit.log_likelihood],
        FIXTURE["fixed_r_fit"],
        atol=1e-6,
        rtol=0,
    )
    repeated = np.repeat(np.arange(t.size), weights)
    expanded = fit_accflf(t[repeated], e[repeated], covariates=x[repeated], p=0.7, q=0.4)
    assert_allclose(expanded.coefficients, fit.coefficients, atol=1e-7)
    scaled = fit_accflf(t * 1e150, e, covariates=x, weights=weights, p=0.7, q=0.4)
    assert_allclose(scaled.coefficients - [np.log(1e150), 0], fit.coefficients, atol=1e-7)
    assert scaled.sigma == pytest.approx(fit.sigma, rel=1e-7)
    assert_allclose(scaled.covariance, fit.covariance, atol=1e-7)
    assert fit.time_log_likelihood == pytest.approx(
        fit.log_likelihood - np.dot(weights * e, np.log(t))
    )
    # Analytic Hessian checked against changes in analytic score at a non-optimum.
    parameters = np.array([0.2, 2.0, 0.1])
    design = np.column_stack((np.ones(t.size), x))
    _, gradient, hessian = _likelihood(parameters, np.log(t), e, design, weights, fit.shape)
    for j in range(3):
        delta = np.eye(3)[j] * 1e-5
        plus, gp, _ = _likelihood(parameters + delta, np.log(t), e, design, weights, fit.shape)
        minus, gm, _ = _likelihood(parameters - delta, np.log(t), e, design, weights, fit.shape)
        assert (plus - minus) / 2e-5 == pytest.approx(gradient[j], abs=1e-8)
        assert_allclose((gp - gm) / 2e-5, hessian[:, j], atol=1e-8)
    exponential = fit_accflf(t, e, p=0, q=1, fixed_sigma=1)
    assert np.exp(exponential.coefficients[0]) == pytest.approx(t.sum() / e.sum(), rel=1e-8)
    assert_allclose(exponential.covariance[0], 0)
    survival = accflf_survival(t, p=0, q=1, sigma=1, coefficients=exponential.coefficients)
    assert_allclose(survival, np.exp(-t / np.exp(exponential.coefficients[0])), atol=1e-9)
    assert exponential.log_likelihood == pytest.approx(
        accflf_loglikelihood(t, e, p=0, q=1, sigma=1, coefficients=exponential.coefficients)
    )
    with pytest.raises(ValueError, match="failure"):
        fit_accflf(t, np.zeros(t.size), p=1, q=0)
