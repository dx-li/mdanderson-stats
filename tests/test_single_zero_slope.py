"""Slope estimation does not require an identifiable quantile dose."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    single_design_correlation,
    single_design_precision,
    single_normal_criterion,
    single_uniform_criterion,
)


@pytest.mark.parametrize("model", ["logistic", "loglog"])
def test_zero_slope_information_and_precision(model):
    r = single_design_precision([-1, 1], [50, 50], [0, 0], model=model, quantile=None)
    weight = 0.25 if model == "logistic" else 1 / np.expm1(1)
    assert_allclose(r.information, np.eye(2) * 100 * weight)
    assert_allclose(r.slope_variance, 1 / (100 * weight))
    assert r.quantile_dose is None and r.quantile_variance is None
    with pytest.raises(ValueError, match="not requested"):
        _ = r.quantile_sd
    with pytest.raises(ValueError, match="Slope"):
        single_design_precision([-1, 1], [50, 50], [0, 0], model=model)


@pytest.mark.parametrize("model", ["logistic", "loglog"])
@pytest.mark.parametrize("form", ["linear", "centered"])
def test_optional_quantile_preserves_existing_slope_precision(model, form):
    args = ([-1, 0, 2], [20, 30, 50], [[0.3, 1.2], [0.5, 0.8]])
    full = single_design_precision(*args, model=model, form=form)
    slope = single_design_precision(*args, model=model, form=form, quantile=None)
    assert_allclose(slope.slope_variance, full.slope_variance)
    assert_allclose(slope.information, full.information)


def test_uniform_slope_prior_crossing_zero_with_odd_order():
    r = single_uniform_criterion(
        [-1, 1], [50, 50], [0, -0.5], [0, 0.5], criterion="slope_variance", order=9
    )
    # Symmetric logistic design: Var(b)=(2+2*cosh(b))/100.
    assert_allclose(r.value, (2 + 2 * np.sinh(0.5) / 0.5) / 100, rtol=1e-13)
    assert np.any(r.parameters[:, 1] == 0)


@pytest.mark.parametrize("criterion,expected", [("slope_sd", 0.2), ("slope_variance", 0.04)])
def test_normal_point_prior_at_zero_slope(criterion, expected):
    r = single_normal_criterion([-1, 1], [50, 50], [0, 0], np.zeros((2, 2)), criterion=criterion)
    assert_allclose(r.value, expected)


def test_normal_odd_order_includes_zero_slope():
    r = single_normal_criterion(
        [-1, 1], [50, 50], [0, 0], [[0, 0], [0, 0.01]], criterion="slope_sd", order=7
    )
    assert np.any(r.parameters[:, 1] == 0)
    assert np.isfinite(r.value) and r.value > 0.2


def test_reference_correlations_do_not_require_a_quantile():
    r = single_design_correlation([-3, 3], [0, 0])
    assert_allclose(r.covariance, np.eye(2) * 0.04)


def test_centered_zero_slope_still_has_singular_information():
    with pytest.raises(ValueError, match="positive definite"):
        single_design_precision([-1, 1], [50, 50], [0, 0], form="centered", quantile=None)
