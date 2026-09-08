"""Uniform-prior quadrature checked against SINGLE and analytic integrals."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_design_precision, single_uniform_criterion

CASES = json.loads((Path(__file__).parent / "fixtures/single_uniform.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_uniform_quadrature(case):
    options = {k: v for k, v in case.items() if k != "value"}
    x, n = [-1, 0, 1, 2], [10, 20, 30, 40]
    if case["comparison"] is not None:
        x, n = (x, x), (n, n)
    result = single_uniform_criterion(x, n, **options)
    # Original quadrature constants are partly initialized from default REAL.
    assert_allclose(result.value, case["value"], rtol=2e-7)
    assert_allclose(result.weights.sum(), 1, atol=1e-14)


def test_analytic_uniform_intercept_integral():
    # For x=+-1, n per dose, b=1: Var(b)=(1+cosh(a)*cosh(1))/n.
    result = single_uniform_criterion(
        [-1, 1], [50, 50], [-0.5, 1], [0.5, 1], criterion="slope_variance"
    )
    expected = (1 + (np.sinh(0.5) / 0.5) * np.cosh(1)) / 50
    assert_allclose(result.value, expected, rtol=1e-13)
    assert result.parameters.shape == (6, 2)
    assert_allclose(result.parameters[:, 1], 1)


def test_average_sd_is_not_sqrt_average_variance():
    args = ([-1, 1], [50, 50], [-2, 1], [2, 1])
    sd = single_uniform_criterion(*args, criterion="slope_sd", order=20)
    variance = single_uniform_criterion(*args, criterion="slope_variance", order=20)
    assert sd.value < np.sqrt(variance.value) - 0.001


@pytest.mark.parametrize(
    "criterion", ["slope_sd", "slope_variance", "quantile_sd", "quantile_variance"]
)
def test_fixed_parameters_reduce_to_point_prior(criterion):
    result = single_uniform_criterion([-1, 1], [30, 70], [0, 1], [0, 1], criterion=criterion)
    fixed = single_design_precision([-1, 1], [30, 70], [0, 1])
    assert_allclose(result.value, getattr(fixed, criterion))
    assert result.parameters.shape == (1, 2)
    assert_allclose(result.weights, [1])


@pytest.mark.parametrize("power,criterion", [(0.5, "quantile_sd"), (1, "quantile_variance")])
def test_allocation_scaling_and_order_convergence(power, criterion):
    args = dict(doses=[-1, 0, 2], lower=[0.2, 0.8], upper=[0.4, 1.2], criterion=criterion)
    reference = single_uniform_criterion(subjects=[20, 30, 50], order=20, **args)
    standard = single_uniform_criterion(subjects=[20, 30, 50], **args)
    scaled = single_uniform_criterion(subjects=[40, 60, 100], order=20, **args)
    assert_allclose(standard.value, reference.value, rtol=1e-7)
    assert_allclose(scaled.value, reference.value / 2**power)


@pytest.mark.parametrize(
    "changes",
    [
        {"lower": [0, 2], "upper": [1, 1]},
        {"lower": [0, 1, 2]},
        {"upper": [np.inf, 2]},
        {"lower": [0, -1]},
        {"lower": [0, 0]},
        {"order": True},
        {"order": 2.5},
        {"order": 0},
        {"order": 33},
        {"criterion": "median"},
        {"quantile": [0.1, 0.2]},
        {"comparison": "location"},
        {"model": "probit"},
    ],
)
def test_invalid_prior(changes):
    args = dict(doses=[-1, 1], subjects=[50, 50], lower=[0, 1], upper=[1, 2])
    args.update(changes)
    with pytest.raises(ValueError):
        single_uniform_criterion(**args)
