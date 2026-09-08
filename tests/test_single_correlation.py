"""Reference-design prior correlations and their statistical interpretation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import single_design_correlation, single_prior_parameters


def test_symmetric_logistic_reference_covariance():
    result = single_design_correlation([-3, 3], [0, 1])
    weight = np.exp(-1) / (1 + np.exp(-1)) ** 2
    assert_allclose(result.covariance, np.eye(2) / (100 * weight), atol=1e-14)
    assert_allclose(result.correlation, np.eye(2), atol=1e-14)


def test_two_sample_midpoint_reference_covariance():
    result = single_design_correlation([-3, 3], [0, 1, 0], comparison="location")
    weight = np.exp(-1) / (1 + np.exp(-1)) ** 2
    assert_allclose(result.covariance, np.diag([1 / (100 * weight), 1 / (100 * weight), 1 / 25]))
    assert_allclose(result.correlation, np.eye(3))
    assert_allclose(result.doses, [[-1, 1], [0, 0]])
    assert_allclose(result.subjects, [[50, 50], [50, 50]])


def test_midpoint_may_not_identify_second_slope():
    with pytest.raises(ValueError, match="identifiable"):
        single_design_correlation([-3, 3], [0, 1, 1], comparison="slope")


@pytest.mark.parametrize("model", ["logistic", "loglog"])
def test_independent_two_parameter_correlation_formula(model):
    result = single_design_correlation([-2, 4], [0.2, 0.8], model=model)
    x = np.array([0.0, 2.0])
    u = 0.2 + 0.8 * x
    if model == "logistic":
        weight = np.exp(u) / (1 + np.exp(u)) ** 2
    else:
        t = np.exp(-u)
        weight = t * t / np.expm1(t)
    # For a 2x2 information matrix, inverse correlation is -I12/sqrt(I11*I22).
    expected = -np.sum(weight * x) / np.sqrt(np.sum(weight) * np.sum(weight * x * x))
    assert_allclose(result.correlation[0, 1], expected)
    mean, covariance = single_prior_parameters([0.2, 0.8], [0.01, 0.02], result.correlation)
    assert_allclose(mean, [0.2, 0.8])
    assert_allclose(covariance[0, 1], expected * np.sqrt(0.01 * 0.02))


@pytest.mark.parametrize("comparison", [None, "location", "slope"])
@pytest.mark.parametrize("form", ["linear", "centered"])
def test_batch_reference_designs(comparison, form):
    parameters = np.array([[0.3, 0.8, 1.2], [0.5, 0.7, 1.1]])[:, : 2 if comparison is None else 3]
    result = single_design_correlation([-1, 3], parameters, comparison=comparison, form=form)
    for i in range(2):
        individual = single_design_correlation(
            [-1, 3], parameters[i], comparison=comparison, form=form
        )
        assert_allclose(result.correlation[i], individual.correlation)
        assert_allclose(result.covariance[i], individual.covariance)


@pytest.mark.parametrize("bounds", [[1, 1], [2, 1], [1], [0, np.inf]])
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        single_design_correlation(bounds, [0, 1])
