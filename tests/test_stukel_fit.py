import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import StukelFitError, fit_stukel

REFERENCE = json.loads((Path(__file__).parent / "fixtures/stukel_fit.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_archived_example_fits(case):
    fit = fit_stukel(case["x"], case["successes"], case["trials"], family=case["family"])
    if case["family"] != 4:
        assert case["status"] in (3, 4, 5, 6)
        assert fit.objective.negative_log_likelihood == pytest.approx(case["objective"], abs=1e-6)
        np.testing.assert_allclose(fit.coefficients, case["coefficients"], rtol=5e-4, atol=1e-5)
    else:
        assert case["status"] == 8
        assert fit.objective.negative_log_likelihood < case["objective"]
        assert fit.alpha[1] == -fit.alpha[0]
    assert np.max(np.abs(fit.objective.gradient)) < 0.001
    assert fit.residual_df == len(case["x"]) - fit.coefficients.size
    assert fit.covariance is not None and np.all(fit.standard_errors > 0)
    np.testing.assert_allclose(
        fit.objective.hessian @ fit.covariance,
        fit.dispersion * np.eye(fit.coefficients.size),
        atol=1e-8,
    )
    np.testing.assert_allclose(fit.predict(case["x"]), fit.objective.probabilities)


def test_intercept_only_closed_form_estimate_covariance_and_dispersion():
    y, n = np.array([2, 3, 4, 5]), np.full(4, 10)
    fit = fit_stukel(np.ones((4, 1)), y, n, intercept=False, scale="fixed")
    p = y.sum() / n.sum()
    assert fit.coefficients[0] == pytest.approx(np.log(p / (1 - p)), abs=1e-7)
    assert fit.covariance[0, 0] == pytest.approx(1 / (n.sum() * p * (1 - p)), rel=1e-7)
    pearson = fit_stukel(np.ones((4, 1)), y, n, intercept=False)
    expected = np.sum((y - n * p) ** 2 / (n * p * (1 - p))) / 3
    assert pearson.dispersion == pytest.approx(expected)
    assert pearson.null_dispersion == pytest.approx(expected)
    assert fit.null_deviance == pytest.approx(fit.deviance, abs=1e-10)
    np.testing.assert_allclose(pearson.covariance, fit.covariance * expected)


def test_nonzero_fixed_shapes_use_transformed_fitted_probabilities():
    fit = fit_stukel(
        np.ones((4, 1)), [2, 3, 4, 5], [10] * 4, intercept=False, fixed_alpha=[-0.4, 0.2]
    )
    np.testing.assert_allclose(fit.objective.probabilities, 0.35, atol=1e-7)
    np.testing.assert_allclose(fit.predict(np.ones(4)), 0.35, atol=1e-7)


@pytest.mark.parametrize("x,y", [([-2, -1, 1, 2], [0, 0, 10, 10]), ([-1, 0, 0, 1], [0, 2, 8, 10])])
def test_complete_and_quasi_separation_are_explicit(x, y):
    with pytest.raises(StukelFitError, match="separation"):
        fit_stukel(x, y, [10] * 4)


def test_shape_bound_omits_unconstrained_covariance():
    case = REFERENCE["cases"][1]
    fit = fit_stukel(case["x"], case["successes"], case["trials"], family=1, shape_bound=0.05)
    assert fit.coefficients[-1] == pytest.approx(0.05)
    assert fit.active_bounds[-1] and fit.covariance is None and fit.standard_errors is None
    assert "bound" in fit.inference_message


def test_failed_convergence_and_rank_deficiency():
    case = REFERENCE["cases"][0]
    with pytest.raises(StukelFitError, match="optimizer failed"):
        fit_stukel(case["x"], case["successes"], case["trials"], max_iterations=1)
    with pytest.raises(StukelFitError, match="full-rank"):
        fit_stukel(np.ones((5, 1)), [2] * 5, [10] * 5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": 6},
        {"family": True},
        {"shape_bound": 0},
        {"shape_bound": np.nan},
        {"scale": "unknown"},
        {"max_iterations": True},
        {"fixed_alpha": [0]},
        {"start_alpha": [0]},
        {"initial": [0]},
        {"tolerance": 0},
    ],
)
def test_invalid_fit_controls(kwargs):
    with pytest.raises(ValueError):
        fit_stukel(np.arange(6), [2] * 6, [10] * 6, **kwargs)
