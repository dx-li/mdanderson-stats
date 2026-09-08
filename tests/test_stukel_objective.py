import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import stukel_objective

REFERENCE = json.loads((Path(__file__).parent / "fixtures/stukel_objective.json").read_text())


def evaluate(case, coefficients=None):
    return stukel_objective(
        case["x"],
        case["successes"],
        case["trials"],
        case["coefficients"] if coefficients is None else coefficients,
        family=case["family"],
        fixed_alpha=case["fixed_alpha"],
    )


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_original_fgh_objective_and_derivatives(case):
    result = evaluate(case)
    reference = case.get("chain_rule_reference", case["reference"])
    assert result.negative_log_likelihood == pytest.approx(reference["objective"], rel=2e-13)
    np.testing.assert_allclose(result.gradient, reference["gradient"], rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(result.hessian, reference["hessian"], rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=1e-14)


@pytest.mark.parametrize(
    "case", [c for c in REFERENCE["cases"] if all(v != 0 for v in c["coefficients"][2:])]
)
def test_independent_finite_difference_gradient_and_hessian(case):
    coef = np.array(case["coefficients"])
    result = evaluate(case)
    for j in range(coef.size):
        step = 1e-5
        left, right = coef.copy(), coef.copy()
        left[j] -= step
        right[j] += step
        lo, hi = evaluate(case, left), evaluate(case, right)
        score = (hi.negative_log_likelihood - lo.negative_log_likelihood) / (2 * step)
        curvature = (hi.gradient - lo.gradient) / (2 * step)
        assert result.gradient[j] == pytest.approx(score, rel=2e-7, abs=2e-8)
        np.testing.assert_allclose(result.hessian[:, j], curvature, rtol=2e-6, atol=2e-7)


def test_opposite_shape_family_corrects_native_sign_error():
    case = next(c for c in REFERENCE["cases"] if c["family"] == 4 and c["coefficients"][-1] == 0.2)
    result = evaluate(case)
    assert not np.isclose(result.gradient[-1], case["reference"]["gradient"][-1])
    assert result.negative_log_likelihood == pytest.approx(case["reference"]["objective"])


def test_standard_logistic_likelihood_and_information():
    x = np.array([[1, -2], [1, -1], [1, 1], [1, 2]])
    y, n = np.array([0, 1, 3, 4]), np.full(4, 4)
    coef = np.array([0.2, 0.7])
    result = stukel_objective(x, y, n, coef)
    p = 1 / (1 + np.exp(-(x @ coef)))
    assert result.negative_log_likelihood == pytest.approx(
        -np.sum(y * np.log(p) + (n - y) * np.log1p(-p))
    )
    np.testing.assert_allclose(result.gradient, x.T @ (n * p - y))
    np.testing.assert_allclose(result.hessian, x.T @ ((n * p * (1 - p))[:, None] * x))


def test_saturated_correct_predictions_retain_tiny_likelihood():
    result = stukel_objective([[1], [-1]], [10, 0], [10, 10], [40])
    assert 0 < result.negative_log_likelihood < 1e-15
    assert result.gradient[0] < 0 and result.hessian[0, 0] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": 6},
        {"family": True},
        {"fixed_alpha": [0]},
        {"coefficients": [1]},
        {"successes": [11] * 6},
        {"trials": [0] * 6},
        {"successes": [0.5] * 6},
        {"x": [[np.nan, 1]] * 6},
    ],
)
def test_invalid_objective_inputs(kwargs):
    case = REFERENCE["cases"][0]
    inputs = {
        key: case[key]
        for key in ("x", "successes", "trials", "coefficients", "family", "fixed_alpha")
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError):
        stukel_objective(**inputs)
