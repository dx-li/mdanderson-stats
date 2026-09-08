import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import predict_stukel, stukel_log_odds, stukel_probability

REFERENCE = json.loads((Path(__file__).parent / "fixtures/stukel.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"])
def test_native_fortran_link(case):
    result = stukel_log_odds(case["eta"], case["alpha1"], case["alpha2"])
    # The native small-product branch truncates its Taylor series after six terms.
    np.testing.assert_allclose(result, case["log_odds"], rtol=3e-13, atol=1e-15)


@pytest.mark.parametrize(
    "eta, a", [(2, -0.5), (-3, 0.25), (1e-9, 1e-9), (7.1e-306, 1e308), (1e308, -1e308)]
)
def test_high_precision_link(eta, a):
    with localcontext() as ctx:
        ctx.prec = 90
        e, shape = Decimal(str(abs(eta))), Decimal(str(a))
        h = ((shape * e).exp() - 1) / shape if a > 0 else -(1 - shape * e).ln() / shape
        expected = float(h) * np.sign(eta)
    assert float(stukel_log_odds(eta, a, a)) == pytest.approx(expected, rel=3e-13, abs=1e-320)


def test_logistic_symmetry_monotonicity_and_broadcasting():
    eta = np.linspace(-10, 10, 101)
    np.testing.assert_allclose(stukel_probability(eta), 1 / (1 + np.exp(-eta)))
    probabilities = stukel_probability(eta[:, None], [-1, 0, 1], [-1, 0, 1])
    assert probabilities.shape == (101, 3)
    assert np.all(np.diff(probabilities, axis=0) >= 0)
    np.testing.assert_allclose(probabilities, 1 - probabilities[::-1], atol=2e-16)
    assert np.all(probabilities[50] == 0.5)


def test_extreme_predictors_saturate_without_nan():
    np.testing.assert_array_equal(stukel_probability([-1e308, 0, 1e308], 2, 2), [0, 0.5, 1])
    # A subnormal product must not erase the ordinary logistic limit.
    assert stukel_log_odds(1e-10, 5e-324) == 1e-10


def test_predictor_bug_is_only_reproduced_when_requested():
    corrected = stukel_log_odds(2, -0.5)
    assert corrected == pytest.approx(2 * np.log(2))
    assert stukel_log_odds(2, -0.5, legacy_prediction=True) == 2
    x = np.array([[[1, 2], [3, 4]], [[-1, 0], [1, -2]]])
    result = predict_stukel(x, [0.5, 1, -1], -0.5, 0.3)
    np.testing.assert_allclose(result, stukel_probability(0.5 + x[..., 0] - x[..., 1], -0.5, 0.3))
    np.testing.assert_allclose(
        predict_stukel(x, [1, -1], intercept=False), stukel_probability(x[..., 0] - x[..., 1])
    )


@pytest.mark.parametrize("kwargs", [{"eta": np.nan}, {"eta": np.inf}, {"eta": 1, "alpha1": np.nan}])
def test_invalid_link_inputs(kwargs):
    with pytest.raises(ValueError):
        stukel_log_odds(**kwargs)


@pytest.mark.parametrize(
    "x, beta, kwargs",
    [
        ([1, 2], [1, 2], {}),
        ([[1, 2]], [1], {}),
        ([[1]], [1, 1], {"alpha1": [1]}),
        ([[1]], [1, 1], {"intercept": 1}),
    ],
)
def test_invalid_prediction_inputs(x, beta, kwargs):
    with pytest.raises(ValueError):
        predict_stukel(x, beta, **kwargs)
