import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.beta_mixture import BetaMixture

REFERENCE = json.loads((Path(__file__).parent / "fixtures/beta_mixture.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["evaluations"])
def test_original_model_density_cdf_posterior_likelihood_and_cvm(case):
    model = BetaMixture(**case["model"])
    x = np.sort(case["pvalues"])
    np.testing.assert_allclose(
        np.exp(model.logpdf(x, legacy_endpoints=True)), case["sorted_density"], rtol=3e-12, atol=0
    )
    np.testing.assert_allclose(model.cdf(x), case["sorted_cdf"], rtol=3e-12, atol=0)
    np.testing.assert_allclose(
        model.null_posterior(x, legacy_endpoints=True),
        case["sorted_null_posterior"],
        rtol=3e-12,
        atol=0,
    )
    assert model.log_likelihood(x, legacy_endpoints=True) == pytest.approx(
        case["log_likelihood"], abs=2e-10
    )
    assert model.cramer_von_mises(x) == pytest.approx(case["cramer_von_mises"], rel=3e-12)


def test_beta_polynomial_and_density_based_bayes_formula():
    x = np.linspace(0, 1, 51)
    model = BetaMixture(0.2, [0.8], [2], [3])
    density = 0.2 + 0.8 * 12 * x * (1 - x) ** 2
    cdf = 0.2 * x + 0.8 * (6 * x**2 - 8 * x**3 + 3 * x**4)
    np.testing.assert_allclose(np.exp(model.logpdf(x)), density, rtol=3e-14)
    np.testing.assert_allclose(model.cdf(x), cdf, rtol=3e-14)
    np.testing.assert_allclose(model.null_posterior(x), 0.2 / density, rtol=3e-14)


def test_uniform_cvm_scaling_and_batches():
    model = BetaMixture(1)
    x = np.array([[0.8, 0.1, 0.5], [0.4, 0.3, 0.9]])
    expected = [
        sum((v - (i + 0.5) / 3) ** 2 for i, v in enumerate(sorted(row))) / 3 + 1 / 108 for row in x
    ]
    np.testing.assert_allclose(model.cramer_von_mises(x), expected)
    np.testing.assert_array_equal(model.log_likelihood(x), 0)
    np.testing.assert_array_equal(model.null_posterior(x), 1)
    assert model.cdf(0.3) == 0.3
    assert model.cdf([]).size == 0


def test_endpoint_limits_and_zero_weight_components():
    model = BetaMixture(1, [0], [0.1], [0.1])
    np.testing.assert_array_equal(model.logpdf([0, 1]), 0)
    np.testing.assert_array_equal(model.cdf([0, 1]), [0, 1])
    singular = BetaMixture(0.5, [0.5], [0.5], [0.5])
    assert np.all(np.isposinf(singular.logpdf([0, 1])))
    np.testing.assert_array_equal(singular.null_posterior([0, 1]), 0)
    assert np.all(np.isfinite(singular.logpdf([0, 1], legacy_endpoints=True)))
    pure = BetaMixture(0, [1], [2], [2])
    assert np.all(np.isneginf(pure.logpdf([0, 1])))
    with pytest.raises(ValueError, match="zero mixture density"):
        pure.null_posterior([0, 1])
    np.testing.assert_array_equal(pure.null_posterior([0, 1], legacy_endpoints=True), 0)


def test_log_density_survives_underflow_with_independent_decimal_calculation():
    x = 1e-200
    model = BetaMixture(0, [1], [1000], [1000])
    with localcontext() as context:
        context.prec = 80
        coefficient = Decimal(math.factorial(1999)) / Decimal(math.factorial(999)) ** 2
        value = Decimal.from_float(x)
        expected = float(coefficient.ln() + 999 * value.ln() + 999 * (1 - value).ln())
    actual = float(model.logpdf(x))
    assert np.isfinite(actual) and np.exp(actual) == 0
    assert actual == pytest.approx(expected, rel=2e-14)


def test_posterior_calibration_under_the_specified_model():
    rng = np.random.default_rng(691)
    is_null = rng.uniform(size=50000) < 0.7
    p = np.where(is_null, rng.uniform(size=50000), rng.beta(0.4, 8, size=50000))
    posterior = BetaMixture(0.7, [0.3], [0.4], [8]).null_posterior(p)
    assert posterior.mean() == pytest.approx(0.7, abs=0.005)
    for low, high in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1)):
        group = (posterior >= low) & (posterior < high)
        assert group.sum() > 300
        assert is_null[group].mean() == pytest.approx(posterior[group].mean(), abs=0.035)


def test_model_copies_inputs_and_component_order_does_not_matter():
    w = np.array([0.3, 0.2])
    model = BetaMixture(0.5, w, [0.4, 2], [8, 3])
    w[0] = 0
    assert model.weights[0] == 0.3
    with pytest.raises(ValueError):
        model.a[0] = 1
    permuted = BetaMixture(0.5, [0.2, 0.3], [2, 0.4], [3, 8])
    x = np.array([[0.1, 0.8], [0.5, 0.2]])
    np.testing.assert_allclose(model.logpdf(x), permuted.logpdf(x), atol=1e-15)
    np.testing.assert_allclose(model.cdf(x), permuted.cdf(x), atol=1e-15)


@pytest.mark.parametrize(
    "args",
    [
        (0.5, [], [], []),
        (-1, [2], [1], [1]),
        (0, [1], [0], [1]),
        (0, [1], [1], [np.nan]),
        (0.5, [0.5], [1, 2], [1]),
        (0, [[1]], [1], [1]),
        (0, [1], [1e308], [1e308]),
    ],
)
def test_invalid_model(args):
    with pytest.raises(ValueError):
        BetaMixture(*args)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-0.1], [1.1]])
def test_invalid_observations(values):
    model = BetaMixture(1)
    for method in (model.logpdf, model.cdf, model.null_posterior, model.cramer_von_mises):
        with pytest.raises(ValueError):
            method(values)
