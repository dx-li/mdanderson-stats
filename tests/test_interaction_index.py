import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import t

from mdanderson_stats import (
    MedianEffectFit,
    fit_median_effect,
    interaction_index,
    interaction_index_ray,
)


def models():
    return [
        MedianEffectFit(0.2, 1.2, np.array([[0.02, -0.003], [-0.003, 0.01]]), 12, 0.1),
        MedianEffectFit(-0.3, 0.8, np.array([[0.01, 0.002], [0.002, 0.02]]), 10, 0.1),
    ]


def numerical_gradient(function, point):
    gradient = np.empty_like(point)
    for i in range(len(point)):
        delta = np.zeros_like(point)
        delta[i] = 1e-5
        gradient[i] = (function(point + delta) - function(point - delta)) / 2e-5
    return gradient


def test_centered_regression_and_dose_inversion():
    x = np.arange(-2.0, 3.0)
    errors = np.array([0.1, -0.2, 0.2, -0.2, 0.1])
    fit = fit_median_effect(np.exp(x), expit(0.3 - 0.7 * x + errors))
    assert fit.intercept == pytest.approx(0.3)
    assert fit.slope == pytest.approx(-0.7)
    assert fit.residual_variance == pytest.approx(0.14 / 3)
    np.testing.assert_allclose(fit.covariance, np.diag([0.14 / 15, 0.14 / 30]), atol=1e-16)
    np.testing.assert_allclose(fit.log_dose(fit.effect(np.exp(x))), x, atol=1e-14)
    shifted = fit_median_effect(np.exp(x) * 1e100, expit(0.3 - 0.7 * x + errors))
    np.testing.assert_allclose(shifted.effect(np.exp(x) * 1e100), fit.effect(np.exp(x)), atol=1e-13)


def test_paper_three_drug_additivity_scenario():
    medians = np.array([1.0, 2.0, 4.0])
    curves = [MedianEffectFit(np.log(d), -1.0, np.zeros((2, 2)), 10, 0) for d in medians]
    effects = np.array([1 / 6, 2 / 7, 0.375, 4 / 9, 0.5, 5 / 9, 5 / 7, 5 / 6])
    result = interaction_index(curves, medians / 3, effects, effect_variance=0)
    np.testing.assert_allclose(result.index, effects / (1 - effects))
    np.testing.assert_allclose(result.interval, np.repeat(result.index[:, None], 2, axis=1))
    assert result.degrees_of_freedom == 24


def test_observed_effect_delta_variance_and_t_interval():
    curves = models()
    doses = np.array([0.3, 0.7])
    effect, variance = 0.4, 0.002
    parameters = np.array([0.2, 1.2, -0.3, 0.8, effect])

    def log_index(p):
        inverse = np.exp((np.log(p[-1] / (1 - p[-1])) - p[:4:2]) / p[1:4:2])
        return np.log(np.sum(doses / inverse))

    gradient = numerical_gradient(log_index, parameters)
    expected = (
        sum(
            gradient[2 * i : 2 * i + 2] @ f.covariance @ gradient[2 * i : 2 * i + 2]
            for i, f in enumerate(curves)
        )
        + gradient[-1] ** 2 * variance
    )
    result = interaction_index(curves, doses, effect, effect_variance=variance)
    assert result.log_standard_error**2 == pytest.approx(expected, rel=2e-8)
    expected_interval = log_index(parameters) + np.array([-1, 1]) * t.isf(0.025, 18) * np.sqrt(
        expected
    )
    np.testing.assert_allclose(result.log_interval, expected_interval, rtol=2e-8)


def test_fixed_ray_delta_variance_with_combination_uncertainty():
    curves = models()
    combination = MedianEffectFit(0.1, 1.1, np.array([[0.03, -0.005], [-0.005, 0.02]]), 15, 0.1)
    effect = 0.7
    parameters = np.array([0.2, 1.2, -0.3, 0.8, 0.1, 1.1])

    def log_index(p):
        inverse = np.exp((np.log(effect / (1 - effect)) - p[::2]) / p[1::2])
        return np.log(inverse[-1] * np.sum(np.array([0.3, 0.7]) / inverse[:2]))

    gradient = numerical_gradient(log_index, parameters)
    expected = sum(
        gradient[2 * i : 2 * i + 2] @ f.covariance @ gradient[2 * i : 2 * i + 2]
        for i, f in enumerate([*curves, combination])
    )
    result = interaction_index_ray(curves, combination, [3, 7], effect)
    assert result.log_index == pytest.approx(log_index(parameters))
    assert result.log_standard_error**2 == pytest.approx(expected, rel=2e-8)
    assert result.degrees_of_freedom == 31
    vector = interaction_index_ray(curves, combination, [3e300, 7e300], [0.2, 0.7])
    assert vector.log_index[1] == pytest.approx(result.log_index)


def test_log_scale_extremes_and_invalid_models():
    fit = MedianEffectFit(1000, 1, np.zeros((2, 2)), 3, 0)
    result = interaction_index([fit, fit], [1.0, 1.0], 0.5, effect_variance=0)
    assert result.log_index == pytest.approx(1000 + np.log(2))
    with pytest.raises(ArithmeticError, match="log outputs"):
        _ = result.index
    with pytest.raises(ValueError, match="positive semidefinite"):
        MedianEffectFit(0, 1, np.array([[1, 2], [2, 1]]), 3, 0)
    with pytest.raises(ValueError, match="variation"):
        fit_median_effect([1, 1, 1], [0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="positive total"):
        interaction_index(models(), [0, 0], 0.5, effect_variance=0)
