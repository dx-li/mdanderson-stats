import numpy as np
import pytest

from mdanderson_stats import ParameterDistribution as Prior
from mdanderson_stats import logistic_regression_ess, normal_regression_ess


def test_logistic_published_table_and_subvectors():
    dose = np.log(np.arange(1, 7) * 100)
    support = np.c_[np.ones(6), dose - dose.mean()]
    expected = [
        [37.1, 22.7, 101.3],
        [9.3, 5.7, 25.3],
        [2.3, 1.4, 6.3],
        [1.0, 0.6, 2.8],
        [0.4, 0.2, 1.0],
    ]
    for sd, target in zip([0.5, 1, 2, 3, 5], expected):
        result = logistic_regression_ess(
            [Prior("normal", -0.1313, sd**2), Prior("normal", 2.398, sd**2)], support
        )
        # Published values are Monte Carlo estimates reported to one decimal.
        np.testing.assert_allclose(np.r_[result.ess, result.component_ess], target, atol=0.06)
        assert result.subvector_ess([0, 1]) == result.ess
        assert result.subvector_ess([1]) == pytest.approx(result.component_ess[1])
    assert not result.component_ess.flags.writeable


def test_normal_regression_published_examples_and_gamma_parameterization():
    for variances, shape, expected in [
        ([1000, 1000], 0.001, [0.001, 0.002]),
        ([100, 10], 1, [0.055, 2]),
        ([1, 1], 2, [1, 4]),
    ]:
        result = normal_regression_ess(
            [Prior("normal", 0, v) for v in variances], [1, 1], Prior("gamma", shape, 1 / shape)
        )
        np.testing.assert_allclose(
            [result.subvector_ess([0, 1]), result.component_ess[-1]],
            np.array(expected) * 0.9999,
            rtol=1e-12,
        )
    mixed = normal_regression_ess([Prior("gamma", 0.5, 2)], [3], Prior("gamma", 2, 0.5))
    # Gamma(shape=.5,scale=2) has variance 2. Precision mean is one.
    assert mixed.component_ess[0] == pytest.approx(0.9999 / 6)
    assert mixed.component_ess[1] == pytest.approx(4 * 0.9999)
    assert mixed.parameter_names[-1] == "precision"


def test_curvature_matches_independent_numerical_derivatives():
    priors = [Prior("normal", 0.2, 2), Prior("gamma", 0.5, 1.6)]
    support = np.array([[1, -1], [1, 2], [1, 0.5]])
    weights = np.array([0.2, 0.3, 0.5])
    result = logistic_regression_ess(priors, support, probabilities=weights)
    theta = np.array([0.2, 0.8])
    y = np.array([0, 1, 1])

    def log_likelihood(value):
        eta = support @ value
        return weights @ (y * eta - np.logaddexp(0, eta))

    def log_prior_ratio(value):
        normal = -0.5 * (value[0] - 0.2) ** 2 * (1 / 2 - 1 / 20000)
        # Negative individual gamma curvatures still have a positive difference.
        gamma = (0.5 - 0.5 / 10000) * np.log(value[1]) - value[1] * (1 / 1.6 - 1 / 16000)
        return normal + gamma

    for j in range(2):
        h = np.eye(2)[j] * 0.0002
        for function, expected in [
            (log_likelihood, np.exp(result.log_observation_information[j])),
            (log_prior_ratio, np.exp(result.log_prior_information_gain[j])),
        ]:
            numerical = (
                -(function(theta + h) - 2 * function(theta) + function(theta - h)) / 0.0002**2
            )
            assert numerical == pytest.approx(expected, rel=2e-6)


def test_extreme_predictors_weights_and_zero_information():
    extreme = logistic_regression_ess([Prior("normal", 1000, 1)], [[1]])
    assert extreme.log_ess == pytest.approx(1000 + np.log(0.9999))
    with pytest.raises(ArithmeticError, match="log ESS"):
        _ = extreme.ess
    priors = [Prior("normal", 0, 1), Prior("normal", 0, 2)]
    unsupported = logistic_regression_ess(priors, [[1, 0]])
    assert np.isinf(unsupported.component_ess[1])
    assert np.isfinite(unsupported.ess)
    result = logistic_regression_ess(priors, [[1, -2], [1, 2]], probabilities=[1e308, 1e308])
    reference = logistic_regression_ess(priors, [[1, -2], [1, 2]])
    np.testing.assert_allclose(result.component_ess, reference.component_ess)
    with pytest.raises(ValueError):
        result.subvector_ess([0, 0])
