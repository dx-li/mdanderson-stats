import numpy as np
import pytest

from mdanderson_stats.toxfinder_model import (
    ToxFinderPrior,
    fit_toxfinder,
    toxfinder_log_likelihood,
    toxfinder_probabilities,
    toxfinder_standardize,
)


def test_probability_matches_model_and_zero_axes() -> None:
    params = np.array([0.4, 1.5, 0.7, 2.0, 0.8, 1.2])
    doses = np.array([[0.0, 0.0], [0.5, 0], [0, 0.25], [0.5, 0.25]])
    q = 0.4 * 0.5**1.5 + 0.7 * 0.25**2 + 0.8 * (0.5**1.5 * 0.25**2) ** 1.2
    expected = np.array(
        [
            0.0,
            0.4 * 0.5**1.5 / (1 + 0.4 * 0.5**1.5),
            0.7 * 0.25**2 / (1 + 0.7 * 0.25**2),
            q / (1 + q),
        ]
    )
    actual = toxfinder_probabilities(doses, params)[:, 1]
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    assert actual[0] == 0
    zero_interaction = params.copy()
    zero_interaction[4] = 0
    assert toxfinder_probabilities(doses[-1], zero_interaction)[0, 1] < actual[-1]
    assert toxfinder_standardize([[10, 20]], [10, 40]).tolist() == [[1, 0.5]]


def test_grouped_likelihood_and_origin_contract() -> None:
    parameters = np.array([0.4, 1.5, 0.7, 2, 0.8, 1.2])
    doses = np.array([[0.5, 0.25], [0, 0]])
    value = toxfinder_log_likelihood(doses, [1, 0], [3, 2], parameters)
    expected = np.log(toxfinder_probabilities(doses[0], parameters)[0, 1]) + 2 * np.log(
        1 - toxfinder_probabilities(doses[0], parameters)[0, 1]
    )
    assert value == pytest.approx(expected)
    with pytest.raises(ValueError, match="origin"):
        toxfinder_log_likelihood(doses, [0, 1], [3, 2], parameters)


def test_tiny_shape_prior_draws_remain_finite_and_fit_is_reproducible() -> None:
    prior = ToxFinderPrior(
        mean=(0.4286, 7.6494, 0.4286, 7.8019, 1, 0.05),
        variance=(0.1054, 5.7145, 0.0791, 3.9933, 3, 3),
    )
    args = dict(
        doses=np.array([[0.0, 0.0], [0.5, 0.5]]),
        toxicities=[0, 1],
        subjects=[1, 3],
        prior=prior,
        draws=8,
        warmup=1,
        chains=2,
    )
    fit1 = fit_toxfinder(**args, rng=np.random.default_rng(12))
    fit2 = fit_toxfinder(**args, rng=np.random.default_rng(12))
    np.testing.assert_array_equal(fit1.log_parameters, fit2.log_parameters)
    assert np.all(np.isfinite(fit1.log_parameters))
    assert fit1.log_parameters[..., 5].min() < -700
    assert not fit1.log_parameters.flags.writeable
    assert not fit1.probabilities.flags.writeable


def test_resource_bounds_are_checked() -> None:
    with pytest.raises(ValueError, match="100 dose pairs"):
        toxfinder_standardize(np.ones((101, 2)), [1, 1])
    with pytest.raises(ValueError, match="10,000"):
        toxfinder_log_likelihood([[1, 1]], [0], [10_001], [1, 1, 1, 1, 1, 1])


def test_extreme_log_parameters_and_subnormal_dose() -> None:
    logs = np.array([-np.inf, 1000, -np.inf, 1000, 0, -1000])
    p = toxfinder_probabilities([[0.5, 0.5]], logs, log_parameters=True)[0, 1]
    assert p == pytest.approx(0.2)
    tiny = np.nextafter(0.0, 1.0)
    from mdanderson_stats.toxfinder_model import toxfinder_log_probabilities

    logp = toxfinder_log_probabilities([[tiny, 0]], [1, 0.25, 0, 1, 0, 1])[0, 1]
    assert logp == pytest.approx(-186.11, abs=0.02)


def test_cartesian_batched_probabilities() -> None:
    parameters = np.ones((2, 3, 6))
    result = toxfinder_probabilities([[0, 0], [1, 0]], parameters)
    assert result.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(result[..., 0, 1], 0)
