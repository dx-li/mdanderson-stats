import numpy as np
import pytest

from mdanderson_stats.bard_response import (
    bard_response_model,
    bard_response_probabilities,
)


def test_joint_profile_calibration_preserves_margins_and_conditional_odds() -> None:
    profiles = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    weights = np.array([0.4, 0.1, 0.2, 0.3])
    odds = np.array([[1.0, 2.5], [1.0, 0.6]])
    model = bard_response_model([0.17, 0.63], profiles, weights, odds)

    assert model.conditional_probabilities.shape == (2, 4)
    assert np.allclose(
        model.conditional_probabilities @ model.profile_probabilities,
        model.population_response,
        rtol=2e-12,
        atol=1e-14,
    )
    for dose in range(2):
        p11 = model.conditional_probabilities[dose, 0]
        p21 = model.conditional_probabilities[dose, 2]
        assert p21 / (1 - p21) / (p11 / (1 - p11)) == pytest.approx(2.5)
        p12 = model.conditional_probabilities[dose, 1]
        p22 = model.conditional_probabilities[dose, 3]
        assert p22 / (1 - p22) / (p12 / (1 - p12)) == pytest.approx(2.5)
    assert not model.intercepts.flags.writeable
    assert not model.profile_probabilities.flags.writeable
    assert model.probability(1, [2, 2]) == pytest.approx(model.conditional_probabilities[0, 3])


def test_probability_evaluator_handles_infinite_intercepts_and_padding_levels() -> None:
    profiles = [[1, 1], [2, 3], [4, 2]]
    odds = [[1.0, 2.0, 1.0, 7.0], [1.0, 3.0, 0.5, 1.0]]
    evaluated = bard_response_probabilities([-np.inf, 0.0, np.inf], profiles, odds)
    assert np.array_equal(evaluated[0], np.zeros(3))
    assert np.array_equal(evaluated[2], np.ones(3))
    assert evaluated[1, 1] == pytest.approx(0.5)
    assert not evaluated.flags.writeable


def test_degenerate_margins_and_tiny_marginal_are_explicit() -> None:
    model = bard_response_model(
        [0.0, 1.0, 1e-100],
        [[1], [2]],
        [0.5, 0.5],
        [[1.0, 4.0]],
    )
    assert model.intercepts[0] == -np.inf
    assert model.intercepts[1] == np.inf
    assert np.array_equal(model.conditional_probabilities[0], [0.0, 0.0])
    assert np.array_equal(model.conditional_probabilities[1], [1.0, 1.0])
    assert model.intercepts[2] < -200
    assert model.marginal_residuals[2] / model.population_response[2] == pytest.approx(0, abs=1e-10)

    subnormal = bard_response_model([1e-310], [[1]], [1.0], [[1.0]])
    assert subnormal.conditional_probabilities[0, 0] > 0.0
    assert subnormal.conditional_probabilities[0, 0] / 1e-310 == pytest.approx(1.0, rel=1e-12)

    # A double-precision plateau cannot identify the intercept, even though
    # the marginal objective appears to equal one half over a wide range.
    with pytest.raises(ArithmeticError, match="too flat"):
        bard_response_model([0.5], [[1], [2]], [0.5, 0.5], [[1.0, np.exp(700.0)]])


def test_correlated_profile_oracle_matches_independent_r_reference() -> None:
    profiles = [[1, 1], [2, 1], [1, 2], [2, 2], [1, 3], [2, 3]]
    weights = [0.40, 0.05, 0.10, 0.15, 0.05, 0.25]
    # The unused third level in factor 1 pads its row to the shared width.
    odds = [[1.0, 2.5, 1.0], [1.0, 0.6, 3.0]]
    model = bard_response_model([0.42], profiles, weights, odds)
    assert model.intercepts[0] == pytest.approx(-0.9637837, abs=5e-8)
    assert model.marginal_residuals[0] == pytest.approx(0.0, abs=1e-12)
    assert model.conditional_probabilities[0] == pytest.approx(
        [0.2761, 0.4881, 0.1863, 0.3640, 0.5337, 0.7410], abs=8e-5
    )


def test_invalid_profiles_or_weights_fail_before_calibration() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        bard_response_model([0.3], [[1], [2]], [0.4, 0.5], [[1.0, 2.0]])
    with pytest.raises(ValueError, match="category labels"):
        bard_response_model([0.3], [[0], [2]], [0.5, 0.5], [[1.0, 2.0]])
