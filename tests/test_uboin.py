import numpy as np
import pytest

from mdanderson_stats.uboin import UBOINPosterior, uboin_allocation, uboin_posterior


def test_binary_posterior_mean_variance_and_marginal_tails() -> None:
    counts = np.array([[[2, 1], [3, 4]], [[0, 0], [0, 0]]])
    prior = np.full((2, 2), 0.25)
    utilities = np.array([[0, 30], [50, 100]])
    result = uboin_posterior(
        counts, prior=prior, utilities=utilities, toxicity_limit=0.5, efficacy_limit=0.5
    )

    assert isinstance(result, UBOINPosterior)
    assert np.array_equal(result.posterior_shape[0], prior + counts[0])
    expected_mean = np.sum((prior + counts[0]) * utilities) / (1 + counts[0].sum())
    assert result.mean_utility[0] == pytest.approx(expected_mean)
    probabilities = (prior + counts[0]) / (1 + counts[0].sum())
    expected_variance = (
        np.sum(probabilities * utilities**2) - expected_mean**2
    ) / (1 + counts[0].sum() + 1)
    assert result.utility_variance[0] == pytest.approx(expected_variance)
    assert result.overdose_probability[1] == pytest.approx(0.5)
    assert result.low_efficacy_probability[1] == pytest.approx(0.5)


def test_categorical_collapse_uses_configured_levels() -> None:
    counts = np.zeros((1, 3, 3), dtype=int)
    counts[0, 0, 2] = 4  # severe toxicity, no response
    counts[0, 2, 0] = 6  # minor toxicity, response
    result = uboin_posterior(
        counts,
        prior=np.full((3, 3), 1 / 9),
        utilities=np.arange(9, dtype=float).reshape(3, 3),
        dlt_level=2,
        response_level=2,
    )
    # Toxicity is columns 2+; efficacy is rows 2+ after the requested collapse.
    assert result.n[0] == 10
    assert 0 < result.overdose_probability[0] < 1
    assert 0 < result.low_efficacy_probability[0] < 1


def test_allocation_tie_lowest_winner_and_empty_selection() -> None:
    posterior = uboin_posterior(
        np.zeros((3, 2, 2), dtype=int),
        prior=np.full((2, 2), 0.25),
        utilities=np.full((2, 2), 50.0),
    )
    assert np.array_equal(uboin_allocation(posterior, eligible=[True, True, True]), [1, 0, 0])
    assert np.array_equal(
        uboin_allocation(posterior, eligible=[True, True, True], method="equal"), [1 / 3] * 3
    )
    assert np.array_equal(uboin_allocation(posterior, eligible=[False, False, False]), [0, 0, 0])


def test_proportional_allocation_and_readonly_outputs() -> None:
    posterior = uboin_posterior(
        np.zeros((2, 2, 2), dtype=int),
        prior=np.full((2, 2), 0.25),
        utilities=np.array([[0, 0], [0, 100]]),
    )
    probabilities = uboin_allocation(
        posterior, eligible=np.array([True, True]), method="proportional"
    )
    assert np.array_equal(probabilities, [0.5, 0.5])
    assert not probabilities.flags.writeable
    assert not posterior.posterior_shape.flags.writeable


def test_invalid_shape_threshold_and_zero_proportional_utility() -> None:
    with pytest.raises(ValueError):
        uboin_posterior(np.zeros((2, 2)), prior=np.ones((2, 2)), utilities=np.ones((2, 2)))
    with pytest.raises(ValueError):
        uboin_posterior(
            np.zeros((1, 2, 2)), prior=np.ones((2, 2)), utilities=np.ones((2, 2)), dlt_level=2
        )
    posterior = uboin_posterior(
        np.zeros((1, 2, 2), dtype=int), prior=np.full((2, 2), 0.25), utilities=np.zeros((2, 2))
    )
    with pytest.raises(ValueError):
        uboin_allocation(posterior, eligible=[True], method="proportional")
