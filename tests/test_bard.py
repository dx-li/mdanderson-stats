import numpy as np
import pytest

from mdanderson_stats.bard import bard_minimization, bard_select_obd


def test_minimization_matches_official_example_rows():
    history_arms = np.array([1, 2, 2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1])
    history_factors = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 1],
            [2, 1, 1],
            [3, 1, 1],
            [1, 1, 2],
            [2, 2, 2],
            [3, 2, 1],
            [1, 1, 2],
            [3, 2, 1],
            [2, 2, 2],
            [2, 1, 2],
            [1, 1, 1],
            [2, 2, 2],
            [1, 2, 1],
        ]
    )
    result = bard_minimization(history_arms, history_factors, [2, 2, 1], seed=1)
    assert result.scores.tolist() == [8, 10]
    assert np.allclose(result.probabilities, [0.95, 0.05])
    assert result.assigned_arm == 1
    assert not result.scores.flags.writeable


def test_minimization_empty_history_and_tie():
    result = bard_minimization(np.array([], dtype=int), np.empty((0, 2), dtype=int), [1, 2], seed=1)
    assert result.scores.tolist() == [2, 2]
    assert result.probabilities.tolist() == [0.5, 0.5]


def test_obd_posterior_and_isotonic_safety():
    result = bard_select_obd(
        [[7, 1, 2, 0], [0, 4, 1, 5]],
        prior=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        safety_weights=[1, 3],
    )
    assert np.allclose(result.mean_utility, [17.3636363636364, 63.5454545454545])
    assert np.allclose(result.adjusted_overdose_probability, [0.314673300856639] * 2)
    assert result.selected_arm == 2
    assert not result.admissible.flags.writeable


def test_noninferiority_uses_guide_negative_margin_and_requires_data():
    result = bard_select_obd(
        [[0, 0, 2, 2], [0, 0, 3, 1]],
        prior=[0.25] * 4,
        safety_weights=[1, 1],
        method="noninferiority",
        margin=0.05,
    )
    assert result.selected_arm == 1
    with pytest.raises(ValueError, match="at least one observed"):
        bard_select_obd(
            [[0, 0, 0, 0], [0, 0, 1, 0]],
            prior=[0.25] * 4,
            safety_weights=[1, 1],
            method="noninferiority",
        )


def test_noninferiority_exact_margin_and_zero_observation_reporting():
    kwargs = dict(
        prior=[0.25] * 4,
        safety_weights=[1, 1],
        method="noninferiority",
        safety_cutoff=1,
        efficacy_cutoff=1,
    )
    counts = [[0, 6, 0, 4], [0, 11, 0, 9]]
    equal = bard_select_obd(counts, margin=0.05, **kwargs)
    below = bard_select_obd(counts, margin=np.nextafter(0.05, 0), **kwargs)
    assert equal.selected_arm == 1
    assert below.selected_arm == 2
    zero = bard_select_obd([[0, 0, 0, 0], [0, 0, 1, 0]], prior=[0.25] * 4, safety_weights=[1, 1])
    assert np.isnan(zero.observed_rates[0])


def test_large_finite_prior_and_weights_are_stable():
    result = bard_select_obd(
        [[7, 1, 2, 0], [0, 4, 1, 5]],
        prior=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        safety_weights=[1e308, 1e308],
    )
    reference = bard_select_obd(
        [[7, 1, 2, 0], [0, 4, 1, 5]],
        prior=[[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
        safety_weights=[1, 1],
    )
    assert np.all(np.isfinite(result.mean_utility))
    assert np.all(np.isfinite(result.adjusted_overdose_probability))
    assert np.allclose(
        result.adjusted_overdose_probability, reference.adjusted_overdose_probability
    )


def test_minimization_probability_and_integer_validation():
    with pytest.raises(ValueError, match="probability"):
        bard_minimization([], np.empty((0, 1), dtype=int), [1], probability=0.49)
    with pytest.raises(ValueError, match="tie_arm"):
        bard_select_obd(
            [[0, 0, 1, 0], [0, 0, 1, 0]], prior=[0.25] * 4, safety_weights=[1, 1], tie_arm=[1, 2]
        )
