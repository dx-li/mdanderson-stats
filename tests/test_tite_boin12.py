import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12 import tite_boin12_decision, tite_boin12_posterior


def design() -> BOIN12Design:
    return BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))


def test_effective_sample_size_and_pending_conditional_mean() -> None:
    result = tite_boin12_posterior(
        design(),
        [1, 1],
        [1, -1],
        [0, -1],
        [1.0, 0.5],
        [2.0, 1.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
    )
    # Toxicity: one event plus 0.5 pending follow-up; efficacy: no event plus
    # 0.5 pending follow-up.
    assert np.allclose(result.ESS[:, 0], [1.5, 0.0])
    assert np.isclose(result.MLE[0, 0], 2 / 3)
    assert np.isnan(result.MLE[1, 0])
    assert np.isclose(result.patient_conditional_means[1, 0], (2 / 3) * 0.5 / (1 - (2 / 3) * 0.5))


def test_complete_joint_cells_and_nonadditive_utility() -> None:
    result = tite_boin12_posterior(
        design(),
        [1, 1, 1, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
    )
    assert np.allclose(result.expected_joint_counts, [[1, 1, 1, 1]])
    assert np.isclose(result.posterior.utility_events[0], 1.95)


def test_strict_pending_fraction_suspends_at_more_than_half() -> None:
    args = ([1, 1, 1], [0, -1, -1], [0, -1, -1], [1, 0.4, 0.4], [2, 0.4, 0.4])
    result = tite_boin12_decision(
        design(),
        *args,
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
        current_dose=1,
    )
    assert result.action == "suspend_pending"
    assert result.posterior is None


def test_elimination_is_sticky() -> None:
    result = tite_boin12_decision(
        design(),
        [1, 2],
        [0, 0],
        [1, 1],
        [1, 1],
        [2, 2],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=1,
        eliminated=[False, True],
    )
    assert bool(result.eliminated[1])


def test_unit_rescaling_followup_and_window_preserves_posterior() -> None:
    kwargs = dict(n_doses=1, current_dose=1)
    first = tite_boin12_decision(
        design(),
        [1, 1],
        [0, -1],
        [1, -1],
        [1, 0.5],
        [2, 1.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        **kwargs,
    )
    second = tite_boin12_decision(
        design(),
        [1, 1],
        [0, -1],
        [1, -1],
        [10, 5],
        [20, 10],
        toxicity_window=10.0,
        efficacy_window=20.0,
        **kwargs,
    )
    assert first.action == second.action
    assert first.posterior is not None and second.posterior is not None
    assert np.allclose(first.posterior.ESS, second.posterior.ESS)
    assert np.allclose(
        first.posterior.posterior.utility_probability,
        second.posterior.posterior.utility_probability,
    )


def test_zero_effective_information_is_explicit_error_for_posterior() -> None:
    with pytest.raises(ValueError, match="zero effective sample size"):
        tite_boin12_posterior(
            design(),
            [1],
            [-1],
            [0],
            [0],
            [2],
            toxicity_window=1.0,
            efficacy_window=2.0,
            n_doses=1,
        )
