import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12 import tite_boin12_decision, tite_boin12_posterior


def design() -> BOIN12Design:
    return BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))


def _runin_data(n_current: int, current_toxicities: int, *, pending: int = 0):
    doses = np.asarray([1, 1, 1] + [2] * n_current)
    toxicity = np.asarray(
        [0, 0, 0] + [1] * current_toxicities + [0] * (n_current - current_toxicities)
    )
    efficacy = np.ones(doses.size, dtype=int)
    toxicity_followup = np.ones(doses.size, dtype=float)
    efficacy_followup = np.full(doses.size, 2.0)
    if pending:
        toxicity[-pending:] = -1
        toxicity_followup[-pending:] = 0.25
    return doses, toxicity, efficacy, toxicity_followup, efficacy_followup


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
    assert np.isclose(result.patient_conditional_means[1, 0], 0.5)


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


def test_no_information_suspends_without_runtime_warning() -> None:
    result = tite_boin12_decision(
        design(),
        [1],
        [-1],
        [-1],
        [0],
        [0],
        toxicity_window=1.0,
        efficacy_window=1.0,
        n_doses=1,
        current_dose=1,
    )
    assert result.action == "suspend_no_information"


def test_all_prior_elimination_stops_before_pending_gate() -> None:
    result = tite_boin12_decision(
        design(),
        [1],
        [-1],
        [-1],
        [0],
        [0],
        toxicity_window=1.0,
        efficacy_window=1.0,
        n_doses=1,
        current_dose=1,
        eliminated=[True],
    )
    assert result.action == "stop_safety"


def test_custom_cutoff_applies_to_untried_prior() -> None:
    d = design()
    object.__setattr__(d, "toxicity_cutoff", 0.5)
    result = tite_boin12_posterior(
        d,
        [1],
        [0],
        [1],
        [1],
        [2],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
    )
    assert not result.admissible[1]


def test_invalid_cutoff_is_rejected_before_pending_suspension() -> None:
    d = design()
    object.__setattr__(d, "toxicity_cutoff", 1.0)
    with pytest.raises(ValueError, match="toxicity_cutoff"):
        tite_boin12_decision(
            d,
            [1],
            [-1],
            [-1],
            [0],
            [0],
            toxicity_window=1.0,
            efficacy_window=1.0,
            n_doses=1,
            current_dose=1,
        )


@pytest.mark.parametrize("n_current", [3, 6])
def test_optional_3plus3_runin_deescalates_at_two_observed_dlt(n_current: int) -> None:
    args = _runin_data(n_current, 2)
    result = tite_boin12_decision(
        BOIN12Design(0.25, 0.25, toxicity_cutoff=0.95),
        *args,
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=2,
        run_in_3plus3=True,
    )
    assert result.action == "deescalate"
    assert result.next_dose == 1


@pytest.mark.parametrize(("n_current", "events"), [(3, 1), (4, 2), (6, 1)])
def test_runin_leaves_other_counts_to_ordinary_conduct(n_current: int, events: int) -> None:
    args = _runin_data(n_current, events)
    common = dict(
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=2,
    )
    ordinary = tite_boin12_decision(BOIN12Design(0.25, 0.25), *args, **common)
    runin = tite_boin12_decision(BOIN12Design(0.25, 0.25), *args, **common, run_in_3plus3=True)
    assert (runin.action, runin.next_dose) == (ordinary.action, ordinary.next_dose)


def test_runin_obeys_pending_gate_and_toxicity_limit_restriction() -> None:
    args = _runin_data(3, 2, pending=1)
    result = tite_boin12_decision(
        BOIN12Design(0.25, 0.25),
        *args,
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=2,
        max_pending_toxicity=0.2,
        run_in_3plus3=True,
    )
    assert result.action == "suspend_pending"
    assert result.posterior is None

    with pytest.raises(ValueError, match="only when toxicity_limit is 0.25"):
        tite_boin12_decision(
            design(),
            *args,
            toxicity_window=1.0,
            efficacy_window=2.0,
            n_doses=2,
            current_dose=2,
            run_in_3plus3=True,
        )


def test_runin_at_lowest_dose_stops_and_does_not_assign_an_eliminated_lower_dose() -> None:
    result = tite_boin12_decision(
        BOIN12Design(0.25, 0.25),
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=1,
        run_in_3plus3=True,
    )
    assert result.action == "stop_safety"
    assert result.next_dose is None

    args = _runin_data(3, 2)
    blocked_lower = tite_boin12_decision(
        BOIN12Design(0.25, 0.25),
        *args,
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=2,
        eliminated=[True, False],
        run_in_3plus3=True,
    )
    assert blocked_lower.action == "stop_no_admissible_neighbor"
    assert blocked_lower.next_dose is None
