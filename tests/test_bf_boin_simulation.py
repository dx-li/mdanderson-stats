import numpy as np
import pytest

from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.bf_boin_simulation import _weibull_endpoint, simulate_bf_boin


def test_weibull_calibration_matches_both_endpoint_probabilities():
    shape, scale = _weibull_endpoint(0.25, 2.0)

    def cdf(t: float) -> float:
        return float(1 - np.exp(-((t / scale) ** shape)))

    assert cdf(2.0) == pytest.approx(0.25)
    assert cdf(1.0) == pytest.approx(0.125)


def test_calendar_simulation_reproducible_and_conserves_assigned_patients():
    kwargs = dict(
        cohorts=2,
        cohort_size=2,
        trials=3,
        n_cap=3,
    )
    first = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        [1.0, 1.0],
        rng=17,
        **{k: v for k, v in kwargs.items() if k != "n_cap"},
    )
    second = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        [1.0, 1.0],
        rng=17,
        **{k: v for k, v in kwargs.items() if k != "n_cap"},
    )
    np.testing.assert_array_equal(first.assigned, second.assigned)
    np.testing.assert_array_equal(first.patients, second.patients)
    np.testing.assert_array_equal(first.assigned, first.patients)
    assert len(first.assigned_history) == 3
    assert all(a.size > 0 for a in first.assessment_history)
    assert np.all(first.trial_duration >= first.escalation_end)


def test_fractional_current_settings_and_invalid_arrival_distribution_rejected():
    with pytest.raises(ValueError):
        simulate_bf_boin(BFBOINDesign(), [0.1, 0.2], [0.5, 0.5], arrival_distribution="bad")


def test_arrivals_continue_as_a_renewal_process_between_cohorts():
    result = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        [0.0, 0.0],
        cohorts=2,
        cohort_size=3,
        trials=1,
        accrual_rate=6,
        rng=1,
    )
    assert result.arrival_history[0][3] > result.assessment_history[0][2]
    assert not np.any(np.isclose(result.arrival_history[0], result.assessment_history[0]))


def test_post_escalation_expansion_stays_one_dose_below_last_escalation_cohort():
    result = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        cohorts=2,
        cohort_size=1,
        trials=1,
        start_dose=1,
        expand_after_escalation=True,
        accrual_rate=0.001,
        rng=21,
    )

    # With no toxicity, the next dose after the second cohort is dose 3, but
    # expansion is anchored to dose 2, the last dose actually treated.
    assert result.expansion_stop_reason == ("assigned_cap",)
    assert result.expansion_patients.tolist() == [2]
    assert result.assigned.tolist() == [[3, 1, 0]]
    assert result.backfill_history[0].sum() == 2
    assert result.expansion_end[0] > result.escalation_end[0]
    assert np.all(result.arrival_history[0][-2:] >= result.escalation_end[0])
    assert result.trial_duration[0] >= result.expansion_end[0]


def test_expansion_without_activity_or_lower_dose_returns_explicit_status():
    no_activity = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        [0.0, 0.0],
        cohorts=2,
        cohort_size=1,
        trials=1,
        expand_after_escalation=True,
        rng=22,
    )
    assert no_activity.expansion_stop_reason == ("activity_unavailable",)
    assert no_activity.expansion_patients.tolist() == [0]
    assert np.isfinite(no_activity.expansion_end[0])

    lowest_only = simulate_bf_boin(
        BFBOINDesign(n_cap=3),
        [0.0, 0.0],
        [1.0, 1.0],
        cohorts=1,
        cohort_size=1,
        trials=1,
        expand_after_escalation=True,
        rng=23,
    )
    assert lowest_only.expansion_stop_reason == ("no_lower_dose",)
    assert lowest_only.expansion_patients.tolist() == [0]
    assert np.isfinite(lowest_only.expansion_end[0])
