import numpy as np

from mdanderson_stats.bf_boin import BFBOINDesign
from mdanderson_stats.bf_boin_simulation import simulate_bf_boin


def _run(**kwargs):
    return simulate_bf_boin(
        BFBOINDesign(n_cap=10),
        kwargs.pop("true_toxicity", [0.0, 0.0, 0.0]),
        kwargs.pop("true_response", [0.0, 0.0, 0.0]),
        cohorts=kwargs.pop("cohorts", 1),
        cohort_size=kwargs.pop("cohort_size", 2),
        trials=1,
        accelerated_titration=True,
        true_grade2=kwargs.pop("true_grade2", [0.0, 0.0, 0.0]),
        grade2_assessment_delay=kwargs.pop("grade2_assessment_delay", 0.1),
        rng=13,
        **kwargs,
    )


def test_lower_cap_without_trigger_starts_full_cohort_at_next_dose():
    result = _run(titration_cap=2)

    assert result.titration_stop_reason == ("lower_cap_no_trigger",)
    assert result.titration_patients.tolist() == [2]
    np.testing.assert_array_equal(result.assigned, [[1, 1, 2]])
    assert not result.backfill_history[0].any()


def test_highest_cap_singleton_is_topped_up_into_first_budgeted_cohort():
    result = _run()

    assert result.titration_stop_reason == ("highest_cap",)
    assert result.titration_patients.tolist() == [3]
    np.testing.assert_array_equal(result.assigned, [[1, 1, 2]])
    assert not result.backfill_history[0].any()


def test_first_dlt_stops_singleton_staircase_then_completes_current_cohort():
    result = _run(true_toxicity=[1.0, 0.0, 0.0], grade2_assessment_delay=10.0)

    assert result.titration_stop_reason == ("first_dlt",)
    assert result.titration_patients.tolist() == [1]
    np.testing.assert_array_equal(result.assigned, [[2, 0, 0]])
    assert result.toxicities[0, 0] == 2
    assert result.titration_grade2.tolist() == [0]
    assert result.titration_end[0] < result.grade2_assessment_history[0][0]
    assert result.trial_duration[0] == result.grade2_assessment_history[0].max()


def test_second_observed_grade2_stops_staircase_and_is_reported_at_exit():
    result = _run(true_grade2=[1.0, 1.0, 1.0])

    assert result.titration_stop_reason == ("second_grade2",)
    assert result.titration_patients.tolist() == [2]
    assert result.titration_grade2.tolist() == [2]
    np.testing.assert_array_equal(result.assigned, [[1, 2, 0]])
    assert result.grade2_history is not None
    assert result.grade2_history[0].sum() == 3
    assert result.grade2_assessment_history is not None
    assert np.all(result.grade2_assessment_history[0] > result.arrival_history[0])


def test_topped_up_cohort_keeps_arrivals_chronological_with_following_cohorts():
    result = _run(
        cohorts=2,
        true_response=[1.0, 1.0, 1.0],
        accrual_rate=100.0,
        dlt_window=0.2,
    )

    assert result.titration_end is not None
    assert result.titration_end[0] == result.arrival_history[0][2]
    assert result.arrival_history[0][3] >= result.titration_end[0]
    assert np.all(np.diff(result.arrival_history[0]) > 0)
    assert result.backfill_history[0].any()
