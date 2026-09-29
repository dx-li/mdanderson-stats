import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12_calendar import _histories, run_tite_boin12_calendar_trial


def test_deadline_ties_are_visible_using_absolute_time():
    start = 0.1
    tox_delay = 0.2
    tox_deadline = start + tox_delay
    tox, eff, tox_follow, eff_follow = _histories(
        tox_deadline,
        [start],
        [tox_delay],
        [float("inf")],
        0.3,
        0.3,
    )
    np.testing.assert_array_equal(tox, [1])
    np.testing.assert_array_equal(eff, [-1])
    assert tox_follow[0] == tox_delay
    assert eff_follow[0] < 0.3


def test_calendar_suspends_until_both_endpoint_histories_resolve():
    design = BOIN12Design(0.6, 0.2, toxicity_cutoff=0.999, efficacy_cutoff=0.999)
    result = run_tite_boin12_calendar_trial(
        design,
        [1.0, 1.0],
        [[3.0], [float("inf")]],
        [[float("inf")], [float("inf")]],
        toxicity_window=10.0,
        efficacy_window=10.0,
        cohort_size=1,
    )
    assert [step.time for step in result.steps[:3]] == [2.0, 4.0, 11.0]
    assert [step.action for step in result.steps[:2]] == ["suspend_pending", "suspend_pending"]
    assert result.enrollment_times[0] == 1.0
    assert result.final_time >= result.accrual_stop_time


def test_staggered_cohort_calendar_restarts_after_suspension_with_decision_lag():
    design = BOIN12Design(0.35, 0.25)
    result = run_tite_boin12_calendar_trial(
        design,
        [1.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        np.full((6, 1), np.inf),
        np.full((6, 1), np.inf),
        toxicity_window=45.0,
        efficacy_window=60.0,
        cohort_size=3,
        decision_lag=1.0,
    )
    np.testing.assert_array_equal(result.enrollment_times, [1.0, 11.0, 21.0, 72.0, 82.0, 92.0])
    assert [step.time for step in result.steps] == [31.0, 46.0, 56.0, 61.0, 66.0, 71.0]
    assert [step.action for step in result.steps[:4]] == [
        "suspend_pending",
        "suspend_pending",
        "suspend_pending",
        "suspend_pending",
    ]


def test_bda_calendar_runs_one_compact_look_and_preserves_sampler_preflight():
    design = BOIN12Design(0.5, 0.25)
    rng = np.random.default_rng(20260929)
    state_before = repr(rng.bit_generator.state)
    with pytest.raises(ValueError, match="per-look BDA work"):
        run_tite_boin12_calendar_trial(
            design,
            [1.0, 2.0],
            [[float("inf")], [float("inf")]],
            [[float("inf")], [float("inf")]],
            toxicity_window=1.0,
            efficacy_window=1.0,
            cohort_size=1,
            method="bda",
            prior_concentrations=[1.0, 1.0, 1.0, 1.0],
            rng=rng,
            draws=20,
            warmup=0,
            chains=2,
            bda_max_work=1,
        )
    assert repr(rng.bit_generator.state) == state_before

    result = run_tite_boin12_calendar_trial(
        design,
        [1.0, 2.0],
        [[float("inf")], [float("inf")]],
        [[float("inf")], [float("inf")]],
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohort_size=1,
        method="bda",
        prior_concentrations=[1.0, 1.0, 1.0, 1.0],
        rng=np.random.default_rng(20260929),
        draws=20,
        warmup=0,
        chains=2,
    )
    assert len(result.steps) == 1
    assert result.steps[0].imputed_toxicity_rate is not None
    assert result.steps[0].toxicity_overdose_probability is not None
    assert not result.steps[0].toxicity_overdose_probability.flags.writeable


def test_safety_termination_suppresses_final_obd_recommendation():
    design = BOIN12Design(0.1, 0.25, toxicity_cutoff=0.5)
    result = run_tite_boin12_calendar_trial(
        design,
        [1.0, 1.0],
        [[0.0], [float("inf")]],
        [[float("inf")], [float("inf")]],
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohort_size=1,
    )
    assert result.stop_reason == "stop_safety"
    assert result.selected_obd is None
    assert result.selection.obd is None
    assert result.enrollment_times.size == 1


def test_dose_elimination_persists_across_later_cohorts():
    design = BOIN12Design(
        0.2,
        0.25,
        toxicity_cutoff=0.9,
        efficacy_cutoff=0.99,
        exploration_patients=3,
    )
    tox = np.full((12, 2), np.inf)
    tox[3:6, 1] = 0.0
    result = run_tite_boin12_calendar_trial(
        design,
        [1.0] + [10.0] * 11,
        tox,
        np.full((12, 2), np.inf),
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohort_size=3,
    )
    assert [step.next_dose for step in result.steps[:3]] == [2, 1, 1]
    assert result.steps[1].eliminated.tolist() == [False, True]
    assert result.steps[2].eliminated.tolist() == [False, True]
    assert result.assigned_doses.tolist() == [1] * 3 + [2] * 3 + [1] * 6


def test_run_in_safety_stop_suppresses_admissible_diagnostic_obd():
    design = BOIN12Design(0.25, 0.25, toxicity_cutoff=0.999, efficacy_cutoff=0.999)
    result = run_tite_boin12_calendar_trial(
        design,
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [[0.0], [0.0], [0.0], [float("inf")], [float("inf")], [float("inf")]],
        np.full((6, 1), np.inf),
        toxicity_window=1.0,
        efficacy_window=1.0,
        cohort_size=3,
        run_in_3plus3=True,
    )
    assert result.stop_reason == "stop_safety"
    assert result.selected_obd is None
    assert result.selection.obd == 1
    assert result.enrollment_times.size == 3
