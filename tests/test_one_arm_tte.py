import numpy as np
import pytest

from mdanderson_stats.one_arm_tte import (
    one_arm_tte_design,
    one_arm_tte_monitor,
    one_arm_tte_trial,
)
from mdanderson_stats.one_arm_tte_simulation import simulate_one_arm_tte


def test_monitor_matches_inverse_gamma_zero_margin_cases():
    design = one_arm_tte_design(
        [4, 8],
        [2, 3],
        cutoff_inferiority=0.5,
        delta_inferiority=0,
        max_patients=10,
    )
    result = one_arm_tte_monitor(design, 2, 2, 5)
    assert result.inferiority_probability == pytest.approx(0.5)
    assert not result.inferior


def test_calendar_trace_stops_at_periodic_check_and_final_followup():
    design = one_arm_tte_design(
        [1, 1],
        [1, 1],
        cutoff_inferiority=0.5,
        cutoff_superiority=0.8,
        max_patients=4,
        minimum_patients=2,
        periodic_interval=1,
        monitor_at_accrual=True,
        followup_period=2,
    )
    trial = one_arm_tte_trial(design, [0.2, 1.2, 2.2, 3.2], [0.5, 100, 100, 100])
    assert trial.early_monitor is not None
    assert trial.early_monitor.patients == 2
    assert trial.early_monitor.events == 1
    assert trial.early_monitor.total_time == pytest.approx(1.3)
    assert trial.final_monitor.total_time == pytest.approx(3.3)
    assert trial.final_time == pytest.approx(4.0)


def test_equal_time_tie_is_observed_before_accrual_check():
    design = one_arm_tte_design(
        [1, 1],
        [1, 1],
        cutoff_inferiority=0.5,
        max_patients=3,
        minimum_patients=1,
        periodic_interval=1,
        monitor_at_accrual=True,
    )
    trial = one_arm_tte_trial(design, [0.5, 1, 2], [0.5, 20, 20])
    assert trial.early_monitor is not None
    assert trial.early_monitor.patients == 1
    assert trial.early_monitor.events == 1


def test_simulation_is_bounded_and_reproducible():
    design = one_arm_tte_design([1, 1], [1, 1], cutoff_inferiority=0.5, max_patients=4)
    first = simulate_one_arm_tte(design, 2, 1, 8, seed=7)
    second = simulate_one_arm_tte(design, 2, 1, 8, seed=7)
    assert first.early_inferior_probability == second.early_inferior_probability
    assert first.sample_size_quantiles.shape == (3,)
    assert np.array_equal(first.final_times, second.final_times)


def test_input_limits_and_both_decisions_are_explicit():
    with pytest.raises(ValueError):
        one_arm_tte_design(
            [1, 1], [1, 1], cutoff_inferiority=None, cutoff_superiority=None, max_patients=2
        )
    design = one_arm_tte_design(
        [1, 1],
        [1, 1],
        cutoff_inferiority=1,
        cutoff_superiority=0,
        max_patients=2,
    )
    result = one_arm_tte_monitor(design, 1, 0, 0)
    assert result.reason == "both"
