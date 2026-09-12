"""Independent probability and calendar references for the native simulator model."""

import csv
from pathlib import Path

import numpy as np

from mdanderson_stats.one_arm_tte import (
    one_arm_tte_design,
    one_arm_tte_monitor,
    one_arm_tte_trial,
)
from mdanderson_stats.one_arm_tte_simulation import simulate_one_arm_tte


def test_signed_margins_goals_and_parameterizations_against_r():
    path = Path(__file__).parent / "fixtures" / "one-arm-tte-reference.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        design = one_arm_tte_design(
            [4, 8],
            [2, 3],
            max_patients=10,
            parameterization=row["parameterization"],
            maximize=row["goal"] == "maximize",
            delta_inferiority=float(row["delta"]),
            cutoff_inferiority=0.2,
            delta_superiority=float(row["delta"]),
            cutoff_superiority=0.8,
        )
        result = one_arm_tte_monitor(design, 5, 2, 5)
        expected = float(row["probability"])
        np.testing.assert_allclose(
            [result.inferiority_probability, result.superiority_probability],
            expected,
            atol=1e-9,
            rtol=0,
        )
        assert result.inferior == (expected < 0.2)
        assert result.superior == (expected > 0.8)
    # Separate margins can legitimately make both configured rules fire.
    distinct = one_arm_tte_design(
        [4, 8],
        [2, 3],
        max_patients=10,
        delta_inferiority=1,
        cutoff_inferiority=0.3,
        delta_superiority=-1,
        cutoff_superiority=0.7,
    )
    result = one_arm_tte_monitor(distinct, 5, 2, 5)
    np.testing.assert_allclose(
        [result.inferiority_probability, result.superiority_probability],
        [0.262938297360843, 0.737061702639157],
        atol=1e-9,
        rtol=0,
    )
    assert result.inferior and result.superior and result.reason == "both"


def test_periodic_stop_and_final_reclassification():
    design = one_arm_tte_design(
        [1, 1],
        [1, 1],
        max_patients=4,
        minimum_patients=2,
        cutoff_inferiority=0.5,
        delta_superiority=0,
        cutoff_superiority=0.8,
        periodic_interval=1,
        followup_period=2,
    )
    trial = one_arm_tte_trial(design, [0.2, 1.2, 2.2, 3.2], [0.5, 100, 100, 100])
    assert trial.accrual_stop_time == 2
    assert trial.final_time == 4
    np.testing.assert_array_equal(trial.enrollment_time, [0.2, 1.2])
    assert [time for time, _ in trial.monitor_history] == [2]
    assert trial.early_monitor.inferior
    assert trial.early_monitor.events == 1
    np.testing.assert_allclose(trial.early_monitor.total_time, 1.3)
    np.testing.assert_allclose(trial.early_monitor.inferiority_probability, (2.3 / 3.3) ** 2)
    assert trial.final_monitor.events == 1
    np.testing.assert_allclose(trial.final_monitor.total_time, 3.3)
    np.testing.assert_allclose(trial.final_monitor.inferiority_probability, (4.3 / 5.3) ** 2)
    assert not trial.final_monitor.inferior and not trial.final_monitor.superior


def test_event_and_monitor_precede_tied_arrival():
    design = one_arm_tte_design(
        [1, 1],
        [1, 1],
        max_patients=3,
        minimum_patients=1,
        cutoff_inferiority=0.5,
        periodic_interval=1,
    )
    trial = one_arm_tte_trial(design, [0.5, 1, 2], [0.5, 20, 20])
    assert trial.accrual_stop_time == 1
    assert trial.early_monitor.patients == 1
    assert trial.early_monitor.events == 1
    np.testing.assert_allclose(trial.early_monitor.inferiority_probability, 0.36)
    assert len(trial.enrollment_time) == 1


def test_simulated_duration_has_poisson_accrual_calendar_mean():
    design = one_arm_tte_design(
        [1, 1],
        [1, 1],
        max_patients=4,
        cutoff_inferiority=0,
        monitor_at_accrual=False,
        followup_period=2,
    )
    result = simulate_one_arm_tte(design, 100, 1, 400, seed=198, credible_level=0.8)
    # Fourth Poisson arrival is Gamma(4, rate=1); add fixed follow-up 2.
    assert abs(np.mean(result.final_times) - 6) < 5 * np.sqrt(4 / 400)
    np.testing.assert_allclose(result.final_times - result.accrual_stop_times, 2)
    np.testing.assert_array_equal(result.sample_sizes, 4)
    assert np.all(result.events <= result.sample_sizes)
    assert np.all(result.exposures >= 0)
    assert not np.any(result.early_inferior)
    assert not np.any(result.early_superior)
    assert not result.final_times.flags.writeable
