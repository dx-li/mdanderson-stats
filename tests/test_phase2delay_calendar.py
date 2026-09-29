from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.phase2delay_calendar import (
    _weibull_delays,
    _weibull_parameters,
    replay_phase2_delay_calendar,
    simulate_phase2_delay_calendar,
    simulate_phase2_delay_calendar_oc,
)

MODEL = dict(
    endpoint="response",
    threshold=0.3,
    cutoff=1.0,
    prior_alpha=0.1,
    prior_beta=0.2,
    intervals=2,
    hazard_c=0.01,
    lambda0=0.1,
    burn_in=0,
    hazard_draws=2,
)


def test_replay_processes_same_time_events_before_look_and_applies_full_window_gate():
    result = replay_phase2_delay_calendar(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, np.inf],
        analysis_times=[1.0, 2.0, 5.0],
        final_analysis="at_enrollment_cap",
        window=3.0,
        minimum_completed=5,
        random_state=17,
        **MODEL,
    )
    assert [look.enrolled for look in result.looks] == [2, 3, 6]
    assert [look.observed_events for look in result.looks] == [1, 2, 5]
    assert [look.completed for look in result.looks] == [0, 0, 3]
    assert np.isnan(result.looks[0].posterior_probability)
    assert result.looks[0].status == "gate_not_reached"
    assert np.isnan(result.looks[1].posterior_probability)
    # The last look has only three full windows, despite five early observed events.
    assert np.isnan(result.looks[-1].posterior_probability)
    assert result.completed_at_latest_followup == 6
    assert result.planned_terminal_time == 5.0
    assert result.latest_enrolled_completion_time == 8.0


def test_replay_adds_terminal_window_look_once_and_stops_without_future_enrollment():
    result = replay_phase2_delay_calendar(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [np.inf] * 6,
        analysis_times=[2.0, 8.0],
        final_analysis="complete_window",
        window=3.0,
        minimum_completed=1,
        random_state=11,
        **MODEL,
    )
    assert [look.time for look in result.looks] == [2.0, 8.0]
    assert result.planned_terminal_time == 8.0
    assert result.enrolled == 6
    assert result.completed_at_latest_followup == 6
    assert result.final_decision == "continue"


def test_absolute_deadlines_handle_nonbinary_times_and_reject_collapsed_windows():
    completed = replay_phase2_delay_calendar(
        [0.1],
        [np.inf],
        analysis_times=[],
        final_analysis="complete_window",
        window=0.1,
        minimum_completed=1,
        random_state=5,
        **MODEL,
    )
    assert completed.looks[-1].time == 0.1 + 0.1
    assert completed.looks[-1].completed == 1
    tied = replay_phase2_delay_calendar(
        [0.1, 1.0],
        [0.2, np.inf],
        analysis_times=[0.1 + 0.2],
        final_analysis="at_enrollment_cap",
        window=1.0,
        minimum_completed=2,
        random_state=5,
        **MODEL,
    )
    assert tied.looks[0].observed_events == 1
    with pytest.raises(ValueError, match="completion time"):
        replay_phase2_delay_calendar(
            [1e308],
            [np.inf],
            analysis_times=[],
            final_analysis="complete_window",
            window=1.0,
            minimum_completed=1,
            random_state=5,
            **MODEL,
        )


def test_early_stop_prevents_later_enrollment_and_decisions():
    arrivals = np.arange(8, dtype=float)
    result = replay_phase2_delay_calendar(
        arrivals,
        [np.inf] * len(arrivals),
        analysis_times=[6.0, 7.0],
        final_analysis="at_enrollment_cap",
        window=2.0,
        minimum_completed=5,
        random_state=21,
        **(MODEL | {"hazard_draws": 2, "cutoff": 1e-300}),
    )
    assert result.stopped
    assert result.final_decision == "stop_futility"
    assert len(result.looks) == 1
    assert result.enrolled == 7
    assert result.stop_time == 6.0
    assert result.planned_terminal_time == 7.0
    assert result.latest_enrolled_completion_time == 8.0


def test_simulation_is_reproducible_and_reports_replay_seeds():
    kwargs = dict(
        event_probability=0.3,
        late_fraction=0.7,
        accrual_rate=2.0,
        max_subjects=6,
        analysis_times=[0.0, 2.0, 4.0],
        final_analysis="at_enrollment_cap",
        window=3.0,
        minimum_completed=1,
        random_state=123,
        max_work=100_000,
        **MODEL,
    )
    first = simulate_phase2_delay_calendar(**kwargs)
    second = simulate_phase2_delay_calendar(**kwargs)
    assert first.seed == second.seed
    assert np.array_equal(first.arrival_times, second.arrival_times)
    assert np.array_equal(first.latent_event_times, second.latent_event_times)
    assert [look.seed for look in first.result.looks] == [look.seed for look in second.result.looks]
    np.testing.assert_allclose(
        [look.posterior_probability for look in first.result.looks],
        [look.posterior_probability for look in second.result.looks],
        equal_nan=True,
    )
    replayed = simulate_phase2_delay_calendar(**(kwargs | {"random_state": first.seed}))
    assert np.array_equal(first.arrival_times, replayed.arrival_times)
    assert np.array_equal(first.latent_event_times, replayed.latent_event_times)
    assert [look.seed for look in first.result.looks] == [
        look.seed for look in replayed.result.looks
    ]
    np.testing.assert_allclose(
        [look.posterior_probability for look in first.result.looks],
        [look.posterior_probability for look in replayed.result.looks],
        equal_nan=True,
    )


def test_oc_runs_serially_and_returns_trial_seed_lineage():
    summary = simulate_phase2_delay_calendar_oc(
        trials=2,
        event_probability=0.3,
        late_fraction=0.7,
        accrual_rate=2.0,
        max_subjects=6,
        analysis_times=[0.0, 2.0, 4.0],
        final_analysis="at_enrollment_cap",
        window=3.0,
        minimum_completed=1,
        random_state=19,
        max_work=100_000,
        **MODEL,
    )
    assert summary.trials == 2
    assert len(summary.trial_seeds) == 2
    assert summary.futility_stops + summary.safety_stops + summary.continue_trials == 2
    one = simulate_phase2_delay_calendar_oc(
        trials=1,
        event_probability=0.3,
        late_fraction=0.7,
        accrual_rate=2.0,
        max_subjects=6,
        analysis_times=[0.0, 2.0, 4.0],
        final_analysis="at_enrollment_cap",
        window=3.0,
        minimum_completed=1,
        random_state=19,
        max_work=100_000,
        **MODEL,
    )
    assert len(one.trial_seeds) == 1
    expected_trial = simulate_phase2_delay_calendar(
        event_probability=0.3,
        late_fraction=0.7,
        accrual_rate=2.0,
        max_subjects=6,
        analysis_times=[0.0, 2.0, 4.0],
        final_analysis="at_enrollment_cap",
        window=3.0,
        minimum_completed=1,
        random_state=one.trial_seeds[0],
        max_work=100_000,
        **MODEL,
    )
    assert one.mean_enrolled == expected_trial.result.enrolled


def test_weibull_calibration_and_inverse_cdf_match_independent_r_fixtures():
    fixture_dir = Path(__file__).parent / "fixtures"
    calibration = np.genfromtxt(
        fixture_dir / "phase2delay-weibull-calibration.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    quantiles = np.genfromtxt(
        fixture_dir / "phase2delay-weibull-quantiles.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    for i, row in enumerate(calibration):
        p = float(row["event_probability"])
        q = float(row["late_fraction"])
        window = float(row["window"])
        shape, scale = _weibull_parameters(p, q, window)
        np.testing.assert_allclose([shape, scale], [row["shape"], row["scale"]], rtol=1e-12)
        f_t = 1.0 - np.exp(-((window / scale) ** shape))
        f_half = 1.0 - np.exp(-((window / 2 / scale) ** shape))
        np.testing.assert_allclose([f_t, (f_t - f_half) / f_t], [p, q], rtol=1e-12)
        # Time-unit changes scale the Weibull scale and window together.
        shape7, scale7 = _weibull_parameters(p, q, 7 * window)
        np.testing.assert_allclose([shape7, scale7], [shape, 7 * scale], rtol=1e-12)
        fixture_rows = quantiles[quantiles["case"] == i + 1]
        uniforms = fixture_rows["probability"]
        unrestricted = scale * (-np.log1p(-uniforms)) ** (1.0 / shape)
        np.testing.assert_allclose(unrestricted, fixture_rows["time"], rtol=1e-10)
        delays = _weibull_delays(uniforms, p, shape, scale, window)
        expected_observed = (
            np.char.lower(np.asarray(fixture_rows["observed_by_window"]).astype(str)) == "true"
        )
        np.testing.assert_array_equal(np.isfinite(delays), expected_observed)
        np.testing.assert_allclose(
            delays[expected_observed], fixture_rows["time"][expected_observed], rtol=1e-10
        )
        np.testing.assert_allclose(
            _weibull_delays(np.array([p]), p, shape, scale, window), [window]
        )
