import numpy as np
import pytest

from mdanderson_stats.dose_schedule_prior import DoseSchedulePrior
from mdanderson_stats.dose_schedule_trial import (
    _event_time,
    _observed_patient,
    run_dose_schedule_trial,
)


def _prior() -> DoseSchedulePrior:
    return DoseSchedulePrior([-1.0, 0.0, 0.0], [0.2, 0.1, 0.1], 1)


def test_event_time_uses_inverse_cumulative_hazard_across_time_scales() -> None:
    uniform = 4e-102
    event_time = _event_time(uniform, np.array([0.0]), 0.4, 2.0, 3.0, 5.0)
    assert event_time == pytest.approx(1e-50, rel=2e-14, abs=0)

    target = -np.log1p(-0.37)
    base = _event_time(0.37, np.array([0.0, 1.0]), 2.0, 0.5, 1.0, 4.0)
    for scale in (1e-6, 1e6):
        scaled = _event_time(
            0.37,
            np.array([0.0, 1.0]) * scale,
            2.0,
            0.5 * scale,
            1.0 * scale,
            4.0 * scale,
        )
        assert scaled / scale == pytest.approx(base, rel=3e-13)
    assert target > 0


def test_event_tie_omits_administration_but_event_free_censor_includes_it() -> None:
    event = _observed_patient(0.0, 3.0, 2.0, 5.0, np.array([0.0, 2.0, 4.0]), 0)
    censor = _observed_patient(0.0, 2.0, float("inf"), 5.0, np.array([0.0, 2.0, 4.0]), 0)
    assert event.event
    np.testing.assert_array_equal(event.administration_times, [0.0])
    assert not censor.event
    np.testing.assert_array_equal(censor.administration_times, [0.0, 2.0])


def test_trial_replay_is_reproducible_and_waits_for_final_followup() -> None:
    kwargs = dict(
        truth_area=[0.2],
        truth_peak=[2.0],
        truth_tail=[2.0],
        prior=_prior(),
        schedules=[[0.0]],
        horizon=3.0,
        arrival_times=[0.0, 1.0],
        max_patients=2,
        toxicity_limit=0.8,
        upper_probability=0.99,
        target=0.3,
        event_uniforms=[0.5, 0.7],
        sampler_seeds=[41, 42],
        draws=8,
        warmup=0,
        chains=2,
    )
    first = run_dose_schedule_trial(**kwargs)
    second = run_dose_schedule_trial(
        **{
            **kwargs,
            "event_uniforms": first.event_uniforms,
            "sampler_seeds": first.sampler_seeds,
        }
    )
    assert first.stop_reason == "max_patients"
    assert first.steps[0].pair == (0, 0)
    assert first.steps[1].decision is not None
    assert first.final_decision.action == "select"
    assert first.final_time == pytest.approx(4.0)
    assert first.stop_time == pytest.approx(1.0)
    assert len(first.sampler_seeds) == 2
    assert first.total_likelihood_evaluations > 0
    np.testing.assert_array_equal(first.event_uniforms, second.event_uniforms)
    np.testing.assert_array_equal(first.final_fit.regimen_risk, second.final_fit.regimen_risk)
    assert first.final_decision.pair == second.final_decision.pair


def test_final_history_uses_exact_relative_horizon_at_large_calendar_origin() -> None:
    result = run_dose_schedule_trial(
        [0.2],
        [2.0],
        [2.0],
        _prior(),
        [[0.0]],
        0.1,
        [1e12],
        max_patients=1,
        toxicity_limit=0.8,
        upper_probability=0.99,
        target=0.3,
        event_uniforms=[0.5],
        sampler_seeds=[43],
        draws=8,
        warmup=0,
        chains=2,
    )
    assert result.patients[0].observed.time == 0.1
    assert result.duration == 0.1
    assert result.final_time > result.patients[0].arrival_time


def test_no_safe_interim_stop_is_permanent_and_budget_preflight_is_early() -> None:
    kwargs = dict(
        truth_area=[1.0],
        truth_peak=[1.0],
        truth_tail=[1.0],
        prior=_prior(),
        schedules=[[0.0]],
        horizon=2.0,
        arrival_times=[0.0, 1.0, 2.0],
        max_patients=3,
        toxicity_limit=0.0,
        upper_probability=0.1,
        target=0.3,
        event_uniforms=[0.8, 0.8, 0.8],
        sampler_seeds=[51, 52, 53],
        draws=8,
        warmup=0,
        chains=2,
    )
    result = run_dose_schedule_trial(**kwargs)
    assert result.stop_reason == "no_safe_regimen"
    assert len(result.patients) == 1
    assert result.stop_time == pytest.approx(1.0)
    assert result.final_time >= result.stop_time
    assert result.final_decision.action == "stop"
    assert result.final_decision.pair is None
    assert result.event_uniforms.shape == (3,)
    assert len(result.sampler_seeds) == 3
    replay = run_dose_schedule_trial(
        **{
            **kwargs,
            "event_uniforms": result.event_uniforms,
            "sampler_seeds": result.sampler_seeds,
        }
    )
    assert replay.stop_reason == result.stop_reason
    assert replay.final_decision.action == result.final_decision.action
    np.testing.assert_array_equal(replay.event_uniforms, result.event_uniforms)

    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    with pytest.raises(ValueError, match="minimum budget"):
        run_dose_schedule_trial(
            **{**kwargs, "rng": rng, "event_uniforms": None, "max_total_work": 1}
        )
    assert rng.bit_generator.state == state
