import copy

import numpy as np
import pytest

from mdanderson_stats.dose_schedule_prior import DoseSchedulePrior
from mdanderson_stats.dose_schedule_simulation import (
    simulate_dose_schedule_operating_characteristics,
)
from mdanderson_stats.dose_schedule_trial import run_dose_schedule_trial


def _configuration(rng, *, scale=1.0, trials=3):
    return dict(
        truth_area=[0.08],
        truth_peak=[scale],
        truth_tail=[2.0 * scale],
        prior=DoseSchedulePrior(np.log([0.08, scale, 2.0 * scale]), [0.0, 0.0, 0.0], 1),
        schedules=[[0.0]],
        horizon=2.0 * scale,
        arrival_times=[0.0, scale],
        max_patients=2,
        toxicity_limit=0.5,
        upper_probability=0.8,
        target=0.2,
        trials=trials,
        rng=rng,
        draws=8,
        warmup=0,
        chains=2,
        max_total_evaluations=10_000,
        max_total_work=100_000,
    )


def test_aggregate_seed_replay_and_counts_match_single_calendar_trials():
    aggregate = simulate_dose_schedule_operating_characteristics(
        **_configuration(np.random.default_rng(18))
    )
    assigned_by_trial = []
    events_by_trial = []
    durations = []
    selected = np.zeros((1, 1), dtype=int)
    stop_counts = {name: 0 for name in aggregate.stop_reasons}
    evaluations = work = 0
    for event_seed, sampler_seed in zip(
        aggregate.event_seeds, aggregate.sampler_seeds, strict=True
    ):
        config = _configuration(np.random.default_rng(0))
        trial = run_dose_schedule_trial(
            config["truth_area"],
            config["truth_peak"],
            config["truth_tail"],
            config["prior"],
            config["schedules"],
            config["horizon"],
            config["arrival_times"],
            max_patients=config["max_patients"],
            toxicity_limit=config["toxicity_limit"],
            upper_probability=config["upper_probability"],
            target=config["target"],
            rng=np.random.default_rng(int(event_seed)),
            sampler_rng=np.random.default_rng(int(sampler_seed)),
            draws=config["draws"],
            warmup=config["warmup"],
            chains=config["chains"],
            max_total_evaluations=10_000,
            max_total_work=100_000,
        )
        assigned_by_trial.append(len(trial.patients))
        events_by_trial.append(sum(patient.observed.event for patient in trial.patients))
        durations.append(trial.duration)
        stop_counts[trial.stop_reason] += 1
        if trial.final_decision.pair is not None:
            selected[trial.final_decision.pair] += 1
        evaluations += trial.total_likelihood_evaluations
        work += trial.total_work_units

    assigned = np.asarray(assigned_by_trial, dtype=float)
    events = np.asarray(events_by_trial, dtype=float)
    pooled = events.sum() / assigned.sum()
    ratio_mcse = np.sqrt(
        aggregate.trials
        / (aggregate.trials - 1)
        * np.sum((events - pooled * assigned) ** 2)
        / assigned.sum() ** 2
    )
    assert np.array_equal(aggregate.selected_count, selected)
    assert aggregate.no_selection_count == 0
    assert aggregate.stop_reason_count.tolist() == [
        stop_counts[key] for key in aggregate.stop_reasons
    ]
    assert np.allclose(aggregate.mean_allocation.sum(), assigned.mean())
    assert aggregate.assigned_patients.sum() == assigned.sum()
    assert aggregate.observed_toxicities.sum() == events.sum()
    assert aggregate.toxicity_probability[0, 0] == pooled
    assert aggregate.toxicity_mcse[0, 0] == pytest.approx(ratio_mcse)
    assert aggregate.mean_enrollment == assigned.mean()
    assert aggregate.mean_duration == np.mean(durations)
    assert aggregate.likelihood_evaluations == evaluations
    assert aggregate.work_units == work


def test_shared_budget_preflight_does_not_advance_rng():
    rng = np.random.default_rng(91)
    before = copy.deepcopy(rng.bit_generator.state)
    config = _configuration(rng)
    # Three trials need at least 3 * 2 * 18 evaluation opportunities.
    config["max_total_evaluations"] = 107
    with pytest.raises(ValueError, match="preflight minimum"):
        simulate_dose_schedule_operating_characteristics(**config)
    assert rng.bit_generator.state == before


def test_duration_moments_remain_finite_and_scale_with_time_units():
    base = simulate_dose_schedule_operating_characteristics(
        **_configuration(np.random.default_rng(301), trials=2)
    )
    scaled = simulate_dose_schedule_operating_characteristics(
        **_configuration(np.random.default_rng(301), scale=1e200, trials=2)
    )
    assert np.isfinite(scaled.mean_duration)
    assert np.isfinite(scaled.duration_mcse)
    assert scaled.mean_duration / 1e200 == pytest.approx(base.mean_duration, rel=2e-14)
    assert scaled.duration_mcse / 1e200 == pytest.approx(base.duration_mcse, rel=2e-14)
