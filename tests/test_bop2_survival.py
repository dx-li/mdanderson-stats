"""Independent gamma identities, time-unit invariance and survival calendar validation."""

from decimal import Decimal, localcontext
from math import factorial

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import bop2_survival_design, run_bop2_survival_trial, simulate_bop2_survival


def test_gamma_posterior_against_high_precision_erlang_identity_and_boundaries():
    design = bop2_survival_design(10, 6, cutoff_scale=0.7, gamma=0.5, looks=[5, 10], prior=[1, 6])
    for d in range(6):
        with localcontext() as ctx:
            ctx.prec = 70
            z = Decimal(1) + Decimal(2).ln() * Decimal(17) / 6
            expected = 1 - (-z).exp() * sum(z**k / Decimal(factorial(k)) for k in range(d + 1))
        state = design.monitor(d, 17, 5)
        assert_allclose(state.success_probability, float(expected), rtol=2e-14, atol=1e-16)
        assert state.posterior_shape == 1 + d
    threshold = design.total_time_boundary(3, 5)
    assert design.monitor(3, threshold * 0.999999, 5).decision == "stop_futility"
    assert design.monitor(3, threshold * 1.000001, 5).decision == "continue"
    probability = float(design.monitor(0, 0, 1).success_probability)
    for equality in [False, True]:
        exact = bop2_survival_design(
            1,
            6,
            cutoff_scale=probability,
            gamma=0,
            looks=[1],
            prior=[1, 6],
            equality_continues=equality,
        )
        assert exact.monitor(0, 0, 1).decision == (
            "final_positive" if equality else "final_negative"
        )


def test_time_unit_invariance_and_elicited_prior():
    values = []
    for unit in [1e-200, 1, 1e200]:
        design = bop2_survival_design(10, 6 * unit, cutoff_scale=0.7, gamma=0.5, looks=[5, 10])
        state = design.monitor_records(np.array([1, 2, 3, 4, 7]) * unit, [1, 0, 0, 1, 0])
        values.append(float(state.success_probability))
        assert_allclose(design.prior_scale_ratio / (design.prior_shape - 1), 1, rtol=2e-15)
        assert_allclose(
            design.total_time_boundary(2, 5) / unit,
            bop2_survival_design(
                10, 6, cutoff_scale=0.7, gamma=0.5, looks=[5, 10]
            ).total_time_boundary(2, 5),
            rtol=2e-15,
        )
    assert_allclose(values, values[1], rtol=2e-15)


def test_calendar_censoring_and_no_future_event_leakage():
    design = bop2_survival_design(6, 1, cutoff_scale=0.2, gamma=1, looks=[3, 6])
    enrolled = np.arange(1, 7)
    first = run_bop2_survival_trial(
        design, enrolled, [0.5, 5, 100, 100, 100, 100], final_followup=2
    )
    changed = run_bop2_survival_trial(design, enrolled, [0.5, 5, 100, 0, 0, 0], final_followup=2)
    assert first.calendar_times[0] == 3
    assert first.states[0].events == 1
    assert first.states[0].total_observation_time == 1.5
    assert first.states[0].success_probability == changed.states[0].success_probability
    assert first.calendar_times[-1] == 8
    stopped = run_bop2_survival_trial(design, enrolled, np.zeros(6), final_followup=2)
    assert len(stopped.states) == 1
    assert stopped.states[-1].decision == "stop_futility"


def test_simulation_matches_replay_and_single_look_analytic_success():
    design = bop2_survival_design(6, 1, cutoff_scale=0.7, gamma=0.5, looks=[3, 6])
    result = simulate_bop2_survival(
        design, 2, accrual_rate=1, final_followup=2, n_trials=128, rng=314
    )
    durations = np.random.default_rng(314).exponential(2 / np.log(2), (128, 6))
    replays = [
        run_bop2_survival_trial(design, np.arange(1, 7), t, final_followup=2) for t in durations
    ]
    assert_array_equal(result.sample_size, [r.states[-1].sample_size for r in replays])
    assert_array_equal(result.events, [r.states[-1].events for r in replays])
    assert_array_equal(result.success, [r.states[-1].decision == "final_positive" for r in replays])
    single = bop2_survival_design(1, 1, cutoff_scale=0.8, gamma=0, looks=[1], prior=[1, 0.1])
    boundary = float(single.total_time_boundary(1, 1))
    assert 0 < boundary < 5
    estimate = simulate_bop2_survival(
        single, 2, accrual_rate=2, final_followup=5, n_trials=25000, rng=99
    )
    expected = np.exp(-np.log(2) * boundary / 2)
    assert abs(estimate.success_probability - expected) < 5 * np.sqrt(
        expected * (1 - expected) / 25000
    )
