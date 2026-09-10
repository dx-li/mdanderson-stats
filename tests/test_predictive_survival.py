"""Inverse-gamma identities, scale invariance and competing calendar stops."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import compare_predictive_survival, predictive_survival
from mdanderson_stats.predictive_survival_simulation import _future_summary, _stop_time


def test_inverse_gamma_ordering_and_extreme_scale_ratio():
    # Unit-shape gamma rates are exponential: P(rate_A<rate_B)=b_A/(b_A+b_B).
    result = compare_predictive_survival([0, 0], [0, 0], prior=[[1, 2], [1, 3]])
    assert_allclose(result.arm_a_probability, 0.4, atol=1e-15)
    assert_allclose(result.arm_b_probability, 0.6, atol=1e-15)
    extreme = compare_predictive_survival([0, 0], [0, 0], prior=[[0.001, 1e-300], [1, 1e300]])
    expected = np.exp(0.001 * (-600 * np.log(10)))
    assert_allclose(extreme.arm_a_probability, expected, rtol=3e-15)
    assert_allclose(extreme.arm_a_probability + extreme.arm_b_probability, 1, atol=1e-15)
    values = []
    for unit in [1e-200, 1, 1e200]:
        r = compare_predictive_survival(
            [3, 8],
            np.array([40, 30]) * unit,
            prior=np.array([[2, 5 * unit], [2, 5 * unit]]),
            method="frequentist",
        )
        values.append([float(r.arm_a_probability), float(r.z_statistic)])
    assert_allclose(values, [values[1]] * 3, rtol=2e-14)
    expected_z = (3 / 40 - 8 / 30) / np.sqrt(3 / 40**2 + 8 / 30**2)
    assert_allclose(values[1][1], expected_z, rtol=1e-15)
    assert (
        compare_predictive_survival(
            [0, 0], [0, 2], prior=[[1, 1], [1, 1]], method="frequentist"
        ).decision
        == "not_evaluable"
    )


def test_competing_stops_and_censored_patient_exposure():
    arrivals = np.array([1.0, 2.0, 4.0])
    arms = np.array([0, 1, 0])
    durations = np.array([3.0, 0.5, 10.0])
    pending = np.array([1.5, 8.0])
    pending_arms = np.array([0, 1])
    # Second new event: pending A at 1.5, then new B at 2.5.
    event_stop = _stop_time(arrivals, arrivals + durations, pending, 2, None, 10)
    assert event_stop == 2.5
    assert _stop_time(arrivals, arrivals + durations, pending, 2, 2, 10) == 2
    assert _stop_time(arrivals, arrivals + durations, pending, 2, 2, 1.75) == 1.75
    n, d, e, final = _future_summary(
        arrivals, arms, durations, pending, pending_arms, event_stop, 1
    )
    assert final == 3.5
    assert_array_equal(n, [1, 1])
    assert_array_equal(d, [1, 1])
    assert_allclose(e, [4, 4])  # A: 1.5+2.5; B: 3.5+0.5.
    # Additional follow-up continues after event-based accrual stopping.
    _, d, _, _ = _future_summary(arrivals, arms, durations, pending, pending_arms, event_stop, 6)
    assert_array_equal(d, [2, 2])


def test_posterior_predictive_pending_event_probability_and_limits():
    # One pending A, accrual already capped. Gamma(a=2,rate=3) posterior.
    # Predict event within t=2: 1-(3/(3+2))**2 = .64.
    result = predictive_survival(
        [1, 0],
        [0, 0],
        [1, 0],
        prior=[[2, 2], [2, 2]],
        accrual_rate=1,
        followup=2,
        max_patients=1,
        n_simulations=15000,
        rng=94,
    )
    observed = result.events[:, 0].mean()
    assert abs(observed - 0.64) < 5 * np.sqrt(0.64 * 0.36 / 15000)
    assert_allclose(result.probability.sum(), 1, atol=1e-15)
    assert_array_equal(result.accrual_stop_time, np.zeros(15000))
    assert_array_equal(result.patients, np.tile([1, 0], (15000, 1)))
    for limits in [dict(max_patients=12), dict(max_duration=4, elapsed_time=1), dict(max_events=4)]:
        r = predictive_survival(
            [2, 2],
            [1, 1],
            [3, 3],
            prior=[[2, 2], [2, 2]],
            accrual_rate=2,
            followup=0,
            n_simulations=200,
            rng=12,
            **limits,
        )
        if "max_patients" in limits:
            assert_array_equal(r.patients.sum(axis=-1), np.full(200, 12))
        elif "max_duration" in limits:
            assert_array_equal(r.accrual_stop_time, np.full(200, 3))
        else:
            assert_array_equal(r.events.sum(axis=-1), np.full(200, 4))
    with pytest.raises(ArithmeticError, match="guard"):
        predictive_survival(
            [0, 0],
            [0, 0],
            [0, 0],
            prior=[[2, 2], [2, 2]],
            accrual_rate=1e6,
            followup=0,
            max_duration=100,
            max_simulated_patients=2,
            n_simulations=1,
            rng=1,
        )
