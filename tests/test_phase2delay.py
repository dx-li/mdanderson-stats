"""Focused checks for delayed-outcome Bayesian monitoring."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import betainc

from mdanderson_stats.phase2delay import phase2_delay_monitor


def test_complete_data_reduces_to_beta_tail_exactly():
    result = phase2_delay_monitor(
        [1, 0, 1],
        [0.2, 0, 0.7],
        [0.2, 1, 1],
        window=1,
        endpoint="response",
        threshold=0.4,
        prior_alpha=0.1,
        prior_beta=0.2,
        hazard_c=0.01,
        lambda0=0.5,
        burn_in=2,
        hazard_draws=8,
        seed=11,
    )
    expected = betainc(0.1 + 2, 0.2 + 1, 0.4)
    assert result.n_pending == 0
    assert_allclose(result.posterior_probability, expected, rtol=0, atol=1e-15)
    assert result.probability_trace.size == result.hazard_trace.size == 0
    equal = phase2_delay_monitor(
        [1],
        [0.2],
        [1],
        window=1,
        threshold=0.5,
        cutoff=0.5,
        prior_alpha=1,
        prior_beta=2,
        hazard_c=0.01,
        lambda0=0.5,
    )
    assert equal.posterior_probability == 0.5
    assert not equal.stopped


def test_upper_tail_endpoints_are_complements_and_seed_reproduces():
    args = dict(
        event=[1, 0, 0],
        event_time=[0.25, 0, 0],
        followup=[0.25, 1, 0.5],
        window=1,
        threshold=0.4,
        intervals=2,
        hazard_c=[2, 3],
        lambda0=0.4,
        prior_alpha=0.3,
        prior_beta=0.7,
        burn_in=10,
        hazard_draws=30,
        seed=123,
    )
    response = phase2_delay_monitor(endpoint="response", **args)
    toxicity = phase2_delay_monitor(endpoint="toxicity", **args)
    repeat = phase2_delay_monitor(endpoint="response", **args)
    assert_allclose(response.posterior_probability + toxicity.posterior_probability, 1, atol=1e-14)
    assert_allclose(response.posterior_probability, repeat.posterior_probability, atol=0, rtol=0)
    assert_array_equal(response.probability_trace, repeat.probability_trace)


def test_input_validation_rejects_future_and_non_event_times():
    with pytest.raises(ValueError, match="event_time"):
        phase2_delay_monitor(
            [1],
            [2],
            [1],
            window=1,
            threshold=0.4,
            prior_alpha=0.1,
            prior_beta=0.2,
            hazard_c=0.01,
            lambda0=0.5,
        )
    with pytest.raises(ValueError, match="non-events"):
        phase2_delay_monitor(
            [0],
            [0.1],
            [0.5],
            window=1,
            threshold=0.4,
            prior_alpha=0.1,
            prior_beta=0.2,
            hazard_c=0.01,
            lambda0=0.5,
        )
    with pytest.raises(ValueError, match="followup"):
        phase2_delay_monitor(
            [0],
            [0],
            [1.1],
            window=1,
            threshold=0.4,
            prior_alpha=0.1,
            prior_beta=0.2,
            hazard_c=0.01,
            lambda0=0.5,
        )


def test_small_paper_c_prior_is_usable():
    result = phase2_delay_monitor(
        np.array([1, 0, 0]),
        np.array([0.25, 0, 0]),
        np.array([0.25, 1, 0.5]),
        window=1,
        threshold=0.4,
        prior_alpha=0.1,
        prior_beta=0.2,
        hazard_c=0.01,
        lambda0=0.5,
        burn_in=100,
        hazard_draws=200,
        seed=2,
    )
    assert np.all(np.isfinite(result.log_hazard_trace))
    assert np.all(np.isfinite(result.hazard_trace))
    assert result.log_hazard_trace.shape == (200, 6)
    assert result.probability_trace.shape == (200,)
    assert_allclose(np.exp(result.log_hazard_trace), result.hazard_trace)
    assert np.all((result.probability_trace >= 0) & (result.probability_trace <= 1))


def test_work_limits_reject_large_or_fractional_runs_before_sampling():
    args = dict(
        event=[0],
        event_time=[0],
        followup=[0.5],
        window=1,
        threshold=0.4,
        prior_alpha=0.1,
        prior_beta=0.2,
        hazard_c=0.01,
        lambda0=0.5,
    )
    for option in (
        {"hazard_draws": 100_000},
        {"burn_in": 100_000},
        {
            "hazard_draws": 1000,
            "imputations_per_draw": 1000,
            "event": [0] * 3,
            "event_time": [0] * 3,
            "followup": [0.5] * 3,
        },
        {"hazard_draws": 3.5},
        {"event": [0] * 1001, "event_time": [0] * 1001, "followup": [0.5] * 1001},
    ):
        with pytest.raises(ValueError):
            phase2_delay_monitor(**(args | option))
