"""Cumulative-incidence native variances and independent probability identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import cumulative_incidence

CASES = json.loads((Path(__file__).parent / "fixtures/cuminc.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_original_cinc(case):
    result = cumulative_incidence(case["time"], case["event"])
    assert_allclose(
        np.column_stack([result.time, result.estimate, result.variance]),
        case["corners"],
        rtol=2e-11,
        atol=2e-15,
    )


def test_competing_events_reduce_survival_instead_of_being_censored():
    result = cumulative_incidence([1, 2, 3, 4], [1, 2, 0, 1])
    estimate, _ = result.at([0, 1, 2, 3, 4, 10])
    assert_allclose(estimate, [0, 0.25, 0.25, 0.25, 0.75, 0.75])
    mistaken = cumulative_incidence([1, 2, 3, 4], [1, 0, 0, 1])
    assert mistaken.estimate[-1] == 1


def test_tied_censoring_remains_at_risk_and_counts_individual_events():
    result = cumulative_incidence([1, 1, 1, 2], [1, 1, 0, 2])
    assert result.n_events == 2
    assert_allclose(result.at([0, 1, 2])[0], [0, 0.5, 0.5])
    # First-jump source Aalen variance: d(r-d)/(r^2(r-1)).
    assert_allclose(result.at(1)[1], 2 * 2 / (16 * 3))


def test_uncensored_incidence_is_empirical_cause_frequency():
    times = np.array([0, 1, 1, 2, 3, 4, 4, 5])
    events = np.array([1, 2, 1, 2, 2, 1, 1, 2])
    for cause in [1, 2]:
        result = cumulative_incidence(times, events, event_of_interest=cause)
        for t in np.arange(6):
            assert_allclose(result.at(t)[0], np.mean((times <= t) & (events == cause)))
    assert_allclose(
        sum(cumulative_incidence(times, events, event_of_interest=c).estimate[-1] for c in [1, 2]),
        1,
    )


def test_permutation_time_scaling_and_event_recoding():
    time = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    event = np.array([1, 2, 0, 1, 0])
    original = cumulative_incidence(time, event)
    perm = np.array([4, 0, 2, 1, 3])
    other = cumulative_incidence(3 * time[perm], event[perm] + 7, event_of_interest=8, censor=7)
    assert_allclose(other.time, 3 * original.time)
    assert_allclose(other.estimate, original.estimate)
    assert_allclose(other.variance, original.variance)
    time[:] = 99
    assert original.time[-1] == 5
    with pytest.raises(ValueError):
        original.estimate[0] = 1


def test_zero_time_and_absent_target():
    r = cumulative_incidence([0, 0, 1], [1, 2, 0])
    assert_allclose(r.at(0)[0], 1 / 3)
    absent = cumulative_incidence([1, 2], [2, 0])
    assert_array_equal(absent.estimate, [0, 0])
    assert_array_equal(absent.variance, [0, 0])
    with pytest.raises(ValueError):
        r.at(-1)


@pytest.mark.parametrize(
    "time,event,kwargs",
    [
        ([], [], {}),
        ([1], [1, 2], {}),
        ([-1], [1], {}),
        ([np.nan], [1], {}),
        ([1], [0.5], {}),
        ([1], [1], {"event_of_interest": 0}),
        ([1], [1], {"censor": True}),
    ],
)
def test_invalid_inputs(time, event, kwargs):
    with pytest.raises(ValueError):
        cumulative_incidence(time, event, **kwargs)
