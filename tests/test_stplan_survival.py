"""Focused checks for censored-survival STPLAN forward power methods."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import ndtr, ndtri
from scipy.stats import poisson

from mdanderson_stats.stplan_continuous import stplan_exponential_one_sample_power
from mdanderson_stats.stplan_survival import (
    stplan_censored_exponential_one_sample_power,
    stplan_george_desu_survival_power,
    stplan_information_survival_power,
    stplan_piecewise_survival_power,
)
from mdanderson_stats.survival_sample_size import exponential_event_probability


def test_censored_one_sample_is_bounded_poisson_event_mixture():
    null_mean, alternative_mean, rate, accrual, followup, alpha = 10, 15, 5, 12, 6, 0.05
    expected_events = (
        rate * accrual * exponential_event_probability(1 / alternative_mean, accrual, followup)
    )
    cutoff = int(np.ceil(poisson.isf(1e-10, expected_events)))
    expected = sum(
        poisson.pmf(events, expected_events)
        * stplan_exponential_one_sample_power(alternative_mean / null_mean, events, alpha=alpha)
        for events in range(1, cutoff + 1)
    )
    assert_allclose(
        stplan_censored_exponential_one_sample_power(
            null_mean, alternative_mean, rate, accrual, followup
        ),
        expected,
        rtol=0,
        atol=1.1e-10,
    )
    null_power = stplan_censored_exponential_one_sample_power(10, 10, rate, accrual, followup)
    assert 0 < null_power < alpha


def test_survival_normal_and_george_desu_nulls_recover_alpha():
    assert_allclose(stplan_george_desu_survival_power(10, 10, 5, 12, 6), 0.05, atol=2e-14)
    assert_allclose(stplan_information_survival_power(10, 10, 5, 12, 6), 0.05, atol=2e-14)


def test_piecewise_large_followup_small_accrual_matches_constant_hazard():
    piecewise = stplan_piecewise_survival_power(
        0.2, 0.8, 1e17, 1.5, 5, 1, 1e16, model_arm="lower_hazard"
    )
    expected = ndtr(np.log(1.5) * np.sqrt(5 / 4) + ndtri(0.05))
    assert_allclose(piecewise, expected, rtol=0, atol=2e-15)
    with pytest.raises(ValueError, match="model_arm"):
        stplan_piecewise_survival_power(0.2, 0.1, 2, 1.5, 5, 10, 5, model_arm="high")
