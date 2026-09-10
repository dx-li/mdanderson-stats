"""Exact path enumeration and independent predictive beta-binomial identities."""

from fractions import Fraction
from itertools import product
from math import comb, factorial

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import betaincc

from mdanderson_stats import (
    posterior_efficacy_design,
    predictive_efficacy_design,
    toxicity_monitoring_design,
)


def test_predictive_manual_example():
    design = predictive_efficacy_design(15)
    state = design.monitor(1, 2)
    assert_allclose(state.final_probability, 0.7476842, atol=1e-7)
    assert_allclose(state.high_probability, 0.6738275, atol=1e-7)
    assert state.decision == "continue"  # before first scheduled look
    assert design.final_positive_min == 6


def test_predictive_recursion_against_exact_beta_binomial_sum():
    design = predictive_efficacy_design(
        8, prior=(2, 3), min_subjects=2, cohort_size=2, target_rate=0.5, final_probability=0.8
    )

    def beta(a, b):
        return Fraction(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))

    for n in range(9):
        for r in range(n + 1):
            remaining = 8 - n
            a, b = 2 + r, 3 + n - r
            expected = sum(
                Fraction(comb(remaining, j)) * beta(a + j, b + remaining - j) / beta(a, b)
                for j in range(remaining + 1)
                if r + j >= design.final_positive_min
            )
            assert_allclose(design.high_probability[n, r], float(expected), atol=2e-15)


@pytest.mark.parametrize(
    "factory", [posterior_efficacy_design, predictive_efficacy_design, toxicity_monitoring_design]
)
def test_operating_characteristics_by_exhaustive_paths(factory):
    design = factory(7, looks=[2, 4, 7])
    p = np.array([0, 0.2, 0.6, 1])
    exact = design.operating_characteristics(p)
    low = np.zeros((4, 3))
    high = np.zeros_like(low)
    positive = np.zeros(4)
    negative = np.zeros(4)
    event_sum = np.zeros(4)
    observed_rate = np.zeros(4)
    for path in product([0, 1], repeat=7):
        mass = p ** sum(path) * (1 - p) ** (7 - sum(path))
        for i, n in enumerate(design.looks):
            r = sum(path[:n])
            if n == 7:
                if r >= design.final_positive_min:
                    positive += mass
                else:
                    negative += mass
            elif r <= design.futility_max[i]:
                low[:, i] += mass
                break
            elif r >= design.positive_min[i]:
                high[:, i] += mass
                break
        event_sum += mass * r
        observed_rate += mass * r / n
    assert_allclose(exact.expected_events, event_sum, atol=1e-14)
    assert_allclose(exact.expected_observed_rate, observed_rate, atol=1e-14)
    assert_allclose(exact.expected_events, p * exact.expected_sample_size, atol=1e-14)
    assert_allclose(exact.stop_low, low, atol=1e-14)
    assert_allclose(exact.stop_high, high, atol=1e-14)
    assert_allclose(exact.complete_positive, positive, atol=1e-14)
    assert_allclose(exact.complete_negative, negative, atol=1e-14)
    pmf = low + high
    pmf[:, -1] += positive + negative
    assert_allclose(exact.sample_size_probability, pmf, atol=1e-14)
    assert_allclose(pmf.sum(axis=-1), 1, atol=1e-14)
    assert_allclose(exact.expected_sample_size, pmf @ design.looks)
    assert np.all(exact.sample_size_quantile(0.5) <= 7)


def test_strict_futility_and_inclusive_efficacy():
    # With a uniform prior and 0/1 observations, P(theta <= .5)=.75 exactly.
    design = posterior_efficacy_design(
        2,
        prior=(1, 1),
        looks=[1, 2],
        futility_rate=0.5,
        futility_probability=0.75,
        efficacy_rate=0.5,
        efficacy_probability=0.75,
        final_rate=0.5,
        final_probability=0.75,
    )
    assert design.monitor(0, 1).decision == "continue"
    assert design.monitor(1, 1).decision == "stop_efficacy"
    tox = toxicity_monitoring_design(
        2, prior=(1, 1), looks=[1, 2], toxicity_rate=0.5, toxicity_probability=0.75
    )
    assert tox.monitor(1, 1).decision == "stop_toxicity"
    prediction = predictive_efficacy_design(
        2,
        prior=(1, 1),
        looks=[1, 2],
        predictive_lower=0,
        predictive_upper=1,
        target_rate=0.5,
        final_probability=0,
    )
    assert_array_equal(prediction.high_probability[np.tril_indices(3)], 1)
    assert prediction.monitor(0, 1).decision == "stop_efficacy"


def test_boundaries_match_all_attainable_posterior_states():
    d = posterior_efficacy_design(15)
    for i, n in enumerate(d.looks[:-1]):
        r = np.arange(n + 1)
        low = 1 - betaincc(0.5 + r, 0.5 + n - r, 0.3)
        high = betaincc(0.5 + r, 0.5 + n - r, 0.3)
        assert_array_equal(r <= d.futility_max[i], low > 0.7)
        assert_array_equal(r >= d.positive_min[i], high >= 0.9)
    assert not d.high_probability.flags.writeable


def test_final_rule_overrides_early_rules_and_path_stays_stopped():
    d = posterior_efficacy_design(
        3, looks=[1, 3], efficacy_probability=0, futility_probability=1, final_probability=1
    )
    assert d.monitor(0, 3).decision == "final_negative"
    history = d.monitor_outcomes([0, 0, 0])
    assert_array_equal(history.decision, ["stop_efficacy"] * 3)
    assert_array_equal(history.events, [0, 0, 0])
    assert history.posterior.alpha.shape == (3,)


def test_impossible_positive_conclusion_and_batch_monitoring():
    d = predictive_efficacy_design(5, min_subjects=1, target_rate=1, final_probability=0.7)
    assert d.final_positive_min == 6
    assert_array_equal(d.high_probability[np.tril_indices(6)], 0)
    oc = d.operating_characteristics(np.array([[0, 0.5], [0.9, 1]]))
    assert_array_equal(oc.positive_conclusion, 0)
    assert_allclose(oc.expected_sample_size, 1)
    assert d.monitor([[0, 1], [1, 2]], 2).decision.shape == (2, 2)


def test_invalid_designs_and_data():
    for build in [
        lambda: predictive_efficacy_design(10, predictive_lower=0.9, predictive_upper=0.2),
        lambda: posterior_efficacy_design(10, futility_probability=0, efficacy_probability=0),
        lambda: toxicity_monitoring_design(10, looks=[5, 5, 10]),
        lambda: posterior_efficacy_design(0),
    ]:
        with pytest.raises(ValueError):
            build()
    with pytest.raises(ValueError):
        posterior_efficacy_design(10).monitor(3, 2)


@pytest.mark.parametrize("factory", [posterior_efficacy_design, predictive_efficacy_design])
def test_disabling_early_stopping_preserves_final_rule(factory):
    design = factory(15, stop_futility=False, stop_efficacy=False)
    result = design.operating_characteristics([0, 0.3, 1])
    assert_allclose(result.stop_low, 0)
    assert_allclose(result.stop_high, 0)
    assert_allclose(result.expected_sample_size, 15)
    assert_allclose(result.sample_size_quantile(0), 15)
    assert_allclose(result.sample_size_quantile(1), 15)
    assert_allclose(result.expected_observed_rate, [0, 0.3, 1])


def test_btox_published_operating_characteristics():
    design = toxicity_monitoring_design(15)
    assert_array_equal(design.positive_min, [2, 4, 6])
    result = design.operating_characteristics([0.1, 0.3, 0.5])
    assert_allclose(result.stop_high.sum(axis=-1), [0.0845, 0.5357, 0.8965], atol=5e-5)
    assert_allclose(result.positive_conclusion, [0.0849, 0.5633, 0.9325], atol=5e-5)
    assert_allclose(result.expected_sample_size, [14.1700, 9.9627, 6.4551], atol=5e-5)
    assert_allclose(result.expected_events, [1.4170, 2.9888, 3.2275], atol=5e-5)
    assert_allclose(result.expected_observed_rate, [0.1178, 0.3602, 0.5400], atol=5e-5)
