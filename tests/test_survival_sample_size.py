"""Nsurvival examples, exposure integration and normal rejection-region checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.special import ndtri

from mdanderson_stats import (
    exponential_event_probability,
    survival_event_power,
    survival_sample_size,
)


@pytest.mark.parametrize(
    "mc,mt,accrual,followup,margin,ratio,objective,sizes,events",
    [
        (3, 7, 3, 6, 0, None, "equality", [17], 9),
        (10, 20, 12, 6, 0, 1, "equality", [59, 59], 52),
        (9, 10, 12, 6, 0.4, 1, "equivalence", [347, 347], 395),
        (10, 9.5, 12, 6, 0.5, 1, "noninferiority", [110, 110], 123),
        (10, 18, 12, 6, 0.2, 1, "superiority", [181, 181], 165),
    ],
)
def test_source_examples(mc, mt, accrual, followup, margin, ratio, objective, sizes, events):
    result = survival_sample_size(
        mc,
        mt,
        accrual=accrual,
        followup=followup,
        margin=margin,
        allocation_ratio=ratio,
        objective=objective,
    )
    assert_array_equal(result.group_sizes, sizes)
    assert result.rounded_events == events
    assert result.expected_events >= result.required_events
    assert result.power >= 0.8
    assert result.total_size == sum(sizes)


def test_uniform_accrual_against_integration_and_event_simulation():
    hazards = np.array([1e-12, 0.03, 0.5, 20])
    actual = exponential_event_probability(hazards, 12, 6)
    expected = [
        quad(lambda u: -np.expm1(-h * (6 + 12 * u)), 0, 1, epsabs=1e-14)[0] for h in hazards
    ]
    assert_allclose(actual, expected, rtol=2e-13, atol=0)
    rng = np.random.default_rng(138)
    followup = 18 - rng.uniform(0, 12, 200_000)
    empirical = np.mean(rng.exponential(1 / 0.03, 200_000) <= followup)
    assert abs(empirical - actual[1]) < 5 * np.sqrt(actual[1] * (1 - actual[1]) / 200_000)
    assert exponential_event_probability(0.1, 0, 0) == 0
    assert_allclose(exponential_event_probability(0.1, 0, 8), -np.expm1(-0.8))


def test_source_approximation_and_extreme_exposure_stability():
    result = survival_sample_size(
        0.1, 0.05, parameter="hazard", accrual=12, followup=6, allocation_ratio=1
    )
    qc = 1 - (np.exp(-0.1 * 6) + 4 * np.exp(-0.1 * 12) + np.exp(-0.1 * 18)) / 6
    assert_allclose(result.event_probabilities, [qc, 1 - (1 - qc) ** 0.5], atol=2e-15)
    tiny = survival_sample_size(
        1e-14, 5e-15, parameter="hazard", accrual=100, followup=0, allocation_ratio=1
    )
    assert_allclose(tiny.event_probabilities, [5e-13, 2.5e-13], rtol=1e-12, atol=0)
    large = survival_sample_size(
        1e308, 5e307, parameter="hazard", accrual=1e308, followup=0, allocation_ratio=1
    )
    assert_allclose(large.event_probabilities, [5 / 6, 1 - np.sqrt(1 / 6)], rtol=2e-14)


def test_joint_equivalence_power_against_normal_rejection_region():
    rng = np.random.default_rng(238)
    events, b, margin = 200, 0.1, 0.4
    se = 1 / np.sqrt(events / 4)
    estimates = rng.normal(b, se, 200_000)
    critical = -ndtri(0.05)
    empirical = np.mean(
        (estimates - critical * se > -margin) & (estimates + critical * se < margin)
    )
    joint = survival_event_power(
        events, b, allocation_ratio=1, objective="equivalence", margin=margin, equivalence="joint"
    )
    conservative = survival_event_power(
        events, b, allocation_ratio=1, objective="equivalence", margin=margin
    )
    assert conservative < joint
    assert abs(empirical - joint) < 5 * np.sqrt(joint * (1 - joint) / 200_000)
    for convention in ["conservative", "joint"]:
        design = survival_sample_size(
            9,
            10,
            accrual=12,
            followup=6,
            allocation_ratio=1,
            objective="equivalence",
            margin=0.4,
            equivalence=convention,
        )
        power = survival_event_power(
            design.required_events,
            design.log_median_ratio,
            allocation_ratio=1,
            objective="equivalence",
            margin=0.4,
            equivalence=convention,
        )
        assert_allclose(power, 0.8, atol=2e-14)
    assert design.rounded_events < 395


def test_units_hazard_entry_and_unequal_allocation_broadcasting():
    scales = np.array([1e-200, 1, 1e200])
    a = survival_sample_size(
        10 * scales,
        20 * scales,
        accrual=12 * scales,
        followup=6 * scales,
        allocation_ratio=1.7,
        event_method="uniform",
    )
    b = survival_sample_size(
        np.log(2) / 10,
        np.log(2) / 20,
        parameter="hazard",
        accrual=12,
        followup=6,
        allocation_ratio=1.7,
        event_method="uniform",
    )
    assert_allclose(a.required_events, b.required_events, rtol=2e-14)
    assert_array_equal(a.group_sizes, np.broadcast_to(b.group_sizes, (3, 2)))
    assert_allclose(
        a.event_probabilities, np.broadcast_to(b.event_probabilities, (3, 2)), rtol=2e-14
    )
    assert np.all(a.power >= 0.8)
    assert np.all(a.expected_events >= a.required_events)
    assert_allclose(a.expected_events, np.sum(a.group_sizes * a.event_probabilities, axis=-1))


def test_invalid_observation_window_and_alternative():
    with pytest.raises(ValueError, match="observable"):
        survival_sample_size(10, 20, accrual=0, followup=0)
    with pytest.raises(ValueError, match="alternative"):
        survival_sample_size(
            10, 9, accrual=12, followup=6, allocation_ratio=1, objective="superiority", margin=0.2
        )
    with pytest.raises(ValueError, match="durations"):
        exponential_event_probability(0.1, 12, -1)
