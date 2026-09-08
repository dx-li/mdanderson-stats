"""Exhaustive first-design checks for fixed and random-family sample sizes."""

import numpy as np
import pytest

from mdanderson_stats import (
    tdtasp_ascertainment,
    tdtasp_fixed_power,
    tdtasp_fixed_sample_size,
    tdtasp_genetics,
    tdtasp_haplotype_frequencies,
    tdtasp_power,
    tdtasp_sample_size,
)


@pytest.mark.parametrize("p", [0, 0.2, 0.35, 0.65, 0.8, 1])
@pytest.mark.parametrize("target", [0.2, 0.8])
@pytest.mark.parametrize("sides,legacy", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("batch", [7, 64])
def test_first_fixed_design_exhaustively(p, target, sides, legacy, batch):
    counts = np.arange(1, 1001)
    forward = tdtasp_fixed_power(counts, p, sides=sides, legacy_two_sided=legacy)
    expected = int(counts[np.flatnonzero(forward.power >= target)[0]])
    result = tdtasp_fixed_sample_size(
        p, target, sides=sides, legacy_two_sided=legacy, max_observations=1000, batch_size=batch
    )
    assert int(result.observations) == expected
    assert float(result.power) >= target


@pytest.mark.parametrize("test", ["tdt", "asp"])
@pytest.mark.parametrize("sampling", ["family", "individual"])
@pytest.mark.parametrize("eligibility", ["father", "one", "both"])
@pytest.mark.parametrize("sides", [1, 2])
def test_first_family_design_exhaustively(test, sampling, eligibility, sides):
    g = tdtasp_genetics(tdtasp_haplotype_frequencies(0.3, 0.3, 1), [1, 0, 0], 0.05)
    a = tdtasp_ascertainment(g, 2, test=test, sampling=sampling, eligibility=eligibility)
    power = np.array([tdtasp_power(a, n, sides=sides).power for n in range(1, 81)])
    expected = int(np.flatnonzero(power >= 0.6)[0]) + 1
    result = tdtasp_sample_size(a, 0.6, sides=sides, max_families=80)
    assert result.design.families == expected
    assert result.design.power == pytest.approx(power[expected - 1], abs=1e-15)
    assert np.all(power[: expected - 1] < 0.6)
    assert result.min_families <= result.search_start <= expected


def test_discrete_drop_and_nontrivial_lower_bound():
    assert tdtasp_fixed_power(5, 0.7).power >= 0.16
    assert tdtasp_fixed_power(6, 0.7).power < 0.16
    first = tdtasp_fixed_sample_size(0.7, 0.16, max_observations=10, batch_size=2)
    assert int(first.observations) == 5
    later = tdtasp_fixed_sample_size(
        0.7, 0.16, min_observations=6, max_observations=10, batch_size=2
    )
    exhaustive = tdtasp_fixed_power(np.arange(6, 11), 0.7)
    assert int(later.observations) == 6 + np.flatnonzero(exhaustive.power >= 0.16)[0]


def test_legacy_options_and_inclusive_search_bounds():
    g = tdtasp_genetics(tdtasp_haplotype_frequencies(0.3, 0.3, 1), [1, 0, 0], 0.05, legacy_asp=True)
    a = tdtasp_ascertainment(g, 2, sampling="individual", all_affected=True, legacy_moments=True)
    options = dict(sides=2, legacy_scale=True, legacy_two_sided=True)
    powers = np.array([tdtasp_power(a, n, **options).power for n in range(7, 60)])
    expected = 7 + int(np.flatnonzero(powers >= 0.8)[0])
    result = tdtasp_sample_size(a, 0.8, min_families=7, max_families=expected, **options)
    assert result.design.families == expected
    with pytest.raises(ValueError, match="no qualifying design"):
        tdtasp_sample_size(a, 0.8, min_families=7, max_families=expected - 1, **options)


def test_resource_limit_caps_prefetch_without_rejecting_valid_search():
    a = tdtasp_ascertainment(tdtasp_genetics([0.5, 0, 0, 0.5], [1, 0, 0]), 2)
    result = tdtasp_sample_size(a, 0.5, max_families=1000, max_terms=31)
    assert result.design.families < 31
    with pytest.raises(ValueError, match="exceeding max_terms"):
        tdtasp_sample_size(a, 0.99, max_families=1000, max_terms=3)


def test_envelope_skips_proven_insufficient_counts():
    a = tdtasp_ascertainment(tdtasp_genetics([0.3, 0.2, 0.1, 0.4], [0.8, 0.5, 0.2], 0.1), 2)
    result = tdtasp_sample_size(a, 0.8, max_families=10000)
    assert result.search_start > 100
    assert result.evaluations < result.design.families // 2
    assert result.design.power >= 0.8
    assert tdtasp_power(a, result.design.families - 1).power < 0.8


@pytest.mark.parametrize("target", [0, 1, float("nan")])
def test_invalid_targets(target):
    a = tdtasp_ascertainment(tdtasp_genetics([0.25] * 4, [0.2] * 3), 2)
    with pytest.raises(ValueError, match="target_power"):
        tdtasp_sample_size(a, target)
    with pytest.raises(ValueError, match="target_power"):
        tdtasp_fixed_sample_size(0.7, target)


@pytest.mark.parametrize("low,high", [(0, 10), (10, 5), (True, 10), (1, 2.5), (1, 2**53)])
def test_invalid_bounds(low, high):
    a = tdtasp_ascertainment(tdtasp_genetics([0.25] * 4, [0.2] * 3), 2)
    with pytest.raises(ValueError):
        tdtasp_sample_size(a, min_families=low, max_families=high)
    with pytest.raises(ValueError):
        tdtasp_fixed_sample_size(0.7, min_observations=low, max_observations=high)


def test_null_and_bounded_unattainability():
    a = tdtasp_ascertainment(tdtasp_genetics([0.25] * 4, [0.2] * 3), 2)
    with pytest.raises(ValueError, match="null power"):
        tdtasp_sample_size(a, 0.8)
    with pytest.raises(ValueError, match="null power"):
        tdtasp_fixed_sample_size(0.5, 0.8)
    with pytest.raises(ValueError, match="no qualifying design"):
        tdtasp_fixed_sample_size(0.7, 0.8, max_observations=2)
    null = tdtasp_fixed_sample_size(0.5, 0.03, max_observations=20)
    assert null.power >= 0.03
