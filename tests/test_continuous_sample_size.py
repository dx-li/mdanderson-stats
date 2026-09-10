"""Source examples, integer enrollment and independent normal-model experiments."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import f, t

from mdanderson_stats import (
    anova_effect_size,
    anova_power,
    anova_sample_size,
    correlation_sample_size,
    normal_mean_power,
    normal_mean_sample_size,
)


@pytest.mark.parametrize(
    "difference,sd,objective,margin,ratio,sides,expected",
    [
        (0.3, 1, "equality", 0, None, 1, (71,)),
        (0.3, 1, "equality", 0, 1, 1, (139, 139)),
        (0.15, 0.1, "equivalence", 0.2, None, 1, (27,)),
        (0.15, 0.1, "equivalence", 0.2, 1, 1, (51, 51)),
        (-0.1, 1, "noninferiority", 0.2, None, 1, (620,)),
        (-0.1, 0.1, "noninferiority", 0.2, 1, 1, (14, 14)),
        (0.2, 0.5, "superiority", 0.15, None, 1, (620,)),
        (0.3, 0.1, "superiority", 0.25, 1, 1, (51, 51)),
        (0.4, 1, "equality", 0, None, 2, (52,)),  # paired differences, SD=1
    ],
)
def test_source_mean_examples(difference, sd, objective, margin, ratio, sides, expected):
    result = normal_mean_sample_size(
        difference, sd, objective=objective, margin=margin, allocation_ratio=ratio, sides=sides
    )
    assert result.group_sizes == expected
    assert result.total_size == sum(expected)
    assert result.previous_power < 0.8 <= result.power


def test_source_correlation_and_anova_examples():
    for result, expected in [
        (correlation_sample_size(0.4), (46,)),
        (correlation_sample_size(0.4, method="z"), (47,)),
        (anova_sample_size(0.25, 4), (45, 45, 45, 45)),
    ]:
        assert result.group_sizes == expected
        assert result.previous_power < 0.8 <= result.power


@pytest.mark.parametrize("n,nt,delta,sd,margin", [(2, None, 0, 1, 0.5), (10, 15, 0.15, 0.3, 0.3)])
def test_exact_equivalence_against_simulated_pooled_tost(n, nt, delta, sd, margin):
    rng = np.random.default_rng(119)
    repetitions = 150_000
    x = rng.normal(0, sd, size=(repetitions, n))
    if nt is None:
        estimate = x.mean(axis=1) + delta
        se = x.std(axis=1, ddof=1) / np.sqrt(n)
        df = n - 1
    else:
        y = rng.normal(delta, sd, size=(repetitions, nt))
        estimate = y.mean(axis=1) - x.mean(axis=1)
        df = n + nt - 2
        variance = ((n - 1) * x.var(axis=1, ddof=1) + (nt - 1) * y.var(axis=1, ddof=1)) / df
        se = np.sqrt(variance * (1 / n + 1 / nt))
    critical = t.isf(0.05, df)
    empirical = np.mean((estimate - critical * se > -margin) & (estimate + critical * se < margin))
    exact = normal_mean_power(
        n, delta, sd, treatment_size=nt, objective="equivalence", margin=margin, exact=True
    )
    planning = normal_mean_power(
        n, delta, sd, treatment_size=nt, objective="equivalence", margin=margin
    )
    assert abs(exact - empirical) < 5 * np.sqrt(exact * (1 - exact) / repetitions)
    assert planning <= exact + 1e-10
    if n == 2:
        assert planning == 0 and exact > 0.02


def test_normal_null_power_and_anova_sampling_distribution():
    assert_allclose(normal_mean_power([3, 15, 100], 0, exact=True), 0.05, atol=2e-14)
    assert_allclose(normal_mean_power([3, 15, 100], 0), 0.025, atol=2e-14)
    assert_allclose(anova_power([2, 10, 100], 0, 4), 0.05, atol=2e-14)
    rng = np.random.default_rng(219)
    means = np.array([-0.5, 0, 0.2, 0.8])
    observations = rng.normal(means[None, :, None], 1, size=(60_000, 4, 9))
    averages = observations.mean(axis=-1)
    between = 9 * ((averages - averages.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1) / 3
    within = ((observations - averages[..., None]) ** 2).sum(axis=(-1, -2)) / 32
    empirical = np.mean(between / within > f.isf(0.05, 3, 32))
    exact = anova_power(9, anova_effect_size(means, 1), 4)
    assert abs(exact - empirical) < 5 * np.sqrt(exact * (1 - exact) / 60_000)


def test_unequal_allocation_broadcasting_and_scale_invariance():
    design = normal_mean_sample_size(0.3, allocation_ratio=1.7)
    nc, nt = design.group_sizes
    assert nt == np.ceil(nc * 1.7)
    assert design.previous_power < 0.8 <= design.power
    # Swapping labels preserves a two-sided equality test at actual group sizes.
    assert_allclose(
        normal_mean_power(nc, 0.3, treatment_size=nt),
        normal_mean_power(nt, -0.3, treatment_size=nc),
    )
    scales = np.array([1e-200, 1, 1e200])
    assert_allclose(normal_mean_power(25, 0.3 * scales, scales), normal_mean_power(25, 0.3))
    assert_allclose(
        anova_effect_size(np.array([-1, 0, 1]) * scales[:, None], scales), np.sqrt(2 / 3)
    )
    # The exact search uses its own chosen power definition and integer predecessor.
    exact = normal_mean_sample_size(0.05, 0.3, objective="equivalence", margin=0.2, exact=True)
    assert exact.previous_power < 0.8 <= exact.power


def test_unattainable_or_invalid_designs_fail():
    with pytest.raises(ValueError, match="alternative"):
        normal_mean_sample_size(0.3, objective="equivalence", margin=0.2)
    with pytest.raises(ValueError, match="max_size"):
        normal_mean_sample_size(0.01, max_size=10)
    with pytest.raises(ValueError, match="sample sizes"):
        normal_mean_power(1, 0.3)
    with pytest.raises(ValueError, match="positive"):
        anova_sample_size(0, 4)
