"""Published planning examples and independent rational Fisher enumeration."""

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    binary_proportion_power,
    binary_proportion_sample_size,
    cohen_kappa,
    fisher_power,
    fisher_sample_size,
    kappa_power,
    kappa_sample_size,
    mcnemar_power,
    mcnemar_sample_size,
)


@pytest.mark.parametrize(
    "pc,pt,objective,margin,ratio,expected",
    [
        (0.2, 0.5, "equality", 0, None, (18,)),
        (0.2, 0.5, "equality", 0, 1, (29, 29)),
        (0.2, 0.35, "equivalence", 0.2, None, (563,)),
        (0.25, 0.35, "equivalence", 0.2, 1, (257, 257)),
        (0.3, 0.2, "noninferiority", 0.2, None, (99,)),
        (0.4, 0.3, "noninferiority", 0.2, 1, (279, 279)),
        (0.3, 0.5, "superiority", 0.15, None, (619,)),
        (0.3, 0.6, "superiority", 0.2, 1, (279, 279)),
    ],
)
def test_source_proportion_examples(pc, pt, objective, margin, ratio, expected):
    result = binary_proportion_sample_size(
        pc, pt, objective=objective, margin=margin, allocation_ratio=ratio
    )
    assert result.group_sizes == expected
    assert result.previous_power < 0.8 <= result.power


def test_source_continuity_fisher_mcnemar_kappa_examples():
    for result, expected in [
        (
            binary_proportion_sample_size(0.2, 0.5, allocation_ratio=1, method="score-continuity"),
            (37, 37),
        ),
        (fisher_sample_size(0.35, 0.6), (56, 56)),
        (mcnemar_sample_size(0.2, 0.5), (46,)),
        (kappa_sample_size(0.4, 0.3, 0.3, 0.5), (136,)),
    ]:
        assert result.group_sizes == expected
        assert result.total_size == sum(expected)
        assert result.previous_power < 0.8 <= result.power


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_fisher_against_rational_table_enumeration(alternative):
    nc, nt = 4, 5
    rates = [(Fraction(1, 3), Fraction(2, 3)), (Fraction(1, 2), Fraction(1, 2)), (0, 1), (1, 0)]
    expected = []
    for pc, pt in rates:
        power = Fraction(0)
        for xc in range(nc + 1):
            for xt in range(nt + 1):
                total = xc + xt
                xs = range(max(0, total - nc), min(nt, total) + 1)
                masses = {
                    x: Fraction(comb(nt, x) * comb(nc, total - x), comb(nc + nt, total)) for x in xs
                }
                if alternative == "greater":
                    selected = [x for x in xs if x >= xt]
                elif alternative == "less":
                    selected = [x for x in xs if x <= xt]
                else:
                    selected = [x for x in xs if masses[x] <= masses[xt]]
                if sum(masses[x] for x in selected) <= Fraction(1, 10):
                    power += (
                        comb(nc, xc)
                        * pc**xc
                        * (1 - pc) ** (nc - xc)
                        * comb(nt, xt)
                        * pt**xt
                        * (1 - pt) ** (nt - xt)
                    )
        expected.append(float(power))
    actual = fisher_power(
        nc,
        nt,
        [float(a) for a, b in rates],
        [float(b) for a, b in rates],
        alpha=0.1,
        alternative=alternative,
    )
    assert_allclose(actual, expected, atol=2e-15)
    assert actual[1] <= 0.1  # true null rejection probability


def test_kappa_delta_variance_against_multinomial_sampling():
    # Fixed binary margins (.4, .3), kappa=.5 give this joint population table.
    probabilities = np.array([0.235, 0.165, 0.065, 0.535])
    rng = np.random.default_rng(139)
    n = 2000
    tables = rng.multinomial(n, probabilities, size=50_000).reshape(-1, 2, 2)
    estimates = cohen_kappa(tables).kappa
    # Recover the alternative variance from planning power at a critical level
    # whose null/alternative standard errors are equal (k0=k1=.5).
    assert_allclose(kappa_power(n, 0.4, 0.3, 0.5, 0.5), 0.05, atol=2e-15)
    theoretical = cohen_kappa(probabilities.reshape(2, 2)).variance / n
    assert_allclose(estimates.var(ddof=1), theoretical, rtol=0.025)
    assert abs(estimates.mean() - 0.5) < 0.001


def test_allocation_and_paired_direction():
    result = binary_proportion_sample_size(
        0.2, 0.45, allocation_ratio=1.3, method="score-continuity"
    )
    nc, nt = result.group_sizes
    assert nt == np.ceil(1.3 * nc)
    ns = np.arange(1, nc)
    assert np.all(
        binary_proportion_power(
            ns, 0.2, 0.45, treatment_size=np.ceil(1.3 * ns), method="score-continuity"
        )
        < 0.8
    )
    assert_allclose(
        mcnemar_power([20, 40, 80], 0.2, 0.5, sides=2),
        mcnemar_power([20, 40, 80], 0.5, 0.2, sides=2),
    )
    assert np.all(mcnemar_power([20, 40, 80], 0.5, 0.2) < 0.05)
    # Two-sided equality planning is deliberately a dominant-tail approximation.
    assert_allclose(binary_proportion_power(100, 0.3, 0.3, sides=2), 0.025, atol=2e-15)


def test_invalid_population_and_search_bounds():
    with pytest.raises(ValueError, match="infeasible"):
        kappa_sample_size(0.9, 0.1, 0, 0.5)
    with pytest.raises(ValueError, match="alternative"):
        mcnemar_sample_size(0.5, 0.2)
    with pytest.raises(ValueError, match="max_size"):
        fisher_sample_size(0.35, 0.6, max_size=10)
    with pytest.raises(ValueError, match="alternative"):
        binary_proportion_sample_size(0.2, 0.5, objective="equivalence", margin=0.1)
