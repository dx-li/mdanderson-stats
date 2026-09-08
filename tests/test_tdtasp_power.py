"""Native powers and independent rational enumeration of discrete tests."""

import json
from fractions import Fraction
from math import comb, floor
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import tdtasp_ascertainment, tdtasp_fixed_power, tdtasp_genetics, tdtasp_power

FIXTURE = json.loads((Path(__file__).parent / "fixtures/tdtasp_fixed_power.json").read_text())


def reference(n, p, alpha, sides=1, legacy=False):
    if n == 0:
        return -1, 1, 0.0, 0.0
    level = Fraction(str(alpha)) / sides
    mass = [Fraction(comb(n, k), 2**n) for k in range(n + 1)]
    included = 0
    while included <= n and sum(mass[: included + 1]) <= level:
        included += 1
    lower = included - 1 if p <= 0.5 or (sides == 2 and not legacy) else -1
    upper = n - included + 1 if p > 0.5 or (sides == 2 and not legacy) else n + 1
    region = [k for k in range(n + 1) if k <= lower or k >= upper]
    pa = Fraction(str(p))
    return (
        lower,
        upper,
        float(sum(mass[k] for k in region)),
        float(sum(Fraction(comb(n, k)) * pa**k * (1 - pa) ** (n - k) for k in region)),
    )


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_fixed_power(case):
    r = tdtasp_fixed_power(case["observations"], case["answer_probability"], case["alpha"])
    assert float(r.power) == pytest.approx(case["power"], abs=2e-12)
    if case["critical"] >= 0:
        critical = r.lower_critical if case["answer_probability"] <= 0.5 else r.upper_critical
        assert float(critical) == case["critical"]


@pytest.mark.parametrize("case", FIXTURE["native_errors"])
def test_native_empty_region_failure_is_zero_power(case):
    r = tdtasp_fixed_power(case["observations"], case["answer_probability"], case["alpha"])
    assert float(r.power) == 0
    assert float(r.actual_size) == 0


@pytest.mark.parametrize("n", [0, 1, 5, 12])
@pytest.mark.parametrize("p", [0, 0.2, 0.5, 0.8, 1])
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.25])
@pytest.mark.parametrize("sides,legacy", [(1, False), (2, False), (2, True)])
def test_rational_binomial_enumeration(n, p, alpha, sides, legacy):
    expected = reference(n, p, alpha, sides, legacy)
    r = tdtasp_fixed_power(n, p, alpha, sides=sides, legacy_two_sided=legacy)
    np.testing.assert_allclose(
        [r.lower_critical, r.upper_critical, r.actual_size, r.power], expected, atol=1e-14
    )


@pytest.mark.parametrize("families", [0, 5, 12, 30])
@pytest.mark.parametrize("eligibility", ["father", "one", "both"])
@pytest.mark.parametrize("sides", [1, 2])
def test_family_count_mixture(families, eligibility, sides):
    g = tdtasp_genetics([0.3, 0.2, 0.1, 0.4], [0.8, 0.5, 0.2], 0.1)
    a = tdtasp_ascertainment(g, 2, eligibility=eligibility)
    r = tdtasp_power(a, families, sides=sides)
    weights = [
        comb(families, k)
        * a.selection_probability**k
        * (1 - a.selection_probability) ** (families - k)
        for k in range(families + 1)
    ]
    expected = [
        reference(floor(k * a.expected_contributions + 0.5), a.answer_probability, 0.05, sides)
        for k in range(families + 1)
    ]
    assert r.power == pytest.approx(sum(w * x[3] for w, x in zip(weights, expected)), abs=2e-14)
    assert r.actual_size == pytest.approx(
        sum(w * x[2] for w, x in zip(weights, expected)), abs=2e-14
    )


def test_half_up_rounding_and_legacy_scale():
    g = tdtasp_genetics([0.25] * 4, [0.2] * 3)
    a = tdtasp_ascertainment(g, 2, eligibility="father")
    r = tdtasp_power(a, 4)
    assert r.contribution_per_family == pytest.approx(1.5)
    np.testing.assert_array_equal(r.conditional.observations, [0, 2, 3, 5, 6])
    individual = tdtasp_ascertainment(g, 2, sampling="individual", all_affected=True)
    modern = tdtasp_power(individual, 20)
    legacy = tdtasp_power(individual, 20, legacy_scale=True)
    assert modern.contribution_per_family > legacy.contribution_per_family
    assert legacy.contribution_per_family == pytest.approx(
        individual.expected_heterozygous_parents * individual.population_average_truncated_mean
    )


def test_null_power_is_size_and_two_sided_tail_accounting():
    r = tdtasp_fixed_power(np.arange(50), 0.5, sides=2)
    np.testing.assert_allclose(r.power, r.actual_size, atol=1e-15)
    both = tdtasp_fixed_power(20, 0.8, sides=2)
    single = tdtasp_fixed_power(20, 0.8, sides=2, legacy_two_sided=True)
    assert both.power > single.power
    assert both.actual_size == pytest.approx(2 * single.actual_size)


def test_power_is_not_monotone_in_observations():
    # A sample-size search must account for discrete changes in the cutoff.
    r = tdtasp_fixed_power(np.arange(5, 30), 0.7)
    assert np.any(np.diff(r.power) < 0)


def test_readonly_and_preallocation_budget():
    a = tdtasp_ascertainment(tdtasp_genetics([0.25] * 4, [0.2] * 3), 2)
    with pytest.raises(ValueError, match="exceeding max_terms"):
        tdtasp_power(a, 10**9)
    r = tdtasp_power(a, 10)
    for values in [r.eligible_families, r.family_count_probability, *vars(r.conditional).values()]:
        if isinstance(values, np.ndarray):
            with pytest.raises(ValueError):
                values.setflags(write=True)


@pytest.mark.parametrize(
    "options",
    [
        {"observations": -1},
        {"observations": 1.2},
        {"observations": 2**53},
        {"answer_probability": -1},
        {"answer_probability": float("nan")},
        {"alpha": 0},
        {"alpha": 1},
        {"sides": True},
        {"sides": 3},
        {"legacy_two_sided": 1},
    ],
)
def test_invalid_fixed_inputs(options):
    arguments = {"observations": 10, "answer_probability": 0.7}
    arguments.update(options)
    with pytest.raises(ValueError):
        tdtasp_fixed_power(**arguments)


@pytest.mark.parametrize(
    "options",
    [
        {"families": True},
        {"families": -1},
        {"families": 1.5},
        {"families": 2**53},
        {"max_terms": True},
        {"max_terms": 0},
        {"max_terms": 1.5},
        {"legacy_scale": 1},
    ],
)
def test_invalid_mixture_inputs(options):
    a = tdtasp_ascertainment(tdtasp_genetics([0.25] * 4, [0.2] * 3), 2)
    arguments = {"ascertainment": a, "families": 10}
    arguments.update(options)
    with pytest.raises(ValueError):
        tdtasp_power(**arguments)
