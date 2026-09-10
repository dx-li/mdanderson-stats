"""Independent rational enumeration, published example and final-decision edge cases."""

from fractions import Fraction
from math import comb, factorial

import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import plan_predictive_binary, predictive_binary


def beta(a, b):
    return Fraction(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))


def mass(n, k, a, b):
    return comb(n, k) * beta(a + k, b + n - k) / beta(a, b)


def order(a, b, c, d):
    return sum(
        (comb(c + d - 1, k) * beta(a + k, b + c + d - 1 - k) / beta(a, b) for k in range(c, c + d)),
        Fraction(0),
    )


def test_bayesian_planning_and_interim_against_rational_enumeration():
    prior = [[1, 2], [2, 1]]
    plan = plan_predictive_binary([2, 3], [5, 6], prior=prior, posterior_cutoff=0.85)
    for sa in range(3):
        for sb in range(4):
            expected = [Fraction(0), Fraction(0), Fraction(0)]
            for ya in range(4):
                for yb in range(4):
                    p = order(1 + sa + ya, 2 + 5 - sa - ya, 2 + sb + yb, 1 + 6 - sb - yb)
                    code = 0 if p > Fraction(85, 100) else 1 if 1 - p > Fraction(85, 100) else 2
                    expected[code] += mass(3, ya, 1 + sa, 2 + 2 - sa) * mass(
                        3, yb, 2 + sb, 1 + 3 - sb
                    )
            result = predictive_binary(
                [sa, sb], [2 - sa, 3 - sb], [5, 6], prior=prior, posterior_cutoff=0.85
            )
            assert_allclose(
                [result.arm_a_superior, result.arm_b_superior, result.inconclusive],
                list(map(float, expected)),
                atol=2e-15,
            )
            assert_allclose(
                [
                    plan.arm_a_superior[sa, sb],
                    plan.arm_b_superior[sa, sb],
                    plan.inconclusive[sa, sb],
                ],
                list(map(float, expected)),
                atol=2e-15,
            )


def test_published_frequentist_example_and_beta_predictive_mass():
    result = predictive_binary(
        [10, 16], [15, 9], [50, 50], prior=[[0.6, 0.4], [0.6, 0.4]], method="frequentist"
    )
    assert_allclose(result.arm_b_superior, 0.6886, atol=5e-5)
    assert_allclose(result.arm_a_superior, 3e-6, atol=1e-6)
    assert_allclose(
        result.arm_a_superior + result.arm_b_superior + result.inconclusive, 1, atol=2e-14
    )
    # Cook's beta(2,3) prediction for 0,1,2 successes on the next two attempts.
    basketball = predictive_binary([0, 0], [0, 0], [2, 0], prior=[[2, 3], [1, 1]])
    assert_allclose(basketball.future_success_probability_a, [0.4, 0.4, 0.2], atol=2e-15)
    plan = plan_predictive_binary([2, 2], [5, 5], method="frequentist")
    for a in range(3):
        for b in range(3):
            cell = predictive_binary([a, b], [2 - a, 2 - b], [5, 5], method="frequentist")
            assert_allclose(plan.arm_a_superior[a, b], cell.arm_a_superior, atol=2e-15)
    assert_allclose(plan.arm_a_superior, plan.arm_b_superior.T, atol=2e-15)


def test_no_future_observations_ties_and_invalid_plans():
    tied = predictive_binary([0, 0], [0, 0], [0, 0], posterior_cutoff=0.5)
    assert tied.inconclusive == 1
    # Beta(19,1) versus Uniform: exact P(A>B)=.95, strict cutoff must not pass.
    boundary = predictive_binary([18, 0], [0, 0], [18, 0])
    assert boundary.inconclusive == 1
    all_success = predictive_binary([5, 5], [0, 0], [5, 5], method="frequentist")
    assert all_success.inconclusive == 1
    decisive = predictive_binary([5, 0], [0, 5], [5, 5], method="frequentist")
    assert decisive.arm_a_superior == 1
    with pytest.raises(ValueError, match="positive planned"):
        predictive_binary([0, 0], [0, 0], [0, 0], method="frequentist")
    with pytest.raises(ValueError, match="at least"):
        predictive_binary([3, 0], [2, 0], [4, 4])
    assert not boundary.final_decision.flags.writeable
