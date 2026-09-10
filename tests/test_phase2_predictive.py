"""Exact predictive identities, complete response paths and grid optimality."""

from fractions import Fraction
from itertools import product
from math import comb, factorial

from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import (
    optimize_phase2_predictive,
    phase2_predictive_design,
    predictive_efficacy_design,
)


def beta(a, b):
    return Fraction(factorial(a - 1) * factorial(b - 1), factorial(a + b - 1))


def test_strict_final_and_early_efficacy_with_predictive_sum():
    design = phase2_predictive_design(
        6, 0.5, looks=[3, 6], theta_final=0.8, theta_lower=0.2, theta_upper=1
    )
    for n in range(7):
        for x in range(n + 1):
            expected = Fraction(0)
            for y in range(6 - n + 1):
                # For uniform prior, final beta SF at .5 is a binomial lower sum.
                posterior = sum(Fraction(comb(7, k), 2**7) for k in range(x + y + 1))
                if posterior > Fraction(8, 10):
                    expected += (
                        comb(6 - n, y) * beta(1 + x + y, 1 + 6 - x - y) / beta(1 + x, 1 + n - x)
                    )
            assert_allclose(design.high_probability[n, x], float(expected), atol=2e-15)
    assert_array_equal(design.positive_min, design.looks + 1)
    strict = phase2_predictive_design(1, 0.5, looks=[1], theta_final=0.75)
    original = predictive_efficacy_design(
        1, prior=[1, 1], looks=[1], target_rate=0.5, final_probability=0.75
    )
    assert strict.final_positive_min == 2
    assert original.final_positive_min == 1
    zero = phase2_predictive_design(100, 0.999, looks=[50, 100], theta_final=0)
    assert zero.final_positive_min == 0
    assert zero.high_probability[0, 0] == 1


def path_oc(design, p):
    success = 0.0
    en = 0.0
    for outcomes in product([0, 1], repeat=design.max_subjects):
        terminal = design.monitor_outcomes(outcomes)
        # Replay is sticky; final array entry carries the first decision.
        decisions = terminal.decision
        first = next((i for i, d in enumerate(decisions) if d != "continue"), len(outcomes) - 1)
        weight = p ** sum(outcomes) * (1 - p) ** (len(outcomes) - sum(outcomes))
        success += weight * (decisions[first] in ["stop_efficacy", "final_positive"])
        en += weight * (first + 1)
    return success, en


def test_grid_and_both_objectives_against_complete_trial_paths():
    candidates = []
    for n, lower, final in product([6, 8], [0.05, 0.2, 0.4], [0.6, 0.8, 0.9]):
        design = phase2_predictive_design(
            n, 0.2, looks=[3, n], theta_lower=lower, theta_final=final, theta_upper=0.9
        )
        null, alt = path_oc(design, 0.2), path_oc(design, 0.6)
        oc = design.operating_characteristics([0.2, 0.6])
        assert_allclose(oc.positive_conclusion, [null[0], alt[0]], atol=2e-14)
        assert_allclose(oc.expected_sample_size, [null[1], alt[1]], atol=2e-13)
        if null[0] <= 0.3 and alt[0] >= 0.4:
            candidates.append((n, lower, final, null[1], alt[0], design))
    for objective in ["power", "expected_sample_size"]:
        fit = optimize_phase2_predictive(
            [6, 8],
            0.2,
            0.6,
            type1_error=0.3,
            minimum_power=0.4,
            theta_lowers=[0.05, 0.2, 0.4],
            theta_finals=[0.6, 0.8, 0.9],
            theta_upper=0.9,
            interim_looks=[3],
            objective=objective,
        )
        key = (
            (lambda c: (-round(c[4], 12), round(c[3], 12), c[0]))
            if objective == "power"
            else (lambda c: (round(c[3], 12), -round(c[4], 12), c[0]))
        )
        expected = min(candidates, key=key)
        assert fit.best.design.max_subjects == expected[0]
        assert_allclose(
            fit.best.operating_characteristics.expected_sample_size[0], expected[3], atol=2e-13
        )
        for lower, final in fit.best.threshold_pairs:
            same = phase2_predictive_design(
                fit.best.design.max_subjects,
                0.2,
                looks=[3, fit.best.design.max_subjects],
                theta_lower=lower,
                theta_final=final,
                theta_upper=0.9,
            )
            assert_array_equal(same.futility_max, fit.best.design.futility_max)
            assert_array_equal(same.positive_min, fit.best.design.positive_min)
            assert same.final_positive_min == fit.best.design.final_positive_min
