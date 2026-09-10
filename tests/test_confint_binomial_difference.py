import numpy as np
from scipy.stats import binom, norm

from mdanderson_stats import (
    confint_binomial_difference_event_limit,
    confint_binomial_difference_probability,
    confint_binomial_difference_sample_size,
)


def test_confint_two_binomial_independent_r_and_midpoint_tail():
    # At the first group's midpoint, both second-group tails still contribute.
    np.testing.assert_allclose(confint_binomial_difference_probability(2, 3, 0.5, 0.4, 1.5), 0.64)
    np.testing.assert_allclose(
        confint_binomial_difference_probability(12, 17, 0.2, 0.4, 0.6), 0.31452557365724548
    )
    assert confint_binomial_difference_sample_size(0.3, 0.2, 0.4) == 74
    np.testing.assert_allclose(
        confint_binomial_difference_probability(74, 74, 0.2, 0.4, 0.3), 0.90808579933027034
    )


def test_confint_two_binomial_full_joint_enumeration_and_symmetries():
    rng = np.random.default_rng(6402)
    for n1, n2 in [(1, 1), (1, 12), (2, 3), (9, 12), (20, 20), (31, 18)]:
        i, j = np.indices((n1 + 1, n2 + 1))
        width = 2 * norm.isf(0.025) * np.sqrt(i * (n1 - i) / n1**3 + j * (n2 - j) / n2**3)
        for p1, p2, length in [(0, 1, 0.01), (0.5, 0.5, 0.8), (*rng.random(2), 0.3)]:
            expected = (binom.pmf(i, n1, p1) * binom.pmf(j, n2, p2) * (width <= length)).sum()
            actual = confint_binomial_difference_probability(n1, n2, p1, p2, length)
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            np.testing.assert_allclose(
                actual, confint_binomial_difference_probability(n2, n1, p2, p1, length), atol=1e-14
            )
            np.testing.assert_allclose(
                actual,
                confint_binomial_difference_probability(n1, n2, 1 - p1, p2, length),
                atol=1e-14,
            )


def test_confint_two_binomial_event_inversion_and_search_range():
    p = confint_binomial_difference_event_limit(40, 60, 0.2, 0.3, assurance=0.8)
    assert p is not None and 0 < p < 0.5
    np.testing.assert_allclose(
        confint_binomial_difference_probability(40, 60, 0.2, p, 0.3), 0.8, atol=1e-12
    )
    assert confint_binomial_difference_event_limit(10, 10, 0.5, 0.001) is None
    assert confint_binomial_difference_event_limit(10, 10, 0.5, 10) == 0.5
    # n=1 always gives zero estimated variance: explicit search range matters.
    assert confint_binomial_difference_sample_size(0.01, 0.5, 0.5, min_sample_size=1) == 1
    values = np.array(
        [confint_binomial_difference_probability(n, n, 0.2, 0.4, 0.3) for n in range(10, 75)]
    )
    assert np.all(values[:-1] < 0.9)
    assert values[-1] >= 0.9
