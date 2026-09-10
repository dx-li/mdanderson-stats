import numpy as np
from scipy.stats import binom

from mdanderson_stats import (
    binomial_interval,
    confint_binomial_event_limit,
    confint_binomial_length,
    confint_binomial_probability,
    confint_binomial_sample_size,
)


def test_confint_binomial_independent_r_and_corrected_manual_examples():
    # R qbeta/dbinom enumeration, without the native fractional-count search.
    for n, p, level, expected in [
        (250, 0.25, 0.9, 0.9842333927173188),
        (162, 0.7, 0.8, 0.75010509662898817),
        (164, 0.7, 0.8, 0.81733777711373934),
    ]:
        np.testing.assert_allclose(
            confint_binomial_probability(n, 0.1, p, confidence=level), expected
        )
    assert confint_binomial_sample_size(0.1, 0.7, confidence=0.8, assurance=0.8) == 164


def test_confint_binomial_cutoffs_against_all_count_outcomes():
    for n in [1, 2, 9, 10, 31, 100]:
        counts = np.arange(n + 1)
        lo, hi = binomial_interval(np.minimum(counts, n - counts), n, 0.9)
        p = np.array([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1])
        for length in [0.01, 0.2, 0.5, 1, float(hi[0] - lo[0])]:
            qualifying = hi - lo <= length
            expected = binom.pmf(counts[:, None], n, p)[qualifying].sum(axis=0)
            actual = confint_binomial_probability(n, length, p, confidence=0.9)
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            assert not actual.flags.writeable
    # Only the zero-event/all-event outcomes have this shortest length.
    width = float(binomial_interval(0, 10, 0.95)[1])
    p = np.array([0, 0.1, 0.5, 0.9, 1])
    np.testing.assert_allclose(confint_binomial_probability(10, width, p), p**10 + (1 - p) ** 10)


def test_confint_binomial_inversions_and_nonmonotone_sample_size():
    for n, p, level in [(10, 0.1, 0.95), (51, 0.3, 0.9), (250, 0.25, 0.9)]:
        for assurance in [0.1, 0.5, 0.9]:
            length = confint_binomial_length(n, p, assurance=assurance, confidence=level)
            assert confint_binomial_probability(n, length, p, confidence=level) >= assurance - 1e-13
            assert (
                confint_binomial_probability(n, np.nextafter(length, 0), p, confidence=level)
                < assurance
            )
    assert confint_binomial_event_limit(10, 0.01) is None
    assert confint_binomial_event_limit(10, 1) == 0.5
    for assurance in [0.05, 0.5, 0.95]:
        p = confint_binomial_event_limit(250, 0.1, confidence=0.9, assurance=assurance)
        assert p is not None
        np.testing.assert_allclose(
            confint_binomial_probability(250, 0.1, p, confidence=0.9), assurance, atol=1e-12
        )
    sizes = np.arange(1, 301)
    probabilities = confint_binomial_probability(sizes, 0.2, 0.1)
    assert np.any(np.diff(probabilities) < -0.01)
    for assurance in [0.5, 0.8, 0.95]:
        expected = int(sizes[np.flatnonzero(probabilities >= assurance)[0]])
        assert (
            confint_binomial_sample_size(0.2, 0.1, assurance=assurance, max_sample_size=300)
            == expected
        )
