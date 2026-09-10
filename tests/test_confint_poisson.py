import numpy as np
from scipy.stats import poisson

from mdanderson_stats import (
    confint_poisson_exposure,
    confint_poisson_length,
    confint_poisson_probability,
    confint_poisson_rate_limit,
    poisson_interval,
)


def test_confint_poisson_manual_and_independent_r_calculations():
    np.testing.assert_allclose(
        confint_poisson_probability(210, 1, 20, confidence=0.9), 0.0053469235111210742
    )
    np.testing.assert_allclose(confint_poisson_length(210, 20, confidence=0.9), 1.0300012109271501)
    np.testing.assert_allclose(
        confint_poisson_rate_limit(210, 1, confidence=0.9), 18.832384614923605
    )
    exposure = confint_poisson_exposure(1, 20, confidence=0.9)
    np.testing.assert_allclose(exposure, 222.5920387855194349, rtol=1e-12)
    np.testing.assert_allclose(
        confint_poisson_probability(exposure, 1, 20, confidence=0.9), 0.90011834565932991
    )


def test_confint_poisson_zero_one_counts_and_full_enumeration():
    rates = np.array([0, 0.1, 1, 10])
    for k in [0, 1, 10]:
        lo, hi = poisson_interval(k)
        width = float(hi - lo)
        actual = confint_poisson_probability(1, width, rates)
        np.testing.assert_allclose(actual, poisson.cdf(k, rates))
        assert not actual.flags.writeable
    assert confint_poisson_probability(1, 0.01, 0) == 0
    assert confint_poisson_rate_limit(1, 0.01) is None
    counts = np.arange(100)
    lo, hi = poisson_interval(counts, 0.9)
    accepted = (hi - lo) / 5 <= 2
    expected = poisson.pmf(counts[accepted], 3 * 5).sum()
    np.testing.assert_allclose(confint_poisson_probability(5, 2, 3, confidence=0.9), expected)


def test_confint_poisson_discrete_inversions_and_change_of_units():
    exposure = confint_poisson_exposure(0.5, 1, assurance=0.8)
    assert confint_poisson_probability(exposure, 0.5, 1) >= 0.8
    # Every earlier jump fails: between jumps the probability only decreases.
    counts = np.arange(1000)
    lo, hi = poisson_interval(counts)
    times = (hi - lo) / 0.5
    assert np.all(
        poisson.cdf(counts[times < exposure - 1e-12], times[times < exposure - 1e-12]) < 0.8
    )
    for scale in [1e-200, 1e200]:
        np.testing.assert_allclose(
            confint_poisson_probability(210 * scale, 1 / scale, 20 / scale, confidence=0.9),
            confint_poisson_probability(210, 1, 20, confidence=0.9),
            rtol=1e-12,
        )
    for assurance in [0.05, 0.5, 0.95]:
        rate = confint_poisson_rate_limit(10, 2, assurance=assurance)
        np.testing.assert_allclose(confint_poisson_probability(10, 2, rate), assurance, atol=1e-12)
        length = confint_poisson_length(10, 1, assurance=assurance)
        assert confint_poisson_probability(10, length, 1) >= assurance
    # The first threshold is finite even if later thresholds in its batch overflow.
    huge = confint_poisson_exposure(3e-308, 0, confidence=0.9)
    np.testing.assert_allclose(huge, -np.log(0.05) / (3e-308))
    assert np.isfinite(huge)
