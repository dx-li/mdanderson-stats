"""Published rounded boundaries, independent mean-coordinate integration and units."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from mdanderson_stats import bayes_factor_survival, bayes_factor_survival_boundaries


def test_guide_boundary_table_and_strict_decisions():
    counts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 47, 48, 49]
    b = bayes_factor_survival_boundaries(counts, null_median=4, alternative_median_mode=5.5)
    assert np.all(np.isneginf(b.inferiority_time[:5]))
    lower_days = [195, 408, 621, 833, 1044, 1256, 8912, 9117, 9321]
    upper_days = [
        574,
        834,
        1085,
        1328,
        1568,
        1804,
        2037,
        2268,
        2498,
        2726,
        2953,
        10933,
        11143,
        11353,
    ]
    # Guide uses integer days. Its display differs by up to ~1.3 days from
    # continuous roots converted using 365.25/12 days per month.
    assert_allclose(b.inferiority_time[5:] * 365.25 / 12, lower_days, atol=1.5, rtol=0)
    assert_allclose(b.superiority_time * 365.25 / 12, upper_days, atol=1.5, rtol=0)
    at_low = bayes_factor_survival(
        counts[5:], b.inferiority_time[5:], null_median=4, alternative_median_mode=5.5
    )
    at_high = bayes_factor_survival(
        counts, b.superiority_time, null_median=4, alternative_median_mode=5.5
    )
    assert_allclose(at_low.alternative_probability, 0.15, atol=1e-9, rtol=0)
    assert_allclose(at_high.alternative_probability, 0.8, atol=1e-9, rtol=0)
    state = bayes_factor_survival(5, [1, 20, 80], null_median=4, alternative_median_mode=5.5)
    assert_array_equal(state.decision, ["inferiority", "continue", "superiority"])
    final = bayes_factor_survival(5, 20, null_median=4, alternative_median_mode=5.5, final=True)
    assert final.decision == "inconclusive"


def test_direct_mean_parameter_integrals_and_no_data():
    theta0, mode, tau = 6.0, 8.0, 6.0
    for d, exposure in [(0, 0), (0, 30), (5, 30), (10, 80), (50, 200)]:

        def integrand(theta):
            gap = theta - theta0
            logprior = np.log(2 * tau) - 3 * np.log(gap) - tau / gap**2
            logratio = -d * np.log(theta / theta0) + exposure * (1 / theta0 - 1 / theta)
            return np.exp(logprior + logratio)

        bf, error = quad(integrand, theta0, np.inf, epsabs=1e-15, epsrel=1e-11, limit=300)
        assert error < max(1e-14, bf * 1e-9)
        value = bayes_factor_survival(
            d, exposure, null_median=theta0 * np.log(2), alternative_median_mode=mode * np.log(2)
        )
        assert_allclose(value.log_bayes_factor, np.log(bf), atol=2e-9, rtol=0)
    no_data = bayes_factor_survival(0, 0, null_median=4, alternative_median_mode=5.5)
    assert no_data.log_bayes_factor == 0
    assert no_data.alternative_probability == 0.5


def test_unit_invariance_disabled_stopping_and_extreme_evidence():
    base = bayes_factor_survival(
        [0, 5, 10], [10, 30, 60], null_median=4, alternative_median_mode=5.5
    )
    boundaries = bayes_factor_survival_boundaries(
        [5, 10], null_median=4, alternative_median_mode=5.5
    )
    for scale in [1e-200, 1e200]:
        value = bayes_factor_survival(
            [0, 5, 10],
            np.array([10, 30, 60]) * scale,
            null_median=4 * scale,
            alternative_median_mode=5.5 * scale,
        )
        assert_allclose(value.log_bayes_factor, base.log_bayes_factor, atol=1e-10, rtol=0)
        b = bayes_factor_survival_boundaries(
            [5, 10], null_median=4 * scale, alternative_median_mode=5.5 * scale
        )
        assert_allclose(b.inferiority_time / scale, boundaries.inferiority_time, rtol=1e-11)
        assert_allclose(b.superiority_time / scale, boundaries.superiority_time, rtol=1e-11)
    disabled = bayes_factor_survival_boundaries(
        [0, 5],
        null_median=4,
        alternative_median_mode=5.5,
        inferiority_cutoff=0,
        superiority_cutoff=1,
    )
    assert np.all(np.isneginf(disabled.inferiority_time))
    assert np.all(np.isposinf(disabled.superiority_time))
    strong = bayes_factor_survival(0, 2000, null_median=4, alternative_median_mode=5.5)
    assert strong.log_bayes_factor > 300
    assert strong.decision == "superiority"
