"""Independent high-precision quantiles and probability calibration checks."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import KeyboardDesign, simulate_tite_keyboard, toxicity_time_quantile


def test_quantiles_against_high_precision_closed_form():
    cases = [(0.01, 0.1), (0.3, 0.5), (0.8, 0.9), (1e-300, 0.75), (1 - 1e-15, 0.2)]
    for family in ("weibull", "log-logistic"):
        for probability, late in cases:
            for fraction in (0.01, 0.3, 0.99):
                quantile = probability * fraction
                with localcontext() as ctx:
                    ctx.prec = 340
                    p, a, q = map(Decimal.from_float, (probability, late, quantile))
                    one, two = Decimal(1), Decimal(2)
                    if family == "weibull":
                        h = -(one - p).ln()
                        half = -(one - (one - a) * p).ln()
                        shape = (h / half).ln() / two.ln()
                        expected = ((-(one - q).ln() / h).ln() / shape).exp()
                    else:
                        odds = p / (one - p)
                        half = (one - a) * p / (one - (one - a) * p)
                        shape = (odds / half).ln() / two.ln()
                        expected = ((q / (one - q) / odds).ln() / shape).exp()
                actual = toxicity_time_quantile(
                    quantile, probability, 1, distribution=family, late_probability=late
                )
                assert_allclose(actual, float(expected), rtol=3e-13, atol=0)


def test_half_window_calibration_censoring_and_time_scaling():
    p = np.array([1e-300, 0.01, 0.3, 0.8, 1 - 1e-15])
    late = np.array([0.01, 0.1, 0.5, 0.9, 0.99])[:, None]
    for family in ("weibull", "log-logistic"):
        for window in (1e-200, 3, 1e200):
            times = toxicity_time_quantile(
                (1 - late) * p, p, window, distribution=family, late_probability=late
            )
            assert_allclose(times / window, 0.5, rtol=2e-13)
        assert_allclose(toxicity_time_quantile(p, p, 3, distribution=family), 3)
        assert np.all(np.isinf(toxicity_time_quantile(1, p, 3, distribution=family)))
        assert np.isinf(toxicity_time_quantile(0, 0, 3, distribution=family))
        assert toxicity_time_quantile(0, 0.3, 3, distribution=family) == 0
    assert np.isinf(toxicity_time_quantile(1, 1e-300, 3, distribution="uniform"))


def test_calibrated_simulations_follow_incidence_and_late_fraction():
    # One patient: no adaptive allocation can obscure the generating laws.
    for family in ("weibull", "log-logistic"):
        result = simulate_tite_keyboard(
            KeyboardDesign(),
            [0.3, 0.6],
            90,
            1,
            cohorts=1,
            cohort_size=1,
            trials=6000,
            event_distribution=family,
            late_probability=0.8,
            rng=135,
        )
        toxic = result.toxicities[:, 0] == 1
        assert abs(toxic.mean() - 0.3) < 6 * np.sqrt(0.3 * 0.7 / 6000)
        assert abs(np.mean(result.duration[toxic] > 46) - 0.8) < 6 * np.sqrt(
            0.8 * 0.2 / toxic.sum()
        )
        assert np.all(result.duration[~toxic] == 91)


def test_nonexistent_calibrations_are_rejected():
    for family in ("weibull", "log-logistic"):
        with pytest.raises(ValueError):
            toxicity_time_quantile(0.5, 1, 90, distribution=family)
        for late in (0, 1):
            with pytest.raises(ValueError):
                toxicity_time_quantile(0.2, 0.3, 90, distribution=family, late_probability=late)
    with pytest.raises(ValueError):
        toxicity_time_quantile(0.2, 0.3, 90, distribution="uniform", late_probability=0.8)
