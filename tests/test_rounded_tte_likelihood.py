"""Independent likelihood checks for positive continuous rounded intervals."""

import math

import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.stats import expon, fisk, gamma, invgamma, lognorm, weibull_min

from mdanderson_stats._rounded_tte_likelihood import rounded_tte_log_probabilities


def _interval(family, coordinates, lower=0.8, upper=1.3, *, work_counter=None, work_limit=0):
    return rounded_tte_log_probabilities(
        [math.log(lower)],
        [math.log(upper)],
        coordinates,
        family,
        work_counter=work_counter,
        work_limit=work_limit,
    )[0][0]


@pytest.mark.parametrize(
    ("family", "coordinates", "distribution"),
    [
        ("exponential", [math.log(1.4)], expon(scale=1.4)),
        ("weibull", [math.log(1.6), math.log(1.4)], weibull_min(1.6, scale=1.4)),
        ("lognormal", [math.log(1.4), math.log(0.7)], lognorm(0.7, scale=1.4)),
        ("gamma", [math.log(2.3), math.log(1.4)], gamma(2.3, scale=1.4)),
        ("inverse_gamma", [math.log(3.1), math.log(1.4)], invgamma(3.1, scale=1.4)),
        ("log_logistic", [math.log(1.6), math.log(1.4)], fisk(1.6, scale=1.4)),
    ],
)
def test_interval_mass_matches_independent_distribution_cdf(family, coordinates, distribution):
    lower, upper = 0.8, 1.3
    result, cdf_lower, cdf_upper = rounded_tte_log_probabilities(
        [math.log(lower)], [math.log(upper)], coordinates, family
    )
    expected_mass = float(distribution.cdf(upper) - distribution.cdf(lower))
    assert result[0] == pytest.approx(math.log(expected_mass), abs=2e-13)
    assert cdf_lower[0] == pytest.approx(math.log(distribution.cdf(lower)), abs=2e-13)
    assert cdf_upper[0] == pytest.approx(math.log(distribution.cdf(upper)), abs=2e-13)

    tail_lower, tail_upper = 5.0, 5.1
    tail_log_mass = _interval(family, coordinates, tail_lower, tail_upper)
    tail_mass = float(distribution.sf(tail_lower) - distribution.sf(tail_upper))
    assert tail_log_mass == pytest.approx(math.log(tail_mass), abs=2e-11)


def test_log_odds_rate_interval_matches_closed_form_and_has_positive_support():
    shape, scale, c = 1.7, 1.4, 0.65
    lower, upper = 0.8, 1.3
    log_mass = _interval(
        "log_odds_rate",
        [math.log(shape), math.log(scale), math.log(c)],
        lower,
        upper,
    )

    def survival(time):
        return (1 + c * (time / scale) ** shape) ** (-1 / c)

    assert log_mass == pytest.approx(math.log(survival(lower) - survival(upper)), abs=2e-13)
    tail_mass = survival(5.0) - survival(5.1)
    assert _interval(
        "log_odds_rate",
        [math.log(shape), math.log(scale), math.log(c)],
        5.0,
        5.1,
    ) == pytest.approx(math.log(tail_mass), abs=2e-11)

    for family, coordinates in (
        ("exponential", [math.log(scale)]),
        ("weibull", [math.log(shape), math.log(scale)]),
        ("lognormal", [math.log(scale), math.log(0.7)]),
        ("gamma", [math.log(shape), math.log(scale)]),
        ("inverse_gamma", [math.log(shape), math.log(scale)]),
        ("log_logistic", [math.log(shape), math.log(scale)]),
        ("log_odds_rate", [math.log(shape), math.log(scale), math.log(c)]),
    ):
        mass, lower_cdf, upper_cdf = rounded_tte_log_probabilities(
            [-np.inf], [math.log(0.5)], coordinates, family
        )
        assert np.isneginf(lower_cdf[0])
        assert np.isfinite(upper_cdf[0]) and upper_cdf[0] <= 0
        assert np.isfinite(mass[0]) and mass[0] == upper_cdf[0]


def test_far_tail_and_close_endpoint_masses_use_log_differences():
    lower, upper = 100.0, 100.0000001
    log_lower, log_upper = math.log(lower), math.log(upper)
    mass = _interval("exponential", [0.0], lower, upper)
    x_lower, x_upper = math.exp(log_lower), math.exp(log_upper)
    expected = -x_lower + math.log(-math.expm1(-(x_upper - x_lower)))
    assert mass == pytest.approx(expected, abs=2e-13)

    # At z=10, ordinary normal CDF values both round to one; upper-tail logs
    # still separate the interval and preserve its finite probability.
    lower_z, upper_z = 10.0, 10.01
    mass, _, _ = rounded_tte_log_probabilities([lower_z], [upper_z], [0.0, 0.0], "lognormal")
    log_sf_lower, log_sf_upper = log_ndtr(-lower_z), log_ndtr(-upper_z)
    expected = log_sf_lower + math.log(-math.expm1(log_sf_upper - log_sf_lower))
    assert np.isfinite(mass[0])
    assert mass[0] == pytest.approx(expected, abs=2e-13)

    symmetric = rounded_tte_log_probabilities(
        [-0.6744897501960817], [0.6744897501960817], [0.0, 0.0], "lognormal"
    )[0][0]
    assert symmetric == pytest.approx(math.log(0.5), abs=2e-14)

    with pytest.raises(ArithmeticError, match="unresolved"):
        rounded_tte_log_probabilities([0.0], [np.nextafter(0.0, 1.0)], [0.0, 0.0], "lognormal")
    # Near z=0, the independent high-precision expansion is
    # Phi(1e-15)-Phi(0) = phi(0)*1e-15 + O(1e-45). The double-precision
    # endpoint log-CDFs do not carry enough relative precision for this mass.
    high_precision_mass = math.exp(-0.5 * math.log(2 * math.pi)) * 1e-15
    assert high_precision_mass > 0
    with pytest.raises(ArithmeticError, match="unresolved"):
        rounded_tte_log_probabilities([0.0], [1e-15], [0.0, 0.0], "lognormal")


def test_distribution_reductions_and_gamma_fallback_work_accounting():
    lower, upper = math.log(0.8), math.log(1.3)
    exponential = rounded_tte_log_probabilities([lower], [upper], [math.log(1.4)], "exponential")[0]
    gamma_exponential = rounded_tte_log_probabilities(
        [lower], [upper], [0.0, math.log(1.4)], "gamma"
    )[0]
    weibull_exponential = rounded_tte_log_probabilities(
        [lower], [upper], [0.0, math.log(1.4)], "weibull"
    )[0]
    np.testing.assert_allclose(gamma_exponential, exponential, rtol=0, atol=2e-14)
    np.testing.assert_allclose(weibull_exponential, exponential, rtol=0, atol=2e-14)

    log_logistic = rounded_tte_log_probabilities(
        [lower], [upper], [math.log(1.6), math.log(1.4)], "log_logistic"
    )[0]
    log_odds_logistic = rounded_tte_log_probabilities(
        [lower],
        [upper],
        [math.log(1.6), math.log(1.4), 0.0],
        "log_odds_rate",
    )[0]
    np.testing.assert_allclose(log_odds_logistic, log_logistic, rtol=0, atol=2e-14)

    log_odds_weibull = rounded_tte_log_probabilities(
        [lower],
        [upper],
        [math.log(1.6), math.log(1.4), math.log(1e-14)],
        "log_odds_rate",
    )[0]
    weibull = rounded_tte_log_probabilities(
        [lower], [upper], [math.log(1.6), math.log(1.4)], "weibull"
    )[0]
    np.testing.assert_allclose(log_odds_weibull, weibull, rtol=0, atol=2e-12)

    counter = [0]
    rounded_tte_log_probabilities(
        [-461.0], [-460.5], [math.log(2.0), 0.0], "gamma", work_counter=counter, work_limit=1000
    )
    assert counter[0] > 0
    with pytest.raises(ArithmeticError, match="work budget"):
        rounded_tte_log_probabilities(
            [-461.0], [-460.5], [math.log(2.0), 0.0], "gamma", work_counter=[0], work_limit=0
        )


def test_rejects_invalid_endpoint_and_parameter_shapes():
    with pytest.raises(ValueError, match="strictly below"):
        rounded_tte_log_probabilities([0.0], [0.0], [0.0], "exponential")
    with pytest.raises(ValueError, match="coordinates"):
        rounded_tte_log_probabilities([0.0], [1.0], [0.0, 0.0], "exponential")
