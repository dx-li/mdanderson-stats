"""Guide examples, analytic identities and independent integration coordinates."""

from fractions import Fraction
from math import comb

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from mdanderson_stats import ParameterDistribution as Distribution
from mdanderson_stats import inequality_probability


def test_guide_normal_and_shifted_beta_polynomial():
    normal = inequality_probability(Distribution("normal", 0, 1), Distribution("normal", 1, 1))
    assert_allclose(
        [normal.x_greater, normal.shifted_y_greater], [0.2397500610934767, 0.7602499389065233]
    )
    # Integrate 2772*x^5*(1-x)^5 times [10*(x-h)^3-15*(x-h)^4+6*(x-h)^5].
    h = Fraction(1, 10)
    exact = Fraction(0)
    for j in range(6):
        for power, coefficient in [(3, 10), (4, -15), (5, 6)]:
            for k in range(power + 1):
                exponent = 6 + j + k
                exact += (
                    2772
                    * (-1) ** j
                    * comb(5, j)
                    * coefficient
                    * comb(power, k)
                    * (-h) ** (power - k)
                    * (1 - h**exponent)
                    / exponent
                )
    result = inequality_probability(
        Distribution("beta", 6, 6), Distribution("beta", 3, 3), delta=0.1
    )
    assert_allclose(result.x_greater, float(exact), atol=1e-9, rtol=0)
    assert_allclose(result.x_greater, 0.342248, atol=5e-7, rtol=0)
    assert abs(result.x_greater + result.shifted_y_greater - 1) <= 2e-9
    assert result.absolute_error <= 1e-9


@pytest.mark.parametrize("family", ["gamma", "weibull"])
def test_shifted_exponentials_and_scale_invariance(family):
    for scale in [1e-100, 1, 1e100]:
        for shift in [-1.0, 0.0, 1.0]:
            result = inequality_probability(
                Distribution(family, 1, 2 * scale),
                Distribution(family, 1, 3 * scale),
                delta=shift * scale,
            )
            expected = 2 / 5 * np.exp(-shift / 2) if shift >= 0 else 1 - 3 / 5 * np.exp(shift / 3)
            assert_allclose(result.x_greater, expected, rtol=0, atol=1e-9)
            assert_allclose(result.shifted_y_greater, 1 - expected, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "family,a,b,c,d,shift",
    [
        ("gamma", 2.5, 3, 0.7, 2, 0.4),
        ("inverse_gamma", 3, 2, 4, 3, 0.2),
        ("lognormal", 0.4, 0.7, -0.2, 1.2, 0.3),
        ("weibull", 0.7, 2, 3, 4, 0.3),
        ("weibull", 0.7, 2, 3, 4, 0),
    ],
)
def test_against_log_density_integral(family, a, b, c, d, shift):
    x, y = Distribution(family, a, b), Distribution(family, c, d)
    dx, dy = x._distribution(), y._distribution()
    lower, upper = np.log(dx.ppf([1e-13, 1 - 1e-13]))
    expected, error = quad(
        lambda t: np.exp(dx.logpdf(np.exp(t)) + t) * dy.cdf(np.exp(t) - shift),
        lower,
        upper,
        epsabs=1e-11,
        epsrel=1e-11,
        limit=300,
    )
    assert error < 1e-10
    result = inequality_probability(x, y, delta=shift)
    reverse = inequality_probability(y, x, delta=-shift)
    assert_allclose(result.x_greater, expected, atol=1e-9, rtol=0)
    assert_allclose(reverse.x_greater, result.shifted_y_greater, atol=1e-12, rtol=0)


def test_analytic_routes_extreme_tails_and_support():
    # Equal-shape gamma ratio has Beta(2,2) CDF 3*r^2-2*r^3.
    result = inequality_probability(Distribution("gamma", 2, 3), Distribution("gamma", 2, 2))
    assert_allclose(result.x_greater, 3 * 0.6**2 - 2 * 0.6**3, rtol=1e-14)
    inverse = inequality_probability(
        Distribution("inverse_gamma", 2, 1 / 3), Distribution("inverse_gamma", 2, 1 / 2)
    )
    assert_allclose(inverse.x_greater, result.shifted_y_greater, rtol=1e-14)
    rare = inequality_probability(Distribution("normal", 0, 1), Distribution("normal", 12, 1))
    assert 0 < rare.x_greater < 1e-16
    logs = inequality_probability(
        Distribution("lognormal", 1000, 1), Distribution("lognormal", 1012, 1)
    )
    assert logs.x_greater == rare.x_greater  # No exponentiation of log-means.
    extreme = inequality_probability(
        Distribution("gamma", 0.001, 1e-300), Distribution("gamma", 0.001, 1e300)
    )
    assert_allclose(extreme.x_greater + extreme.shifted_y_greater, 1, atol=1e-14)
    for delta in [-2, -1, 1, 2]:
        beta = inequality_probability(
            Distribution("beta", 0.2, 0.3), Distribution("beta", 5, 4), delta=delta
        )
        assert beta.x_greater == float(delta < 0)
    with pytest.raises(ValueError, match="same distribution"):
        inequality_probability(Distribution("gamma", 1, 2), Distribution("weibull", 1, 2))
