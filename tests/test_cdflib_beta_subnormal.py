"""Independent beta integrals at subnormal complementary coordinates."""

import numpy as np
import pytest
from test_cdflib_beta_series_reference import independent

from mdanderson_stats import ccum_beta, cdf_beta, cum_beta


@pytest.mark.parametrize("a", [1e-10, 0.01, 0.1, 0.5, 1.0, 1.01, 2.0, 10.0])
@pytest.mark.parametrize("b", [0.25, 0.5, 2.0, 1e10])
@pytest.mark.parametrize(
    "x", [5e-324, 1e-320, 1e-310, np.nextafter(np.finfo(float).tiny, 0), np.finfo(float).tiny]
)
def test_subnormal_coordinate_beta_integrals(a, b, x):
    p, q = independent(2, a, b, float(x)), independent(1, a, b, float(x))
    result = cdf_beta(x=x, a=a, b=b)
    np.testing.assert_allclose(result.cum, p, rtol=5e-13, atol=5e-324)
    np.testing.assert_allclose(result.ccum, q, rtol=5e-13, atol=5e-324)
    if p > 0:
        assert result.cum > 0
    if q > 0:
        assert result.ccum > 0
    reflected = cdf_beta(x=1, cx=x, a=b, b=a)
    np.testing.assert_array_equal([reflected.cum, reflected.ccum], [result.ccum, result.cum])
    assert result.cum + result.ccum == 1


def test_public_tail_helpers_preserve_the_smaller_supplied_coordinate():
    x = np.array([5e-324, 1e-320, 1e-310])
    p = np.array([independent(2, 0.01, 0.5, float(z)) for z in x])
    q = np.array([independent(1, 0.01, 0.5, float(z)) for z in x])
    np.testing.assert_allclose(cum_beta(x, 0.01, 0.5), p, rtol=5e-13, atol=0)
    np.testing.assert_allclose(ccum_beta(x, 0.01, 0.5), q, rtol=5e-13, atol=0)
    np.testing.assert_allclose(cum_beta(None, 0.5, 0.01, cx=x), q, rtol=5e-13, atol=0)
    np.testing.assert_allclose(ccum_beta(None, 0.5, 0.01, cx=x), p, rtol=5e-13, atol=0)


@pytest.mark.parametrize("x", [5e-324, 1e-310, 1e-300, np.nextafter(1e-300, np.inf)])
@pytest.mark.parametrize("a,b", [(1e-10, 0.5), (0.01, 0.25), (0.01, 2.0), (1.01, 0.5), (2.0, 1e10)])
def test_legacy_beta_uses_stable_coordinate_tails(a, b, x):
    from mdanderson_stats import cumbet

    expected = [independent(2, a, b, float(x)), independent(1, a, b, float(x))]
    np.testing.assert_allclose(cumbet(x, a, b), expected, rtol=5e-13, atol=5e-324)
    np.testing.assert_allclose(cumbet(1, b, a, cx=x), expected[::-1], rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("a", [0.01, 0.5, 1.01])
@pytest.mark.parametrize("x", [5e-324, 1e-310])
def test_negative_binomial_beta_identity(a, x):
    from mdanderson_stats import cdf_neg_binomial, cumnbn

    expected = [independent(2, a, 2.0, x), independent(1, a, 2.0, x)]
    result = cdf_neg_binomial(f=1, s=a, pr=x)
    np.testing.assert_allclose([result.cum, result.ccum], expected, rtol=5e-13, atol=5e-324)
    np.testing.assert_allclose(cumnbn(1, a, x), expected, rtol=5e-13, atol=5e-324)


@pytest.mark.parametrize("x", [5e-324, 1e-310])
@pytest.mark.parametrize("which", [3, 4])
def test_shape_inversion_uses_independent_subnormal_target(x, which):
    a, b = 0.01, 0.5
    p, q = independent(2, a, b, x), independent(1, a, b, x)
    inputs = {"a": a} if which == 4 else {"b": b}
    result = cdf_beta(which, cum=p, ccum=q, x=x, **inputs)
    np.testing.assert_allclose([result.a, result.b], [a, b], rtol=5e-11, atol=0)


@pytest.mark.parametrize(
    "a,b,x",
    [(1e-10, 5e-324, 5e-324), (0.01, 1e300, 5e-324), (1.01, 1e290, 1e-310), (10.0, 1e290, 1e-310)],
)
def test_shared_tail_wider_shapes_with_negligible_coordinate_correction(a, b, x):
    from mdanderson_stats.cdflib_beta import _tails

    expected = [independent(2, a, b, x), independent(1, a, b, x)]
    actual = _tails(*map(np.asarray, (x, 1.0, a, b)))
    np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-324)
