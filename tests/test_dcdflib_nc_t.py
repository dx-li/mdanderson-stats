"""Legacy signed noncentral t: native evidence and independent probabilities."""

import json
from math import erfc, exp, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdftnc, cumtnc

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_nc_t.json").read_text())
CASES = [(lang, c) for lang, d in FIXTURE["profiles"].items() for c in d["cases"]]


def phi(x):
    return erfc(-x / sqrt(2)) / 2


def df_two(t, nc):
    r = abs(t) / sqrt(t * t + 2)
    term = r * exp(-nc * nc / (t * t + 2))
    delta = term * phi(nc * r) if t >= 0 else -term * phi(-nc * r)
    return phi(-nc) + delta, phi(nc) - delta


@pytest.mark.parametrize("language,case", CASES)
def test_ordinary_native_contracts(language, case):
    w, p, _, t, df, nc = case["input"]
    assert case["execution_outcome"] == "completed"
    kw = dict(t=t, df=df, pnonc=nc)
    if w != 1:
        kw.pop({2: "t", 3: "df", 4: "pnonc"}[w])
        kw["p"] = p
    if w == 3:
        kw["df_bracket"] = (df * 0.95, df * 1.05)
    r = cdftnc(w, **kw)
    if w == 1:
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], atol=1e-7, rtol=0)
    else:
        check = cdftnc(t=r.t, df=r.df, pnonc=r.pnonc)
        np.testing.assert_allclose(check.p, p, rtol=3e-10, atol=0)
        if w == 2 and t == 0:
            assert float(r.t) == 0


@pytest.mark.parametrize("t", [-2, -1, -0.5, 0, 0.5, 1, 2])
@pytest.mark.parametrize("nc", [-10, -3, 0, 3, 10])
def test_independent_df_two_closed_form(t, nc):
    np.testing.assert_allclose(cumtnc(t, 2, nc), df_two(t, nc), rtol=3e-10, atol=0)


@pytest.mark.parametrize("df", [0.5, 2, 10, 1e-100, 1e200])
@pytest.mark.parametrize("nc", [-10, -3, 0, 3, 10])
def test_exact_zero_coordinate_and_normal_mean_inverse(df, nc):
    r = cdftnc(t=0, df=df, pnonc=nc)
    np.testing.assert_allclose([r.p, r.q], [phi(-nc), phi(nc)], rtol=3e-14, atol=0)
    if float(r.p) <= 1 - 1e-16:
        assert float(cdftnc(2, p=r.p, df=df, pnonc=nc).t) == 0
        np.testing.assert_allclose(cdftnc(4, p=r.p, t=0, df=df).pnonc, nc, atol=1e-13)


@pytest.mark.parametrize("df", [1e30, 1e100, 1e200, 1e308])
@pytest.mark.parametrize("t,nc", [(1, 3), (-1, -3), (3, 3), (2, -3)])
def test_large_df_conditional_normal_bounds(df, t, nc):
    epsilon = 1e-6
    candidates = [phi(t * sqrt(1 + e) - nc) for e in [-epsilon, epsilon]]
    allowance = 2 / df / epsilon**2
    value = float(cdftnc(t=t, df=df, pnonc=nc).p)
    assert min(candidates) - allowance <= value <= max(candidates) + allowance


@pytest.mark.parametrize("size", [1e20, 1e100, 1e200, 1e308])
def test_large_signed_numerator_and_df_two_limit(size):
    # (Z+nc)/nc converges to one; for df=2, V/2 is Exp(1).
    p, q = cumtnc(size, 2, size)
    np.testing.assert_allclose([p, q], [exp(-1), -np.expm1(-1)], rtol=3e-13)
    np.testing.assert_allclose(cumtnc(-size, 2, -size), [q, p], rtol=3e-13)
    assert float(cdftnc(t=1, df=2, pnonc=size).p) == 0


@pytest.mark.parametrize("size", [1e100, 1e150])
def test_large_coordinate_df_two_second_moment_tail(size):
    # For df=2, Q(t) ~ E[(Z+nc)_+**2]/t**2.
    nc = 3
    moment = (1 + nc * nc) * phi(nc) + nc * exp(-nc * nc / 2) / sqrt(2 * np.pi)
    expected = moment / size / size
    np.testing.assert_allclose(cdftnc(t=size, df=2, pnonc=nc).q, expected, rtol=3e-12, atol=0)


def test_multiple_df_roots_and_actual_legacy_bound():
    r = cdftnc(3, p=0.03, t=1, pnonc=3, df_bracket=([0.001, 0.2], [0.2, 100]))
    assert float(r.df[0]) < 0.2 < float(r.df[1])
    np.testing.assert_allclose(cdftnc(t=1, df=r.df, pnonc=3).p, 0.03, rtol=3e-12)
    with pytest.raises(ValueError):
        cdftnc(3, p=0.4999995, t=1, pnonc=1)


def test_ignored_q_owned_broadcast_empty():
    for q in [None, "ignored", np.nan, [[1, 2], [3, 4]], -123]:
        r = cdftnc(2, p=0.5, q=q, df=2, pnonc=-3)
        assert r.t.shape == () and float(r.q) == 0.5
    t = np.array([[-2.0], [2.0]])
    r = cdftnc(t=t, df=[2, 10], pnonc=[-3, 3])
    t[:] = 9
    assert r.t[0, 0] == -2
    for name in ["p", "q", "t", "df", "pnonc"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w in range(1, 5):
        kw = dict(t=[], df=[], pnonc=[])
        if w != 1:
            kw.pop({2: "t", 3: "df", 4: "pnonc"}[w])
            kw["p"] = []
        assert cdftnc(w, **kw).p.shape == (0,)


@pytest.mark.parametrize(
    "kw",
    [
        dict(which=True),
        dict(which=0),
        dict(which=5),
        dict(which=2.0),
        dict(t=None, df=2, pnonc=3),
        dict(t=1, df=0, pnonc=3),
        dict(t=1, df=2, pnonc=None),
        dict(t=np.inf, df=2, pnonc=3),
        dict(t=1, df=2, pnonc=np.nan),
        dict(t=1, df=2, pnonc=3, p=0.5),
        dict(which=2, p=1, df=2, pnonc=3),
        dict(which=2, p=0, df=2, pnonc=3),
        dict(which=2, q=0.5, df=2, pnonc=3),
        dict(which=2, p=-0.1, df=2, pnonc=3),
        dict(which=3, p=0.5, t=0, pnonc=0),
        dict(which=3, p=0.5, t=1, pnonc=0, df_bracket=(1, 1e5)),
        dict(which=3, p=0.5, t=1, pnonc=0, df_bracket=(2, 1)),
        dict(which=3, p=0.5, t=1, pnonc=0, df_bracket=(1,)),
        dict(which=2, p=0.5, df=2, pnonc=3, df_bracket=(1, 2)),
    ],
)
def test_invalid_inputs(kw):
    with pytest.raises(ValueError):
        cdftnc(**kw)


@pytest.mark.parametrize("df", [1e-100, 1e-308, np.nextafter(0.0, 1.0)])
@pytest.mark.parametrize("nc", [40.0, 1e20])
@pytest.mark.parametrize("t", [1.0, 1e100])
def test_tiny_df_positive_tail_retains_gamma_contribution(df, nc, t):
    from decimal import Decimal, localcontext

    # With tiny df, E1(df*z*z/(2*t*t)) = -Euler-log(df/2)-2*log(z/t).
    # Expand E[log(nc+Z)] in even normal moments. At nc>=40, truncation
    # after 20 terms is negligible here; omitted negative-normal mass is
    # below 4e-350. Decimal preserves the minimum-subnormal df/2 product.
    with localcontext() as ctx:
        ctx.prec = 800
        d, n, x = map(Decimal.from_float, (df, nc, t))
        mean_log, moment = n.ln(), Decimal(1)
        for k in range(1, 21):
            moment *= 2 * k - 1
            mean_log -= moment / (2 * k * n ** (2 * k))
        euler = Decimal("0.577215664901532860606512090082402431042159335939923598805767")
        expected = float(d * (-d.ln() + Decimal(2).ln() + 2 * x.ln() - euler - 2 * mean_log) / 2)
    r = cdftnc(t=t, df=df, pnonc=nc)
    np.testing.assert_allclose(r.p, expected, rtol=3e-12, atol=np.nextafter(0.0, 1.0))
    assert float(r.p) > 0
    np.testing.assert_allclose(cumtnc(-t, df, -nc), [r.q, r.p], rtol=3e-12, atol=0)


@pytest.mark.parametrize("df", [1e30, 1e100, 1e200, 1e308])
def test_huge_df_and_numerator_must_retain_denominator_fluctuation(df):
    nc = 1e150
    # Choose a coordinate separated on the denominator-fluctuation scale.
    t = nc * (1 + sqrt(2 / df))
    r = cdftnc(t=t, df=df, pnonc=nc)
    if t == nc:
        np.testing.assert_allclose(r.p, 0.5, rtol=0, atol=2e-15)
    else:
        delta = ((t - nc) / t) * sqrt(2 * df)
        np.testing.assert_allclose(r.p, phi(delta), rtol=3e-13)


def test_search_bounds_and_signed_noncentrality_endpoints():
    for nc in [-1e4, 1e4]:
        r = cdftnc(t=nc, df=2, pnonc=nc)
        np.testing.assert_allclose(cdftnc(4, p=r.p, t=nc, df=2).pnonc, nc, rtol=3e-12)
    r = cdftnc(2, p=1 - 1e-16, df=2, pnonc=0)
    assert 0 < float(r.t) < 1e100
    with pytest.raises(ValueError):
        cdftnc(2, p=1e-200, df=0.5, pnonc=0)


@pytest.mark.parametrize("p", [1e-30, 1e-100, 1e-190])
@pytest.mark.parametrize("nc", [-3, 3])
def test_extreme_quantiles_against_df_two_truncated_normal_moment(p, nc):
    moment = (1 + nc * nc) * phi(-nc) - nc * exp(-nc * nc / 2) / sqrt(2 * np.pi)
    expected = -sqrt(moment / p)
    r = cdftnc(2, p=p, df=2, pnonc=nc)
    np.testing.assert_allclose(r.t, expected, rtol=3e-11)
    np.testing.assert_allclose(cdftnc(t=r.t, df=2, pnonc=nc).p, p, rtol=3e-11, atol=0)


@pytest.mark.parametrize("df", [0.5, 1, 4, 10])
@pytest.mark.parametrize("t,nc", [(-2, 3), (0.5, 4), (2, -3)])
def test_independent_chi_square_conditioning(df, t, nc):
    from math import gamma

    def integrate(n):
        r = np.linspace(0, 8, n + 1)
        a = df / 2
        if df < 1:
            # Transform the gamma coordinate by its shape to remove the
            # integrable singularity at zero for fractional df.
            density = np.exp(-(r ** (1 / a))) / gamma(a + 1)
            coordinate = r ** (1 / (2 * a)) / sqrt(a)
        else:
            density = 2 * a**a / gamma(a) * r ** (df - 1) * np.exp(-a * r * r)
            coordinate = r
        probabilities = [
            np.array([phi(t * v - nc) for v in coordinate]),
            np.array([phi(nc - t * v) for v in coordinate]),
        ]
        return np.array(
            [
                (v[0] + v[-1] + 4 * v[1:-1:2].sum() + 2 * v[2:-1:2].sum()) * (8 / n) / 3
                for v in [probability * density for probability in probabilities]
            ]
        )

    first, second = integrate(8192), integrate(16384)
    np.testing.assert_allclose(first, second, rtol=3e-10, atol=0)
    np.testing.assert_allclose(cumtnc(t, df, nc), second, rtol=3e-10, atol=0)


@pytest.mark.parametrize("df", [1e10, 1e14, 1e18, 1e20, 1e24, 1e28, 1e30])
def test_large_numerator_gamma_mean_probability_and_skewness(df):
    # When nc=t is enormous, numerator noise vanishes and P=Pr(V>=df).
    # Gamma's mean-tail expansion is 1/2 - 1/(3*sqrt(pi*df)) + O(df**(-3/2)).
    expected = 0.5 - 1 / (3 * sqrt(np.pi * df))
    np.testing.assert_allclose(cdftnc(t=1e200, df=df, pnonc=1e200).p, expected, atol=2e-15, rtol=0)


def test_large_df_coordinate_rounding_regression():
    # Both normal numerator and denominator contribute here. Symmetry holds
    # up to O(1/sqrt(df)); forming exp(log(df/2)) previously gave about .529.
    np.testing.assert_allclose(cdftnc(t=1e14, df=1e28, pnonc=1e14).p, 0.5, atol=1e-13, rtol=0)
