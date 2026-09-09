"""Unchanged legacy beta references, overlap and independent failure evidence."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_beta

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_beta.json").read_text())
CASES = [(lang, c) for lang, d in FIXTURE["profiles"].items() for c in d["cases"]]
INTEGER = [
    (lang, c)
    for lang, c in CASES
    if c["input"][0] == 1
    and float(c["input"][5]).is_integer()
    and float(c["input"][6]).is_integer()
]


@pytest.mark.parametrize("language,case", CASES)
def test_native_overlap_against_existing_f95_port(language, case):
    w, p, q, x, cx, a, b = case["input"]
    kw = dict(x=x, cx=cx, a=a, b=b)
    if w != 1:
        kw.update(cum=p, ccum=q)
        for name in {2: ["x", "cx"], 3: ["a"], 4: ["b"]}[w]:
            kw.pop(name)
    r = cdf_beta(w, **kw)
    assert FIXTURE["profiles"][language]["adaptations"] == []
    assert case["execution_outcome"] == "completed" and case["status"] == 0
    if w == 1:
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=3e-12, atol=0)
    else:
        for name in {2: ["x", "cx"], 3: ["a"], 4: ["b"]}[w]:
            truth = dict(x=x, cx=cx, a=a, b=b)[name]
            np.testing.assert_allclose(getattr(r, name), truth, rtol=3e-11, atol=0)
        _, _, nx, ny, na, nb, _ = case["result"]
        check = cdf_beta(x=nx, cx=ny, a=na, b=nb)
        np.testing.assert_allclose([check.cum, check.ccum], [p, q], rtol=1e-6, atol=0)


@pytest.mark.parametrize("language,case", INTEGER)
def test_integer_shapes_against_120_digit_binomial_sum(language, case):
    _, _, _, x, cx, a, b = case["input"]
    with localcontext() as ctx:
        ctx.prec = 120
        xx = Decimal.from_float(x) if x <= cx else 1 - Decimal.from_float(cx)
        yy = 1 - xx
        n = int(a + b - 1)
        term = yy**n
        lower = Decimal(0)
        upper = term
        for k in range(1, n + 1):
            term *= Decimal(n - k + 1) / k * xx / yy
            if k >= a:
                lower += term
            else:
                upper += term
        expected = [float(lower), float(upper)]
    np.testing.assert_allclose(case["result"][:2], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_invalid_endpoints_and_independent_false_successes(language):
    d = FIXTURE["profiles"][language]
    assert [c["status"] for c in d["invalid_cases"]] == [-1, -1, -2, -3, -4, -5, -6, -7, 3, 4]
    # Uniform beta has quantile x=p and complementary quantile cx=q exactly.
    for index, coordinate in [(6, "x"), (7, "cx")]:
        bad = d["wide_cases"][index]
        assert bad["status"] == 0
        value = bad["result"][2 if coordinate == "x" else 3]
        assert value > 1e40 * 1e-100
        r = cdf_beta(2, cum=bad["input"][1], ccum=bad["input"][2], a=1, b=1)
        np.testing.assert_allclose(getattr(r, coordinate), 1e-100, rtol=3e-13, atol=0)
    for index, coordinate in [(2, "x"), (3, "cx")]:
        bad = d["boundary_cases"][index]
        assert bad["status"] == 0
        assert bad["result"][2 if coordinate == "x" else 3] > 0
        r = cdf_beta(2, cum=bad["input"][1], ccum=bad["input"][2], a=1, b=1)
        assert float(getattr(r, coordinate)) == 0
    # Power-law inverse near the native lower shape bound.
    for index, column in [(8, 4), (9, 5)]:
        bad = d["wide_cases"][index]
        assert bad["status"] == 0
        with localcontext() as ctx:
            ctx.prec = 150
            true_shape = (1 - Decimal.from_float(1e-100)).ln() / Decimal(".5").ln()
            assert abs(Decimal.from_float(bad["result"][column]) / true_shape - 1) > Decimal(".3")
    for index, w in [(4, 3), (5, 4), (6, 3), (7, 4)]:
        bad = d["boundary_cases"][index]
        assert bad["status"] == 0
        kw = dict(cum=bad["input"][1], ccum=bad["input"][2], x=bad["input"][3])
        kw["b" if w == 3 else "a"] = 1
        with pytest.raises(ValueError):
            cdf_beta(w, **kw)
    # Beta mean/variance and Chebyshev bound refute the huge shape result.
    bad = d["wide_cases"][10]
    assert bad["status"] == 0
    with localcontext() as ctx:
        ctx.prec = 150
        a, b = map(Decimal.from_float, bad["result"][4:6])
        mean = a / (a + b)
        variance = a * b / ((a + b) ** 2 * (a + b + 1))
        gap = mean - Decimal(".5")
        assert gap > 0
        assert variance / gap**2 < Decimal("1e-70")


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_wide_native_successes_against_power_and_symmetry_identities(language):
    cases = FIXTURE["profiles"][language]["wide_cases"]
    assert cases[0]["execution_outcome"] == "timeout"
    assert cases[0]["status"] is None and cases[0]["result"] is None
    assert cases[0]["timeout_seconds"] == 3
    assert cases[1]["status"] == 0
    np.testing.assert_array_equal(cases[1]["result"][:2], [0.5, 0.5])
    for index in (2, 3, 5):
        c = cases[index]
        _, _, _, x, cx, a, b = c["input"]
        with localcontext() as ctx:
            ctx.prec = 800
            xx = Decimal.from_float(x) if x <= cx else 1 - Decimal.from_float(cx)
            if b == 1:
                p = xx ** Decimal.from_float(a)
                expected = [float(p), float(1 - p)]
            else:
                q = (1 - xx) ** Decimal.from_float(b)
                expected = [float(1 - q), float(q)]
        assert c["status"] == 0
        np.testing.assert_allclose(
            c["result"][:2], expected, rtol=3e-13, atol=np.nextafter(0.0, 1.0)
        )
    # As a,b tend to zero at fixed ratio and interior x, P -> b/(a+b).
    # Here the first-order interior correction is O(1e-308), below one ulp.
    assert cases[4]["status"] == 0
    np.testing.assert_allclose(cases[4]["result"][:2], [2 / 3, 1 / 3], rtol=3e-15, atol=0)
