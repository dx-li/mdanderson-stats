"""Native binomial contract evidence on the existing F95 port's overlap."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_binomial

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_binomial.json").read_text())
CASES = [(lang, c) for lang, d in FIXTURE["profiles"].items() for c in d["cases"]]
INTEGER = [
    (lang, c)
    for lang, c in CASES
    if c["input"][0] == 1
    and float(c["input"][3]).is_integer()
    and float(c["input"][4]).is_integer()
]


@pytest.mark.parametrize("language,case", CASES)
def test_existing_python_port_against_native_overlap(language, case):
    w, p, q, s, n, pr, cpr = case["input"]
    kw = dict(s=s, n=n, pr=pr, cpr=cpr)
    if w != 1:
        kw.update(cum=p, ccum=q)
        for name in {2: ["s"], 3: ["n"], 4: ["pr", "cpr"]}[w]:
            kw.pop(name)
    r = cdf_binomial(w, **kw)
    assert FIXTURE["profiles"][language]["adaptations"] == []
    if w == 1:
        assert case["status"] == 0
        np.testing.assert_allclose([r.cum, r.ccum], case["result"][:2], rtol=3e-12, atol=0)
    else:
        for name in {2: ["s"], 3: ["n"], 4: ["pr", "cpr"]}[w]:
            truth = dict(s=s, n=n, pr=pr, cpr=cpr)[name]
            np.testing.assert_allclose(getattr(r, name), truth, rtol=3e-11, atol=1e-14)
        if case["execution_outcome"] == "process_error":
            assert language == "c" and w == 2 and n < 5
            assert case["status"] is None and case["result"] is None
            assert case["exit_code"] == 1 and "not monotone in INVR" in case["stderr"]
        else:
            assert case["status"] == 0
            _, _, ns, nn, nx, ny, _ = case["result"]
            check = cdf_binomial(s=ns, n=nn, pr=nx, cpr=ny)
            # Native parameter tolerance does not promise matching relative
            # tail accuracy: ordinary chance inverses show errors near 4e-7.
            np.testing.assert_allclose([check.cum, check.ccum], [p, q], rtol=1e-6, atol=1e-15)


@pytest.mark.parametrize("language,case", INTEGER)
def test_native_tails_against_independent_120_digit_pmf_sum(language, case):
    _, _, _, s, n, pr, cpr = case["input"]
    with localcontext() as ctx:
        ctx.prec = 120
        x = Decimal.from_float(pr) if pr <= cpr else 1 - Decimal.from_float(cpr)
        y = 1 - x
        term = total = y ** int(n)
        for k in range(1, int(s) + 1):
            term *= Decimal(int(n) - k + 1) / k * x / y
            total += term
        expected = [1.0, 0.0] if s == n else [float(total), float(1 - total)]
    np.testing.assert_allclose(case["result"][:2], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_invalid_modes_and_independent_false_success_evidence(language):
    d = FIXTURE["profiles"][language]
    assert [c["status"] for c in d["invalid_cases"]] == [
        -999,
        -999,
        -4,
        -4,
        -5,
        -6,
        -7,
        -2,
        -3,
        3,
        4,
    ]
    for w in (0, 5):
        with pytest.raises(ValueError):
            cdf_binomial(w, cum=0.5, ccum=0.5, s=1, n=2, pr=0.5)
    # At s=0,n=1, Q=pr exactly, including the complementary small target.
    bad = d["wide_cases"][4]
    assert bad["status"] == 0
    assert bad["result"][4] > 1e90 * bad["input"][2]
    r = cdf_binomial(4, ccum=1e-100, s=0, n=1)
    np.testing.assert_allclose(r.pr, 1e-100, rtol=3e-13, atol=0)
    # At s=0, P=(1-pr)**n, so solve the real trial count independently.
    bad = d["wide_cases"][5]
    with localcontext() as ctx:
        ctx.prec = 150
        exact = (1 - Decimal("1e-100")).ln() / Decimal(".5").ln()
        returned = Decimal.from_float(bad["result"][3])
        assert abs(returned / exact - 1) > Decimal(".3")
    assert bad["status"] == 0
    # P=1 at s=n leaves the chance unidentified; Python rejects it.
    assert d["boundary_cases"][7]["status"] == 0
    with pytest.raises(ValueError):
        cdf_binomial(4, cum=1, s=1, n=1)
    if language == "c":
        # The returned huge success count lies many standard deviations above
        # the mean. Chebyshev proves its CDF cannot match the requested 1/2.
        bad = d["wide_cases"][6]
        assert bad["status"] == 0
        with localcontext() as ctx:
            ctx.prec = 150
            n = Decimal.from_float(bad["input"][4])
            s = Decimal.from_float(bad["result"][2])
            gap = s - n / 2
            assert gap > 0
            upper_bound = (n / 4) / (gap * gap)
            assert upper_bound < Decimal("1e-180")
