"""Audit native negative-binomial results within the existing F95 port's domain.

These tests do not claim that the wider legacy API is implemented.
"""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_neg_binomial

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_neg_binomial.json").read_text())
CASES = [(language, c) for language, p in FIXTURE["profiles"].items() for c in p["cases"]]
INTEGER_CASES = [
    (language, c)
    for language, c in CASES
    if c["input"][0] == 1
    and float(c["input"][3]).is_integer()
    and float(c["input"][4]).is_integer()
]


@pytest.mark.parametrize("language,case", CASES)
def test_ordinary_legacy_reference_against_f95_domain(language, case):
    w, p, q, f, s, pr, cpr = case["input"]
    kwargs = dict(f=f, s=s, pr=pr, cpr=cpr)
    if w != 1:
        kwargs.update(cum=p, ccum=q)
        if w == 4:
            kwargs.pop("pr")
            kwargs.pop("cpr")
        else:
            kwargs.pop({2: "f", 3: "s"}[w])
    result = cdf_neg_binomial(w, **kwargs)
    assert case["status"] == 0
    assert FIXTURE["profiles"][language]["adaptations"] == []
    if w == 1:
        np.testing.assert_allclose(
            [result.cum, result.ccum], case["result"][:2], rtol=3e-12, atol=0
        )
    else:
        _, _, nf, ns, nx, ny, _ = case["result"]
        native_forward = cdf_neg_binomial(f=nf, s=ns, pr=nx, cpr=ny)
        np.testing.assert_allclose(
            [native_forward.cum, native_forward.ccum], [p, q], rtol=3e-7, atol=1e-15
        )
        names = ["pr", "cpr"] if w == 4 else [{2: "f", 3: "s"}[w]]
        for name in names:
            truth = dict(f=f, s=s, pr=pr, cpr=cpr)[name]
            np.testing.assert_allclose(getattr(result, name), truth, rtol=3e-7, atol=1e-14)


@pytest.mark.parametrize("language,case", INTEGER_CASES)
def test_native_tails_against_independent_120_digit_pmf_sum(language, case):
    _, _, _, f, s, pr, cpr = case["input"]
    with localcontext() as ctx:
        ctx.prec = 120
        # Normalize from the smaller supplied coordinate, as for the Python pair.
        if pr <= cpr:
            x = Decimal.from_float(pr)
            y = 1 - x
        else:
            y = Decimal.from_float(cpr)
            x = 1 - y
        term = total = x ** int(s)
        for k in range(1, int(f) + 1):
            term *= (Decimal(int(s) + k - 1) / k) * y
            total += term
        expected = [float(total), float(1 - total)]
    np.testing.assert_allclose(case["result"][:2], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_recorded_contract_differences_and_independent_false_successes(language):
    profile = FIXTURE["profiles"][language]
    assert [c["status"] for c in profile["invalid_cases"]] == [-1, -4, -5, -6, -7, -2, -3, 3, 4]
    for case in profile["boundary_cases"]:
        _, _, _, f, s, pr, cpr = case["input"]
        if s == 0 and pr == 0:
            assert case["status"] == 0 and case["result"][:2] == [0, 1]
            # The existing documented F95 Python convention needs no trials
            # to achieve zero successes, even when the success chance is zero.
            assert float(cdf_neg_binomial(f=f, s=s, pr=pr, cpr=cpr).cum) == 1
    wide = profile["wide_cases"]
    timeout = wide[1]
    assert timeout["execution_outcome"] == "timeout"
    assert timeout["status"] is None and timeout["result"] is None
    assert timeout["timeout_seconds"] == 3
    # f=0, s=1 gives P=pr exactly; native root tolerance loses small targets.
    bad = wide[6]
    assert bad["status"] == 0
    target, native_pr = bad["input"][1], bad["result"][4]
    assert native_pr > 1e40 * target
    assert float(cdf_neg_binomial(4, cum=target, f=0, s=1).pr) == pytest.approx(
        target, rel=1e-13, abs=0
    )
    # f=0 gives P=pr**s, hence a positive upper target requires positive s.
    bad = wide[5]
    assert bad["status"] == 0 and bad["result"][3] == 0
    with localcontext() as ctx:
        ctx.prec = 150
        expected_s = (1 - Decimal("1e-100")).ln() / Decimal("0.5").ln()
    assert expected_s > 0
