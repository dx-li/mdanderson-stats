"""Legacy binomial implementation against native and independent references."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdfbin

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
        kw.update(p=p, q=q)
        for name in {2: ["s"], 3: ["n"], 4: ["pr", "cpr"]}[w]:
            kw.pop(name)
    r = cdfbin(w, **kw)
    assert FIXTURE["profiles"][language]["adaptations"] == []
    if w == 1:
        assert case["status"] == 0
        np.testing.assert_allclose([r.p, r.q], case["result"][:2], rtol=3e-12, atol=0)
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
            check = cdfbin(s=ns, n=nn, pr=nx, cpr=ny)
            # Native parameter tolerance does not promise matching relative
            # tail accuracy: ordinary chance inverses show errors near 4e-7.
            np.testing.assert_allclose([check.p, check.q], [p, q], rtol=1e-6, atol=1e-15)


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
    r = cdfbin(s=s, n=n, pr=pr, cpr=cpr)
    np.testing.assert_allclose([r.p, r.q], expected, rtol=3e-12, atol=0)


@pytest.mark.parametrize(
    "n,pr,cpr",
    [
        (1e-308, 0.5, 0.5),
        (1e-100, 0.5, 0.5),
        (1e100, 1e-100, 1.0),
        (1e200, 1e-200, 1.0),
        (1, 1e-300, 1.0),
        (1e-320, 0.5, 0.5),
    ],
)
def test_independent_800_digit_zero_success_power(n, pr, cpr):
    with localcontext() as ctx:
        ctx.prec = 800
        y = Decimal.from_float(cpr) if cpr <= pr else 1 - Decimal.from_float(pr)
        p = y ** Decimal.from_float(n)
        pp, qq = float(p), float(1 - p)
    tiny = np.nextafter(0.0, 1.0)
    r = cdfbin(s=0, n=n, pr=pr, cpr=cpr)
    np.testing.assert_allclose([r.p, r.q], [pp, qq], rtol=3e-13, atol=tiny)
    if pp > 0 and qq > 0:
        chance = cdfbin(4, p=pp, q=qq, s=0, n=n)
        np.testing.assert_allclose(
            cdfbin(s=0, n=n, pr=chance.pr, cpr=chance.cpr).q, qq, rtol=3e-12, atol=tiny
        )
        assert float(cdfbin(2, p=pp, q=qq, n=n, pr=pr, cpr=cpr).s) == 0
        if 1e-100 <= n <= 1e100:
            np.testing.assert_allclose(
                cdfbin(3, p=pp, q=qq, s=0, pr=pr, cpr=cpr).n, n, rtol=3e-12, atol=0
            )
        else:
            with pytest.raises(ValueError):
                cdfbin(3, p=pp, q=qq, s=0, pr=pr, cpr=cpr)


@pytest.mark.parametrize("n", [1e90, 1e100, 1e200, 1e308])
def test_symmetric_large_success_bound_is_not_capped_at_1e100(n):
    assert float(cdfbin(s=n / 2, n=n, pr=0.5).p) == 0.5
    assert float(cdfbin(2, p=0.5, n=n, pr=0.5).s) == n / 2
    if n <= 1e100:
        np.testing.assert_allclose(cdfbin(3, p=0.5, s=n / 2, pr=0.5).n, n, rtol=3e-14)
    else:
        with pytest.raises(ValueError):
            cdfbin(3, p=0.5, s=n / 2, pr=0.5)


def test_native_repairs_endpoint_semantics_and_mixed_rows():
    np.testing.assert_allclose(cdfbin(4, q=1e-100, s=0, n=1).pr, 1e-100, rtol=3e-13, atol=0)
    np.testing.assert_allclose(
        cdfbin(3, q=1e-100, s=0, pr=0.5).n, 1e-100 / np.log(2), rtol=3e-13, atol=0
    )
    assert float(cdfbin(4, p=0, s=0, n=1).pr) == 1
    assert float(cdfbin(4, q=0, s=0, n=1).pr) == 0
    assert float(cdfbin(2, q=0, n=5, pr=1).s) == 5
    assert float(cdfbin(3, q=0, s=5, pr=1).n) == 5
    r = cdfbin(s=[0, 0.125, 1, 1e90], n=[1, 0.5, 5, 2e90], pr=0.5)
    np.testing.assert_allclose(
        cdfbin(2, p=r.p, q=r.q, n=r.n, pr=0.5).s, r.s, rtol=3e-12, atol=1e-14
    )
    np.testing.assert_allclose(cdfbin(3, p=r.p, q=r.q, s=r.s, pr=0.5).n, r.n, rtol=3e-12)
    np.testing.assert_array_equal(cdfbin(s=[0, 1], n=1, pr=[0, 1]).p, 1)


@pytest.mark.parametrize("n", [1e-100, 1e-308])
def test_tiny_trials_nonzero_fraction(n):
    # At s=n/2 the first beta shape is n/2 and the second differs
    # from one by n/2. The latter correction is negligible at these scales.
    with localcontext() as ctx:
        ctx.prec = 400
        gap = Decimal.from_float(n) - Decimal.from_float(n / 2)
        q = float(-gap * Decimal(".5").ln())
    r = cdfbin(s=n / 2, n=n, pr=0.5)
    np.testing.assert_allclose(r.q, q, rtol=3e-13, atol=np.nextafter(0.0, 1.0))
    inv = cdfbin(2, q=q, n=n, pr=0.5)
    np.testing.assert_allclose(inv.s, n / 2, rtol=3e-10, atol=0)
    chance = cdfbin(4, q=q, s=n / 2, n=n)
    np.testing.assert_allclose(chance.pr, 0.5, rtol=3e-10)


def test_owned_broadcast_empty_and_pair_helper():
    from mdanderson_stats import cumbin

    s = np.array([[0.0], [1.0]])
    n = np.array([2.0, 5.0])
    r = cdfbin(s=s, n=n, pr=0.5)
    s[:] = 8
    n[:] = 8
    assert r.s[0, 0] == 0 and r.n[0, 0] == 2
    for name in ["p", "q", "s", "n", "pr", "cpr"]:
        with pytest.raises(ValueError):
            getattr(r, name).setflags(write=True)
    for w, name in [(1, "p"), (2, "s"), (3, "n"), (4, "pr")]:
        kw = dict(s=[], n=[], pr=[])
        if w != 1:
            kw.pop(name)
            kw["p"] = []
        assert getattr(cdfbin(w, **kw), name).size == 0
    np.testing.assert_allclose(cumbin(0, 1, 0.2), [0.8, 0.2], rtol=3e-14)


@pytest.mark.parametrize(
    "w,kw",
    [
        (True, {}),
        (0, {}),
        (5, {}),
        (1.0, {}),
        (1, dict(s=0)),
        (1, dict(s=0, n=0, pr=0.5)),
        (1, dict(s=-1, n=1, pr=0.5)),
        (1, dict(s=2, n=1, pr=0.5)),
        (1, dict(s=0, n=np.inf, pr=0.5)),
        (1, dict(s=0, n=1, pr=0.4, cpr=0.4)),
        (1, dict(s=0, n=1, pr=-1)),
        (2, dict(p=0, n=1, pr=1)),
        (2, dict(q=0, n=1, pr=0)),
        (2, dict(p=0.5, n=1, pr=0)),
        (2, dict(p=0.01, n=1, pr=0.5)),
        (3, dict(q=0, s=0, pr=0.5)),
        (3, dict(p=0, s=0, pr=0.5)),
        (3, dict(p=0.5, s=1e101, pr=0.5)),
        (4, dict(p=1, s=1, n=1)),
        (4, dict(p=0.4, q=0.4, s=0, n=1)),
    ],
)
def test_invalid_unidentified_and_outside_bounds(w, kw):
    with pytest.raises(ValueError):
        cdfbin(w, **kw)
