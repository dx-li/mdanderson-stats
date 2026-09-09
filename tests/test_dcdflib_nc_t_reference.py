"""Legacy noncentral-t evidence, signed reflection and independent identities."""

import json
from math import erfc, exp, sqrt
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import cdf_nc_t

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_nc_t.json").read_text())
CASES = [(lang, c) for lang, d in FIXTURE["profiles"].items() for c in d["cases"]]
DF_TWO = [(lang, c) for lang, c in CASES if c["input"][0] == 1 and c["input"][4] == 2]


def phi(x):
    return erfc(-x / sqrt(2)) / 2


def df_two(t, nc):
    ratio = abs(t) / sqrt(t * t + 2)
    term = ratio * exp(-nc * nc / (t * t + 2))
    delta = term * phi(nc * ratio) if t >= 0 else -term * phi(-nc * ratio)
    return phi(-nc) + delta, phi(nc) - delta


def python_tails(t, df, nc):
    r = cdf_nc_t(t=t if nc >= 0 else -t, df=df, pnonc=abs(nc))
    return (float(r.cum), float(r.ccum)) if nc >= 0 else (float(r.ccum), float(r.cum))


@pytest.mark.parametrize("language,case", CASES)
def test_native_ordinary_calls_against_signed_python_reflection(language, case):
    w, p, q, t, df, nc = case["input"]
    assert not FIXTURE["profiles"][language]["adaptations"]
    assert case["execution_outcome"] == "completed"
    if case["status"] != 0:
        assert w == 3 and case["status"] == 2
        assert case["result"][3] == 1e4 and case["result"][-1] == 1e100
        # Local brackets find valid roots missed by the native global search.
        r = cdf_nc_t(
            3,
            cum=p if nc >= 0 else q,
            t=t if nc >= 0 else -t,
            pnonc=abs(nc),
            df_bracket=(df * 0.95, df * 1.05),
        )
        np.testing.assert_allclose(python_tails(t, float(r.df), nc)[0], p, rtol=3e-8, atol=0)
        return
    if w == 1:
        np.testing.assert_allclose(python_tails(t, df, nc), case["result"][:2], rtol=0, atol=1e-7)
    elif w == 2 and t == 0 and nc != 0:
        # F(0)=Phi(-nc) for every positive df and the CDF is strictly increasing.
        np.testing.assert_allclose(p, phi(-nc), rtol=3e-14, atol=0)
        assert abs(case["result"][2]) > 1e-3
        assert abs(python_tails(*case["result"][2:5])[0] - p) > 1e-5
    else:
        np.testing.assert_allclose(python_tails(*case["result"][2:5])[0], p, rtol=0, atol=1e-7)


@pytest.mark.parametrize("language,case", DF_TWO)
def test_independent_df_two_closed_form(language, case):
    _, _, _, t, _, nc = case["input"]
    expected = df_two(t, nc)
    # Native truncation gives absolute errors around 1.03e-8 here;
    # retain a separate, much tighter independent check of the Python tails.
    np.testing.assert_allclose(case["result"][:2], expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(python_tails(t, 2, nc), expected, rtol=3e-10, atol=0)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_ignored_q_and_misleading_status_bounds(language):
    d = FIXTURE["profiles"][language]
    assert [c["status"] for c in d["invalid_cases"]] == [-1, -1, -2, -2, -5]
    assert d["invalid_cases"][1]["result"][-1] == 5  # Actual maximum mode is four.
    roots = []
    for c in d["ignored_q_cases"]:
        assert c["status"] == 0 and c["result"][1] == c["input"][2]
        roots.append(c["result"][2])
    np.testing.assert_array_equal(roots, np.full(4, roots[0]))
    assert d["boundary_cases"][3]["status"] == 0 and d["boundary_cases"][3]["result"][3] == 5
    with pytest.raises(ValueError):
        cdf_nc_t(3, cum=0.5, t=0, pnonc=0)
    assert d["boundary_cases"][5]["status"] == 0  # Inclusive executable upper p bound.


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_independent_wide_failures_and_signed_tails(language):
    wide = FIXTURE["profiles"][language]["wide_cases"]
    for index in (3, 4):
        c = wide[index]
        _, _, _, t, df, nc = c["input"]
        expected = df_two(t, nc)
        column = 0 if t < 0 else 1
        assert c["status"] == 0
        assert c["result"][column] > expected[column] * 10
        np.testing.assert_allclose(python_tails(t, df, nc), expected, rtol=3e-10, atol=0)
    c = wide[2]
    # For V~chi-square(df), Chebyshev bounds V/df in 1+-epsilon.
    # Conditional normal probabilities then rigorously bound this CDF.
    df = c["input"][4]
    epsilon = 1e-6
    lower = phi(sqrt(1 - epsilon) - 3) - 2 / (df * epsilon**2)
    upper = phi(sqrt(1 + epsilon) - 3) + 2 / (df * epsilon**2)
    assert lower < phi(-2) < upper
    assert c["status"] == 0 and c["result"][0] < lower
    # The small-df limit is Phi(-nc), unlike the large-df limit Phi(t-nc).
    np.testing.assert_allclose(wide[1]["result"][0], phi(-3), rtol=3e-14, atol=0)


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_multiple_df_roots_and_executable_upper_bound(language):
    wide = FIXTURE["profiles"][language]["wide_cases"]
    for c in wide[6:]:
        assert c["status"] == 2
        assert c["result"][3] == 1e4 and c["result"][-1] == 1e100
    r = cdf_nc_t(3, cum=0.03, t=1, pnonc=3, df_bracket=([0.001, 0.2], [0.2, 100]))
    assert float(r.df[0]) < 0.2 < float(r.df[1])
    np.testing.assert_allclose(cdf_nc_t(t=1, df=r.df, pnonc=3).cum, 0.03, rtol=3e-12, atol=0)
    r = cdf_nc_t(3, cum=0.4999995, t=1, pnonc=1, df_bracket=(1e4, 1e8))
    assert 1e4 < float(r.df) < 1e10
    np.testing.assert_allclose(cdf_nc_t(t=1, df=r.df, pnonc=1).cum, 0.4999995, rtol=3e-12, atol=0)
