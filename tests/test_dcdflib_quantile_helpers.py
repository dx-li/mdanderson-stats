"""Native formulas and independent normal-tail/Decimal series verification."""

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest
from test_cdflib_error_exponential_reference import independent_error

from mdanderson_stats import dcdflib_support as legacy

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_quantile_helpers.json").read_text())
CASES = [(lang, c) for lang, row in FIXTURE["profiles"].items() for c in row["cases"]]


def starting_normal(p):
    with localcontext() as ctx:
        ctx.prec = 100
        prob = Decimal.from_float(float(p))
        small = min(prob, 1 - prob)
        y = (-2 * small.ln()).sqrt()
        numerator = sum(
            Decimal(c) * y**i
            for i, c in enumerate(
                ["-.322232431088", "-1", "-.342242088547", "-.0204231210245", "-.0000453642210148"]
            )
        )
        denominator = sum(
            Decimal(c) * y**i
            for i, c in enumerate(
                [
                    ".0993484626060",
                    ".588581570495",
                    ".531103462366",
                    ".103537752850",
                    ".0038560700634",
                ]
            )
        )
        value = y + numerator / denominator
        return float(-value if prob <= Decimal(".5") else value)


def t_expansion(z, df):
    # The normal inverse is independently checked below. Evaluate the specified
    # t expansion using high-precision powers, not the production summation.
    with localcontext() as ctx:
        ctx.prec = 800
        z, d = Decimal.from_float(float(z)), Decimal.from_float(float(df))
        w = z * z
        terms = [
            (1 + w) / (4 * d),
            (3 + 16 * w + 5 * w * w) / (96 * d * d),
            (-15 + 17 * w + 19 * w * w + 3 * w**3) / (384 * d**3),
            (-945 - 1920 * w + 1482 * w * w + 776 * w**3 + 79 * w**4) / (92160 * d**4),
        ]
        return float(z * (1 + sum(terms)))


def verify_inverse(p, q, z):
    if p == q:
        assert z == 0
        return
    # Independent Decimal erfcx series plus log-domain Gaussian scaling avoids
    # rounding the true probability to zero or to a single subnormal bin.
    x = abs(z) / math.sqrt(2)
    log_tail = math.log(independent_error(x, scaled=True)) - z * z / 2 - math.log(2)
    assert abs(log_tail - math.log(min(p, q))) < 3e-12
    assert (z < 0) == (p < q)


@pytest.mark.parametrize("language,case", CASES)
def test_native_contracts_and_documented_numerical_repairs(language, case):
    op, p, q, df = case["op"], case["p"], case["q"], case["df"]
    fn = legacy.stvaln if op == 1 else legacy.dinvnr if op == 2 else legacy.dt1
    args = (p,) if op == 1 else (p, q) if op == 2 else (p, q, df)
    if (op == 1 and not 0 < p < 1) or (op != 1 and min(p, q) <= 0):
        with pytest.raises(ValueError):
            fn(*args)
        return
    native = float(case["result"])
    if op == 1:
        actual = float(fn(*args))
        np.testing.assert_allclose(actual, native, rtol=3e-14, atol=3e-15)
        np.testing.assert_allclose(actual, starting_normal(p), rtol=3e-14, atol=3e-15)
    elif op == 2:
        actual = float(fn(*args))
        verify_inverse(p, q, actual)
        if min(p, q) >= 1e-300:
            np.testing.assert_allclose(actual, native, rtol=3e-12, atol=2e-13)
        else:
            assert (
                abs(actual - native) > 1e-8
            )  # Native Newton falls back to its initial approximation.
    else:
        z = float(legacy.dinvnr(p, q))
        target = t_expansion(z, df)
        if not math.isfinite(target):
            with pytest.raises(ArithmeticError):
                fn(*args)
            return
        actual = float(fn(*args))
        np.testing.assert_allclose(actual, target, rtol=3e-13, atol=5e-324)
        if df >= 0.1 and abs(p - 0.5) > 1e-14:
            np.testing.assert_allclose(actual, native, rtol=3e-12, atol=2e-13)
        if p == 0.5:
            assert actual == 0


def test_initial_approximations_remain_distinct_from_refined_quantiles():
    assert legacy.stvaln(0.5) != 0 and legacy.dinvnr(0.5) == 0
    # The prescribed low-df expansion can have the opposite sign to the exact quantile.
    assert legacy.dt1(0.49, 0.51, 0.1) > 0
    # Cauchy quantile at p=.25 is -1; DT1 is an approximation rather than that inverse.
    assert abs(float(legacy.dt1(0.25, 0.75, 1)) + 1) > 0.01


def test_small_df_does_not_require_representable_fourth_power():
    p = np.nextafter(0.5, 1)
    with np.errstate(all="raise"):
        result = legacy.dt1(p, 1 - p, 1e-80)
        assert np.isfinite(result) and result < 0
        assert legacy.dt1(0.5, 0.5, 5e-324) == 0
    np.testing.assert_allclose(result, t_expansion(legacy.dinvnr(p, 1 - p), 1e-80), rtol=3e-13)
    with pytest.raises(ArithmeticError):
        legacy.dt1(0.99, 0.01, 1e-100)


def test_broadcasting_probability_completion_and_owned_results():
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    q = 1 - p
    degrees = np.array([[1.0], [10.0]])
    result = legacy.dt1(p, q, degrees)
    expected = np.array(
        [
            [t_expansion(legacy.dinvnr(pp, qq), df) for pp, qq in zip(p, q, strict=True)]
            for df in [1.0, 10.0]
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=3e-13, atol=0)
    np.testing.assert_allclose(legacy.dinvnr(None, q), legacy.dinvnr(p), rtol=3e-15, atol=0)
    for out in [result, legacy.dinvnr(p, q), legacy.stvaln(p)]:
        assert not out.flags.writeable and not np.shares_memory(out, p)
        with pytest.raises(ValueError):
            out.setflags(write=True)
    assert legacy.dt1(np.empty((0, 1)), None, np.ones((1, 3))).shape == (0, 3)
    assert legacy.stvaln([]).shape == (0,)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0, -1, 1])
def test_invalid_start_probability(bad):
    with pytest.raises(ValueError):
        legacy.stvaln(bad)


@pytest.mark.parametrize(
    "p,q", [(None, None), (0.1, 0.1), (0, 1), (1, 0), (np.nan, 0.5), (0.5, np.inf), (-1, 2)]
)
def test_invalid_refined_probabilities(p, q):
    with pytest.raises(ValueError):
        legacy.dinvnr(p, q)
    with pytest.raises(ValueError):
        legacy.dt1(p, q, 1)


@pytest.mark.parametrize("df", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_degrees_of_freedom(df):
    with pytest.raises(ValueError):
        legacy.dt1(0.5, 0.5, df)
