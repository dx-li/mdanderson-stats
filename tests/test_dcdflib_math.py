"""Legacy mathematical contracts, with independent checks for native failures."""

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from test_cdflib_beta_factors_reference import independent as beta_factor
from test_cdflib_beta_remaining_reference import expected as beta_increment
from test_cdflib_beta_support_reference import independent as beta_log
from test_cdflib_bratio import tails
from test_cdflib_error_exponential_reference import independent_error
from test_cdflib_gamma_factor import expected_factor
from test_cdflib_gamma_support_reference import independent as gamma_value

from mdanderson_stats import dcdflib_support as legacy
from mdanderson_stats import exparg as f95_exparg

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_math.json").read_text())


@pytest.mark.parametrize("language", ["c", "fortran"])
def test_legacy_exparg_retains_its_conservative_machine_margin(language):
    cases = [c for c in FIXTURE["profiles"][language]["cases"] if c["op"] == "exparg"]
    for case in cases:
        expected = float(case["values"][0])
        assert float(legacy.exparg(case["k"])) == expected
        assert float(legacy.exparg(case["k"])) != float(f95_exparg(case["k"]))


SINGLE = "alnrel rexp rlog rlog1 alngam gamln gamln1 gam1 Xgamm psi erf1".split()
TWO = "algdiv bcorr betaln gsumln".split()
ARGUMENTS = {
    "rcomp": "a x",
    "erfc1": "k x",
    "esum": "k x",
    "exparg": "k",
    "apser": "a b x eps",
    "fpser": "a b x eps",
    "bpser": "a b x eps",
    "basym": "a b lam eps",
    "bfrac": "a b x y eps",
    "brcomp": "a b x y",
    "brcmp1": "k a b x y",
    "bup": "a b x y k eps",
    "bgrat": "a b x y w eps",
    "bratio": "a b x y",
    "grat1": "a x r eps",
    "gratio": "a x k",
}
CASES = [(language, c) for language, row in FIXTURE["profiles"].items() for c in row["cases"]]


def call(case):
    op = case["op"]
    fields = ["x"] if op in SINGLE else ["a", "b"] if op in TWO else ARGUMENTS[op].split()
    return getattr(legacy, op)(*(case[key] for key in fields))


def extreme_target(case):
    op, x = case["op"], float(case["x"])
    a, b, y = (float(case[k]) for k in ["a", "b", "y"])
    if op == "rlog1":
        with localcontext() as ctx:
            ctx.prec = 800
            z = Decimal.from_float(x)
            return float(z - (1 + z).ln())
    if op == "erfc1":
        return independent_error(x, scaled=case["k"] != 0, complement=True)
    if op in ["gamln", "alngam", "Xgamm", "psi"]:
        return gamma_value({"gamln": 2, "alngam": 1, "Xgamm": 6, "psi": 7}[op], x)
    if op in ["algdiv", "bcorr", "betaln"]:
        return beta_log({"algdiv": 1, "bcorr": 2, "betaln": 3}[op], a, b)
    if op == "rcomp":
        return expected_factor(a, x)
    if op == "brcomp":
        return beta_factor(1, a, b, x, y)
    if op == "bratio":
        return tails(a, b, x, y)
    if op == "bgrat":
        return beta_increment(dict(case, mode=3, initial=case["w"]))
    if op == "basym":
        assert a == b and case["lam"] == 0
        return 0.5
    raise AssertionError(f"Missing independent oracle: {op}")


@pytest.mark.parametrize("language,case", CASES)
def test_direct_legacy_contracts_and_independently_verified_repairs(language, case):
    assert case["outcome"] == "completed"
    op = case["op"]
    if case["tag"] == "invalid":
        if op == "bratio":
            assert case["status"] in range(1, 8)
            message = f"IERR={case['status']}"
        else:
            message = None
            if op == "gratio":
                assert case["values"][0] == 2
            elif op in ["Xgamm", "psi"]:
                assert case["values"][0] == 0
            else:
                assert math.isinf(float(case["values"][0]))
        with pytest.raises(ValueError, match=message):
            call(case)
        return
    if case["tag"] == "rounded_product":
        assert op == "apser" and case["b"] * case["x"] == 1
        assert Fraction(case["b"]) * Fraction(case["x"]) > 1
        assert case["status"] == 0 and math.isfinite(case["values"][0])
        with pytest.raises(ValueError, match=r"b\*x<=1"):
            call(case)
        return
    actual = np.asarray(call(case)).reshape(-1)
    assert case["status"] == 0
    if case["tag"] == "repair":
        expected = np.asarray(extreme_target(case)).reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-324)
        assert np.all(actual[expected != 0] != 0)
        np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
        return
    assert case["tag"] == "ordinary"
    expected = np.array([float(v) for v in case["values"][: actual.size]])
    # The source explicitly offers reduced-accuracy modes; Python retains full precision.
    rtol = {1: 2e-5, 2: 5e-3}.get(case["k"], 5e-13) if op == "gratio" else 5e-13
    exact_log_zero = (op in ["alngam", "gamln"] and case["x"] in [1, 2]) or (
        op in ["gsumln", "betaln"] and case["a"] == case["b"] == 1
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=2e-14 if exact_log_zero else 0)


def test_legacy_exponential_limits_are_safe_batched_and_immutable():
    selectors = np.array([[0, 1], [-2, 2]], dtype=np.int32)
    values = legacy.exparg(selectors)
    assert np.all(np.isfinite(np.exp(values))) and np.all(np.exp(values) > 0)
    assert values[0, 0] < f95_exparg(0) and values[0, 1] > f95_exparg(1)
    selectors[:] = 99
    assert not values.flags.writeable and not np.shares_memory(values, selectors)
    with pytest.raises(ValueError):
        values.setflags(write=True)
    assert legacy.exparg(np.empty((0, 3))).shape == (0, 3)
    for bad in [0.5, -(2**31) - 1, 2**31, np.nan, np.inf]:
        with pytest.raises(ValueError):
            legacy.exparg(bad)


def test_tail_outputs_accumulators_and_renamed_functions_through_legacy_namespace():
    x = np.array([0, 0.25, 0.5, 0.75, 1])
    p, q = legacy.bratio(1, 1, x)
    np.testing.assert_allclose(p, x, rtol=3e-15, atol=0)
    np.testing.assert_allclose(q, 1 - x, rtol=3e-15, atol=0)
    assert not p.flags.writeable and not q.flags.writeable and not np.shares_memory(p, q)
    np.testing.assert_allclose(legacy.bgrat(15, 1, x, w=-0.5), -0.5 + x**15, rtol=1e-14)
    np.testing.assert_allclose(legacy.bfrac(2, 2, x), 3 * x * x - 2 * x * x * x, rtol=1e-14)
    np.testing.assert_array_equal(legacy.Xgamm([1, 2, 3, 4]), [1, 1, 2, 6])
    np.testing.assert_array_equal(legacy.erf1([-0.0, 0.0]), [-0.0, 0.0])
    assert np.signbit(legacy.erf1(-0.0))
