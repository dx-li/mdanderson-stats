"""Legacy native root protocols, bracket invariants and independent equations."""

import json
import math
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from mdanderson_stats import dcdflib_support as legacy

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dcdflib_root.json").read_text())
CASES = [(lang, c) for lang, p in FIXTURE["profiles"].items() for c in p["cases"]]


def equation(case, x):
    fn, target, scale = case["fn"], case["target"], case["scale"]
    if fn == 1:
        return (x - target) * scale
    if fn == 2:
        return (target - x) * scale
    if fn == 3:
        return (x / scale) ** 2 - target
    if fn == 4:
        return target
    if fn == 5:
        return math.exp(x) - target
    if fn == 6:
        return (x - target) ** 3 * scale
    return -scale if x < target else scale


def setup(case):
    if case["mode"] == 1:
        return legacy.dstinv(
            case["low"],
            case["high"],
            case["abs_step"],
            case["rel_step"],
            case["multiplier"],
            case["abs_tol"],
            case["rel_tol"],
        )
    return legacy.dstzr(case["low"], case["high"], case["abs_tol"], case["rel_tol"])


def run(state, fn, initial=None):
    advance = legacy.dzror if initial is None else legacy.dinvr
    result = advance(state) if initial is None else advance(state, initial=initial)
    calls = []
    while result.status == 1:
        assert result.xlo is result.xhi is result.qleft is result.qhi is None
        calls.append(result.x)
        result = advance(state, fn(result.x))
    assert state.result is result and result.evaluations == len(calls)
    return result, calls


def check_bracket(result, fn, absolute, relative):
    assert result.status == 0 and result.xlo == result.x
    assert result.qleft is result.qhi is None
    fa, fb = fn(result.xlo), fn(result.xhi)
    assert fa == 0 or fb == 0 or (fa < 0) != (fb < 0)
    assert abs(fa) <= abs(fb)
    with localcontext() as ctx:
        ctx.prec = 800
        a, b = Decimal.from_float(result.xlo), Decimal.from_float(result.xhi)
        tolerance = max(Decimal.from_float(absolute), Decimal.from_float(relative) * abs(a))
        assert abs(a - b) <= tolerance or math.nextafter(float(a), float(b)) == float(b)


@pytest.mark.parametrize("language,case", CASES)
def test_native_contracts_and_mathematical_brackets(language, case):
    state = setup(case)
    if case["outcome"] == "native_stop":
        with pytest.raises(ValueError, match="inside"):
            legacy.dinvr(state, initial=case["initial"])
        assert state.result is None
        return
    result, _ = run(
        state, lambda x: equation(case, x), case["initial"] if case["mode"] == 1 else None
    )
    no_bracket = (
        case["fn"] in [1, 2]
        and not case["low"] <= case["target"] <= case["high"]
        or case["fn"] == 4
        and case["target"] != 0
    )
    if no_bracket:
        assert result.status == case["status"] == -1
        assert (result.qleft, result.qhi) == (case["qleft"], case["qhi"])
        assert result.x == case["x"] == case["high"]
        assert result.xlo is result.xhi is None
        return
    check_bracket(result, lambda x: equation(case, x), case["abs_tol"], case["rel_tol"])
    # DZROR multiplies tiny residuals to test signs and reports a false failure.
    native_false_failure = case["mode"] == 2 and case["fn"] == 1 and case["scale"] == 1e-300
    assert case["status"] == (-1 if native_false_failure else 0)
    if case["fn"] == 4:  # Every coordinate is a root of the identically zero function.
        return
    with localcontext() as ctx:
        ctx.prec = 800
        target = Decimal.from_float(case["target"])
        root = (
            target.sqrt() * Decimal.from_float(case["scale"])
            if case["fn"] == 3
            else target.ln()
            if case["fn"] == 5
            else target
        )
        tolerance = max(
            Decimal.from_float(case["abs_tol"]),
            Decimal.from_float(case["rel_tol"]) * abs(root),
            Decimal.from_float(math.ulp(float(root))),
        )
        assert abs(Decimal.from_float(result.x) - root) <= tolerance
        if not native_false_failure:
            assert abs(Decimal.from_float(case["x"]) - root) <= tolerance


@pytest.mark.parametrize("initial", [None, 0.5])
@pytest.mark.parametrize(
    "absolute,relative", [(0.1, 0.1), (0.0, 1e-12), (1e-12, 0.0), (5e-324, 0.0)]
)
def test_legacy_max_tolerance_and_best_endpoint(initial, absolute, relative):
    state = (
        legacy.dstzr(0, 2, absolute, relative)
        if initial is None
        else legacy.dstinv(0, 2, 0.5, 0.5, 5, absolute, relative)
    )
    result, _ = run(state, lambda x: x * x - 2, initial)
    check_bracket(result, lambda x: x * x - 2, absolute, relative)


def test_overflow_safe_tolerance_and_unavoidable_float_spacing():
    def fn(x):
        return -1.0 if x < 0.3 else 1.0

    result, calls = run(legacy.dstzr(-1.7e308, 1e308, 0, 1.1), fn)
    assert len(calls) > 2  # The unhalved relative product overflows at the first bracket.
    check_bracket(result, fn, 0.0, 1.1)
    low = 1e308
    high = math.nextafter(low, math.inf)
    result, _ = run(legacy.dstzr(low, high, 5e-324, 0), lambda x: -1 if x == low else 1)
    assert {result.xlo, result.xhi} == {low, high}


def test_interleaved_modes_and_frozen_snapshots():
    left = legacy.dstzr(0, 2, 1e-12, 0)
    right = legacy.dstinv(0, 3, 0.5, 0.5, 5, 1e-12, 0)
    a, b = legacy.dzror(left), legacy.dinvr(right, initial=1)
    saved = a
    while a.status == 1 or b.status == 1:
        if a.status == 1:
            a = legacy.dzror(left, a.x * a.x - 2)
        if b.status == 1:
            b = legacy.dinvr(right, math.exp(b.x) - 2)
    assert abs(a.x - math.sqrt(2)) < 1e-12 and abs(b.x - math.log(2)) < 1e-12
    assert saved.status == 1
    with pytest.raises(FrozenInstanceError):
        saved.status = 0


def test_protocol_errors_do_not_consume_the_pending_request():
    state = legacy.dstzr(0, 2, 1e-12, 0)
    with pytest.raises(ValueError):
        legacy.dzror(state, 1)
    pending = legacy.dzror(state)
    for value in [None, math.nan, math.inf, [1]]:
        with pytest.raises(ValueError):
            legacy.dzror(state, value)
        assert state.result is pending
    with pytest.raises(ValueError):
        legacy.dinvr(state, 1)
    result = legacy.dzror(state, 0)
    assert result.status == 0
    with pytest.raises(ValueError):
        legacy.dzror(state, 0)


def test_budget_is_terminal_and_last_allowed_evaluation_can_finish():
    state = legacy.dstzr(0, 2, 1e-12, 0, max_evaluations=2)
    with pytest.raises(ArithmeticError, match="max_evaluations"):
        run(state, lambda x: x * x - 2)
    assert state.result.status == -2 and state.result.evaluations == 2
    with pytest.raises(ValueError):
        legacy.dzror(state, 0)
    result, _ = run(legacy.dstzr(0, 2, 1e-12, 0, max_evaluations=2), lambda x: x - 2)
    assert result.status == 0 and result.evaluations == 2


@pytest.mark.parametrize(
    "low,high,absolute,relative",
    [(1, 0, 1e-6, 0), (0, math.inf, 1e-6, 0), (0, 1, -1, 0), (0, 1, 0, 0), (0, 1, 0, math.nan)],
)
def test_invalid_configuration(low, high, absolute, relative):
    with pytest.raises(ValueError):
        legacy.dstzr(low, high, absolute, relative)
    with pytest.raises(ValueError):
        legacy.dstinv(low, high, 0.5, 0.5, 5, absolute, relative)


@pytest.mark.parametrize("steps", [(-1, 0.5, 5), (0.5, -1, 5), (0.5, 0.5, 1)])
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        legacy.dstinv(0, 1, *steps, 1e-6, 0)


@pytest.mark.parametrize("step", [False, True])
@pytest.mark.parametrize("residual", [-1.0, 0.0, 1.0])
def test_single_point_interval(step, residual):
    state = legacy.dstinv(0, 0, 0, 0, 2, 1e-12, 0) if step else legacy.dstzr(0, 0, 1e-12, 0)
    result, _ = run(state, lambda x: residual, 0 if step else None)
    if residual == 0:
        assert result.status == 0 and result.xlo == result.xhi == result.x == 0
    else:
        assert result.status == -1 and result.x == 0
        assert result.qhi == (residual > 0)
        assert result.qleft == (step and residual < 0)


def test_step_start_and_mode_validation():
    state = legacy.dstinv(0, 2, 0.5, 0.5, 5, 1e-12, 0)
    for initial in [-1, 3, math.nan, None]:
        with pytest.raises(ValueError):
            legacy.dinvr(state, initial=initial)
        assert state.result is None
    with pytest.raises(ValueError):
        legacy.dzror(state)
    pending = legacy.dinvr(state, initial=1)
    with pytest.raises(ValueError):
        legacy.dinvr(state, -1, initial=1)
    assert state.result is pending
    with pytest.raises(ValueError, match="step"):
        legacy.dinvr(legacy.dstinv(0, 2, 0, 0, 5, 1e-12, 0), initial=1)
