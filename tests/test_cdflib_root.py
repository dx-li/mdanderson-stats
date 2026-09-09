"""Numerical and continuation contracts for corrected CDFLIB root searches."""

from dataclasses import FrozenInstanceError
from math import exp, isfinite, nextafter, sqrt

import pytest
from test_cdflib_root_reference import CASES, equation, roots

from mdanderson_stats.cdflib_root import (
    final_zf_state,
    interval_zf,
    rc_interval_zf,
    rc_step_zf,
    set_zero_finder,
    step_zf,
)


@pytest.mark.parametrize("case", CASES)
def test_archived_equations_with_corrected_root_and_state(case):
    state = set_zero_finder(
        low_limit=case["low"],
        hi_limit=case["high"],
        abs_tol=case["abs_tol"],
        rel_tol=case["rel_tol"],
    )

    def f(x):
        return equation(case["function_id"], x)

    mode = case["mode"]
    if mode == 1:
        result = interval_zf(f, case["target"], local=state)
    elif mode == 3:
        result = step_zf(f, case["initial"], case["target"], local=state)
    else:
        advance = rc_interval_zf if mode == 2 else rc_step_zf
        result = advance(state) if mode == 2 else advance(state, initial=case["initial"])
        while result.status == 1:
            result = advance(state, f(result.x) - case["target"])
    assert final_zf_state(state) is result
    assert result.status == case["status"]
    if result.status == 0:
        candidates = [float(r) for r in roots(case) if case["low"] <= r <= case["high"]]
        tolerance = case["abs_tol"] + case["rel_tol"] * abs(result.x)
        assert min(abs(result.x - r) for r in candidates) <= tolerance
        assert result.bound_low <= result.x <= result.bound_high
        assert (f(result.bound_low) - case["target"]) * (f(result.bound_high) - case["target"]) <= 0
        assert result.crash_hi is result.crash_left is None
    else:
        assert result.crash_hi == case["failure"]["crash_hi"]
        assert result.crash_left == case["failure"]["crash_left"]


@pytest.mark.parametrize("step", [False, True])
@pytest.mark.parametrize("target", [0, 0.125, 0.5, 1, 2])
def test_direct_reverse_identical_requests_and_exact_roots(step, target):
    calls = []

    def f(x):
        calls.append(x)
        return x

    settings = dict(low_limit=0, hi_limit=2, abs_tol=1e-14, rel_tol=0)
    state = set_zero_finder(**settings)
    direct = step_zf(f, 1, target, local=state) if step else interval_zf(f, target, local=state)
    other = set_zero_finder(**settings)
    reverse = rc_step_zf(other, initial=1) if step else rc_interval_zf(other)
    requests = []
    while reverse.status == 1:
        requests.append(reverse.x)
        reverse = (
            rc_step_zf(other, reverse.x - target)
            if step
            else rc_interval_zf(other, reverse.x - target)
        )
    assert calls == requests
    assert direct == reverse
    assert direct.x == target
    assert direct.bound_low == direct.bound_high == target
    assert direct.evaluations == len(calls)


@pytest.mark.parametrize("step", [False, True])
def test_tolerance_changes_work_and_bounds_error(step):
    results = []
    for tolerance in (0.01, 1e-14):
        state = set_zero_finder(low_limit=0, hi_limit=2, abs_tol=tolerance, rel_tol=0)
        r = (
            step_zf(lambda x: x * x, 1, 2, local=state)
            if step
            else interval_zf(lambda x: x * x, 2, local=state)
        )
        assert abs(r.x - sqrt(2)) <= tolerance
        results.append(r)
    assert results[0].evaluations < results[1].evaluations
    assert results[1].evaluations < 50


@pytest.mark.parametrize("power", [1, 3, 7, 19])
@pytest.mark.parametrize("root", [-0.9, -0.123456789, 0, 0.23456789, 0.99])
@pytest.mark.parametrize("scale", [1e-280, 1, 1e280])
def test_flat_odd_polynomials(power, root, scale):
    r = interval_zf(
        lambda x: scale * (x - root) ** power,
        local=set_zero_finder(low_limit=-1, hi_limit=1, abs_tol=1e-12, rel_tol=0),
    )
    # A rounded residual may be exactly zero before x reaches the mathematical
    # root for the smallest scale/highest powers. Require a zero residual or
    # a bracket-based positional guarantee, as the caller's data permit.
    assert abs(r.x - root) <= 1e-12 or scale * (r.x - root) ** power == 0
    assert r.evaluations <= 200


def test_extreme_bounds_residuals_and_adjacent_floats():
    r = interval_zf(lambda x: x, local=set_zero_finder(low_limit=-1.7e308, hi_limit=1.7e308))
    assert r.x == 0
    a, b = 1e308, nextafter(1e308, float("inf"))
    r = interval_zf(
        lambda x: -1 if x == a else 1,
        local=set_zero_finder(low_limit=a, hi_limit=b, abs_tol=5e-324, rel_tol=0),
    )
    assert r.status == 0 and a <= r.x <= b and isfinite(r.x)
    r = interval_zf(
        lambda x: -1e308 if x < 0.3 else 1e308,
        local=set_zero_finder(low_limit=-1, hi_limit=1, abs_tol=1e-12, rel_tol=0),
    )
    assert abs(r.x - 0.3) <= 1e-12


def test_interleaved_and_nested_states_are_independent():
    a = set_zero_finder(low_limit=0, hi_limit=2, abs_tol=1e-12, rel_tol=0)
    b = set_zero_finder(low_limit=0, hi_limit=3, abs_tol=1e-12, rel_tol=0)
    ra, rb = rc_interval_zf(a), rc_step_zf(b, initial=1)
    while ra.status == 1 or rb.status == 1:
        if ra.status == 1:
            ra = rc_interval_zf(a, ra.x * ra.x - 2)
        if rb.status == 1:
            rb = rc_step_zf(b, exp(rb.x) - 2)
    assert abs(ra.x - sqrt(2)) < 1e-12
    assert abs(exp(rb.x) - 2) < 3e-12

    def outer(x):
        inner = interval_zf(lambda z: z - x, local=set_zero_finder(low_limit=-2, hi_limit=2))
        return inner.x

    assert interval_zf(outer, 0.5, local=set_zero_finder(low_limit=-1, hi_limit=1)).x == 0.5


def test_protocol_and_snapshot_immutability():
    state = set_zero_finder(low_limit=0, hi_limit=2)
    assert final_zf_state(state) is None
    with pytest.raises(ValueError, match="first call"):
        rc_interval_zf(state, 1)
    pending = rc_interval_zf(state)
    with pytest.raises(FrozenInstanceError):
        pending.status = 0
    with pytest.raises(ValueError, match="outstanding"):
        rc_interval_zf(state)
    with pytest.raises(ValueError, match="switch"):
        rc_step_zf(state, 1)
    for invalid in (float("nan"), float("inf"), [1]):
        with pytest.raises(ValueError):
            rc_interval_zf(state, invalid)
        assert state.result is pending
    done = rc_interval_zf(state, 0)
    assert done.status == 0 and pending.status == 1
    with pytest.raises(ValueError, match="finished"):
        rc_interval_zf(state, 1)


def test_budget_is_exact_and_terminal():
    state = set_zero_finder(low_limit=0, hi_limit=2, max_evaluations=2)
    calls = []

    def f(x):
        calls.append(x)
        return x * x - 2

    with pytest.raises(ArithmeticError, match="max_evaluations"):
        interval_zf(f, local=state)
    assert calls == [0, 2]
    assert state.result.status == -2 and state.result.evaluations == 2
    with pytest.raises(ValueError, match="finished"):
        rc_interval_zf(state, 0)
    # The last allowed evaluation may still finish successfully.
    r = interval_zf(
        lambda x: x - 2, local=set_zero_finder(low_limit=0, hi_limit=2, max_evaluations=2)
    )
    assert r.status == 0 and r.evaluations == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_limit": 2, "hi_limit": 1},
        {"low_limit": float("inf")},
        {"abs_tol": -1},
        {"rel_tol": -1},
        {"abs_tol": 0, "rel_tol": 0},
        {"abs_step": -1},
        {"rel_step": -1},
        {"step_multiplier": 1},
        {"max_evaluations": 0},
        {"max_evaluations": True},
        {"max_evaluations": 1.5},
    ],
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        set_zero_finder(**kwargs)


def test_invalid_initial_and_nonfinite_callback():
    for initial in (-1, 3, float("nan")):
        with pytest.raises(ValueError):
            step_zf(lambda x: x, initial, local=set_zero_finder(low_limit=0, hi_limit=2))
    with pytest.raises(ValueError, match="step"):
        step_zf(lambda x: x, 0, local=set_zero_finder(abs_step=0))
    with pytest.raises(ValueError):
        interval_zf(lambda x: float("inf"))
    with pytest.raises(ValueError):
        interval_zf(lambda x: 1e308, -1e308)


@pytest.mark.parametrize("root", [-90.25, -3.125, 4.25, 90.75])
@pytest.mark.parametrize("direction", [-1, 1])
def test_step_expands_geometrically_in_both_directions(root, direction):
    calls = []

    def f(x):
        calls.append(x)
        return direction * (x - root)

    r = step_zf(
        f,
        0,
        local=set_zero_finder(
            low_limit=-100,
            hi_limit=100,
            abs_step=0.25,
            rel_step=0,
            abs_tol=1e-12,
            rel_tol=0,
        ),
    )
    assert r.x == root
    assert calls[:3] == [-100, 100, 0]
    assert calls[3:6] == ([0.25, 0.75, 1.75] if root > 0 else [-0.25, -0.75, -1.75])
    assert calls.count(-100) == calls.count(100) == 1
    assert r.evaluations < 20


@pytest.mark.parametrize("root", [1e-300, 1e-100, 1e100, 1e300])
@pytest.mark.parametrize("step", [False, True])
def test_relative_only_tolerance_across_scales(root, step):
    state = set_zero_finder(low_limit=0, hi_limit=2 * root, abs_tol=0, rel_tol=1e-12)

    def f(x):
        return (x / root) ** 2 - 1

    r = step_zf(f, root / 2, local=state) if step else interval_zf(f, local=state)
    assert abs(r.x / root - 1) <= 1e-12


def test_step_smaller_than_float_spacing_makes_progress():
    low = 1e100
    high = low + 1e86
    calls = []

    def f(x):
        calls.append(x)
        return x - high

    r = step_zf(
        f,
        low,
        local=set_zero_finder(
            low_limit=low,
            hi_limit=high * 2,
            abs_step=5e-324,
            rel_step=0,
            abs_tol=5e-324,
            rel_tol=0,
        ),
    )
    assert r.x == high
    assert len(calls) == len(set(calls))
    assert r.evaluations < 100
