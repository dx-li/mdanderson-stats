"""Independent equations classify native root results without endorsing defects."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cdflib_root.json").read_text())
CASES = REFERENCE["cases"]


def equation(function_id, x):
    return x if function_id == 1 else (x * x if function_id == 2 else -x)


def roots(case):
    with localcontext() as ctx:
        ctx.prec = 80
        target = Decimal.from_float(case["target"])
        if case["function_id"] == 1:
            return [target]
        if case["function_id"] == 3:
            return [-target]
        if target < 0:
            return []
        root = target.sqrt()
        return [root] if root == 0 else [-root, root]


def known_wrong_success(case):
    mode, function_id, target = (case[k] for k in ("mode", "function_id", "target"))
    if function_id == 2:
        return target == 0
    if function_id == 1:
        return target in (0, 0.5) or (target == 1 and mode <= 2)
    return target == -1 and mode <= 2


@pytest.mark.parametrize("case", CASES)
def test_native_result_against_equation_and_failure_contract(case):
    assert case["execution_outcome"] == "completed"
    assert case["status"] == case["global_status"]
    initial_root = (
        case["mode"] >= 3 and equation(case["function_id"], case["initial"]) == case["target"]
    )
    if initial_root:
        assert case["status"] == 0 and case["final_status"] == 1
    else:
        assert case["status"] == case["final_status"]
    lo, hi = (Decimal.from_float(case[k]) for k in ("low", "high"))
    target = Decimal.from_float(case["target"])
    fa, fb = equation(case["function_id"], lo) - target, equation(case["function_id"], hi) - target
    assert case["evaluations"] == len(case["requests"])
    assert all(case["low"] <= x <= case["high"] for x in case["requests"])
    if case["status"] == -1:
        assert fa * fb > 0
        failure = case["failure"]
        assert (failure["low"], failure["high"]) == (case["low"], case["high"])
        assert failure["crash_hi"] == (fa > 0)
        if case["mode"] == 1:
            # INTENT(OUT) answer is undefined on direct interval failure.
            assert case["answer"] is None
        elif case["mode"] == 3:
            assert case["answer"] == case["initial"]
        else:
            assert case["answer"] == (case["low"] if failure["crash_left"] else case["high"])
        if case["low"] == -2:
            assert roots(case) == [Decimal(-1), Decimal(1)]
        return
    assert case["status"] == 0 and fa * fb <= 0
    assert case["failure"] is None
    in_bounds = [r for r in roots(case) if lo <= r <= hi]
    answer = Decimal.from_float(case["answer"])
    error = min(abs(answer - r) for r in in_bounds)
    if known_wrong_success(case):
        assert error > Decimal("0.1")
        assert abs(equation(case["function_id"], answer) - target) > Decimal("0.1")
    else:
        assert error < Decimal("1e-12")


def key(case):
    return tuple(
        case[k] for k in ("function_id", "low", "high", "target", "initial", "abs_tol", "rel_tol")
    )


@pytest.mark.parametrize("direct,reverse", [(1, 2), (3, 4)])
def test_direct_and_reverse_communication_reproduce_same_requests(direct, reverse):
    reverse_cases = {key(c): c for c in CASES if c["mode"] == reverse}
    for case in (c for c in CASES if c["mode"] == direct):
        other = reverse_cases[key(case)]
        for field in (
            "status",
            "final_status",
            "global_status",
            "evaluations",
            "requests",
            "failure",
        ):
            assert case[field] == other[field]
        if case["status"] == 0:
            assert case["answer"] == other["answer"]


@pytest.mark.parametrize("mode", [1, 2, 3, 4])
def test_configured_tolerances_do_not_change_native_square_root_trace(mode):
    selected = [
        c for c in CASES if c["mode"] == mode and c["function_id"] == 2 and c["target"] == 2
    ]
    assert {c["abs_tol"] for c in selected} == {0.01, 1e-12}
    assert {c["rel_tol"] for c in selected} == {0.01, 1e-12}
    assert selected[0]["requests"] == selected[1]["requests"]
    assert selected[0]["answer"] == selected[1]["answer"]
