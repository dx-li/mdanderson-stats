"""Native menu actions plus independent extreme-value and state invariants."""

import io
import json
import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import CDFConsole, CDFConsoleError, CDFNumberList

REFERENCE = json.loads((Path(__file__).parent / "fixtures/cdflib_console.json").read_text())
CASES = [case for case in REFERENCE["cases"] if case["mode"] == "list"]


@pytest.mark.parametrize("case", CASES)
def test_all_native_list_menu_actions_with_documented_repairs(case):
    state = CDFNumberList([3, 1, 1], max_size=5)
    console = CDFConsole(io.StringIO(case["input"]), io.StringIO())
    if case["input"] == "4\n8\n":
        # Printing pauses: the supplied 8 is consumed by hold, then menu sees EOF.
        with pytest.raises(EOFError):
            console.get_list_double(state)
        np.testing.assert_array_equal(state.values, [3, 1, 1])
        return
    result = console.get_list_double(state)
    assert result.size == state.size
    match = re.search(r"RESULT\s+F\s+(\d+)\s*\n(.*)", case["stdout"], re.S)
    assert match
    native = np.array([float(x) for x in match[2].split()])
    if "1e308 1.5e308" in case["input"]:
        assert native.size == 1
        np.testing.assert_array_equal(result, [1e308, 1.5e308])
    else:
        np.testing.assert_allclose(result, native, rtol=1e-15, atol=0)
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize(
    "start,stop",
    [
        (1, 100),
        (100, 1),
        (1e-300, 1e300),
        (1e300, 1e-300),
        (1e300, 1.00000001e300),
        (1e-300, 1.00000001e-300),
        (1e308, 1.7e308),
    ],
)
def test_spacing_against_decimal_arithmetic(start, stop, log):
    state = CDFNumberList(max_size=17)
    state.append_spaced(start, stop, 16, log=log)
    actual = state.values
    with localcontext() as ctx:
        ctx.prec = 100
        a, b = Decimal.from_float(start), Decimal.from_float(stop)
        expected = []
        for i in range(17):
            t = Decimal(i) / 16
            value = ((1 - t) * a.ln() + t * b.ln()).exp() if log else (1 - t) * a + t * b
            expected.append(float(value))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-13 if log and max(start, stop) / 2 > min(start, stop) else 3e-16,
        atol=0,
    )
    assert actual[0] == start and actual[-1] == stop
    assert np.all((actual >= min(start, stop)) & (actual <= max(start, stop)))


def test_linear_opposite_extremes_and_equal_endpoints():
    state = CDFNumberList(max_size=9)
    state.append_spaced(-1.7e308, 1.7e308, 8)
    assert np.all(np.isfinite(state.values))
    assert state.values[4] == 0
    state = CDFNumberList(max_size=3)
    state.append_spaced(1.7e308, 1.7e308, 2)
    np.testing.assert_array_equal(state.values, [1.7e308] * 3)


def decimal_unique(values):
    with localcontext() as ctx:
        ctx.prec = 800
        epsilon = Decimal.from_float(1e-14)
        floor = Decimal.from_float(1e-100)
        kept = []
        for value in sorted(values):
            x = Decimal.from_float(float(value))
            if not kept or abs(x - kept[-1]) / max(abs(x) + abs(kept[-1]), floor) >= epsilon:
                kept.append(x)
        return np.array([float(v) for v in kept])


@pytest.mark.parametrize("scale", [1e-110, 1e-100, 1e-10, 1, 1e100, 1e308])
def test_duplicate_groups_use_retained_representative_and_stable_ratio(scale):
    values = np.array([1 + 3e-14, 1 + 1.5e-14, 1, 1, 1.5, -1, -1 - 3e-14]) * scale
    state = CDFNumberList(values, max_size=len(values))
    state.sort_unique()
    np.testing.assert_array_equal(state.values, decimal_unique(values))


def test_state_is_owned_and_rejected_actions_are_atomic():
    source = np.array([1.0, 2.0, 3.0])
    state = CDFNumberList(source, max_size=5, lo=0, hi=10)
    snapshot = state.values
    source[:] = 9
    actions = [
        lambda: state.append([4, 5, 6]),
        lambda: state.append([4, np.nan]),
        lambda: state.append([-1]),
        lambda: state.delete(0),
        lambda: state.delete(4),
        lambda: state.append_spaced(1, 2, 2),
        lambda: state.append_spaced(0, 2, 1, log=True),
        lambda: state.append_spaced(1, 2, 1.5),
    ]
    for action in actions:
        with pytest.raises(ValueError):
            action()
        np.testing.assert_array_equal(state.values, [1, 2, 3])
    state.delete(3, 2)
    state.append([4, 5])
    np.testing.assert_array_equal(state.values, [1, 4, 5])
    np.testing.assert_array_equal(snapshot, [1, 2, 3])


def test_successful_edits_survive_eof_and_action_budget_exhaustion():
    state = CDFNumberList(max_size=5)
    console = CDFConsole(io.StringIO("1\n2\n3 4\n"), io.StringIO())
    with pytest.raises(EOFError):
        console.get_list_double(state)
    np.testing.assert_array_equal(state.values, [3, 4])
    console = CDFConsole(io.StringIO("5\n1\n"), io.StringIO())
    with pytest.raises(CDFConsoleError, match="max_actions"):
        console.get_list_double(state, max_actions=1)
    np.testing.assert_array_equal(state.values, [4])


def test_rejected_actions_and_numeric_retries_follow_distinct_budgets():
    state = CDFNumberList(max_size=5, lo=0, hi=1)
    # Invalid values retry inside ADD, then an invalid log spacing returns to menu.
    console = CDFConsole(io.StringIO("1\n1\n-1\n2\n0.5\n3\n0 1 1\n8\n"), io.StringIO())
    np.testing.assert_array_equal(console.get_list_double(state), [0.5])
    console = CDFConsole(io.StringIO("7\n7\n7\n7\n"), io.StringIO())
    with pytest.raises(CDFConsoleError, match="failed list actions"):
        console.get_list_double(CDFNumberList())
    # Exhausted menu input terminates immediately rather than spending action failures.
    console = CDFConsole(io.StringIO("9\n9\n9\n8\n"), io.StringIO())
    with pytest.raises(CDFConsoleError, match="numeric"):
        console.get_list_double(state)


def test_print_paginates_and_does_not_duplicate_last_pause():
    state = CDFNumberList([1, 2, 3, 4])
    output = io.StringIO()
    c = CDFConsole(io.StringIO("4\n\n\n8\n"), output)
    c.get_list_double(state, page_size=2)
    assert output.getvalue().count("Press the Return") == 2
    c = CDFConsole(io.StringIO("4\n8\n"), io.StringIO())
    c.get_list_double(state, page_size=0)


def test_fractional_intervals_rejected_without_truncation():
    state = CDFNumberList(max_size=3)
    c = CDFConsole(io.StringIO("2\n0 1 1.5\n8\n"), io.StringIO())
    assert c.get_list_double(state).size == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_size=-1),
        dict(max_size=True),
        dict(max_size=2**31),
        dict(lo=2, hi=1),
        dict(lo=1, hi=1, hi_eq_ok=False),
        dict(lo=np.inf),
    ],
)
def test_invalid_list_configuration(kwargs):
    with pytest.raises(ValueError):
        CDFNumberList(**kwargs)


@pytest.mark.parametrize("start,stop", [(5e-324, 1e-323), (-1e-323, -5e-324), (-5e-324, 1e-323)])
def test_linear_subnormal_midpoint_rounds_only_after_combining_terms(start, stop):
    state = CDFNumberList(max_size=3)
    state.append_spaced(start, stop, 2)
    with localcontext() as ctx:
        ctx.prec = 800
        expected = float((Decimal.from_float(start) + Decimal.from_float(stop)) / 2)
    assert state.values[1] == expected


@pytest.mark.parametrize("allowance", [0, 3])
def test_exhausted_add_input_counts_as_one_failed_action(allowance):
    state = CDFNumberList([0.5], max_size=3)
    c = CDFConsole(io.StringIO("1\n1\nbad\nbad\nbad\n8\n"), io.StringIO())
    if allowance == 0:
        with pytest.raises(CDFConsoleError, match="failed list actions"):
            c.get_list_double(state, max_failures=allowance)
        # The next menu record remains available to resume this same state.
        np.testing.assert_array_equal(c.get_list_double(state), [0.5])
    else:
        np.testing.assert_array_equal(c.get_list_double(state, max_failures=allowance), [0.5])


def test_overlapping_middle_deletion_reclaims_capacity():
    state = CDFNumberList(list(range(1, 9)), max_size=8)
    state.delete(3, 2)
    np.testing.assert_array_equal(state.values, [1, 4, 5, 6, 7, 8])
    state.append([9, 10])
    np.testing.assert_array_equal(state.values, [1, 4, 5, 6, 7, 8, 9, 10])
