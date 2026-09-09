"""Source requests, transactional reuse and table/session isolation."""

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from test_stattab_reference import CASES, normal
from test_stattab_results import INPUTS, TARGETS

from mdanderson_stats import (
    STATTAB_DISTRIBUTIONS,
    CDFNumberList,
    STATTABRequest,
    STATTABResult,
    STATTABSession,
    parse_stattab_request,
    stattab_solve,
)


def result(session, line, **kwargs):
    value = session.execute(line, **kwargs)
    assert isinstance(value, STATTABResult)
    return value


def request_line(name, compute, inputs):
    descriptor = STATTAB_DISTRIBUTIONS[name]
    return " ".join(
        "?" if key == compute else str(inputs[key]) if key in inputs else "."
        for key in descriptor.parameters
    )


@pytest.mark.parametrize("name", INPUTS)
def test_native_forward_request_and_table_for_each_family(name):
    session = STATTABSession(name)
    line = CASES[name + "_which_1"]["input"].splitlines()[3]
    actual = result(session, line)
    expected = stattab_solve(name, **INPUTS[name])
    np.testing.assert_array_equal(actual.values, expected.values)
    table_line = CASES[name + "_table"]["input"].splitlines()[3]
    values = [0.25, 0.5, 1] if name == "beta" else [0.5, 1, 2]
    table = result(session, table_line, table=values)
    assert table.values.shape[0] == 3
    assert session.previous[next(iter(INPUTS[name]))] == values[-1]
    scalar = result(session, line)
    np.testing.assert_array_equal(scalar.values, expected.values)


@pytest.mark.parametrize(
    "name,group", [(n, g) for n, d in STATTAB_DISTRIBUTIONS.items() for g in d.groups]
)
def test_all_42_computed_groups_through_positional_requests(name, group):
    inputs = INPUTS[name] | dict(cum=TARGETS[name], ccum=1 - TARGETS[name])
    for pair in [("x", "cx"), ("pr", "cpr"), ("cum", "ccum")]:
        if pair[0] in inputs:
            inputs.pop(pair[1], None)
    inputs = {k: v for k, v in inputs.items() if k not in group}
    line = request_line(name, group[0], inputs)
    bracket = dict(df_bracket=(4.9, 5.1)) if name == "nc_t" and group == ("df",) else {}
    parsed = parse_stattab_request(name, line)
    assert parsed.compute == group[0] and parsed.action == "solve"
    actual = result(STATTABSession(name), line, **bracket)
    for key, expected in INPUTS[name].items():
        assert actual.parameters[key] == pytest.approx(expected, rel=2e-8, abs=2e-9)


def test_numeric_forms_and_comment_grammar():
    expected = stattab_solve("normal", x=-1, mean=0, sd=2)
    for text in ["-1 0 2 ? .", " - 1, +0, 2D0, ?, . # comment", "-1\t0\t2e+0\t?\t."]:
        np.testing.assert_array_equal(
            result(STATTABSession("normal"), text).values, expected.values
        )
    # Integer tokens wider than int32 retain their finite real value.
    request = parse_stattab_request("poisson", "2147483648 3 ? .")
    assert request.parameters["s"] == 2147483648
    assert parse_stattab_request("normal", "1 0 1 . ?").compute == "ccum"
    assert parse_stattab_request("beta", ". ? 2 2 . 0.5").compute == "cx"


@pytest.mark.parametrize(
    "line",
    [
        "1 0 1 ? . 9 9",
        "1 0 ? .",
        "? 0 1 ? .",
        "1 0 1 . .",
        "1 0 1 0.5 0.5",
        "1 0 1 ? =",
        ". 0 1 ? .",
        "T T 1 ? .",
        "1e999 0 1 ? .",
        "1e-999 0 1 ? .",
        "1 0 1 0.5 .",
        "SHELPER",
        "HELP now",
        '"1" 0 1 ? .',
        "1 0 1 '?' .",
        "1;0;1;?;.",
        "1 0 1 ? .\n",
        "1 0 1 ? .\r",
        "NaN 0 1 ? .",
        "inf 0 1 ? .",
    ],
)
def test_invalid_grammar_fails_before_solving(line):
    with pytest.raises(ValueError):
        parse_stattab_request("normal", line)


def test_help_menu_and_selection_have_explicit_state_effects():
    session = STATTABSession("normal")
    result(session, "1 0 1 ? .")
    previous = session.previous
    for line in ["help", " HELP # note"]:
        command = session.execute(line)
        assert isinstance(command, STATTABRequest) and command.action == "help"
        assert session.previous is previous
    for bad in ["not_a_distribution", None]:
        with pytest.raises(ValueError):
            session.select(bad)
        assert session.previous is previous
    for line in ["", "   # menu"]:
        result(session, "1 0 1 ? .")
        assert session.execute(line).action == "menu"
        assert not session.previous and session.last_result is None
    session.select("gamma")
    assert session.distribution == "gamma"
    with pytest.raises(ValueError, match="previous"):
        session.execute("= = = ? .")
    result(session, "1 2 3 ? .")
    session.select("gamma")
    assert not session.previous


def test_poisson_reuse_keeps_original_cdf_and_continuous_inverse():
    session = STATTABSession("poisson")
    forward = result(session, "2 3 ? .")
    assert session.previous["cum"] == pytest.approx(8.5 * math.exp(-3))
    inverse = result(session, "? = = .")
    assert inverse.parameters["s"] == pytest.approx(2, abs=2e-8)
    assert session.previous["s"] == inverse.parameters["s"]
    assert session.previous["s"] != inverse.neighbors[1].candidate
    np.testing.assert_array_equal(forward.values, stattab_solve("poisson", s=2, mean=3).values)


def test_reuse_preserves_tiny_saved_complement_even_when_primary_rounds_to_one():
    session = STATTABSession("normal")
    first = result(session, "9 0 1 ? .")
    assert session.previous["cum"] == 1 and session.previous["ccum"] > 0
    inverse = result(session, "? = = = .")
    assert inverse.parameters["x"] == pytest.approx(9, rel=2e-14)
    assert inverse.parameters["ccum"] == first.parameters["ccum"]
    inverse = result(session, "? = = . =")
    assert inverse.parameters["x"] == pytest.approx(9, rel=2e-14)
    beta = STATTABSession("beta")
    result(beta, ". 1e-20 2 2 ? .")
    next_result = result(beta, "= . = = ? .")
    assert next_result.parameters["cx"] == 1e-20


def test_gamma_reuse_retains_input_rate_shape_order():
    session = STATTABSession("gamma")
    result(session, "1 2 3 ? .")
    assert tuple(session.previous) == ("x", "rate", "shape", "cum", "ccum")
    second = result(session, "2 = = ? .")
    assert second.parameters["rate"] == 2 and second.parameters["shape"] == 3
    assert second.parameters["cum"] == pytest.approx(1 - 13 * math.exp(-4))


def test_rejected_requests_leave_last_successful_state_untouched():
    session = STATTABSession("normal")
    with pytest.raises(ValueError, match="previous"):
        session.execute("= 0 1 ? .")
    initial = result(session, "1 0 1 ? .")
    previous = session.previous
    for line, kwargs in [
        ("1 0 -1 ? .", {}),
        ("bad", {}),
        ("T = = ? .", {}),
        ("T = = ? .", dict(table=[1, math.nan])),
        ("T = = ? .", dict(table=[[1, 2]])),
        ("2 = = ? .", dict(table=[2])),
        ("HELP", dict(table=[2])),
        ("T = = ? .", dict(table=list(range(101)))),
        ("2 = = ? .", dict(df_bracket=(1, 2))),
    ]:
        with pytest.raises(ValueError):
            session.execute(line, **kwargs)
        assert session.previous is previous and session.last_result is initial
    np.testing.assert_array_equal(
        result(session, "2 = = ? .").values, stattab_solve("normal", x=2, mean=0, sd=1).values
    )


def test_tables_reset_between_calls_and_reuse_last_row():
    session = STATTABSession("normal")
    table = result(session, "T 0 1 ? .", table=[0, 1, 2])
    assert session.previous["x"] == 2
    assert session.previous["cum"] == pytest.approx(normal(2))
    scalar = result(session, "3 = = ? .")
    assert scalar.values.ndim == 1 and session.previous["x"] == 3
    previous = session.previous
    empty = result(session, "T = = ? .", table=[])
    assert empty.values.shape == (0, 6) and session.previous is previous
    assert session.last_result is empty
    assert table.values[:, 0].tolist() == [0, 1, 2]


def test_list_editor_snapshot_and_session_isolation():
    numbers = CDFNumberList([0, 1, 2], max_size=10)
    one, two = STATTABSession("normal"), STATTABSession("normal")
    actual = result(one, "T 0 1 ? .", table=numbers)
    np.testing.assert_array_equal(numbers.values, [0, 1, 2])
    with pytest.raises(ValueError):
        two.execute("= = = ? .")
    result(two, "4 1 2 ? .")
    assert one.previous["mean"] == 0 and two.previous["mean"] == 1
    with pytest.raises(ValueError):
        actual.values.setflags(write=True)
    with pytest.raises(TypeError):
        one.previous["x"] = 5
    parsed = parse_stattab_request("normal", "1 0 1 ? .")
    with pytest.raises(TypeError):
        parsed.parameters["x"] = 5
    with pytest.raises(FrozenInstanceError):
        parsed.compute = "x"


@pytest.mark.parametrize(
    "name,key", [(n, k) for n, d in STATTAB_DISTRIBUTIONS.items() for k in d.parameters]
)
def test_table_selection_at_every_source_input_position(name, key):
    descriptor = STATTAB_DISTRIBUTIONS[name]
    # Tail inputs require an inverse request; other positions use forward tails.
    compute = descriptor.groups[1][0] if key in ("cum", "ccum") else "cum"
    computed = next(g for g in descriptor.groups if compute in g)
    base = INPUTS[name] | dict(cum=TARGETS[name], ccum=1 - TARGETS[name])
    inputs = {k: v for k, v in base.items() if k not in computed}
    for pair in [g for g in descriptor.groups if len(g) == 2]:
        keep = key if key in pair else pair[0]
        inputs.pop(next(k for k in pair if k != keep), None)
    values = [inputs[key]] * 2
    inputs[key] = "T"
    actual = result(STATTABSession(name), request_line(name, compute, inputs), table=values)
    assert actual.values.shape[0] == 2
    assert actual.parameters[key].tolist() == values


def test_limits_and_unsupported_distribution_requests():
    for n in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            STATTABSession("normal", max_table_size=n)
        with pytest.raises(ValueError):
            parse_stattab_request("normal", "1 0 1 ? .", max_length=n)
    with pytest.raises(ValueError):
        parse_stattab_request("normal", "1 0 1 ? .", max_length=3)
    with pytest.raises(ValueError):
        parse_stattab_request("normal", None)
    with pytest.raises(ValueError):
        parse_stattab_request("bad", "HELP")
    with pytest.raises(ValueError):
        parse_stattab_request("f", "1 ? 4 . 0.5")
    with pytest.raises(ValueError):
        parse_stattab_request("nc_f", "1 4 ? 2 . 0.5")
    session = STATTABSession("normal", max_table_size=101)
    assert result(session, "T 0 1 ? .", table=np.arange(101)).values.shape == (101, 6)


def test_noncentral_t_bracket_execution():
    session = STATTABSession("nc_t")
    a = result(session, "1 ? 3 0.03 .", df_bracket=(0.001, 0.2))
    b = result(session, "1 ? 3 0.03 .", df_bracket=(0.2, 100))
    assert a.parameters["df"] < 0.2 < b.parameters["df"]
    with pytest.raises(ValueError):
        session.execute("1 ? 3 0.03 .", df_bracket=([0.001, 0.2], [0.2, 100]))


def test_failed_table_and_root_search_do_not_commit_partial_state():
    session = STATTABSession("normal")
    first = result(session, "1 0 1 ? .")
    previous = session.previous
    with pytest.raises(ValueError):
        session.execute("1 0 T ? .", table=[1, -1])
    assert session.previous is previous and session.last_result is first
    session = STATTABSession("nc_t")
    first = result(session, "1 5 0.5 ? .")
    previous = session.previous
    with pytest.raises(ValueError):
        session.execute("1 ? 3 0.99 .", df_bracket=(4.9, 5.1))
    assert session.previous is previous and session.last_result is first


def test_smallest_representable_input_and_request_resource_limit():
    parsed = parse_stattab_request("normal", "? 0 1 . 5e-324")
    assert parsed.parameters["ccum"] == np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError):
        parse_stattab_request("normal", "HELP #" + ("x" * 4096))


@pytest.mark.parametrize("line", ["1 0 1 ? '.", '1 0 1 ? ".', "? 0 1 . 'T", '? 0 1 . "='])
def test_unterminated_quoted_placeholders_are_rejected(line):
    with pytest.raises(ValueError):
        parse_stattab_request("normal", line)
