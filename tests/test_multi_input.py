import json
from pathlib import Path

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import parse_multi_data, read_multi_data
from mdanderson_stats._multi_lexer import tokens

CASES = json.loads((Path(__file__).parent / "fixtures/multi_input.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr(c["line"]))
def test_native_qlex_tokens_and_values(case):
    actual = list(tokens(case["line"]))
    expected = case["tokens"]
    assert [(t.kind, t.text.rstrip()) for t in actual] == [(t["kind"], t["text"]) for t in expected]
    for token, native in zip(actual, expected, strict=True):
        if token.kind in ("IN", "RL") and native["overflow"] < 3:
            value = native["value"]
            if token.kind == "IN" and token.text.startswith("-"):
                value = -value  # Deliberately fix the original signed-integer error.
            assert_allclose(token.value, value, atol=0, rtol=2e-15)


def test_sorted_values_original_order_and_diagnostics():
    result = parse_multi_data('0.8,0.2 0.2\n"name" -1 .1 1e-2')
    assert_allclose(result.pvalues, [0.1, 0.2, 0.2, 0.8])
    assert_array_equal(result.order, [3, 1, 2, 0])
    assert_allclose(result.entered_values, [0.8, 0.2, 0.2, 0.1])
    assert [(w.line, w.column, w.token) for w in result.warnings] == [
        (1, 4, ","),
        (2, 1, "name"),
        (2, 8, "-1"),
        (2, 14, "1e-2"),
    ]
    assert result.warnings[2].reason == "P-value not in range [0,1]"
    assert not result.pvalues.flags.writeable and not result.order.flags.writeable


def test_terminal_quit_and_file_mode_differ():
    text = '.1 .2 .3 .4 "Quit now" .5\n.6'
    result = parse_multi_data(text, terminal=True)
    assert len(result.pvalues) == 4 and not result.warnings
    file = parse_multi_data(text)
    assert len(file.pvalues) == 6 and len(file.warnings) == 1


def test_file_read_and_crlf(tmp_path):
    path = tmp_path / "pvalues.txt"
    path.write_bytes(b".1 .2\r\n.3 .4\r\n")
    assert_allclose(read_multi_data(path).pvalues, [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(FileNotFoundError):
        read_multi_data(tmp_path / "missing")


def test_long_lines_do_not_lose_data_and_limit_is_enforced():
    assert len(parse_multi_data(".1 " * 10000).pvalues) == 10000
    with pytest.raises(ValueError, match="Too many"):
        parse_multi_data(".1 " * 10001)


def test_overflow_underflow_endpoints_and_negative_integer_correction():
    result = parse_multi_data("0 1 -1 -0 .5 1.0e-999999 1.0e999999")
    assert_allclose(result.pvalues, [0, 0, 0, 0.5, 1])
    assert [w.token for w in result.warnings] == ["-1", "1.0e999999"]


@pytest.mark.parametrize("text", ["", ".1 .2 .3", ".1 .2 q .3 .4"])
def test_insufficient_terminal_data(text):
    with pytest.raises(ValueError, match="Too few"):
        parse_multi_data(text, terminal=True)


def test_tabs_are_diagnosed_not_silently_removed():
    result = parse_multi_data(".1\t.2 .3 .4")
    assert len(result.warnings) == 1 and result.warnings[0].token == "\t"


def test_input_validation():
    with pytest.raises(ValueError):
        parse_multi_data(None)
    with pytest.raises(ValueError):
        parse_multi_data(".1 .2 .3 .4", terminal=1)


@pytest.mark.parametrize("case", CASES, ids=lambda c: repr(c["line"]))
def test_rddata_acceptance_from_native_tokens(case):
    accepted = []
    rejected = []
    for token in case["tokens"]:
        value = token["value"]
        if token["kind"] == "IN" and token["text"].startswith("-"):
            value = -value
        if token["kind"] in ("IN", "RL") and token["overflow"] == 0 and 0 <= value <= 1:
            accepted.append(value)
        else:
            rejected.append(token["text"])
    # Ensure every lexical case can also exercise RDDATA's >=4 validation.
    parsed = parse_multi_data(case["line"] + "\n0 0 0 0")
    assert_allclose(parsed.entered_values, accepted + [0] * 4, atol=0, rtol=2e-15)
    assert [w.token.rstrip() for w in parsed.warnings] == rejected
