"""CDFLIB string APIs, native comparisons and independent defect regressions."""

import json
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import pytest

from mdanderson_stats import (
    lower_case_char,
    lower_case_string,
    qlex,
    upper_case_char,
    upper_case_string,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_strings.json").read_text())


def pairs(text):
    return [(t.kind, t.text) for t in qlex(text)]


@pytest.mark.parametrize("code,lower,upper", FIXTURE["conversion"][0]["result"])
def test_native_ascii_case_mapping(code, lower, upper):
    assert lower_case_char(chr(code)) == chr(lower)
    assert upper_case_char(chr(code)) == chr(upper)


@pytest.mark.parametrize("row", FIXTURE["conversion"][1:])
def test_native_whole_string_mapping(row):
    text = "".join(map(chr, row["codes"]))
    assert lower_case_string(text) == "".join(map(chr, row["result"][0]))
    assert upper_case_string(text) == "".join(map(chr, row["result"][1]))


@pytest.mark.parametrize("row", FIXTURE["lexical"])
def test_native_token_contracts_with_documented_repairs(row):
    text = "".join(map(chr, row["codes"]))
    actual = list(qlex(text))
    if row["execution_outcome"] != "completed":
        assert row["label"] in ("exact-buffer", "buffer-or-byte-bound", "numeric-6")
        if row["label"] == "numeric-6":
            assert actual[0].real == 0 and actual[0].underflow
        elif not text.strip(" "):
            assert actual == []
        elif text.startswith("'"):
            assert [(t.kind, t.text) for t in actual] == [("QS", text[1:-1])]
        else:
            expected_kind = (
                "UC"
                if ord(text[0]) > 127
                else ("IN" if text.isdigit() else "DL" if text == "(" else "ID")
            )
            assert [(t.kind, t.text) for t in actual] == [(expected_kind, text)]
        return
    expected = [(t["kind"], "".join(map(chr, t["codes"]))) for t in row["tokens"]]
    label = row["label"]
    repairs = {
        "ordinary-6": [
            ("OP", s) for s in ["+", "-", "*", "/", "<", ">", "=", "<=", ">=", "==", "**"]
        ],
        "ordinary-8": [
            ("DL", "("),
            ("IN", "-2"),
            ("DL", ")"),
            ("DL", "["),
            ("IN", "+3"),
            ("DL", "]"),
            ("ID", "x"),
            ("OP", "="),
            ("IN", "-4"),
        ],
        "ordinary-12": [("QS", "don't")],
        "ordinary-13": [("QS", 'a"b')],
        "ordinary-15": [("OS", "unterminated")],
        "ordinary-17": [("OS", s) for s in ["1e", "1e+", "1e-", "1.2.3"]],
        "ascii-34": [("OS", "")],
        "ascii-39": [("OS", "")],
    }
    if label in repairs:
        expected = repairs[label]
    elif label in ("ordinary-10", "ordinary-11", "ordinary-14"):
        expected = expected[:-1]  # Discard the native uninitialized end-of-input token.
    assert [(t.kind, t.text) for t in actual] == expected
    for token in actual:
        assert token.length == len(token.text)
        assert 0 <= token.start < token.stop <= len(text)
        if token.kind in ("IN", "RL"):
            exact = Decimal(token.text.replace("D", "e").replace("d", "e"))
            value = float(exact)
            if value in (float("inf"), float("-inf")):
                assert token.overflow == 2 and token.real is None and token.integer is None
            else:
                assert token.real == value
                truncated = int(value)
                assert token.integer == (truncated if -(2**31) <= truncated < 2**31 else None)
                assert token.underflow == (value == 0 and exact != 0)
        else:
            assert token.integer is None and token.real is None and token.overflow == 0


@pytest.mark.parametrize(
    "text,expected",
    [
        ("a+2", [("ID", "a"), ("OP", "+"), ("IN", "2")]),
        ("a + 2", [("ID", "a"), ("IN", "+2")]),
        ("x=-4", [("ID", "x"), ("OP", "="), ("IN", "-4")]),
        ("a(-2)", [("ID", "a"), ("DL", "("), ("IN", "-2"), ("DL", ")")]),
        ("1e-2+3", [("RL", "1e-2"), ("OP", "+"), ("IN", "3")]),
        ("+ -", [("OP", "+"), ("OP", "-")]),
        ('"a""b"', [("QS", 'a"b')]),
        ("'x''y''z'", [("QS", "x'y'z")]),
        ('"α\0ß"', [("QS", "α\0ß")]),
        ("α\0ß", [("UC", "α\0ß")]),
        ('"a""', [("OS", 'a"')]),
        ('"last"', [("QS", "last")]),
        ("1e 2D+ .", [("OS", "1e"), ("OS", "2D+"), ("OS", ".")]),
    ],
)
def test_independent_intended_tokens(text, expected):
    assert pairs(text) == expected


def test_independent_streams_restart_spans_and_frozen_results():
    text = "  x= -  2 'a''b'  "
    first, second = qlex(text), qlex("9 z")
    assert next(first).text == "x"
    assert next(second).integer == 9
    assert next(first).text == "="
    assert next(second).text == "z"
    number, quoted = list(first)
    assert text[number.start : number.stop] == "-  2" and number.text == "-2"
    assert text[quoted.start : quoted.stop] == "'a''b'" and quoted.text == "a'b"
    assert len(list(qlex(text))) == 4
    with pytest.raises(FrozenInstanceError):
        number.text = "changed"
    assert list(first) == list(second) == []


@pytest.mark.parametrize(
    "text,value,overflow,underflow",
    [
        ("-2147483648", -2147483648.0, 0, False),
        ("2147483648", 2147483648.0, 1, False),
        ("1e309", None, 2, False),
        ("1e-1000000000", 0.0, 0, True),
        ("0e9999999999999999999999999", 0.0, 0, False),
        ("1e9999999999999999999999999", None, 2, False),
        ("1e-9999999999999999999999999", 0.0, 0, True),
        ("1e-320", 1e-320, 0, False),
    ],
)
def test_numeric_extremes_without_exponent_sized_work(text, value, overflow, underflow):
    token = next(qlex(text))
    assert token.real == value and token.overflow == overflow and token.underflow == underflow


def test_long_tokens_and_ascii_only_unicode_conversion():
    word = "a" * 10000
    assert next(qlex(word)).text == word
    token = next(qlex("9" * 10000))
    assert token.overflow == 2 and len(token.text) == 10000
    assert lower_case_string("ABC ßİΣ\0") == "abc ßİΣ\0"
    assert upper_case_string("abc ßİΣ\0") == "ABC ßİΣ\0"


@pytest.mark.parametrize(
    "function", [qlex, lower_case_char, upper_case_char, lower_case_string, upper_case_string]
)
@pytest.mark.parametrize("value", [None, 1, b"abc", ["a"]])
def test_input_type_validation(function, value):
    with pytest.raises(ValueError):
        function(value)


@pytest.mark.parametrize("function", [lower_case_char, upper_case_char])
@pytest.mark.parametrize("value", ["", "ab"])
def test_character_length_validation(function, value):
    with pytest.raises(ValueError):
        function(value)


@pytest.mark.parametrize("text", ["x=- 4", "x= - 4", "x=-4"])
def test_spaced_sign_after_operator_uses_same_numeric_policy(text):
    assert pairs(text) == [("ID", "x"), ("OP", "="), ("IN", "-4")]


def test_signed_zero_and_truncated_real_int32_boundaries():
    import math

    tokens = list(qlex("-0 -1e-9999 2147483647.5 -2147483648.5"))
    assert math.copysign(1, tokens[0].real) == -1
    assert math.copysign(1, tokens[1].real) == -1 and tokens[1].underflow
    assert tokens[2].integer == 2147483647 and tokens[2].overflow == 0
    assert tokens[3].integer == -2147483648 and tokens[3].overflow == 0
