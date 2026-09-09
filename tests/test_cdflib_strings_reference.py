"""Independent string rules and explicit evidence of native lexical defects."""

import json
from decimal import Decimal
from pathlib import Path

import pytest

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_strings.json").read_text())
ASCII = [r for r in FIXTURE["lexical"] if r["label"].startswith("ascii-")]
ORDINARY = {r["label"]: r for r in FIXTURE["lexical"] if r["label"].startswith("ordinary-")}
NUMERIC = {r["label"]: r for r in FIXTURE["lexical"] if r["label"].startswith("numeric-")}


def tokens(row):
    return [(t["kind"], "".join(map(chr, t["codes"]))) for t in row["tokens"]]


@pytest.mark.parametrize("code,lower,upper", FIXTURE["conversion"][0]["result"])
def test_independent_ascii_only_character_mapping(code, lower, upper):
    assert lower == (code + 32 if 65 <= code <= 90 else code)
    assert upper == (code - 32 if 97 <= code <= 122 else code)


@pytest.mark.parametrize("case", FIXTURE["conversion"][1:])
def test_whole_strings_preserve_bytes_and_length(case):
    assert case["execution_outcome"] == "completed"
    expected = [
        [c + 32 if 65 <= c <= 90 else c for c in case["codes"]],
        [c - 32 if 97 <= c <= 122 else c for c in case["codes"]],
    ]
    assert case["result"] == expected


@pytest.mark.parametrize("case", ASCII)
def test_independent_single_ascii_token_classes(case):
    code = case["codes"][0]
    char = chr(code)
    assert case["execution_outcome"] == "completed" and case["termination"] == ["END F"]
    if char == " ":
        assert case["tokens"] == []
        return
    if "0" <= char <= "9":
        kind = "IN"
        assert case["tokens"][0]["integer"] == int(char)
        assert case["tokens"][0]["real"] == int(char)
    elif "a" <= char <= "z" or "A" <= char <= "Z":
        kind = "ID"
    elif char in "()[]{},:;":
        kind = "DL"
    elif char in "+-*/<>=":
        kind = "OP"
    elif char in "._$\"'":
        kind = "OS"
    else:
        kind = "UC"
    # An unmatched quote includes the synthetic trailing blank in native CVAL.
    expected = " " if char in "\"'" else char
    assert tokens(case) == [(kind, expected)]
    assert not case["tokens"][0]["qstart"]


@pytest.mark.parametrize(
    "index,expected",
    [
        (2, [("IN", "0"), ("IN", "1"), ("IN", "-2"), ("IN", "+3")]),
        (3, [("RL", v) for v in ["1.25", "-0.5", ".25", "5.", "1e3", "1D-3"]]),
        (4, [("ID", v) for v in ["abc", "ABC", "a1", "a.b", "a_b", "a$b"]]),
        (5, [("DL", v) for v in "()[]{},:;"]),
        (16, [("OS", v) for v in ["123abc", ".", "_", "$"]]),
        (19, [("UC", "\t\r\n")]),
        (
            20,
            [
                ("IN", "1"),
                ("DL", ","),
                ("IN", "2"),
                ("DL", ";"),
                ("IN", "3"),
                ("DL", ":"),
                ("IN", "4"),
            ],
        ),
        (21, [("IN", "+12"), ("RL", "-.25")]),
        (22, [("ID", "a"), ("UC", "\0"), ("ID", "b")]),
    ],
)
def test_command_language_token_boundaries(index, expected):
    row = ORDINARY[f"ordinary-{index}"]
    assert tokens(row) == expected
    for t in row["tokens"]:
        if t["kind"] in ("IN", "RL"):
            text = "".join(map(chr, t["codes"])).replace("D", "e")
            expected_value = Decimal(text)
            assert t["real"] == pytest.approx(float(expected_value), rel=3e-14, abs=0)
            assert t["integer"] == int(expected_value)


def test_executable_sign_rules_differ_from_header():
    assert tokens(ORDINARY["ordinary-7"]) == [
        ("ID", "a"),
        ("OP", "+"),
        ("IN", "2"),
        ("ID", "a"),
        ("OP", "-"),
        ("IN", "2"),
        ("ID", "a"),
        ("IN", "+2"),
        ("ID", "a"),
        ("IN", "-2"),
    ]
    assert tokens(ORDINARY["ordinary-8"])[-3:] == [("ID", "x"), ("OP", "=-"), ("IN", "4")]
    assert tokens(ORDINARY["ordinary-6"])[0] == ("OP", "+-")


@pytest.mark.parametrize("index,correct,broken", [(12, "don't", "don''"), (13, 'a"b', 'a""')])
def test_doubled_quote_data_loss_and_spurious_empty_token(index, correct, broken):
    row = ORDINARY[f"ordinary-{index}"]
    assert tokens(row) == [("QS", broken), ("OS", "")]
    assert broken != correct  # The documented doubled-delimiter rule loses data.


def test_missing_exponent_is_not_a_valid_number():
    row = ORDINARY["ordinary-17"]
    assert tokens(row)[:3] == [("RL", "1e"), ("RL", "1e+"), ("RL", "1e-")]
    assert all(t["real"] == 0 for t in row["tokens"][:3])
    for _, value in tokens(row)[:3]:
        with pytest.raises(ValueError):
            float(value)


def test_integer_and_double_overflow_contract_discrepancies():
    row = NUMERIC["numeric-0"]
    assert [t["overflow"] for t in row["tokens"]] == [0, 1, 1, 1]
    assert int("-2147483648") == -(2**31)
    assert row["tokens"][2]["integer"] == 0  # Fits signed int32, but magnitude test rejects it.
    large = NUMERIC["numeric-1"]["tokens"]
    assert large[-1]["real"] == large[-2]["real"]
    assert Decimal("9007199254740993") != Decimal.from_float(large[-1]["real"])
    overflow = NUMERIC["numeric-2"]["tokens"]
    assert [t["overflow"] for t in overflow] == [1, 1, 1, 3]
    assert overflow[-1]["real"] == 0  # Source uses undocumented code 3 for exponent overflow.
    assert NUMERIC["numeric-7"]["tokens"][0]["real"] == "inf"
    assert NUMERIC["numeric-7"]["tokens"][0]["overflow"] == 2


def test_subnormal_rounding_and_exponent_runtime():
    row = NUMERIC["numeric-3"]
    for token in row["tokens"]:
        text = "".join(map(chr, token["codes"]))
        assert token["real"] == pytest.approx(float(Decimal(text)), rel=3e-14, abs=5e-324)
    assert NUMERIC["numeric-6"]["execution_outcome"] == "timeout"
    assert NUMERIC["numeric-6"]["timeout_seconds"] == 3
    # The exact decimal value is below half a float subnormal; a future
    # parser can resolve this without a billion repeated multiplications.
    assert float(Decimal("1e-1000000000")) == 0


@pytest.mark.parametrize(
    "case", [r for r in FIXTURE["lexical"] if r["execution_outcome"] == "process_error"]
)
def test_explicit_buffer_and_character_domain_failures(case):
    assert case["label"] in ("exact-buffer", "buffer-or-byte-bound")
    assert "bounds" in case["stderr"] or "above upper bound" in case["stderr"]
    assert case["exit_code"] != 0
