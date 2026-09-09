"""Independent integer invariants and unchanged C/F77 helper contracts."""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import dcdflib_support as support

REFERENCE = json.loads((Path(__file__).parent / "fixtures/dcdflib_support.json").read_text())
CASES = [
    (language, case) for language, row in REFERENCE["profiles"].items() for case in row["cases"]
]


@pytest.mark.parametrize("language,case", CASES)
def test_native_contracts(language, case):
    op = case["op"]
    if op == "ftnstop":
        assert case["returncode"] == 1 and case["stderr"] == "reference stop\n"
        with pytest.raises(RuntimeError, match="reference stop"):
            support.ftnstop("reference stop")
        return
    assert case["returncode"] == 0 and not case["stderr"]
    fn = getattr(support, op)
    if op in ("ipmpar", "spmpar"):
        result = fn(case["index"])
    elif op == "devlpl":
        result = fn([1, -2, 3, 1e99], case["index"], case["a"])
    elif op in ("fifdint", "fifidint"):
        result = fn(case["a"])
    else:
        result = fn(case["a"], case["b"])
    if op in ("ipmpar", "fifidint", "fifmod"):
        assert int(result) == int(case["stdout"])
    else:
        expected = float(case["stdout"])
        assert struct.pack(">d", float(result)) == struct.pack(">d", expected)


def test_machine_model_is_int32_binary32_binary64():
    assert [support.ipmpar(i) for i in (1, 2, 3)] == [2, 31, np.iinfo(np.int32).max]
    assert support.ipmpar(4) == 2
    for dtype, start in [(np.float32, 5), (np.float64, 8)]:
        info = np.finfo(dtype)
        assert [support.ipmpar(i) for i in range(start, start + 3)] == [
            info.nmant + 1,
            info.minexp + 1,
            info.maxexp,
        ]
    assert support.spmpar(1) == 2.0**-52
    assert support.spmpar(2) == 2.0**-1022 > np.nextafter(0.0, 1.0)
    assert support.spmpar(3) == float.fromhex("0x1.fffffffffffffp+1023")


@pytest.mark.parametrize("index", [0, -1, 11, True, 1.0, "1"])
def test_bad_machine_indices(index):
    with pytest.raises(ValueError):
        support.ipmpar(index)
    with pytest.raises(ValueError):
        support.spmpar(index)


def test_polynomial_prefix_and_batched_horner_contract():
    x = np.array([[-2.0, -0.5], [0.5, 2.0]])
    result = support.devlpl([1, -2, 3, 1e99], 3, x)
    np.testing.assert_array_equal(result, 1 - 2 * x + 3 * x * x)
    np.testing.assert_array_equal(support.devlpl([2], 1, x), np.full(x.shape, 2))
    for a, n in [([], 1), ([1], 0), ([1], 2), ([1], True), ([[1]], 1)]:
        with pytest.raises(ValueError):
            support.devlpl(a, n, 1)
    with pytest.raises(ArithmeticError):
        support.devlpl([1e308, 1e308], 2, 2)


def test_wide_float_truncation_and_checked_integer_conversion():
    values = [-1.75, -0.5, -0.0, 0.5, 1.75, -1e308, 1e308]
    actual = support.fifdint(values)
    assert [int(v) for v in actual] == [math.trunc(v) for v in values]
    assert not np.any(np.signbit(actual[1:4]))
    upper = np.nextafter(2.0**63, 0)
    np.testing.assert_array_equal(
        support.fifidint([-(2.0**63), upper, -1.75, 1.75]), [-(2**63), int(upper), -1, 1]
    )
    for value in (2.0**63, np.nextafter(-(2.0**63), -np.inf), 1e308):
        with pytest.raises(ValueError):
            support.fifidint(value)


def test_remainder_is_exact_for_full_int64_and_uses_dividend_sign():
    edges = [-(2**63), -(2**63) + 1, -5, -1, 0, 1, 5, 2**63 - 2, 2**63 - 1]
    pairs = [(a, b) for a in edges for b in edges if b]
    rng = np.random.default_rng(948)
    pairs += list(
        zip(rng.integers(-(2**63), 2**63 - 1, 1000), rng.integers(-(2**63), 2**63 - 1, 1000))
    )
    a, b = np.asarray(pairs, dtype=np.int64).T
    with np.errstate(all="raise"):
        actual = support.fifmod(a, b)
    for aa, bb, remainder in zip(a, b, actual, strict=True):
        aa, bb, remainder = int(aa), int(bb), int(remainder)
        quotient = (abs(aa) // abs(bb)) * (-1 if (aa < 0) != (bb < 0) else 1)
        assert aa == quotient * bb + remainder
        assert abs(remainder) < abs(bb)
        assert remainder == 0 or (remainder < 0) == (aa < 0)
    assert int(support.fifmod(-(2**63), -1)) == 0
    assert int(support.fifmod(np.uint64(2**63 - 1), 3)) == 1


@pytest.mark.parametrize(
    "a,b",
    [
        (1, 0),
        (1.0, 2),
        (True, 2),
        (2**63, 2),
        (1, -(2**63) - 1),
        (np.array([2**64 - 1], dtype=np.uint64), 2),
    ],
)
def test_remainder_rejects_invalid_integer_inputs(a, b):
    with pytest.raises(ValueError):
        support.fifmod(a, b)


def test_broadcasting_selection_and_output_ownership():
    a = np.array([[-2.0], [0.0], [2.0]])
    b = np.array([-3.0, -0.0, 3.0])
    np.testing.assert_array_equal(support.fifdmax1(a, b), np.maximum(a, b))
    np.testing.assert_array_equal(support.fifdmin1(a, b), np.minimum(a, b))
    np.testing.assert_array_equal(
        support.fifdsign(a, b), [[-2, 2, 2], [-0.0, 0.0, 0.0], [-2, 2, 2]]
    )
    np.testing.assert_array_equal(support.fifmod([[5], [-5]], [2, -2]), [[1, 1], [-1, -1]])
    for result in (
        support.fifdint(a),
        support.fifidint(a),
        support.fifmod([1, 2], 2),
        support.fifdmax1(a, b),
        support.fifdmin1(a, b),
        support.fifdsign(a, b),
        support.devlpl([1, 2], 2, a),
    ):
        assert not result.flags.writeable and not np.shares_memory(result, a)
        with pytest.raises(ValueError):
            result.setflags(write=True)
    assert support.fifmod(np.array([], dtype=np.int64), 2).size == 0


@pytest.mark.parametrize(
    "op", ["fifdint", "fifidint", "fifdmax1", "fifdmin1", "fifdsign", "devlpl"]
)
def test_nonfinite_rejected(op):
    for value in (np.nan, np.inf, -np.inf):
        args = (
            ([1], 1, value)
            if op == "devlpl"
            else (value,)
            if op in ("fifdint", "fifidint")
            else (value, 1)
        )
        with pytest.raises(ValueError):
            getattr(support, op)(*args)


def test_stop_becomes_exception_without_stream_side_effects(capsys):
    with pytest.raises(RuntimeError, match="Fortran STOP"):
        support.ftnstop()
    with pytest.raises(ValueError):
        support.ftnstop(123)
    assert capsys.readouterr() == ("", "")
