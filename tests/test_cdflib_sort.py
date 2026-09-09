"""All native sorting overloads and independent ordering/permutation checks."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import sort_list

FIXTURE = json.loads((Path(__file__).parent / "fixtures/cdflib_sort.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["cases"])
def test_native_overloads_and_defect_repairs(case):
    assert FIXTURE["adaptations"] == []
    kind, values, count = case["kind"], case["values"], case["ncol"]
    if kind != 4:
        values = np.array(values, dtype={1: np.float64, 2: np.float32, 3: np.int32}[kind])
    compare = (lambda a, b: a < b) if case["descending"] else None
    result = sort_list(values, count, compare)
    if kind == 4:
        actual = [v.ljust(case["width"]) for v in result]
        original = [v.ljust(case["width"]) for v in values]
        expected = sorted(original[:count], reverse=case["descending"]) + original[count:]
    else:
        actual = result.tolist()
        expected = (
            sorted(values[:count].tolist(), reverse=case["descending"]) + values[count:].tolist()
        )
    assert actual == expected
    if case["execution_outcome"] == "process_error":
        assert case["label"] in ("ordinary-5", "ordinary-6", "ties")
        assert not case["descending"]
        assert "above upper bound" in case["stderr"]
    elif case["label"] == "long-strings":
        assert case["execution_outcome"] == "completed"
        assert Counter(case["result"]) != Counter(original)
        assert Counter(actual) == Counter(original)
    else:
        assert case["execution_outcome"] == "completed"
        assert actual == case["result"]


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("size", [0, 1, 2, 10, 11, 16, 64, 257])
def test_order_permutation_prefix_and_dtype(dtype, size):
    values = np.random.default_rng(47 + size).integers(0, 20, size=size).astype(dtype)
    original = values.copy()
    count = size // 2
    result = sort_list(values, count)
    np.testing.assert_array_equal(
        result, sorted(original[:count].tolist()) + original[count:].tolist()
    )
    assert result.dtype == values.dtype
    values[:] = 99
    np.testing.assert_array_equal(result[count:], original[count:])
    with pytest.raises(ValueError):
        result.setflags(write=True)


@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("strings", [False, True])
def test_custom_ordering_stable_ties_and_long_strings(inclusive, strings):
    values = [31, 12, 21, 22, 11, 32]
    if strings:
        values = [str(v) + "x" * 300 for v in values]

    def key(v):
        return int(str(v)[:2]) % 10

    def compare(a, b):
        return key(a) >= key(b) if inclusive else key(a) > key(b)

    result = sort_list(values, a_gt_b=compare)
    assert list(result) == sorted(values, key=key)


def test_blank_padding_ascii_unicode_and_nuls():
    values = ["a", "a\t", "a ", "a!", "A", "ß", "a\0", "a\0\0"]
    width = max(map(len, values))
    result = sort_list(values)
    assert result == tuple(sorted(values, key=lambda s: s.ljust(width)))
    assert result.index("a\t") < result.index("a") < result.index("a ")
    assert Counter(result) == Counter(values)


def test_integer_precision_and_stable_signed_zero():
    values = np.array([2**63 - 1, 2**53 + 1, 2**53, -(2**63)], dtype=np.int64)
    assert sort_list(values).tolist() == sorted(values.tolist())
    result = sort_list(np.array([0.0, -0.0, 0.0, -0.0]))
    np.testing.assert_array_equal(np.signbit(result), [False, True, False, True])


def test_callback_receives_blank_padded_strings_and_propagates_errors():
    seen = []

    def compare(a, b):
        seen.append((a, b))
        return a > b

    assert sort_list(["b", "a", "long"], 2, compare) == ("a", "b", "long")
    assert all(len(a) == len(b) == 4 for a, b in seen)

    def broken(a, b):
        raise RuntimeError("caller comparator failed")

    with pytest.raises(RuntimeError, match="caller comparator"):
        sort_list([2, 1], a_gt_b=broken)


@pytest.mark.parametrize(
    "values,ncol,compare",
    [
        ("abc", None, None),
        ([[1, 2]], None, None),
        ([1, "2"], None, None),
        ([True, 1], None, None),
        ([1 + 2j], None, None),
        ([np.nan], None, None),
        ([np.inf], None, None),
        ([1], -1, None),
        ([1], 2, None),
        ([1], True, None),
        ([1], 0.5, None),
        ([1], None, 1),
        ([2, 1], None, lambda a, b: 1),
        ([2, 1], None, lambda a, b: [True]),
        (np.array([1], dtype=object), None, None),
    ],
)
def test_invalid_inputs(values, ncol, compare):
    with pytest.raises(ValueError):
        sort_list(values, ncol, compare)
