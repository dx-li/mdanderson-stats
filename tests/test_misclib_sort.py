import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mdanderson_stats import permutation_sort_matrix, permute_matrix, sort_matrix


def test_native_matrix_sort_and_all_permutation_options():
    cases = json.loads((Path(__file__).parent / "fixtures/misclib-sort-native.json").read_text())
    for dtype in (np.float64, np.float32, np.int64):
        a = np.array([[30, 10, 40, 20, 99], [7, 8, 2, 6, 0], [11, 12, 13, 14, 15]], dtype=dtype)
        original = a.copy()
        for case in cases["numeric"]:

            def compare(x, y, irow):
                assert irow == case["mode"]
                assert not x.flags.writeable and not y.flags.writeable
                return x[0] > y[0] if irow == 1 else x[0] + 10 * x[1] > y[0] + 10 * y[1]

            order = permutation_sort_matrix(a, ncol=4, irow=case["mode"], a_gt_b=compare)
            assert_array_equal(order, case["index"])
            out = sort_matrix(a, ncol=4, nrowus=2, irow=case["mode"], a_gt_b=compare)
            assert_array_equal(out, case["sorted"])
            assert out.dtype == dtype and not out.flags.writeable
            for opt, key in ((1, "direct"), (0, "copy"), (-1, "reversed")):
                assert_array_equal(permute_matrix(a, order, ncol=4, nrowus=2, opt=opt), case[key])
        assert_array_equal(a, original)
    assert_array_equal(permutation_sort_matrix(a, ncol=4), cases["numeric"][0]["index"])


def test_native_padded_character_order_and_stable_numeric_ties():
    case = json.loads((Path(__file__).parent / "fixtures/misclib-sort-native.json").read_text())[
        "character"
    ]
    a = np.array([["b", "a", "a!", "a\t", "z"], ["B", "A", "C", "T", "Z"]])
    assert_array_equal(permutation_sort_matrix(a), case["index"])
    out = sort_matrix(a)
    assert [str(x).ljust(2) for x in out.flat] == case["sorted"]
    assert_array_equal(sort_matrix(a, a_gt_b=lambda x, y, row: x[row] > y[row]), out)
    large = 2**60
    a = np.array([[large + 1, large, large + 1], [0, 1, 2]], dtype=np.int64)
    assert_array_equal(permutation_sort_matrix(a), [1, 0, 2])
    assert_array_equal(sort_matrix(a)[1], [1, 0, 2])
    assert_array_equal(permute_matrix(a, [1, 2, 0], opt=-1), a[:, [0, 2, 1]])


def test_permutations_are_checked_and_results_do_not_alias():
    a = np.array([[3, 1, 2], [30, 10, 20]])
    out = permute_matrix(a, [1, 2, 0])
    a[:] = 0
    assert_array_equal(out, [[1, 2, 3], [10, 20, 30]])
    with pytest.raises(ValueError):
        out.setflags(write=True)
    for order in ([0, 0, 2], [0, 1, 3], [0.0, 1.0, 2.0]):
        with pytest.raises(ValueError):
            permute_matrix(a, order)
    assert_array_equal(permute_matrix(a, None, opt=0), a)
    assert permute_matrix(np.empty((2, 0)), []).shape == (2, 0)
    for invalid in ([[1, "2"]], [[1, np.nan]], [1, 2]):
        with pytest.raises(ValueError):
            sort_matrix(invalid)
