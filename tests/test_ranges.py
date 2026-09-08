import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.ranges import kwrange, range2


def test_original_range_programs():
    cases = json.loads((Path(__file__).parent / "fixtures/ranges.json").read_text())["cases"]
    for case in cases:
        if case["program"] == "range2":
            result = range2(
                case["values"],
                case["sizes"],
                case["error_mean_square"],
                case["critical_value"],
                legacy=True,
            )
        else:
            result = kwrange(
                case["values"],
                case["sizes"],
                case["critical_value"],
                rank_sums=case["rank_sums"],
                legacy=True,
            )
        np.testing.assert_array_equal(result.order, case["order"])
        np.testing.assert_array_equal(result.reject, case["reject"])
        assert result.similarity_groups == tuple(map(tuple, case["similarity_groups"]))


def test_range2_independent_pairwise_calculation():
    means, sizes, mse = [9, 1, 4], [3, 20, 5], 2
    result = range2(means, sizes, mse, 8)
    for i, j in itertools.product(range(3), repeat=2):
        q = abs(means[i] - means[j]) / math.sqrt(mse * (1 / sizes[i] + 1 / sizes[j]) / 2)
        assert result.statistic[i, j] == pytest.approx(q, rel=1e-14)
        assert result.reject[i, j] == (q > 8)
    # The original loses the association between means and group sizes when sorting.
    legacy = range2(means, sizes, mse, 8, legacy=True)
    assert not result.reject[0, 2]
    assert legacy.reject[0, 2]


def test_range2_does_not_suppress_significant_interior_comparisons():
    means, sizes = [0, 0.2, 0.8, 1], [1, 100, 100, 1]
    result = range2(means, sizes, 1, 1.5)
    legacy = range2(means, sizes, 1, 1.5, legacy=True)
    assert result.reject[1, 2]
    assert result.statistic[1, 2] == pytest.approx(6)
    assert not legacy.reject[1, 2]
    assert legacy.similarity_groups == ((0, 1, 2, 3),)


def test_permutation_invariance_of_corrected_comparisons():
    means = np.array([9, 1, 4, 4, 8])
    sizes = np.array([3, 20, 5, 15, 2])
    original = range2(means, sizes, 2, 4)
    for permutation in itertools.permutations(range(5)):
        perm = np.array(permutation)
        new = range2(means[perm], sizes[perm], 2, 4)
        np.testing.assert_allclose(new.statistic, original.statistic[np.ix_(perm, perm)])
        np.testing.assert_array_equal(new.reject, original.reject[np.ix_(perm, perm)])


def test_similarity_groups_against_exhaustive_contiguous_subsets():
    random = np.random.default_rng(731)
    for _ in range(50):
        means = random.normal(size=8)
        sizes = random.integers(1, 50, size=8)
        result = range2(means, sizes, 1, 3)
        ordered_reject = result.reject[np.ix_(result.order, result.order)]
        candidates = [
            (i, j)
            for i in range(8)
            for j in range(i + 1, 8)
            if not np.any(ordered_reject[i : j + 1, i : j + 1])
        ]
        maximal = [
            (i, j)
            for i, j in candidates
            if not any(a <= i and j <= b and (i, j) != (a, b) for a, b in candidates)
        ]
        expected = tuple(tuple(result.order[i : j + 1]) for i, j in maximal)
        assert result.similarity_groups == expected


def test_kwrange_mean_and_sum_inputs_and_denominator():
    means, sizes = np.array([17, 2, 8.5]), np.array([7, 3, 10])
    result = kwrange(means, sizes, 3.5)
    sums = kwrange(means * sizes, sizes, 3.5, rank_sums=True)
    np.testing.assert_allclose(result.statistic, sums.statistic)
    np.testing.assert_array_equal(result.reject, sums.reject)
    total = sizes.sum()
    for i, j in itertools.combinations(range(3), 2):
        expected = abs(means[i] - means[j]) / math.sqrt(
            total * (total + 1) * (1 / sizes[i] + 1 / sizes[j]) / 12
        )
        assert result.statistic[i, j] == pytest.approx(expected)


def test_critical_equality_and_zero_variance():
    result = range2([0, 1], [1, 1], 1, 1)
    assert result.statistic[0, 1] == pytest.approx(1)
    assert not result.reject[0, 1]
    zero = range2([1, 1, 2], [2, 3, 4], 0, 3)
    assert zero.statistic[0, 1] == 0
    assert zero.statistic[0, 2] == np.inf
    assert zero.reject[0, 2]
    np.testing.assert_array_equal(np.diag(zero.reject), False)


@pytest.mark.parametrize(
    "values,sizes",
    [
        ([1], [2]),
        ([1, 2], [2]),
        ([[1, 2]], [[2, 2]]),
        ([1, 2], [0, 1]),
        ([1, 2], [-1, 1]),
        ([1, 2], [1.5, 2]),
        ([1, np.nan], [1, 2]),
        ([1, 2], [np.inf, 2]),
    ],
)
def test_invalid_group_inputs(values, sizes):
    for fn, extra in [(range2, (1, 3.5)), (kwrange, (3.5,))]:
        with pytest.raises(ValueError):
            fn(values, sizes, *extra)


@pytest.mark.parametrize("mse,critical", [(-1, 3), (np.nan, 3), (1, 0), (1, -1), (1, np.inf)])
def test_invalid_variance_and_critical_values(mse, critical):
    with pytest.raises(ValueError):
        range2([1, 2], [3, 4], mse, critical)
