"""Exact split membership, alignment, immutable data and empty groups."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mdanderson_stats import exploratory_survival, survival_cutpoint


def test_equality_goes_lower_and_row_ids_preserve_unsorted_input():
    data = survival_cutpoint([3, 1, 4, 2], [1, 0, 1, 1], [2, 1, 2, 3])
    result = data.compare(2)
    assert_array_equal(result.lower_indices, [0, 1, 2])
    assert_array_equal(result.upper_indices, [3])
    expected = exploratory_survival([3, 1, 4], [1, 0, 1])
    assert_allclose(result.lower.survival, expected.survival)
    assert result.upper.n_observations == 1
    assert data.baseline.n_observations == 4


@pytest.mark.parametrize(
    "cut,lower,upper", [(-1, 0, 3), (0, 0, 3), (1, 1, 2), (3, 3, 0), (4, 3, 0)]
)
def test_boundaries_and_empty_groups(cut, lower, upper):
    r = survival_cutpoint([1, 2, 3], [1, 0, 1], [1, 2, 3]).compare(cut)
    assert r.lower_indices.size == lower and r.upper_indices.size == upper
    assert (r.lower is None) == (lower == 0)
    assert (r.upper is None) == (upper == 0)


def test_data_is_copied_and_legacy_ties_are_retained():
    t = np.array([1.0, 1.0, 2.0])
    x = np.array([0.0, 0.0, 1.0])
    data = survival_cutpoint(t, [0, 1, 1], x, legacy=True)
    t[0] = 99
    x[0] = 99
    r = data.compare(0)
    assert_allclose(r.lower.survival, [1, 0])
    assert_array_equal(data.time, [1, 1, 2])
    for array in (data.time, data.status, data.covariate, r.lower_indices, r.upper_indices):
        assert not array.flags.writeable


@pytest.mark.parametrize("covariate", [[1], [1, np.nan], [1, np.inf], [[1, 2]]])
def test_invalid_covariates(covariate):
    with pytest.raises(ValueError):
        survival_cutpoint([1, 2], [1, 0], covariate)


@pytest.mark.parametrize("cut", [np.nan, np.inf, -np.inf])
def test_invalid_cut(cut):
    with pytest.raises(ValueError):
        survival_cutpoint([1, 2], [1, 0], [1, 2]).compare(cut)
