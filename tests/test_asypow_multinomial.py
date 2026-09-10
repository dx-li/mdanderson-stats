from fractions import Fraction

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    asypow_group_information,
    asypow_information,
    asypow_multinomial_information,
    asypow_ordinal_information,
    asypow_reparameterize,
)


def test_expected_category_scores_and_implicit_last_category():
    p = np.array([0.2, 0.3, 0.1])
    full = np.r_[p, 1 - p.sum()]
    scores = np.vstack((np.diag(1 / p), -np.ones((1, 3)) / full[-1]))
    expected = scores.T @ (full[:, None] * scores)
    info = asypow_multinomial_information(p)
    assert_allclose(info, expected, rtol=1e-14)
    # Native K>2 output is diag(1/p), omitting the last category's contribution.
    assert_allclose(info - np.diag(1 / p), np.full((3, 3), 2.5), atol=1e-14)
    assert_allclose(full @ scores, 0, atol=1e-15)


def test_equivalent_ordinal_parameterization_and_binary_reduction():
    p = np.array([0.2, 0.3, 0.1])
    info = asypow_multinomial_information(p)
    jacobian = np.tril(np.ones((3, 3)))
    ordinal = asypow_ordinal_information(np.cumsum(p))
    assert_allclose(asypow_reparameterize(info, jacobian), ordinal, rtol=1e-14, atol=1e-14)
    a = asypow_information(p, info, [1, 0, 0], 0.1)
    b = asypow_information(np.cumsum(p), ordinal, [1, 0, 0], 0.1)
    assert_allclose(a.power(100), b.power(100), atol=1e-14)
    assert_allclose(
        asypow_multinomial_information([[0.2], [0.7]], group_size=[1, 3]),
        asypow_group_information([0.2, 0.7], group_size=[1, 3]),
        rtol=1e-14,
    )


def test_allocations_and_invalid_simplex():
    near_boundary = [np.nextafter(np.nextafter(1.0, 0.0), 0.0), 1e-16]
    residual = float(1 - sum(Fraction(float(p)) for p in near_boundary))
    near = asypow_multinomial_information(near_boundary)
    assert_allclose(near[0, 1], 1 / residual, rtol=1e-14)
    p = [[0.2, 0.3], [0.1, 0.4]]
    a = asypow_multinomial_information(p, group_size=[1, 3])
    b = asypow_multinomial_information(p, group_size=[1e300, 3e300])
    assert_allclose(a, b, rtol=1e-13)
    assert_allclose(a[:2, 2:], 0)
    with pytest.raises(ValueError, match="sum to less"):
        asypow_multinomial_information([0.5, 0.5])
    assert not a.flags.writeable
