import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    asypow_information,
    asypow_ordinal_information,
    asypow_ordinal_regression_information,
    asypow_regression_information,
)


def test_raw_ordinal_multinomial_information():
    p = np.array([0.2, 0.6])
    derivative = np.array([[1, 0], [-1, 1], [0, -1]])
    expected = derivative.T @ np.diag(1 / np.diff(np.r_[0, p, 1])) @ derivative
    assert_allclose(asypow_ordinal_information(p), expected, rtol=1e-14)
    grouped = asypow_ordinal_information([p, [0.3, 0.8]], group_size=[1, 3])
    assert_allclose(grouped[:2, :2], expected / 4, rtol=1e-14)
    assert_allclose(grouped[:2, 2:], 0)
    with pytest.raises(ValueError, match="increase strictly"):
        asypow_ordinal_information([0.2, 0.2])


@pytest.mark.parametrize("link", ["logistic", "cloglog"])
def test_original_r_ordinal_quadratic_information(link):
    references = json.loads((Path(__file__).parent / "fixtures/asypow-ordinal.json").read_text())
    theta = [[-1, 0.5, 0.2, -0.1], [-0.7, 0.8, -0.3, 0.1]]
    info = asypow_ordinal_regression_information(
        theta,
        [[-1, 0, 1, 2], [0, 1, 2, 3]],
        quadratic=True,
        link=link,
        observations=[[1, 2, 3, 4], [4, 3, 2, 1]],
        group_size=[1, 2],
    )
    assert_allclose(info, references[link], rtol=1e-13, atol=1e-15)
    power = asypow_information(np.ravel(theta), info, [0, 0, 1, 0, 0, 0, -1, 0])
    assert_allclose(power.power(power.sample_size()), 0.8, atol=1e-12)


def test_binary_reduction_and_tail_information():
    for link in ("logistic", "cloglog"):
        for theta, x in (([0.3, -0.2], [-1, 0, 1]), ([-1000, 0], [1e200])):
            ordinal = asypow_ordinal_regression_information(theta, x, link=link)
            binary = asypow_regression_information(theta, x, family=link)
            assert_allclose(ordinal, binary, rtol=1e-12, atol=0)
    tail = asypow_ordinal_regression_information([40, 41, 0], [-1, 0, 1])
    assert np.all(np.linalg.eigvalsh(tail) > 0)
    with pytest.raises(ValueError, match="intercepts"):
        asypow_ordinal_regression_information([1, 0, 0.2], [0, 1])
