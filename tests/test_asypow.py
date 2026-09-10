import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_group_information, asypow_information


def test_original_r_group_power_and_accurate_inverse():
    for model, theta, w, power in (
        ("binomial", [0.2, 0.4], 0.05, 0.6087794846454565),
        ("poisson", [2, 4], 1 / 3, 0.99993150447084356),
    ):
        fit = asypow_information(theta, asypow_group_information(theta, model=model), [1, -1])
        assert_allclose(fit.noncentrality_per_observation, w, rtol=1e-14)
        assert_allclose(fit.power(100), power, atol=1e-14)
        size = fit.sample_size(0.8, 0.05)
        assert_allclose(fit.power(size), 0.8, atol=1e-13)
        assert fit.power(np.floor(size)) < 0.8 <= fit.power(np.ceil(size))
    fit = asypow_information([0.2, 0.4], asypow_group_information([0.2, 0.4]), [1, -1])
    assert_allclose(fit.sample_size(), 156.977210186524, rtol=1e-12)
    assert_allclose(fit.null_parameters, [0.28, 0.28], atol=1e-15)


def test_projection_multiple_constraints_and_invariance():
    fit = asypow_information([1, 2], [[2, 1], [1, 2]], [1, 0])
    assert_allclose(fit.null_parameters, [0, 2.5], atol=1e-14)
    assert_allclose(fit.noncentrality_per_observation, 1.5, atol=1e-14)
    base = asypow_information([0.1, 0.2, 0.3], np.eye(3), [[1, -1, 0], [0, 1, -1]])
    for scale in (1e-100, 1e100):
        x = asypow_information(
            np.array([0.1, 0.2, 0.3]) * scale,
            np.eye(3) / scale**2,
            np.array([[1, -1, 0], [0, 1, -1]]) * [[1e-200], [1e200]],
        )
        assert_allclose(
            x.noncentrality_per_observation, base.noncentrality_per_observation, rtol=1e-13
        )
        assert_allclose(x.null_parameters / scale, base.null_parameters, atol=1e-14)
    alpha = base.significance([100, 200], [0.8, 0.9])
    assert_allclose(base.power([100, 200], alpha), [0.8, 0.9], atol=1e-13)
    with pytest.raises(ValueError, match="dependent"):
        asypow_information([1, 2], np.eye(2), [[1, -1], [2, -2]])
    null = asypow_information([1, 1], np.eye(2), [1, -1])
    assert_allclose(null.power([0, 100], [1e-20, 0.05]), [1e-20, 0.05], rtol=1e-13)
    with pytest.raises(ValueError, match="unattainable"):
        null.sample_size()


def test_exponential_information_small_rates_and_weight_scaling():
    x = asypow_group_information([0.1, 0.2], model="exponential", duration=5, group_size=[1, 2])
    expected = (
        np.array([1 / 3, 2 / 3])
        * (1 + np.expm1(-np.array([0.1, 0.2]) * 5) / (np.array([0.1, 0.2]) * 5))
        / np.array([0.1, 0.2]) ** 2
    )
    assert_allclose(np.diag(x), expected, rtol=1e-14)
    y = asypow_group_information(
        [0.1, 0.2], model="exponential", duration=5, group_size=[1e300, 2e300]
    )
    assert_allclose(x, y, rtol=1e-13)
    # Rate*duration underflows, while information tends to duration/(2*rate).
    z = asypow_group_information([1e-300], model="exponential", duration=1e-100)
    assert_allclose(z[0, 0], 5e199, rtol=1e-13)
