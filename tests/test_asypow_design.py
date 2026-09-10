import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_design_information, asypow_information, asypow_reparameterize


@pytest.mark.parametrize(
    "model,theta,expected",
    [
        (
            "logistic",
            [-0.3, 0.4, 0.2],
            [
                [0.23554445674300045, 0.23001182336304385, 0.13856035735582198],
                [0.23001182336304385, 0.45172469665615284, 0.1773702986344872],
                [0.13856035735582198, 0.1773702986344872, 0.13856035735582198],
            ],
        ),
        (
            "loglinear",
            [0.2, 0.8, 1.1],
            [
                [5.31089487793398, 0.96817637669593, 0.55436183927804],
                [0.96817637669593, 0.551052474062251, 0.148975791433892],
                [0.55436183927804, 0.148975791433892, 0.100793061686916],
            ],
        ),
    ],
)
def test_original_r_general_design(model, theta, expected):
    x = np.column_stack((np.ones(4), [-1, 0, 1, 2], [0, 1, 0, 1]))
    info = asypow_design_information(theta, x, model=model, observations=[1, 2, 3, 4])
    assert_allclose(info, expected, rtol=1e-13, atol=1e-15)
    null = 0 if model == "logistic" else 1
    power = asypow_information(theta, info, [0, 1, 0], null)
    assert_allclose(power.power(power.sample_size()), 0.8, atol=1e-12)


def test_multiplicative_probability_limits_and_unused_points():
    rare = asypow_design_information([1e-300], [[1]], model="loglinear")
    assert_allclose(rare, [[1e300]], rtol=1e-13)
    p = np.nextafter(1.0, 0.0)
    near_one = asypow_design_information([p], [[1]], model="loglinear")
    assert_allclose(near_one, [[1 / (p * (1 - p))]], rtol=1e-13)
    assert_allclose(
        asypow_design_information([0.2], [[1], [0]], model="loglinear", observations=[1, 0]),
        [[6.25]],
    )
    with pytest.raises(ValueError, match="below one"):
        asypow_design_information([2], [[1]], model="loglinear")


def test_delta_information_and_reduced_parameters():
    info = np.array([[3.0, 1.0], [1.0, 4.0]])
    theta = np.array([0.2, 0.8])
    transformed = asypow_reparameterize(info, np.diag(1 / theta))
    assert_allclose(transformed, [[0.12, 0.16], [0.16, 2.56]], atol=1e-14)
    assert_allclose(asypow_reparameterize(info, [1, -1]), [[11 / 9]], atol=1e-14)
    for scale in (1e-100, 1e100):
        x = asypow_reparameterize(info, np.diag([scale, 1 / scale]))
        assert_allclose(x * np.array([[scale**2, 1], [1, 1 / scale**2]]), info, rtol=1e-13)
    with pytest.raises(ValueError, match="dependent"):
        asypow_reparameterize(info, [[1, 1], [2, 2]])
    assert not transformed.flags.writeable
