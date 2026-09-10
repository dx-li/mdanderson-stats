import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_regression


@pytest.mark.parametrize(
    "family,null,w,n",
    [
        ("logistic", -0.093281846302436008, 0.037965321010678954, 233.07745789472395),
        ("poisson", -0.027003488090893299, 0.1277553120648481, 69.26412973602655),
    ],
)
def test_native_slope_null_and_inversion(family, null, w, n):
    x = asypow_smo_regression(
        [-0.5, 0.4],
        [-1, 0, 1, 2],
        family=family,
        constraints=[1, 2, 0],
        lower=-3,
        upper=3,
        observations=[1, 2, 3, 4],
    )
    assert_allclose(x.null_parameters, [null, 0], atol=1e-8)
    assert_allclose(x.divergence_per_observation, w, rtol=1e-12)
    assert_allclose(x.sample_size(), n, rtol=1e-12)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
    assert not x.null_parameters.flags.writeable


def test_poisson_small_means_and_zero_observation_points():
    p = np.array([-0.5, 0.4])
    original = asypow_smo_regression(
        p,
        [-1, 0, 1, 2],
        family="poisson",
        constraints=[1, 2, 0],
        lower=-3,
        upper=3,
        observations=[1, 2, 3, 4],
    )
    rare = asypow_smo_regression(
        p + [-600, 0],
        [-1, 0, 1, 2, 1e300],
        family="poisson",
        constraints=[1, 2, 0],
        lower=[-603, -3],
        upper=[-597, 3],
        observations=[1, 2, 3, 4, 0],
    )
    assert_allclose(rare.null_parameters + [600, 0], original.null_parameters, atol=1e-8)
    assert_allclose(
        rare.divergence_per_observation / np.exp(-600),
        original.divergence_per_observation,
        rtol=1e-11,
    )
    with pytest.raises(ValueError, match="identify"):
        asypow_smo_regression([0, 1], [1, 1], constraints=[1, 2, 0], lower=-3, upper=3)


def test_original_quadratic_likelihood_and_group_weighting():
    for family, expected in [("logistic", 0.0353613203150256), ("poisson", 0.105966528307318)]:
        x = asypow_smo_regression(
            [[-0.7, 0.3, -0.1], [0.2, -0.4, 0.15]],
            [[-1, 0, 1, 2], [0, 1, 2, 3]],
            family=family,
            constraints=[[1, i, 0] for i in range(1, 7)],
            lower=-2,
            upper=2,
            observations=[[1, 2, 3, 4], [4, 3, 2, 1]],
            group_size=[1e300, 2e300],
        )
        assert x.degrees_of_freedom == 6
        assert_allclose(x.divergence_per_observation, expected, rtol=1e-12)


def test_nearly_certain_logistic_events_retain_tail_information():
    logistic = asypow_smo_regression(
        [40, 0.4], [-1, 0, 1, 2], constraints=[1, 2, 0], lower=[35, -2], upper=[45, 2]
    )
    poisson = asypow_smo_regression(
        [-40, -0.4],
        [-1, 0, 1, 2],
        family="poisson",
        constraints=[1, 2, 0],
        lower=[-45, -2],
        upper=[-35, 2],
    )
    # In this rare-failure limit Bernoulli KL agrees with Poisson KL to O(exp(-40)).
    assert_allclose(
        logistic.divergence_per_observation, poisson.divergence_per_observation, rtol=1e-11
    )
    assert_allclose(logistic.null_parameters, -poisson.null_parameters, atol=1e-8)
