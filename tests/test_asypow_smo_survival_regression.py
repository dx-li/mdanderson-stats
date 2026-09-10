import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_regression


def test_native_survival_slope_null_and_quadratic_design():
    x = asypow_smo_regression(
        [-0.5, 0.4],
        [-1, 0, 1, 2],
        family="exponential",
        duration=5,
        constraints=[1, 2, 0],
        lower=-3,
        upper=3,
        observations=[1, 2, 3, 4],
    )
    assert_allclose(x.null_parameters, [-0.13234022960878544, 0], atol=1e-8)
    assert_allclose(x.divergence_per_observation, 0.12104360619579624, rtol=1e-12)
    assert_allclose(x.sample_size(), 73.10473297542525, rtol=1e-12)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
    quadratic = asypow_smo_regression(
        [[-0.7, 0.3, -0.1], [0.2, -0.4, 0.15]],
        [[-1, 0, 1, 2], [0, 1, 2, 3]],
        family="exponential",
        duration=[2, 5],
        constraints=[[1, i, 0] for i in range(1, 7)],
        lower=-2,
        upper=2,
        observations=[[1, 2, 3, 4], [4, 3, 2, 1]],
        group_size=[1, 2],
    )
    assert quadratic.degrees_of_freedom == 6
    assert_allclose(quadratic.divergence_per_observation, 0.07580219137112798, rtol=1e-12)


def test_survival_time_units_and_rare_event_limit():
    def fit(shift, duration):
        return asypow_smo_regression(
            [-0.5 + shift, 0.4],
            [-1, 0, 1, 2],
            family="exponential",
            duration=duration,
            constraints=[1, 2, 0],
            lower=[-3 + shift, -3],
            upper=[3 + shift, 3],
            observations=[1, 2, 3, 4],
        )

    original = fit(0, 5)
    for scale in [1e-200, 1e200]:
        shifted = fit(-np.log(scale), 5 * scale)
        assert_allclose(
            shifted.divergence_per_observation, original.divergence_per_observation, rtol=1e-11
        )
        assert_allclose(
            shifted.null_parameters + [np.log(scale), 0], original.null_parameters, atol=1e-8
        )
    rare = fit(-600, 5)
    poisson = asypow_smo_regression(
        [-0.5, 0.4],
        [-1, 0, 1, 2],
        family="poisson",
        constraints=[1, 2, 0],
        lower=-3,
        upper=3,
        observations=[1, 2, 3, 4],
    )
    # d≈rate*L/2 makes the censored expected likelihood proportional to Poisson's.
    assert_allclose(
        rare.divergence_per_observation / (2.5 * np.exp(-600)),
        poisson.divergence_per_observation,
        rtol=1e-11,
    )
    with pytest.raises(ValueError, match="duration"):
        asypow_smo_regression(
            [0, 1], [0, 1], family="exponential", constraints=[1, 2, 0], lower=-3, upper=3
        )
