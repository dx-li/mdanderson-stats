import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_regression


def test_original_cloglog_linear_and_quadratic_likelihood():
    x = asypow_smo_regression(
        [-0.5, 0.4],
        [-1, 0, 1, 2],
        family="cloglog",
        constraints=[1, 2, 0],
        lower=-3,
        upper=3,
        observations=[1, 2, 3, 4],
    )
    assert_allclose(x.null_parameters, [-0.089424265273979037, 0], atol=1e-8)
    assert_allclose(x.divergence_per_observation, 0.080851511919384444, rtol=1e-12)
    assert_allclose(x.sample_size(), 109.44582604898264, rtol=1e-12)
    assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
    quadratic = asypow_smo_regression(
        [[-0.7, 0.3, -0.1], [0.2, -0.4, 0.15]],
        [[-1, 0, 1, 2], [0, 1, 2, 3]],
        family="cloglog",
        constraints=[[1, i, 0] for i in range(1, 7)],
        lower=-2,
        upper=2,
        observations=[[1, 2, 3, 4], [4, 3, 2, 1]],
        group_size=[1, 2],
    )
    assert_allclose(quadratic.divergence_per_observation, 0.07424557446553437, rtol=1e-12)


def test_cloglog_rare_events_match_poisson_limit():
    kwargs = dict(
        covariates=[-1, 0, 1, 2],
        constraints=[1, 2, 0],
        lower=[-603, -3],
        upper=[-597, 3],
        observations=[1, 2, 3, 4],
    )
    cloglog = asypow_smo_regression([-600.5, 0.4], family="cloglog", **kwargs)
    poisson = asypow_smo_regression([-600.5, 0.4], family="poisson", **kwargs)
    assert_allclose(cloglog.null_parameters, poisson.null_parameters, atol=1e-8)
    assert_allclose(
        cloglog.divergence_per_observation, poisson.divergence_per_observation, rtol=1e-11, atol=0
    )


def test_cloglog_probabilities_rounding_to_one_retain_failure_information():
    from decimal import Decimal, localcontext

    eta = 4 + 0.2 * np.array([-1, 0, 1, 2])
    with localcontext() as ctx:
        ctx.prec = 100
        survival = [(-Decimal(float(value)).exp()).exp() for value in eta]
        mean = sum(survival) / 4
        expected_intercept = float((-mean.ln()).ln())
        expected_w = float(
            sum((1 - s) * ((1 - s) / (1 - mean)).ln() + s * (s / mean).ln() for s in survival) / 2
        )
    x = asypow_smo_regression(
        [4, 0.2],
        [-1, 0, 1, 2],
        family="cloglog",
        constraints=[1, 2, 0],
        lower=[3, -0.5],
        upper=[5, 0.5],
    )
    assert_allclose(x.null_parameters, [expected_intercept, 0], atol=1e-8)
    assert_allclose(x.divergence_per_observation, expected_w, rtol=1e-10, atol=0)
