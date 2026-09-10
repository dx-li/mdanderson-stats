import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import asypow_smo_ordinal_regression, asypow_smo_regression


def test_original_ordinal_likelihood_and_pooled_null():
    for link, intercepts, divergence in (
        ("logistic", [-0.684429553555737535, 0.784139816287566127], 0.026151127604835622),
        ("cloglog", [-0.677557839554262076, 0.750850928095994186], 0.067370094489672905),
    ):
        fitted = asypow_smo_ordinal_regression(
            [-1, 0.5, 0.3],
            [-1, 0, 1, 2],
            link=link,
            observations=[1, 2, 3, 4],
            constraints=[1, 3, 0],
            lower=[-2, 0, -1],
            upper=[-0.1, 2, 1],
        )
        assert_allclose(fitted.null_parameters, [*intercepts, 0], atol=1e-8)
        assert_allclose(fitted.divergence_per_observation, divergence, rtol=1e-12)
        assert_allclose(fitted.power(fitted.sample_size()), 0.8, atol=1e-12)


def test_binary_reduction_including_rare_categories_and_quadratic_groups():
    for link in ("logistic", "cloglog"):
        for intercept in (-0.5, -600.5):
            kwargs = dict(
                covariates=[-1, 0, 1, 2],
                observations=[1, 2, 3, 4],
                constraints=[1, 2, 0],
                lower=[intercept - 2, -1],
                upper=[intercept + 2, 1],
            )
            ordinal = asypow_smo_ordinal_regression([intercept, 0.4], link=link, **kwargs)
            binary = asypow_smo_regression([intercept, 0.4], family=link, **kwargs)
            assert_allclose(ordinal.null_parameters, binary.null_parameters, atol=1e-8)
            assert_allclose(
                ordinal.divergence_per_observation,
                binary.divergence_per_observation,
                rtol=1e-10,
                atol=0,
            )
        kwargs = dict(
            parameters=[[-0.7, 0.3, -0.1], [0.2, -0.4, 0.15]],
            covariates=[[-1, 0, 1, 2], [0, 1, 2, 3]],
            observations=[[1, 2, 3, 4], [4, 3, 2, 1]],
            group_size=[1, 2],
            constraints=[[1, i, 0] for i in range(1, 7)],
            lower=-2,
            upper=2,
        )
        ordinal = asypow_smo_ordinal_regression(quadratic=True, link=link, **kwargs)
        binary = asypow_smo_regression(family=link, **kwargs)
        assert_allclose(
            ordinal.divergence_per_observation, binary.divergence_per_observation, rtol=1e-12
        )
        assert np.isfinite(ordinal.sample_size())
