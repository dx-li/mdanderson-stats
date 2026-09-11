from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import fit_prt_model, prt_isotonic_projection


def test_random_walk_posterior_against_independent_r_quadrature_and_prior_covariance():
    reference = np.genfromtxt(
        Path(__file__).parent / "fixtures/prt-fit-r.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    fit = fit_prt_model(
        [[3, 2]],
        [[1, 3]],
        prior_mean=0.3,
        prior_variance=1.2,
        draws=3000,
        warmup=1000,
        rng=np.random.default_rng(6910),
    )
    means = np.r_[fit.beta_summary.mean.ravel(), fit.risk_summary.mean.ravel()]
    mcse = np.r_[fit.beta_summary.batch_mean_mcse.ravel(), fit.risk_summary.batch_mean_mcse.ravel()]
    assert np.all(abs(means - reference["mean"]) <= 6 * mcse + 1e-10)
    assert np.max(fit.beta_summary.split_rhat) < 1.02
    assert_allclose(fit.conditional_toxicity[..., -1, :], 0)
    assert not fit.beta.flags.writeable
    prior = fit_prt_model(
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        prior_mean=[-0.5, 0.5],
        prior_variance=[1, 2],
        draws=5000,
        warmup=0,
        rng=np.random.default_rng(6913),
    )
    assert prior.likelihood_evaluations == 0
    for interval, variance in enumerate([1, 2]):
        cov = np.cov(prior.beta[:, :, interval].reshape(-1, 3), rowvar=False)
        assert_allclose(cov, variance * np.minimum.outer([1, 2, 3], [1, 2, 3]), atol=0.15, rtol=0)


def test_full_covariance_isotonic_transform_against_r_and_singular_input():
    reference = np.genfromtxt(
        Path(__file__).parent / "fixtures/prt-isotonic-r.csv", delimiter=",", skip_header=1
    )
    result = prt_isotonic_projection(reference[:, None, :3])
    assert_allclose(result.probability[:, 0], reference[:, 3:], atol=2e-14, rtol=0)
    assert np.all(np.diff(result.probability, axis=-1) >= 0)
    assert not result.probability.flags.writeable
    with pytest.raises(ArithmeticError, match="singular"):
        prt_isotonic_projection(np.ones((30, 2, 3)) * 0.3)
