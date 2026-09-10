import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import anovaddp_curve, anovaddp_loglikelihood, fit_anovaddp


def test_full_chain_thinning_and_saved_state_consistency():
    times = np.tile([0, 1, 2, 3, 4, 6], 3)
    ids = np.repeat(np.arange(3), 6)
    initial = np.tile([2, 0.5, 1.8, 1, 3, 0.2], (3, 1))
    observations = np.tile(anovaddp_curve(initial[0], [0, 1, 2, 3, 4, 6]), 3) + 0.15 * np.sin(
        np.arange(18)
    )
    options = dict(
        initial_parameters=initial,
        initial_covariance=np.diag([1, 1, 1, 0.2, 0.2, 0.1]),
        base_prior=[0.5, 1.8, 1, 3, 0.2],
        base_covariance=np.eye(5),
        covariance_df=8,
        alpha0=6,
        beta0=2,
        concentration_shape=2,
        concentration_rate=1,
        iterations=12,
        burn_in=4,
        seed=67,
    )
    fit = fit_anovaddp(times, observations, ids, np.ones((3, 1)), **options)
    thinned = fit_anovaddp(times, observations, ids, np.ones((3, 1)), thin=3, **options)
    assert np.array_equal(fit.parameters[::3], thinned.parameters)
    assert np.array_equal(thinned.saved_iterations, [5, 8, 11])
    assert np.all(fit.observation_variance > 0) and np.all(fit.concentration > 0)
    assert np.all(np.linalg.eigvalsh(fit.residual_covariance) > 0)
    assert np.all((fit.acceptance_fraction >= 0) & (fit.acceptance_fraction <= 1))
    assert not np.array_equal(fit.residual_covariance[0], fit.residual_covariance[-1])
    for j in range(len(fit.atoms)):
        assert fit.atoms[j].shape == (fit.cluster_count[j], 5)
        assert np.all(np.bincount(fit.labels[j], minlength=fit.cluster_count[j]) > 0)
        expected = sum(
            float(
                anovaddp_loglikelihood(
                    fit.parameters[j, i],
                    times[ids == i],
                    observations[ids == i],
                    fit.observation_variance[j],
                    normalized=True,
                )
            )
            for i in range(3)
        )
        assert_allclose(fit.log_likelihood[j], expected, atol=1e-13, rtol=0)
    with pytest.raises(ValueError):
        fit.parameters[0, 0, 0] = 0
    with pytest.raises(ValueError):
        fit.atoms[0][0, 0] = 0
    assert np.array_equal(initial, np.tile([2, 0.5, 1.8, 1, 3, 0.2], (3, 1)))


def test_invalid_design_fails_before_sampling():
    with pytest.raises(ValueError, match="each design row"):
        fit_anovaddp(
            [0, 1],
            [2, 2],
            [0, 0],
            [[1], [1]],
            initial_parameters=np.ones((2, 6)),
            initial_covariance=np.eye(6),
            base_prior=np.zeros(5),
            base_covariance=np.eye(5),
            iterations=5,
            burn_in=2,
        )
