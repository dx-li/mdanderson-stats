import numpy as np
import pytest

from mdanderson_stats.bchm import bchm_borrow, bchm_cluster, bchm_fit
from mdanderson_stats.bchm_clustering import _silhouette


def test_weighted_cluster_returns_raw_allocations_and_d0_floor():
    result = bchm_cluster([1, 2, 8], [15, 18, 20], iterations=16, burn_in=8, d0=0.2, seed=4)
    assert result.result.allocations.shape == (16, 3)
    assert np.all(result.result.similarity >= 0.2)
    assert np.allclose(np.diag(result.result.similarity), 1)


def test_borrowing_keeps_target_samples_and_similarity_floor():
    similarity = np.array([1.0, 0.4, 0.001])
    result = bchm_borrow(
        [1, 2, 8],
        [15, 18, 20],
        similarity,
        target=2,
        draws=8,
        warmup=2,
        chains=2,
        seed=2,
    )
    assert result.samples.shape == (2, 8)
    assert np.all((result.samples >= 0) & (result.samples <= 1))
    assert result.similarity.shape == (3,)


def test_fit_exposes_raw_and_native_rounded_decisions():
    result = bchm_fit(
        [1, 2, 8],
        [15, 18, 20],
        iterations=12,
        burn_in=8,
        draws=8,
        warmup=2,
        chains=2,
        seed=3,
    )
    assert np.allclose(result.native_probability, np.round(result.raw_probability, 3))
    assert np.array_equal(result.decision, result.native_probability > 0.5)
    assert all(not x.flags.writeable for x in (result.posterior_mean, result.raw_probability))


@pytest.mark.parametrize("successes,trials", [([0, 0], [5, 8]), ([5, 8], [5, 8])])
def test_empirical_prior_boundary_is_rejected(successes, trials):
    with pytest.raises(ValueError, match="empirical prior"):
        bchm_borrow(successes, trials, np.ones(2))
    cluster = bchm_cluster(successes, trials, iterations=8, burn_in=0, seed=158)
    np.testing.assert_array_equal(np.diag(cluster.result.similarity), 1)


def test_representative_partition_silhouette_counts_singletons_as_zero():
    rates = np.array([0.0, 0.1, 0.9])
    assert _silhouette(np.array([1, 1, 2]), rates) == pytest.approx((8 / 9 + 7 / 8) / 3)
    assert _silhouette(np.array([1, 2, 1]), rates) == pytest.approx(-1 / 3)
    assert _silhouette(np.array([1, 1, 1]), rates) == -0.1
    assert _silhouette(np.array([1, 2, 3]), rates) == -0.1


def test_work_limits_and_invalid_fit_parameters_fail_before_sampling():
    for arguments in ({"draws": 8.5}, {"warmup": True}, {"beta1": -1}, {"thetaT": 2}):
        with pytest.raises(ValueError):
            bchm_fit([1, 2], [10, 10], **arguments)
    with pytest.raises(ValueError, match="work budget"):
        bchm_fit([1] * 20, [10] * 20, draws=10000)
    with pytest.raises(ValueError, match="allocation MCMC budget"):
        bchm_cluster([1] * 20, [10] * 20, iterations=1000000)
    with pytest.raises(ValueError, match="target out of range"):
        bchm_borrow([1, 2], [10, 10], np.eye(2), target=2)


def test_probability_threshold_endpoints_remain_exact_for_logistic_samples():
    for threshold in (0.0, 1.0):
        result = bchm_borrow(
            [0],
            [1],
            [1],
            prior_mean=-1000,
            tau2=1e12,
            phi1=threshold,
            deltaT=0,
            draws=8,
            warmup=8,
            seed=158,
        )
        assert result.probability == 1 - threshold
