import numpy as np
import pytest

from mdanderson_stats.bchm import bchm_borrow, bchm_cluster, bchm_fit


def test_weighted_cluster_returns_raw_allocations_and_d0_floor():
    result = bchm_cluster([1, 2, 8], [15, 18, 20], iterations=16, burn_in=8, d0=0.2, seed=4)
    assert result.result.allocations.shape == (16, 3)
    assert np.all(result.result.similarity >= 0.2)
    assert np.allclose(np.diag(result.result.similarity), 1)


def test_borrowing_keeps_target_samples_and_similarity_floor():
    similarity = np.array([1.0, 0.4, 0.001])
    result = bchm_borrow(
        [1, 2, 8], [15, 18, 20], similarity, target=2,
        draws=8, warmup=2, chains=2, seed=2,
    )
    assert result.samples.shape == (2, 8)
    assert np.all((result.samples >= 0) & (result.samples <= 1))
    assert result.similarity.shape == (3,)


def test_fit_exposes_raw_and_native_rounded_decisions():
    result = bchm_fit(
        [1, 2, 8], [15, 18, 20], iterations=12, burn_in=8,
        draws=8, warmup=2, chains=2, seed=3,
    )
    assert np.allclose(result.native_probability, np.round(result.raw_probability, 3))
    assert np.array_equal(result.decision, result.native_probability > 0.5)
    assert all(not x.flags.writeable for x in (result.posterior_mean, result.raw_probability))


@pytest.mark.parametrize("successes,trials", [([0, 0], [5, 8]), ([5, 8], [5, 8])])
def test_empirical_prior_boundary_is_rejected(successes, trials):
    with pytest.raises(ValueError, match="empirical prior"):
        bchm_cluster(successes, trials)
