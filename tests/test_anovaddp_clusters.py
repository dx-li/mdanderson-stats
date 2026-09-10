import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import anovaddp_atom_posterior, anovaddp_cluster_sweep


def test_atom_and_full_sweep_against_independent_r():
    ref = json.loads((Path(__file__).parent / "fixtures/anovaddp-clusters.json").read_text())
    c = ref["atom"]
    post = anovaddp_atom_posterior(
        c["values"],
        c["design"],
        residual_covariance=0.25 ** np.abs(np.arange(5)[:, None] - np.arange(5)),
        base_mean=np.full(35, 0.1),
        base_covariance=0.2 ** np.abs(np.arange(35)[:, None] - np.arange(35)),
    )
    assert_allclose(post.mean, c["mean"], atol=1e-13, rtol=0)
    assert_allclose(post.covariance, c["covariance"], atol=1e-13, rtol=0)
    fit = anovaddp_cluster_sweep(
        [[3, 1], [-2, 0], [-1, 1], [4, 2]],
        [[1, 0], [1, 1], [1, -1], [1, 2]],
        [0, 1, 1, 2],
        [[3, 1, 0, 0], [-2, 0, 0.5, 0.2], [4, 2, 0, 0]],
        residual_covariance=[[1, 0.2], [0.2, 0.7]],
        base_mean=np.zeros(4),
        base_covariance=0.2 ** np.abs(np.arange(4)[:, None] - np.arange(4)),
        concentration=1.3,
        seed=70,
    )
    assert np.array_equal(fit.labels, ref["sweep"]["labels"])
    assert_allclose(fit.atoms, ref["sweep"]["atoms"], atol=1e-13, rtol=0)
    assert_allclose(fit.assignment_uniforms, ref["sweep"]["uniforms"], atol=0, rtol=0)
    assert_allclose(fit.atom_normal_draws, ref["sweep"]["normals"], atol=0, rtol=0)
    assert fit.removed == fit.created == 1
    assert np.array_equal(fit.counts, np.bincount(fit.labels))
    with pytest.raises(ValueError):
        fit.atoms[0, 0] = 0


def test_singleton_and_density_underflow():
    single = anovaddp_cluster_sweep(
        [[1]],
        [[1]],
        [0],
        [[0]],
        residual_covariance=[[1]],
        base_mean=[0],
        base_covariance=[[1]],
        concentration=1,
        seed=81,
    )
    assert single.removed == single.created == 1
    assert np.array_equal(single.labels, [0])
    assert np.array_equal(single.counts, [1])
    tail = anovaddp_cluster_sweep(
        [[1000], [1001], [1002]],
        [[1], [1], [1]],
        [0, 0, 0],
        [[0]],
        residual_covariance=[[1]],
        base_mean=[0],
        base_covariance=[[1]],
        concentration=1,
        seed=81,
    )
    assert tail.counts.sum() == 3 and np.all(tail.counts > 0)
    assert np.all(np.isfinite(tail.atoms))
    assert len(tail.counts) == 1 + tail.created - tail.removed
    with pytest.raises(ValueError, match="occupied"):
        anovaddp_cluster_sweep(
            [[1]],
            [[1]],
            [0],
            [[0], [1]],
            residual_covariance=[[1]],
            base_mean=[0],
            base_covariance=[[1]],
            concentration=1,
        )
