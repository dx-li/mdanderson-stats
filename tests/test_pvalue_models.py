import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from mdanderson_stats import clustered_pvalues, order_statistic_diagnostics

REFERENCE = json.loads((Path(__file__).parent / "fixtures/multi.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["order_statistics"])
def test_original_osfit(case):
    result = order_statistic_diagnostics(case["pvalues"])
    np.testing.assert_allclose(
        result.cumulative[result.order], case["sorted_cumulative"], rtol=3e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        result.legacy_transformed_cdf[result.order],
        case["sorted_legacy_transformed_cdf"],
        rtol=3e-12,
        atol=1e-15,
    )
    assert result.combined_score == pytest.approx(case["combined_score"], rel=3e-12)


def test_order_cdf_matches_binomial_tail_by_enumeration():
    values = np.array([0.7, 0.01, 0.9, 0.1, 0.1, 0.3])
    n = values.size
    result = order_statistic_diagnostics(values)
    for rank, index in enumerate(result.order, start=1):
        p = values[index]
        expected = sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(rank, n + 1))
        assert result.cumulative[index] == pytest.approx(expected, rel=1e-14)


def test_legacy_denominator_and_uncapped_score_are_explicit():
    result = order_statistic_diagnostics([0.2, 0.3, 0.4])
    np.testing.assert_allclose(
        result.legacy_transformed_cdf,
        [1 - 0.8**3, 1 - (1 - 0.3 / 0.7) ** 2, 0.4 / 0.6],
    )
    assert order_statistic_diagnostics([1, 1, 1]).combined_score == 3
    tiny = order_statistic_diagnostics([1e-30, 2e-30])
    assert tiny.legacy_transformed_cdf[0] == pytest.approx(2e-30, rel=1e-14, abs=0)


def test_diagnostics_batches_permutations_endpoints():
    values = np.array([[[0, 0.5, 1], [0.4, 0.2, 0.3]], [[1, 1, 1], [0, 0, 0]]])
    result = order_statistic_diagnostics(values)
    assert result.combined_score.shape == (2, 2)
    for i, j in np.ndindex(2, 2):
        single = order_statistic_diagnostics(values[i, j])
        np.testing.assert_array_equal(result.cumulative[i, j], single.cumulative)
        assert result.combined_score[i, j] == single.combined_score
    permutation = [2, 0, 1]
    changed = order_statistic_diagnostics(values[..., permutation])
    np.testing.assert_array_equal(changed.cumulative, result.cumulative[..., permutation])
    assert np.all(np.isfinite(result.legacy_transformed_cdf))


@pytest.mark.parametrize("values", [[], 0.1, [np.nan], [np.inf], [-0.01], [1.01]])
def test_diagnostic_validation(values):
    with pytest.raises(ValueError):
        order_statistic_diagnostics(values)


@pytest.mark.parametrize("rho", [-0.3, 0, 0.6])
def test_cluster_distribution_and_latent_covariance(rho):
    p = clustered_pvalues(60000, 60000, cluster_size=4, correlation=rho, rng=421)
    z = -ndtri(p)
    covariance = np.full((4, 4), rho)
    np.fill_diagonal(covariance, 1)
    for group, mean in ((z[:60000], 0), (z[60000:], 1.96)):
        np.testing.assert_allclose(group.mean(axis=0), mean, atol=0.015)
        np.testing.assert_allclose(np.cov(group, rowvar=False), covariance, atol=0.025)
    # Uniform marginals and independent clusters under the null.
    ordered = np.sort(p[:60000, 0])
    assert np.max(np.abs(ordered - np.arange(1, 60001) / 60000)) < 0.012
    assert abs(np.corrcoef(z[:59999, 0], z[1:60000, 0])[0, 1]) < 0.02
    # Nonlinear transformation changes Pearson correlation of the p-values.
    expected_p_correlation = 6 / np.pi * np.arcsin(rho / 2)
    assert np.corrcoef(p[:60000, :2], rowvar=False)[0, 1] == pytest.approx(
        expected_p_correlation, abs=0.015
    )


def test_one_sided_pvalues_row_order_and_explicit_rng():
    draws = np.random.default_rng(29).standard_normal(8)
    draws[3:] += 1.96
    expected = [math.erfc(z / math.sqrt(2)) / 2 for z in draws]
    actual = clustered_pvalues(3, 5, rng=29)
    np.testing.assert_allclose(actual[:, 0], expected, rtol=2e-14)
    np.testing.assert_array_equal(actual, clustered_pvalues(3, 5, correlation=-1, rng=29))
    rng = np.random.default_rng(29)
    first = clustered_pvalues(10, cluster_size=2, rng=rng)
    second = clustered_pvalues(10, cluster_size=2, rng=rng)
    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, clustered_pvalues(10, cluster_size=2, rng=29))


def test_empty_clusters_large_clusters_and_near_singular_covariance():
    assert clustered_pvalues(0, cluster_size=3).shape == (0, 3)
    assert clustered_pvalues(1, cluster_size=150, rng=1).shape == (1, 150)
    for rho in (np.nextafter(-1 / 3, 0), np.nextafter(1, 0)):
        p = clustered_pvalues(100, cluster_size=4, correlation=rho, rng=21)
        assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("rho", [-0.2, 0, 0.8])
def test_linear_projection_matches_dense_covariance_square_root(rho):
    covariance = np.full((5, 5), rho)
    np.fill_diagonal(covariance, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    z = np.random.default_rng(32).standard_normal((20, 5)) @ root
    z[12:] += 1.96
    result = clustered_pvalues(12, 8, cluster_size=5, correlation=rho, rng=32)
    np.testing.assert_allclose(result, ndtr(-z), rtol=1e-13, atol=2e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"null_clusters": -1},
        {"null_clusters": True},
        {"alternative_clusters": 1.5},
        {"cluster_size": 0},
        {"cluster_size": 2.5},
        {"correlation": np.nan},
        {"correlation": 1.01},
        {"alternative_mean": np.inf},
        {"cluster_size": 4, "correlation": -1 / 3},
        {"cluster_size": 3, "correlation": -0.7},
        {"cluster_size": 2, "correlation": 1},
    ],
)
def test_invalid_cluster_parameters(kwargs):
    with pytest.raises(ValueError):
        clustered_pvalues(**({"null_clusters": 1} | kwargs))
