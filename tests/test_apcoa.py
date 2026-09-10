import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import adjusted_pcoa


def distances(y):
    return np.sqrt(((y[:, None] - y[None, :]) ** 2).sum(axis=2))


def test_euclidean_adjustment_equals_multivariate_regression_residuals():
    rng = np.random.default_rng(147)
    x = rng.normal(size=(20, 2)) + [3, 5]
    y = x @ rng.normal(size=(2, 4)) + rng.normal(size=(20, 4))
    for intercept in [False, True]:
        fit = adjusted_pcoa(distances(y), x, components=None, intercept=intercept)
        design = np.column_stack((np.ones(20), x)) if intercept else x
        centered = y - y.mean(axis=0)
        residual = centered - design @ np.linalg.lstsq(design, centered, rcond=None)[0]
        expected = residual @ residual.T
        np.testing.assert_allclose(fit.adjusted.gram_matrix, expected, atol=2e-13)
        np.testing.assert_allclose(
            fit.adjusted.coordinates @ fit.adjusted.coordinates.T, expected, atol=2e-13
        )
        np.testing.assert_allclose(design.T @ fit.adjusted.coordinates, 0, atol=3e-13)
        assert fit.covariate_rank == design.shape[1]


def test_unmodified_native_r_function_non_euclidean_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/apcoa-native.json").read_text())
    fit = adjusted_pcoa(fixture["distances"], fixture["covariates"], components=None)
    np.testing.assert_allclose(
        fit.adjusted.coordinates @ fit.adjusted.coordinates.T, fixture["positive_gram"], atol=2e-14
    )
    assert fit.adjusted.negative_inertia_fraction == pytest.approx(0.03605176287785776)
    assert np.any(fit.adjusted.eigenvalues < 0)
    assert fit.adjusted.positive_fraction.sum() == pytest.approx(1)
    # Positive coordinates do not reconstruct the signed non-Euclidean kernel.
    assert np.linalg.norm(fit.adjusted.gram_matrix - fixture["positive_gram"]) > 0.01


def test_distance_units_covariate_units_and_dependent_columns():
    rng = np.random.default_rng(147)
    y, x = rng.normal(size=(12, 3)), rng.normal(size=(12, 2))
    d = distances(y)
    base = adjusted_pcoa(d, x, components=None)
    for unit in [1e-200, 1e200]:
        expanded = np.column_stack((x[:, 0] * 1e200, x[:, 1] * 1e-200, x[:, 0], np.zeros(12)))
        fit = adjusted_pcoa(d * unit, expanded, components=None)
        assert fit.covariate_rank == 2
        np.testing.assert_allclose(
            fit.adjusted.scaled_gram_matrix, base.adjusted.scaled_gram_matrix, atol=2e-15
        )
        recovered = fit.adjusted.coordinates / unit
        np.testing.assert_allclose(
            recovered @ recovered.T,
            base.adjusted.coordinates @ base.adjusted.coordinates.T,
            atol=3e-14,
        )
        with pytest.raises(ArithmeticError, match="scaled values"):
            _ = fit.adjusted.eigenvalues
    empty = adjusted_pcoa(d, np.eye(12), components=None)
    assert empty.adjusted.coordinates.shape == (12, 0)
    zero = adjusted_pcoa(np.zeros((3, 3)))
    assert zero.original.coordinates.shape == (3, 0)
    assert zero.original.negative_inertia_fraction == 0
    assert not base.adjusted.coordinates.flags.writeable
    bad = d.copy()
    bad[0, 1] += 0.01
    with pytest.raises(ValueError, match="symmetric"):
        adjusted_pcoa(bad)
