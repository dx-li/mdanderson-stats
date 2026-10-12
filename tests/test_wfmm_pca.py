"""Native retained subspaces and independent rectangular reconstruction checks."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    wfmm_basis,
    wfmm_compress_coefficients,
    wfmm_covariance,
    wfmm_inverse,
    wfmm_pca_basis,
    wfmm_restore_coefficients,
    wfmm_select_coefficients,
    wfmm_summarize,
    wfmm_summarize_covariance,
    wfmm_transform,
)

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-pca.json").read_text())["cases"]
SUCCESS = [c for c in CASES if not c["exit_code"]]


@pytest.mark.parametrize("case", SUCCESS, ids=lambda c: c["case"])
def test_native_subspace_scores_reconstruction_and_partitions(case):
    y = np.array(case["y"])
    settings = case["settings"]
    kind = settings["transformtype"]
    wave = None if kind == "PC" else wfmm_basis(y.shape[1], filter_length=2, levels=1)
    result = wfmm_pca_basis(
        y,
        kind=kind,
        alpha=settings["alphaPC"],
        wavelet_basis=wave,
        partition_sizes=None
        if "partitions" not in settings
        else np.array(settings["partitions"], dtype=int),
        npartitions=None if "npartitions" not in settings else int(settings["npartitions"]),
        partition_base=settings.get("PCpartitionbase", 10),
    )
    a = np.array(case["analysis"])
    k = result.retained_components
    assert k == a.shape[1]
    assert result.partition_sizes == tuple(np.atleast_1d(case["basis"]["partitions"]))
    if kind == "wPC":
        a = wfmm_transform(np.eye(y.shape[1]), wave).coefficients @ a
    actual_a = result.basis.custom_matrix[:, :k]
    np.testing.assert_allclose(actual_a @ actual_a.T, a @ a.T, atol=5e-13)
    signs = np.sign(np.sum(a * actual_a, axis=0))
    expected = np.array(case["D"])
    expected[:, :k] *= signs
    d = wfmm_transform(y, result.basis).coefficients
    np.testing.assert_allclose(d, expected, atol=6e-13)
    restored = wfmm_inverse(d, result.basis)
    np.testing.assert_allclose(restored, y if kind == "PCw" else y @ a @ a.T, atol=8e-13)
    assert not result.basis.custom_matrix.flags.writeable
    assert not result.basis.synthesis_matrix.flags.writeable


@pytest.mark.parametrize("kind", ["PC", "PCw", "wPC"])
def test_rectangular_covariance_posterior_and_selection_compose(kind):
    y = np.random.default_rng(22).normal(size=(8, 4))
    wave = None if kind == "PC" else wfmm_basis(4, filter_length=2, levels=1)
    basis = wfmm_pca_basis(y, kind=kind, alpha=0.6, wavelet_basis=wave).basis
    k = basis.coefficient_count
    b = basis.synthesis_matrix
    draws = np.random.default_rng(31).normal(size=(2, 4, 2, k))
    expected = np.einsum("cdpk,kt->cdpt", draws, b)
    summary = wfmm_summarize(draws, basis, retain_curves=True)
    np.testing.assert_allclose(summary.curves, expected, atol=1e-14)
    np.testing.assert_allclose(summary.mean, expected.reshape(8, 2, 4).mean(axis=0), atol=1e-14)
    omega = np.abs(draws) + 0.1
    cov = np.einsum("kt,cdpk,ku->cdptu", b, omega, b)
    np.testing.assert_allclose(wfmm_covariance(omega, basis), cov, atol=1e-14)
    moments = wfmm_summarize_covariance(
        omega, basis, include_covariance=True, retain_covariance_draws=True
    )
    np.testing.assert_allclose(moments.covariance_draws, cov, atol=1e-14)
    np.testing.assert_allclose(
        moments.variance_function_mean,
        np.diagonal(cov, axis1=-2, axis2=-1).mean(axis=(0, 1)),
        atol=1e-14,
    )
    selected = wfmm_select_coefficients(draws, basis, indices=[0])
    restored = wfmm_restore_coefficients(selected.coefficients, selected)
    assert restored.shape == draws.shape
    np.testing.assert_array_equal(restored[..., 0], draws[..., 0])
    np.testing.assert_array_equal(restored[..., 1:], 0)
    compression = wfmm_compress_coefficients(wfmm_transform(y, basis).coefficients, basis)
    assert compression.selection.original_coefficient_count == k


def test_analytic_diagonal_projection_rank_and_full_energy_correction():
    y = np.diag([4.0, 2.0, 1.0, 0.5])
    result = wfmm_pca_basis(y, alpha=0.8)
    assert result.retained_components == 2
    assert result.energy_fraction == pytest.approx(20 / 21.25)
    np.testing.assert_allclose(
        result.basis.custom_matrix @ result.basis.synthesis_matrix,
        np.diag([1, 1, 0, 0]),
        atol=1e-15,
    )
    all_directions = wfmm_pca_basis(y, alpha=1)
    np.testing.assert_allclose(
        wfmm_inverse(wfmm_transform(y, all_directions.basis).coefficients, all_directions.basis),
        y,
        atol=1e-15,
    )
    rank_one = wfmm_pca_basis([[1, 2, 3], [2, 4, 6]], alpha=1)
    assert rank_one.retained_components == rank_one.numerical_rank == 1


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_extreme_scale_retained_subspace_and_energy_are_stable(scale):
    y = np.diag([4.0, 2.0, 1.0, 0.5]) * scale
    result = wfmm_pca_basis(y, alpha=0.8, partition_sizes=[2])
    assert result.retained_components == 2
    np.testing.assert_allclose(
        result.basis.custom_matrix @ result.basis.synthesis_matrix,
        np.diag([1, 1, 0, 0]),
        atol=1e-14,
    )
    assert result.energy_fraction == pytest.approx(20 / 21.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1},
        {"alpha": 1.1},
        {"alpha": True},
        {"alpha": np.nan},
        {"partition_base": 1},
        {"partition_base": np.inf},
        {"partition_sizes": [1, 1]},
        {"npartitions": 0},
        {"kind": "pca"},
        {"kind": "PCw"},
        {"max_work": 1},
    ],
)
def test_invalid_controls_fail(kwargs):
    with pytest.raises(ValueError):
        wfmm_pca_basis(np.diag([4.0, 2.0, 1.0, 0.5]), **kwargs)


def test_unidentified_directions_zero_energy_and_nonfinite_inputs_fail():
    for y in [np.zeros((4, 4)), np.full((4, 4), np.nan), np.eye(4, dtype=complex)]:
        with pytest.raises(ValueError):
            wfmm_pca_basis(y)
    with pytest.raises(ValueError, match="tied"):
        wfmm_pca_basis(np.eye(4), alpha=0.1)


def test_basis_owns_training_results_and_reuses_them_without_refitting():
    y = np.diag([4.0, 2.0, 1.0, 0.5])
    basis = wfmm_pca_basis(y, alpha=0.8).basis
    y[:] = 99
    out = wfmm_transform([[3, 7, 2, 5]], basis).coefficients
    np.testing.assert_array_equal(out, [[3, 7]])


def test_low_rank_dense_covariance_still_preflights_time_space_workspace():
    basis = wfmm_pca_basis(np.ones((2, 1000)), alpha=1).basis
    assert basis.coefficient_count == 1
    # Few retained components do not eliminate the dense T-by-T covariance
    # buffers and symmetrization temporaries.
    with pytest.raises(ValueError, match="live-cell"):
        wfmm_covariance([1.0], basis)
