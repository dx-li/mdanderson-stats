import numpy as np
import pytest

from mdanderson_stats.wfmm_basis import wfmm_basis, wfmm_inverse, wfmm_transform
from mdanderson_stats.wfmm_selection import (
    wfmm_restore_coefficients,
    wfmm_select_coefficients,
)


def test_partition_selection_preserves_packed_order_and_inverse_fill() -> None:
    basis = wfmm_basis(8, levels=2, filter_length=2)
    curves = np.arange(16.0).reshape(2, 8)
    transformed = wfmm_transform(curves, basis)
    selected = wfmm_select_coefficients(transformed.coefficients, basis, partitions=[0, 2])

    expected_indices = np.array([0, 1, 4, 5, 6, 7])
    np.testing.assert_array_equal(selected.retained_indices, expected_indices)
    np.testing.assert_array_equal(
        selected.coefficient_partition, basis.coefficient_partition[expected_indices]
    )
    np.testing.assert_array_equal(
        selected.coefficient_scale, basis.coefficient_scale[expected_indices]
    )
    np.testing.assert_array_equal(
        selected.coefficients, transformed.coefficients[..., expected_indices]
    )

    restored = wfmm_restore_coefficients(selected.coefficients, selected)
    expected = np.zeros_like(transformed.coefficients)
    expected[..., expected_indices] = transformed.coefficients[..., expected_indices]
    np.testing.assert_array_equal(restored, expected)
    np.testing.assert_array_equal(wfmm_inverse(restored, basis), wfmm_inverse(expected, basis))
    assert not selected.coefficients.flags.writeable
    assert not restored.flags.writeable


def test_explicit_indices_retain_identity_for_posterior_arrays() -> None:
    basis = wfmm_basis(6, transform="identity")
    complete = wfmm_select_coefficients(np.arange(6.0), basis, partitions=[0])
    np.testing.assert_array_equal(complete.retained_indices, np.arange(6))
    np.testing.assert_array_equal(
        wfmm_restore_coefficients(complete.coefficients, complete), np.arange(6.0)
    )

    selected = wfmm_select_coefficients(np.arange(6.0), basis, indices=[5, 1, 3])
    np.testing.assert_array_equal(selected.retained_indices, [1, 3, 5])
    np.testing.assert_array_equal(selected.coefficients, [1.0, 3.0, 5.0])

    posterior = np.arange(24.0).reshape(2, 4, 3)
    restored = wfmm_restore_coefficients(posterior, selected)
    expected = np.zeros((2, 4, 6))
    expected[..., [1, 3, 5]] = posterior
    np.testing.assert_array_equal(restored, expected)


def test_selection_validation_and_restore_preflight() -> None:
    basis = wfmm_basis(8, transform="identity")
    with pytest.raises(ValueError, match="exactly one"):
        wfmm_select_coefficients(np.zeros(8), basis)
    with pytest.raises(ValueError, match="duplicates"):
        wfmm_select_coefficients(np.zeros(8), basis, indices=[1, 1])
    with pytest.raises(ValueError, match="within the original"):
        wfmm_select_coefficients(np.zeros(8), basis, indices=[8])
    with pytest.raises(ValueError, match="within the original"):
        wfmm_select_coefficients(np.zeros(8), basis, indices=[-1])
    with pytest.raises(ValueError, match="finite"):
        wfmm_select_coefficients(np.array([0.0, 1.0, np.nan, 3, 4, 5, 6, 7]), basis, indices=[0])

    large_basis = wfmm_basis(4096, transform="identity")
    one_column = wfmm_select_coefficients(np.zeros(4096), large_basis, indices=[4095])
    with pytest.raises(ValueError, match="bounded cell limit"):
        wfmm_restore_coefficients(np.zeros((501, 1)), one_column)
