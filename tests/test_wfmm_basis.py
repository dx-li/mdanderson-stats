import numpy as np
import pytest

from mdanderson_stats.pinnacle_wavelet import pinnacle_daubechies_filter
from mdanderson_stats.wfmm_basis import wfmm_basis, wfmm_inverse, wfmm_transform


def test_haar_packing_is_hand_computable_and_parseval_holds():
    curve = np.arange(1.0, 9.0)[None, :]
    basis = wfmm_basis(8, transform="wavelet", filter_length=2, levels=2)
    transformed = wfmm_transform(curve, basis)

    expected = np.asarray([[5.0, 13.0, -2.0, -2.0, *([-1 / np.sqrt(2)] * 4)]])
    np.testing.assert_allclose(transformed.coefficients, expected, rtol=0, atol=2e-15)
    np.testing.assert_array_equal(transformed.coefficient_partition, [0, 0, 1, 1, 2, 2, 2, 2])
    np.testing.assert_array_equal(transformed.coefficient_scale, [2, 2, 2, 2, 1, 1, 1, 1])
    np.testing.assert_allclose(
        np.sum(transformed.coefficients**2), np.sum(curve**2), rtol=2e-15, atol=0
    )
    np.testing.assert_allclose(
        wfmm_inverse(transformed.coefficients, basis), curve, rtol=0, atol=5e-15
    )
    assert not transformed.coefficients.flags.writeable


def test_db4_periodic_transform_inverts_and_custom_orthogonal_basis_roundtrips():
    curves = np.random.default_rng(71).normal(size=(4, 32))
    wavelet = wfmm_basis(32, transform="wavelet", filter_length=8, levels=3)
    coefficients = wfmm_transform(curves, wavelet)

    # Independent scalar-loop reference for the periodic even-index analysis
    # and [a_J,d_J,...,d_1] packing convention.
    low_taps, high_taps = pinnacle_daubechies_filter(8)
    reference_rows = []
    for row in curves[:1]:
        current = row.copy()
        details = []
        for _ in range(3):
            half = current.size // 2
            low = np.asarray(
                [
                    sum(low_taps[j] * current[(2 * k + j) % current.size] for j in range(8))
                    for k in range(half)
                ]
            )
            detail = np.asarray(
                [
                    sum(high_taps[j] * current[(2 * k + j) % current.size] for j in range(8))
                    for k in range(half)
                ]
            )
            current = low
            details.append(detail)
        reference_rows.append(np.concatenate((current, *details[::-1])))
    np.testing.assert_allclose(coefficients.coefficients[:1], reference_rows, rtol=0, atol=3e-15)

    np.testing.assert_allclose(
        np.sum(coefficients.coefficients**2, axis=1), np.sum(curves**2, axis=1), rtol=0, atol=2e-12
    )
    np.testing.assert_allclose(
        wfmm_inverse(coefficients.coefficients, wavelet), curves, rtol=0, atol=2e-12
    )

    orthogonal, _ = np.linalg.qr(np.random.default_rng(72).normal(size=(32, 32)))
    custom = wfmm_basis(32, transform="custom", custom_matrix=orthogonal)
    custom_coefficients = wfmm_transform(curves, custom)
    np.testing.assert_allclose(
        wfmm_inverse(custom_coefficients.coefficients, custom), curves, rtol=0, atol=2e-14
    )
    np.testing.assert_array_equal(custom.coefficient_partition, np.zeros(32, dtype=np.int64))


def test_identity_and_level_preflight_are_explicit():
    values = np.arange(12.0).reshape(2, 6)
    identity = wfmm_basis(6, transform="identity")
    np.testing.assert_array_equal(wfmm_transform(values, identity).coefficients, values)
    np.testing.assert_array_equal(wfmm_inverse(values, identity), values)
    with pytest.raises(ValueError, match=r"2\*\*levels must divide"):
        wfmm_basis(12, transform="wavelet", levels=3)
    non_power_of_two = np.random.default_rng(73).normal(size=(2, 12))
    default_basis = wfmm_basis(12, transform="wavelet", filter_length=4)
    assert default_basis.levels == 2
    np.testing.assert_allclose(
        wfmm_inverse(wfmm_transform(non_power_of_two, default_basis).coefficients, default_basis),
        non_power_of_two,
        rtol=0,
        atol=2e-14,
    )
    with pytest.raises(ValueError, match="orthonormal"):
        wfmm_basis(6, transform="custom", custom_matrix=np.ones((6, 6)))
