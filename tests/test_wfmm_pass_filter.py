"""Source pass flags, independent frequency decomposition and safe failures."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    wfmm_basis,
    wfmm_compress_coefficients,
    wfmm_inverse,
    wfmm_pass_filter,
    wfmm_restore_coefficients,
    wfmm_transform,
)

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-pass-filter.json").read_text())[
    "cases"
]
RUNS = [(c, r) for c in CASES for r in c["runs"]]


@pytest.mark.parametrize("case,run", RUNS)
def test_native_pass_flags_and_original_coefficient_identity(case, run):
    y = np.array(case["Y"])
    basis = wfmm_basis(y.shape[1], levels=case["levels"], filter_length=2)
    d = wfmm_transform(y, basis).coefficients

    def selected():
        filtered = wfmm_pass_filter(d, basis, highpass=run["highpass"], lowpass=run["lowpass"])
        return wfmm_compress_coefficients(
            d, basis, alpha=run["alpha"], t=run["t"], eligible_indices=filtered.retained_indices
        ).selection

    if run["exit_code"]:
        with pytest.raises(ValueError):
            selected()
        return
    result = selected()
    np.testing.assert_array_equal(result.retained_indices, run["indices"])
    np.testing.assert_allclose(result.coefficients, run["D"], atol=3e-14)
    if run["alpha"] == 1:
        assert len(np.unique(result.coefficient_partition)) == run["J"]
    restored = wfmm_restore_coefficients(result.coefficients, result)
    np.testing.assert_array_equal(restored[:, result.retained_indices], result.coefficients)
    np.testing.assert_allclose(
        wfmm_transform(wfmm_inverse(restored, basis), basis).coefficients, restored, atol=4e-14
    )


def test_independent_haar_pair_averages_and_details_reconstruct_split():
    y = np.array([[1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0]])
    basis = wfmm_basis(8, filter_length=2, levels=2)
    d = wfmm_transform(y, basis).coefficients
    low = wfmm_pass_filter(d, basis, lowpass=1)
    high = wfmm_pass_filter(d, basis, highpass=1)
    smooth = wfmm_inverse(wfmm_restore_coefficients(low.coefficients, low), basis)
    detail = wfmm_inverse(wfmm_restore_coefficients(high.coefficients, high), basis)
    expected_smooth = np.repeat(y.reshape(1, 4, 2).mean(axis=-1), 2, axis=-1)
    np.testing.assert_allclose(smooth, expected_smooth, atol=1e-14)
    np.testing.assert_allclose(detail, y - expected_smooth, atol=1e-14)
    np.testing.assert_allclose(smooth + detail, y, atol=1e-14)


@pytest.mark.parametrize("high,low", [(-1, 0), (0, -1), (1.5, 0), (True, 0), (0, 3)])
def test_invalid_flags_fail(high, low):
    with pytest.raises(ValueError):
        wfmm_pass_filter(np.ones((2, 8)), wfmm_basis(8, levels=2), highpass=high, lowpass=low)


def test_nonwavelet_basis_is_not_misinterpreted_as_frequency_bands():
    with pytest.raises(ValueError, match="wavelet"):
        wfmm_pass_filter(np.ones((2, 8)), wfmm_basis(8, transform="identity"))
