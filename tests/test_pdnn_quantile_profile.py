"""Tests for PerfectMatch's five per-array QC quantiles."""

import numpy as np
import pytest

from mdanderson_stats.pdnn_quantile_profile import pdnn_array_quantiles


def test_array_quantiles_use_linear_interpolation_and_keep_sample_order():
    intensities = np.array([[8, 21], [1, 13], [6, 34], [4, 55], [9, 89]], dtype=float)
    original = intensities.copy()
    profile = pdnn_array_quantiles(intensities, sample_names=("A.CEL", "B.CEL"))

    assert profile.sample_names == ("A.CEL", "B.CEL")
    assert profile.probabilities == (0.02, 0.25, 0.5, 0.75, 0.98)
    np.testing.assert_allclose(
        profile.values,
        np.vstack(
            [
                np.quantile(intensities[:, column], profile.probabilities, method="linear")
                for column in range(intensities.shape[1])
            ]
        ),
    )
    np.testing.assert_array_equal(intensities, original)
    assert profile.values.flags.writeable is False


def test_single_probe_ties_and_large_finite_intensities():
    one_probe = pdnn_array_quantiles([[7.0, 1e308]])
    np.testing.assert_array_equal(one_probe.values, [[7.0] * 5, [1e308] * 5])

    tied = pdnn_array_quantiles([[1, 3], [1, 3], [1, 3]])
    np.testing.assert_array_equal(tied.values, [[1] * 5, [3] * 5])


def test_invalid_or_oversized_matrix_is_rejected_before_materialization():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        pdnn_array_quantiles([[1.0, -1.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        pdnn_array_quantiles([[1.0, np.nan]])
    with pytest.raises(ValueError, match="rectangular"):
        pdnn_array_quantiles([[1.0, 2.0], [3.0]])
    oversized = np.lib.stride_tricks.as_strided(
        np.array([1.0]), shape=(1_000_001, 2), strides=(0, 0)
    )
    with pytest.raises(ValueError, match="workspace limit"):
        pdnn_array_quantiles(oversized)
    with pytest.raises(ValueError, match="one label per sample"):
        pdnn_array_quantiles([[1, 2]], sample_names=("one",))
