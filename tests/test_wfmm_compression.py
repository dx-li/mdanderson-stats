import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    wfmm_basis,
    wfmm_compress_coefficients,
    wfmm_inverse,
    wfmm_restore_coefficients,
    wfmm_transform,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/wfmm-native-compression.json").read_text()
)
RUNS = [(case, run) for case in REFERENCE["cases"] for run in case["runs"]]


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["case"])
def test_periodic_haar_matches_original_full_native_transform(case):
    curves = np.asarray(case["Y"])
    basis = wfmm_basis(curves.shape[1], levels=case["levels"], filter_length=2)
    transformed = wfmm_transform(curves, basis)
    np.testing.assert_allclose(transformed.coefficients, case["D_full"], atol=2e-14)
    np.testing.assert_allclose(wfmm_inverse(transformed.coefficients, basis), curves, atol=2e-14)


@pytest.mark.parametrize(
    "case,run",
    RUNS,
    ids=[f"{case['case']}-a{run['alpha']}-t{run['t']}" for case, run in RUNS],
)
def test_compression_matches_native_retained_columns_and_values(case, run):
    coefficients = np.asarray(case["D_full"])
    basis = wfmm_basis(coefficients.shape[1], levels=case["levels"], filter_length=2)
    if "error" in run:
        with pytest.raises(ValueError, match="no coefficients"):
            wfmm_compress_coefficients(coefficients, basis, alpha=run["alpha"], t=run["t"])
        return
    # The native 16-column insertion sort preserves these synthetic ties.
    # This explicit Python policy is not a general promise about std::sort.
    tie_policy = "original_index" if case["case"] == "paperless16" else "reject_split"
    result = wfmm_compress_coefficients(
        coefficients, basis, alpha=run["alpha"], t=run["t"], tie_policy=tie_policy
    )
    np.testing.assert_array_equal(result.selection.retained_indices, run["indices"])
    np.testing.assert_allclose(result.selection.coefficients, run["D"], atol=1e-13)
    expected_energy = np.square(coefficients[:, run["indices"]]).sum(axis=1) / np.square(
        coefficients
    ).sum(axis=1)
    np.testing.assert_allclose(result.energy_fraction, expected_energy, atol=5e-16)
    restored = wfmm_restore_coefficients(result.selection.coefficients, result.selection)
    np.testing.assert_array_equal(restored[:, run["indices"]], run["D"])
    omitted = np.setdiff1d(np.arange(basis.time_count), run["indices"])
    np.testing.assert_array_equal(restored[:, omitted], 0)


def test_strict_cumulative_threshold_excludes_crossing_coefficient_and_exact_equality():
    # Energies 9 and 3 total 12, exactly .75 for the leading coefficient.
    basis = wfmm_basis(4, transform="identity")
    coefficients = np.array([[3, 1, 1, 1], [3, 1, 1, 1]], dtype=float)
    with pytest.raises(ValueError, match="no coefficients"):
        wfmm_compress_coefficients(coefficients, basis, alpha=0.75)
    result = wfmm_compress_coefficients(coefficients, basis, alpha=0.8)
    np.testing.assert_array_equal(result.selection.retained_indices, [0])
    np.testing.assert_array_equal(result.vote_counts, [2, 0, 0, 0])
    np.testing.assert_allclose(result.energy_fraction, 0.75)
    assert np.all(result.energy_fraction < result.alpha)


def test_curve_votes_are_strict_and_not_pooled_energy():
    basis = wfmm_basis(4, transform="identity")
    coefficients = [[3, 1, 1, 1], [1, 3, 1, 1], [3, 1, 1, 1]]
    result = wfmm_compress_coefficients(coefficients, basis, alpha=0.8, t=1)
    np.testing.assert_array_equal(result.vote_counts, [2, 1, 0, 0])
    np.testing.assert_array_equal(result.selection.retained_indices, [0])
    assert result.energy_fraction[1] < 0.1


def test_split_ties_require_explicit_policy():
    basis = wfmm_basis(4, transform="identity")
    coefficients = [[1, -1, 0, 0]]
    with pytest.raises(ValueError, match="splits tied"):
        wfmm_compress_coefficients(coefficients, basis, alpha=0.75)
    result = wfmm_compress_coefficients(
        coefficients, basis, alpha=0.75, tie_policy="original_index"
    )
    np.testing.assert_array_equal(result.selection.retained_indices, [0])
    assert result.tie_policy == "original_index"


def test_bypass_and_zero_energy_have_explicit_diagnostics():
    basis = wfmm_basis(4, transform="identity")
    coefficients = [[0, 0, 0, 0], [3, 1, 1, 1]]
    result = wfmm_compress_coefficients(coefficients, basis, t=2)
    assert result.vote_counts is None
    np.testing.assert_array_equal(result.selection.retained_indices, np.arange(4))
    np.testing.assert_array_equal(result.energy_fraction, [0, 1])
    compressed = wfmm_compress_coefficients(coefficients, basis, alpha=0.8)
    np.testing.assert_array_equal(compressed.vote_counts, [1, 0, 0, 0])
    np.testing.assert_allclose(compressed.energy_fraction, [0, 0.75])


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e300])
def test_energy_calculation_is_scale_invariant_without_squaring_overflow(scale):
    basis = wfmm_basis(4, transform="identity")
    result = wfmm_compress_coefficients(np.array([[3, 1, 1, 1]]) * scale, basis, alpha=0.8)
    np.testing.assert_array_equal(result.selection.retained_indices, [0])
    np.testing.assert_allclose(result.energy_fraction, [0.75])


def test_selection_owns_values_and_preserves_original_partition_metadata():
    coefficients = np.zeros((2, 8))
    coefficients[:, [0, 3]] = [3, 1]
    basis = wfmm_basis(8, levels=2, filter_length=2)
    result = wfmm_compress_coefficients(coefficients, basis, alpha=0.99)
    coefficients[:] = 99
    np.testing.assert_array_equal(result.selection.coefficients, [[3], [3]])
    assert result.selection.original_coefficient_count == 8
    np.testing.assert_array_equal(result.selection.coefficient_partition, [0])
    assert not result.selection.coefficients.flags.writeable
    assert not result.selection.retained_indices.flags.writeable
    assert result.vote_counts is not None and not result.vote_counts.flags.writeable
    assert not result.energy_fraction.flags.writeable


@pytest.mark.parametrize(
    "change",
    [
        {"alpha": 0},
        {"alpha": 1.01},
        {"alpha": float("nan")},
        {"alpha": True},
        {"alpha": 1j},
        {"alpha": [0.9]},
        {"t": -1},
        {"t": 3},
        {"t": 0.5},
        {"t": True},
        {"tie_policy": "native"},
        {"coefficients": [3, 1, 1, 1]},
        {"coefficients": np.ones((2, 3))},
        {"coefficients": np.ones((0, 4))},
        {"coefficients": np.ones((2, 4), dtype=bool)},
        {"coefficients": np.ones((2, 4), dtype=complex)},
        {"coefficients": np.full((2, 4), float("nan"))},
        {"coefficients": np.full((2, 4), float("inf"))},
        {"basis": None},
    ],
)
def test_invalid_inputs_are_rejected(change):
    arguments = {"coefficients": np.ones((2, 4)), "basis": wfmm_basis(4, transform="identity")}
    arguments.update(change)
    with pytest.raises(ValueError):
        wfmm_compress_coefficients(**arguments)


def test_work_limit_is_checked_before_energy_or_sorting(monkeypatch):
    basis = wfmm_basis(4096, transform="identity")
    coefficients = np.broadcast_to(1.0, (489, 4096))

    def forbidden(*args, **kwargs):
        raise AssertionError("sorting called before preflight")

    monkeypatch.setattr(np, "argsort", forbidden)
    with pytest.raises(ValueError, match="cell limit"):
        wfmm_compress_coefficients(coefficients, basis, alpha=0.9)
