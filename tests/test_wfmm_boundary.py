"""Original-source impulse maps and independent extended-transform identities."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    WFMMStudy,
    read_wfmm_study,
    run_wfmm_study,
    wfmm_covariance,
    wfmm_extended_wavelet_basis,
    wfmm_inverse,
    wfmm_pass_filter,
    wfmm_transform,
)

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-native-boundary.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"db{c['order']}-{c['size']}-{c['boundary']}")
def test_native_impulse_analysis_synthesis_and_covariance(case):
    n = case["size"]
    basis = wfmm_extended_wavelet_basis(
        n, levels=case["levels"], filter_length=2 * case["order"], boundary=case["boundary"]
    )
    a, b = np.array(case["analysis"]), np.array(case["synthesis"])
    np.testing.assert_allclose(basis.custom_matrix, a, rtol=0, atol=2e-11)
    np.testing.assert_allclose(basis.synthesis_matrix, b, rtol=0, atol=2e-11)
    np.testing.assert_allclose(a @ b, np.eye(n), rtol=0, atol=4e-12)
    rng = np.random.default_rng(763)
    y = rng.normal(size=(3, n))
    d = wfmm_transform(y, basis).coefficients
    np.testing.assert_allclose(d, y @ a, rtol=0, atol=2e-10)
    np.testing.assert_allclose(wfmm_inverse(d, basis), y, rtol=0, atol=3e-11)
    arbitrary = rng.normal(size=(3, basis.coefficient_count))
    np.testing.assert_allclose(wfmm_inverse(arbitrary, basis), arbitrary @ b, rtol=0, atol=2e-10)
    omega = np.linspace(0.2, 2, basis.coefficient_count)
    np.testing.assert_allclose(
        wfmm_covariance(omega, basis), b.T @ np.diag(omega) @ b, rtol=0, atol=2e-10
    )
    selected = wfmm_pass_filter(d, basis, lowpass=1)
    assert np.all(selected.coefficient_partition < basis.levels)


@pytest.mark.parametrize("boundary", ["periodic", "zero", "reflection"])
def test_odd_haar_boundary_values(boundary):
    y = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    basis = wfmm_extended_wavelet_basis(5, filter_length=2, boundary=boundary)
    last = dict(periodic=1.0, zero=0.0, reflection=5.0)[boundary]
    expected = np.array([[3.0, 7.0, 5.0 + last, -1.0, -1.0, 5.0 - last]]) / np.sqrt(2)
    np.testing.assert_allclose(wfmm_transform(y, basis).coefficients, expected, atol=1e-14)
    np.testing.assert_allclose(wfmm_inverse(expected, basis), y, atol=1e-14)


@pytest.mark.parametrize(
    "size,levels,length", [(2, 5, 20), (3, 3, 4), (7, 2, 6), (16, 4, 2), (65, 3, 12)]
)
@pytest.mark.parametrize("boundary", ["periodic", "zero", "reflection"])
def test_independent_perfect_reconstruction(size, levels, length, boundary):
    b = wfmm_extended_wavelet_basis(size, levels=levels, filter_length=length, boundary=boundary)
    np.testing.assert_allclose(
        b.custom_matrix @ b.synthesis_matrix, np.eye(size), rtol=0, atol=3e-11
    )
    assert not b.custom_matrix.flags.writeable
    assert not b.synthesis_matrix.flags.writeable
    assert b.coefficient_partition.size == b.coefficient_count


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(time_count=513),
        dict(levels=0),
        dict(levels=True),
        dict(filter_length=3),
        dict(boundary="symmetric"),
        dict(max_work=1),
    ],
)
def test_invalid_or_bounded(kwargs):
    with pytest.raises(ValueError):
        wfmm_extended_wavelet_basis(**(dict(time_count=32) | kwargs))


@pytest.mark.parametrize("boundary", ["periodic", "zero", "reflection"])
def test_saved_odd_extended_workflow(boundary):
    y = np.random.default_rng(5).normal(size=(8, 9))
    study = WFMMStudy(
        y,
        np.ones((8, 1)),
        transform="extended_wavelet",
        boundary=boundary,
        levels=2,
        filter_length=4,
        draws=8,
        warmup=4,
        seed=194,
        prediction_design=[[1.0]],
    )
    reopened = read_wfmm_study(study.to_json())
    first, second = run_wfmm_study(study), run_wfmm_study(reopened)
    assert first.fit is not None
    np.testing.assert_array_equal(first.fit.coefficients, second.fit.coefficients)
    assert first.summary.mean.shape == (1, 9)
    np.testing.assert_allclose(first.prediction.mean, first.summary.mean, atol=1e-14)


def test_previous_version_one_study_without_boundary_reopens():
    y = np.arange(32, dtype=float).reshape(8, 4)
    original = WFMMStudy(y, np.ones((8, 1)))
    data = json.loads(original.to_json())
    del data["boundary"]
    reopened = read_wfmm_study(json.dumps(data))
    assert reopened.to_json() == original.to_json()
