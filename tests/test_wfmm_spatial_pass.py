"""Native pyramid-level pass references and complete orientation selection."""

import json
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats import (
    WFMMStudy,
    read_wfmm_study,
    run_wfmm_study,
    wfmm_inverse,
    wfmm_pass_filter,
    wfmm_restore_coefficients,
    wfmm_spatial_wavelet_basis,
    wfmm_transform,
)

CASES = json.loads((Path(__file__).parent / "fixtures/wfmm-spatial-pass.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES)
def test_native_spatial_selection_and_empty_failures(case):
    b = wfmm_spatial_wavelet_basis(
        case["dims"],
        levels=(case["levels"], 1) if case["rectangular"] else case["levels"],
        wavelets=case["wavelet"],
        boundaries=case["boundary"],
        layout="rectangular" if case["rectangular"] else "pyramid",
    )
    if case["rectangular"] or "failure" in case:
        with pytest.raises(ValueError):
            wfmm_pass_filter(b.custom_matrix, b, highpass=case["highpass"], lowpass=case["lowpass"])
    else:
        selected = wfmm_pass_filter(
            b.custom_matrix, b, highpass=case["highpass"], lowpass=case["lowpass"]
        )
        np.testing.assert_array_equal(selected.retained_indices, case["indices"])


@pytest.mark.parametrize("shape", [(8, 8), (4, 4, 4)])
def test_low_and_high_pyramid_levels_form_complete_orthogonal_decomposition(shape):
    b = wfmm_spatial_wavelet_basis(shape, levels=2)
    y = np.random.default_rng(138).normal(size=(2, b.time_count))
    d = wfmm_transform(y, b).coefficients
    lo = wfmm_pass_filter(d, b, lowpass=1)
    hi = wfmm_pass_filter(d, b, highpass=1)
    a = wfmm_inverse(wfmm_restore_coefficients(lo.coefficients, lo), b)
    c = wfmm_inverse(wfmm_restore_coefficients(hi.coefficients, hi), b)
    np.testing.assert_allclose(a + c, y, atol=2e-15, rtol=1e-15)
    np.testing.assert_allclose(np.sum(a * c, axis=1), 0, atol=1e-14)


def test_saved_spatial_pass_replay_preserves_full_grid():
    y = np.random.default_rng(46).normal(size=(8, 16)) + 5 * np.random.default_rng(411).normal(
        size=16
    )
    s = WFMMStudy(
        y,
        np.ones((8, 1)),
        transform="spatial_wavelet",
        spatial_shape=(4, 4),
        levels=2,
        lowpass=1,
        draws=8,
        warmup=4,
        optimizer_iterations=50,
    )
    r = run_wfmm_study(read_wfmm_study(s.to_json()))
    assert r.compression.selection.retained_indices.size == 4
    if r.fit is not None:
        assert r.summary.mean.shape == (1, 16)
    else:
        assert r.initialization.holds
    assert r.report_json() == run_wfmm_study(s).report_json()


def test_saved_axis_levels_control_pass_bounds():
    y = np.ones((8, 16))
    study = WFMMStudy(
        y,
        np.ones((8, 1)),
        transform="spatial_wavelet",
        spatial_shape=(4, 4),
        axis_levels=(2, 2),
        highpass=1,
    )
    assert study.highpass == 1
