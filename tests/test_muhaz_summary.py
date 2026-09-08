"""Metadata, bypasses, score semantics and report file output."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mdanderson_stats import muhaz_global, muhaz_knn, muhaz_local, summarize_muhaz


@pytest.mark.parametrize(
    "function,method", [(muhaz_global, "global"), (muhaz_local, "local"), (muhaz_knn, "knn")]
)
def test_summary_retains_selected_sample_settings_and_convergence(function, method):
    t = np.r_[np.linspace(0.1, 3, 20), np.nan]
    d = np.r_[np.tile([1, 1, 0, 1], 5), np.nan]
    kwargs = {"neighbors": [6, 7]} if method == "knn" else {"bandwidths": [0.4, 0.8]}
    fit = function(
        t,
        d,
        subset=np.r_[np.ones(20, dtype=bool), False],
        bounds=(0.05, 4),
        pilot_bandwidth=0.6,
        n_min_grid=7,
        n_est_grid=9,
        kernel="triquadratic",
        boundary="left",
        max_refinements=1,
        **kwargs,
    )
    s = summarize_muhaz(fit)
    assert s.n_observations == 20 and s.n_censored == 5
    assert s.method == method
    assert s.bounds == (0.05, 3)
    assert s.kernel == "triquadratic" and s.boundary == "left"
    assert s.n_min_grid == 7 and s.n_est_grid == 9
    assert s.pilot_bandwidth == 0.6
    assert s.score == fit.score
    assert s.converged_cells == 0 and s.diagnostic_cells == 14
    assert "0/14" in s.report()
    if method == "local":
        assert s.score == fit.minimum_mse.sum()
        assert "pointwise minimum MSE" in s.report()
    elif method == "knn":
        assert s.neighbors == fit.neighbors and s.neighbor_method == "survival"
    else:
        assert s.constant_bandwidth == fit.bandwidth
    with pytest.raises(FrozenInstanceError):
        s.score = 0


@pytest.mark.parametrize("function", [muhaz_global, muhaz_local])
def test_fixed_bypass_does_not_invent_mse_or_pilot(function):
    fit = function([1, 2], bandwidths=0.123456, bounds=(0, 2), n_min_grid=17, n_est_grid=3)
    s = summarize_muhaz(fit)
    assert s.n_min_grid == 17 and s.n_est_grid == 3
    assert s.pilot_bandwidth is s.score is s.smoothing_bandwidth is None
    assert s.constant_bandwidth == 0.123456
    assert s.diagnostic_cells == 0
    report = s.report()
    assert "bypassed (single candidate)" in report
    assert "Pilot bandwidth: not used" in report
    assert "Summed grid MSE:" not in report


def test_single_neighbor_keeps_pilot_even_without_mse():
    fit = muhaz_knn(
        np.linspace(0.1, 3, 20), neighbors=3, bounds=(0, 3), pilot_bandwidth=0.65, n_min_grid=7
    )
    s = summarize_muhaz(fit)
    assert fit.diagnostics is None
    assert s.pilot_bandwidth == 0.65
    assert s.smoothing_bandwidth == 3.25
    assert s.n_min_grid == 7
    assert s.neighbors == 3
    assert s.score is None


def test_legacy_sentinel_is_identified_as_legacy_and_not_a_valid_zero():
    fit = muhaz_global(
        [1, 2], [0, 0], bandwidths=[0.4, 0.8], pilot_bandwidth=0.5, bounds=(0, 2), legacy=True
    )
    s = summarize_muhaz(fit)
    assert s.score == 1e30 and s.legacy
    assert "Legacy conventions: yes" in s.report()
    assert "Summed grid MSE: 1e+30" in s.report()


def test_small_bandwidth_report_preserves_significant_digits_and_writes_utf8(tmp_path):
    fit = muhaz_global([1, 2], bandwidths=1.234567e-8, bounds=(0, 2))
    s = summarize_muhaz(fit)
    report = s.report(digits=8)
    assert "1.234567e-08" in report
    path = tmp_path / "hazard summary.txt"
    s.write_report(path, digits=8)
    assert path.read_text(encoding="utf-8") == report
    with pytest.raises(IsADirectoryError):
        s.write_report(tmp_path)


@pytest.mark.parametrize("digits", [0, 18, True, 1.2, "6"])
def test_invalid_report_precision(digits):
    s = summarize_muhaz(muhaz_global([1, 2], bandwidths=0.5, bounds=(0, 2)))
    with pytest.raises(ValueError):
        s.report(digits=digits)


def test_invalid_fit_type_is_explicit():
    with pytest.raises(TypeError, match="MUHAZ fit"):
        summarize_muhaz(object())
