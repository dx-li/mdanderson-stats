"""Hazard geometry, overlays, stratification and missing estimates."""

from dataclasses import replace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from numpy.testing import assert_array_equal  # noqa: E402

from mdanderson_stats import (  # noqa: E402
    kphaz,
    muhaz_fixed,
    muhaz_global,
    muhaz_knn,
    muhaz_local,
    pehaz,
    plot_kphaz,
    plot_muhaz,
    plot_pehaz,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize("method", ["fixed", "global", "local", "knn"])
def test_kernel_curve_uses_estimation_points(method):
    t = np.linspace(0.1, 3, 20)
    if method == "fixed":
        fit = muhaz_fixed(t, bandwidth=0.5)
    else:
        fn = {"global": muhaz_global, "local": muhaz_local, "knn": muhaz_knn}[method]
        fit = fn(t, bounds=(0, 3))
    ax = plot_muhaz(fit, label="Estimate")
    assert_array_equal(ax.lines[0].get_xdata(), fit.time)
    assert_array_equal(ax.lines[0].get_ydata(), fit.hazard)
    assert ax.get_xlabel() == "Follow-up Time"
    assert ax.get_ylim()[0] == 0
    ax.figure.canvas.draw()


def test_overlay_preserves_existing_labels_and_manual_limits():
    _, ax = plt.subplots()
    ax.set(xlabel="Days", ylabel="Rate", xlim=(0, 10), ylim=(0, 5))
    fit = muhaz_fixed([1, 2, 3], bandwidth=0.5)
    assert plot_muhaz(fit, ax=ax, label="Kernel") is ax
    assert plot_pehaz(pehaz([1, 2, 3], width=1), ax=ax, label="Binned") is ax
    assert ax.get_xlim() == (0, 10) and ax.get_ylim() == (0, 5)
    assert ax.get_xlabel() == "Days" and ax.get_ylabel() == "Rate"
    assert len(ax.lines) == 1 and len(ax.patches) == 1


def test_piecewise_geometry_has_exact_edges_and_no_artificial_baseline():
    fit = pehaz([0.2, 1.3, 2.5], width=1, bounds=(0, 2.5))
    ax = plot_pehaz(fit)
    data = ax.patches[0].get_data()
    assert_array_equal(data.edges, fit.cuts)
    assert_array_equal(data.values, fit.hazard)
    assert data.baseline is None
    ax.figure.canvas.draw()


def test_piecewise_missing_bin_is_a_gap():
    fit = pehaz([0.2, 1.3, 2.5], width=1, bounds=(0, 2.5))
    fit = replace(fit, hazard=np.array([1.0, np.nan, 2.0]))
    ax = plot_pehaz(fit)
    assert np.count_nonzero(ax.patches[0].get_path().codes == Path.MOVETO) == 2
    assert np.isnan(ax.patches[0].get_data().values[1])
    ax.figure.canvas.draw()


def test_stratified_terminal_infinities_are_gaps_for_every_stratum():
    fit = kphaz([1, 2, 3, 1, 2, 3], [1] * 6, strata=["A"] * 3 + ["B"] * 3, method="product-limit")
    ax = plot_kphaz(fit)
    assert len(ax.lines) == 2
    for line in ax.lines:
        assert_array_equal(line.get_xdata(), [0, 1.5, 2.5])
        assert line.get_ydata()[0] == 0
        assert np.isnan(line.get_ydata()[-1])
        assert line.get_drawstyle() == "steps-post"
    assert [text.get_text() for text in ax.get_legend().texts] == ["A", "B"]
    ax.figure.canvas.draw()


def test_empty_or_nonfinite_results_are_rejected_before_creating_figure():
    with pytest.raises(ValueError):
        plot_muhaz(muhaz_fixed([1, 2], bandwidth=0.5, grid=[]))
    with pytest.raises(ValueError):
        plot_kphaz(kphaz([1, 2], [1, 1], method="product-limit"))
    fit = pehaz([1, 2], width=1)
    with pytest.raises(ValueError):
        plot_pehaz(replace(fit, hazard=np.full_like(fit.hazard, np.nan)))
    assert plt.get_fignums() == []


def test_zero_hazard_curve_has_valid_limits():
    ax = plot_muhaz(muhaz_fixed([1, 2], [0, 0], bandwidth=0.5))
    assert ax.get_ylim() == (0, 1)
    ax.figure.canvas.draw()
