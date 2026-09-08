"""Rendered CUMINC plot geometry, intervals, selection and style isolation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from mdanderson_stats import cuminc, plot_cuminc  # noqa: E402


@pytest.fixture
def study():
    return cuminc(
        [0, 1, 2, 3, 4, 5, 6, 7],
        ["a", "b", "c", "a", "b", "a", "c", "b"],
        ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
        censor="c",
        confidence=0.8,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_overlay_preserves_native_corners_labels_and_style(study, tmp_path):
    before = dict(matplotlib.rcParams)
    axes = plot_cuminc(study, xlabel="Months", ylabel="Event probability")
    assert len(axes) == 1
    ax = axes[0]
    assert len(ax.lines) == 4
    for line, curve in zip(ax.lines, study.curves.values(), strict=True):
        assert_array_equal(line.get_xdata(), curve.time)
        assert_array_equal(line.get_ydata(), curve.estimate)
    assert ax.get_xlabel() == "Months"
    assert ax.get_ylabel() == "Event probability"
    assert len(ax.get_legend().get_texts()) == 4
    ax.figure.savefig(tmp_path / "overlay.png")
    assert (tmp_path / "overlay.png").stat().st_size > 1000
    assert dict(matplotlib.rcParams) == before


def test_panels_match_pointwise_intervals_and_grid(study, tmp_path):
    axes = plot_cuminc(study, overlay=False)
    assert len(axes) == 4
    for ax, ((cause, group), curve) in zip(axes, study.curves.items(), strict=True):
        summary = curve.summary(confidence=0.8)
        assert len(ax.lines) == 3
        assert_array_equal(ax.lines[0].get_ydata(), summary.rows[:, 1])
        assert_array_equal(ax.lines[1].get_ydata(), summary.rows[:, 3])
        assert_array_equal(ax.lines[2].get_ydata(), summary.rows[:, 4])
        assert ax.get_title() == f"Cause = {cause}, Group = {group}"
        assert ax.get_ylim() == (0, 1)
        grid = ax.get_subplotspec().get_gridspec()
        assert (grid.nrows, grid.ncols) == (2, 2)
    axes[0].figure.savefig(tmp_path / "panels.svg")
    assert (tmp_path / "panels.svg").stat().st_size > 1000


def test_one_group_panels_are_horizontal_and_selection_order_is_retained(study):
    axes = plot_cuminc(study, causes=["b", "a"], groups="X", overlay=False)
    assert len(axes) == 2
    assert axes[0].get_title() == "Cause = b, Group = X"
    grid = axes[0].get_subplotspec().get_gridspec()
    assert (grid.nrows, grid.ncols) == (1, 2)


def test_data_coordinate_legend_position(study):
    ax = plot_cuminc(study, causes="a", legend_at=[1, 0.9])[0]
    bbox = ax.get_legend().get_bbox_to_anchor()
    assert_allclose([bbox.x0, bbox.y0], ax.transData.transform([1, 0.9]))


def test_zero_time_and_absent_cause_within_selected_group_have_valid_limits():
    study = cuminc([0, 0], [1, 0], ["A", "B"])
    ax = plot_cuminc(study, groups="B")[0]
    assert ax.get_xlim() == (0, 1)
    assert ax.get_ylim() == (0, 1)
    assert_array_equal(ax.lines[0].get_ydata(), [0, 0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"causes": []},
        {"groups": []},
        {"causes": "unknown"},
        {"groups": ["X", "X"]},
        {"overlay": 1},
        {"legend_at": [1]},
        {"legend_at": [1, np.nan]},
        {"overlay": False, "legend_at": [0, 0.5]},
    ],
)
def test_invalid_plot_request_creates_no_figure(study, kwargs):
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_cuminc(study, **kwargs)
    assert plt.get_fignums() == before


def test_all_censored_plot_has_no_curve_to_draw():
    with pytest.raises(ValueError, match="At least one cause"):
        plot_cuminc(cuminc([1, 2], [0, 0]))
