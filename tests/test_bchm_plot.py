import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.bchm import BCHMBorrowResult, BCHMCluster, BCHMFit
from mdanderson_stats.bchm_clustering import BCHMClusterResult
from mdanderson_stats.bchm_plot import (
    _bw_nrd0,
    _density_grid,
    _hpd_interval,
    plot_bchm_cluster,
    plot_bchm_density,
    plot_bchm_posterior,
)


def _rows(name: str) -> list[dict[str, str]]:
    with (Path(__file__).parent / "fixtures" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fit() -> BCHMFit:
    samples = (np.array([[0.1, 0.2, 0.3, 0.4]]), np.array([[0.2, 0.3, 0.4, 0.5]]))
    cluster_result = BCHMClusterResult(
        allocations=np.array([[1, 2]], dtype=np.int16),
        raw_similarity=np.eye(2),
        similarity=np.eye(2),
        representative=np.array([1, 2], dtype=np.int16),
        representative_score=0.5,
    )
    cluster = BCHMCluster(np.array([0.15, 0.25]), np.array([10.0, 10.0]), cluster_result)
    borrowing = tuple(
        BCHMBorrowResult(i, sample, float(sample.mean()), 0.5, 0.5, False, np.ones(2), None)
        for i, sample in enumerate(samples)
    )
    return BCHMFit(
        cluster,
        borrowing,
        np.array([0.25, 0.35]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([False, False]),
        np.eye(2),
        np.eye(2),
        np.eye(2),
        np.array([[1, 2]], dtype=np.int16),
        (None, None),
    )


def test_boa_hpd_and_r_bandwidth_match_independent_native_fixtures() -> None:
    samples: dict[str, list[float]] = defaultdict(list)
    for row in _rows("bchm-plot-input.csv"):
        samples[row["case"]].append(float(row["sample"]))
    for row in _rows("bchm-plot-hpd.csv"):
        actual = _hpd_interval(samples[row["case"]], float(row["level"]))
        np.testing.assert_allclose(
            actual, [float(row["lower"]), float(row["upper"])], atol=0, rtol=0
        )
    density_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _rows("bchm-plot-density.csv"):
        density_rows[row["case"]].append(row)
    for case, rows in density_rows.items():
        grid, density = _density_grid(samples[case], points=len(rows))
        np.testing.assert_allclose(grid, [float(row["x"]) for row in rows], atol=1e-14, rtol=0)
        np.testing.assert_allclose(
            density,
            [float(row["direct_gaussian"]) for row in rows],
            atol=1e-12,
            rtol=1e-12,
        )
        assert _bw_nrd0(samples[case]) == pytest.approx(float(rows[0]["bandwidth"]), rel=1e-14)


def test_plot_helpers_return_axes_with_native_titles_and_subgroup_traces() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = _fit()
    cluster_ax = plot_bchm_cluster(fit)
    assert cluster_ax.get_title() == "Subgroup Clusters"
    assert cluster_ax.get_ylabel() == "Observed Response Rates"
    assert len(cluster_ax.collections) == 1

    posterior_ax = plot_bchm_posterior(fit, hpd=0.5, observed_mean=True)
    assert posterior_ax.get_title() == "Posterior Probability HPD = 0.5"
    assert posterior_ax.get_legend() is not None
    assert len(posterior_ax.collections) == 6

    density_ax = plot_bchm_density(fit, points=64, ylim=None)
    assert density_ax.get_title() == "Posterior Distribution"
    assert len(density_ax.lines) == 2
    assert [line.get_label() for line in density_ax.lines] == ["Subg. 1", "Subg. 2"]
    plt.close("all")


def test_density_work_rejected_before_creating_a_figure() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = _fit()
    before = len(plt.get_fignums())
    with pytest.raises(ValueError, match="work"):
        plot_bchm_density(fit, points=64, max_work=1)
    assert len(plt.get_fignums()) == before
