"""Optional plots for BCHM cluster and target-specific posterior results."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, pi, sqrt
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .bchm import BCHMCluster, BCHMFit, _native_probability

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _fit_colors(labels: np.ndarray, colors: Sequence[str] | None) -> list[str]:
    groups = len(labels)
    if colors is not None:
        if len(colors) != groups:
            raise ValueError("colors must contain one value per subgroup")
        return list(colors)
    palette = (
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    )
    return [palette[(int(label) - 1) % len(palette)] for label in labels]


def _axes(ax: Axes | None, *, figsize: tuple[float, float]) -> Axes:
    if ax is not None:
        return ax
    from matplotlib import pyplot as plt

    _, result = plt.subplots(figsize=figsize, layout="constrained")
    return result


def _bounds(
    value: tuple[float, float] | None, default: tuple[float, float], name: str
) -> tuple[float, float]:
    if value is None:
        return default
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or np.any(~np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError(f"{name} must be a finite increasing pair")
    return float(bounds[0]), float(bounds[1])


def _cluster_data(result: BCHMCluster | BCHMFit) -> BCHMCluster:
    if isinstance(result, BCHMFit):
        return result.cluster
    if isinstance(result, BCHMCluster):
        return result
    raise TypeError("result must be BCHMCluster or BCHMFit")


def _samples(fit: BCHMFit) -> tuple[np.ndarray, ...]:
    if not isinstance(fit, BCHMFit):
        raise TypeError("fit must be a BCHMFit")
    group_count = len(fit.borrowing)
    if not 1 <= group_count <= 20 or np.shape(fit.posterior_mean) != (group_count,):
        raise ValueError("fit must contain results for 1 to 20 subgroups")
    raw = tuple(np.asarray(item.samples) for item in fit.borrowing)
    if (
        any(
            sample.ndim != 2
            or sample.shape[0] < 1
            or sample.shape[1] < 2
            or np.iscomplexobj(sample)
            or not np.issubdtype(sample.dtype, np.number)
            for sample in raw
        )
        or sum(sample.size for sample in raw) > 200_000
        or any(
            np.any(~np.isfinite(sample)) or np.any((sample < 0) | (sample > 1)) for sample in raw
        )
    ):
        raise ValueError("fit must contain finite posterior samples for every subgroup")
    return tuple(sample.astype(float, copy=False).reshape(-1) for sample in raw)


def _hpd_interval(samples: ArrayLike, probability: float) -> tuple[float, float]:
    """Apply CRAN boa.hpd's order-statistic interval rule exactly."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(~np.isfinite(x)):
        raise ValueError("samples must be a finite vector with at least two values")
    if not np.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("hpd must be in (0,1]")
    m = max(1, ceil((1 - probability) * x.size))
    ordered = np.sort(x)
    lower, upper = ordered[:m], ordered[x.size - m :]
    index = int(np.argmin(upper - lower))
    return float(lower[index]), float(upper[index])


def _bw_nrd0(samples: ArrayLike) -> float:
    """R stats::bw.nrd0 rule, including its constant-sample fallbacks."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(~np.isfinite(x)):
        raise ValueError("density samples must be a finite vector with at least two values")
    hi = float(np.std(x, ddof=1))
    q1, q3 = np.quantile(x, [0.25, 0.75], method="linear")
    lo = min(hi, float(q3 - q1) / 1.34)
    if lo == 0:
        lo = hi
    if lo == 0:
        lo = abs(float(x[0]))
    if lo == 0:
        lo = 1.0
    bandwidth = 0.9 * lo * x.size ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ArithmeticError("R-compatible Gaussian bandwidth is not representable")
    return bandwidth


def _density_grid(
    samples: ArrayLike, *, points: int, chunk_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Direct Gaussian-kernel evaluation on R density()'s default support."""
    x = np.asarray(samples, dtype=float)
    bandwidth = _bw_nrd0(x)
    grid = np.linspace(float(np.min(x)) - 3 * bandwidth, float(np.max(x)) + 3 * bandwidth, points)
    density = np.empty(points)
    normalizer = bandwidth * sqrt(2 * pi) * x.size
    for start in range(0, points, chunk_size):
        stop = min(points, start + chunk_size)
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            z = (grid[start:stop, None] - x[None, :]) / bandwidth
            kernel = np.exp(-0.5 * z * z)
            density[start:stop] = kernel.sum(axis=1) / normalizer
    if np.any(~np.isfinite(density)):
        raise ArithmeticError("Gaussian density evaluation is non-finite")
    return grid, density


def plot_bchm_cluster(
    result: BCHMCluster | BCHMFit,
    *,
    colors: Sequence[str] | None = None,
    marker: str = "o",
    marker_size: float = 6.0,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot observed subgroup rates colored by the representative partition.

    This corresponds to native ``BCHMplot_cluster``. Returns the axes for
    customization and saving; it never calls ``show`` or changes global style.
    """
    cluster = _cluster_data(result)
    rates = np.asarray(cluster.rates, dtype=float)
    labels = np.asarray(cluster.result.representative, dtype=int)
    if rates.ndim != 1 or labels.shape != rates.shape or not rates.size:
        raise ValueError("cluster result contains inconsistent subgroup arrays")
    size = float(marker_size)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("marker_size must be positive and finite")
    color_values = _fit_colors(labels, colors)
    axis = _axes(ax, figsize=(7, 4.5))
    x = np.arange(1, rates.size + 1)
    plotted_rates = np.asarray([_native_probability(float(value) + 1e-9) for value in rates])
    axis.scatter(x, plotted_rates, c=color_values, marker=marker, s=size**2)
    axis.set(
        xlim=_bounds(xlim, (0.0, float(rates.size + 2)), "xlim"),
        ylim=_bounds(ylim, (0.0, 1.0), "ylim"),
        xlabel="Subgroup ID",
        ylabel="Observed Response Rates",
        title="Subgroup Clusters",
    )
    axis.set_xticks(x)
    return axis


def plot_bchm_posterior(
    fit: BCHMFit,
    *,
    hpd: float | None = 0.95,
    observed_mean: bool = False,
    colors: Sequence[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot posterior means, native-rule HPD bars, and optionally observed rates.

    This corresponds to native ``BCHMplot_post_value``. ``hpd=None`` suppresses
    intervals, paralleling native ``HPD=NA``. The interval rule is the cached
    CRAN ``boa.hpd`` implementation, applied to pooled retained Python chains.
    """
    samples = _samples(fit)
    count = len(samples)
    labels = np.asarray(fit.cluster.result.representative, dtype=int)
    mean = np.asarray(fit.posterior_mean, dtype=float)
    observed = np.asarray(fit.cluster.rates, dtype=float)
    if mean.shape != (count,) or observed.shape != (count,) or labels.shape != (count,):
        raise ValueError("fit subgroup summaries are inconsistent")
    color_values = _fit_colors(labels, colors)
    x = np.arange(1, count + 1)
    mean = np.asarray([_native_probability(float(value)) for value in mean])
    if hpd is not None:
        try:
            coverage = float(hpd)
        except (TypeError, ValueError) as exc:
            raise ValueError("hpd must be in (0,1] or None") from exc
        if not np.isfinite(coverage) or not 0 < coverage <= 1:
            raise ValueError("hpd must be in (0,1] or None")
        bounds = np.asarray([_hpd_interval(s, coverage) for s in samples])
    if not isinstance(observed_mean, (bool, np.bool_)):
        raise ValueError("observed_mean must be boolean")
    horizontal = _bounds(xlim, (0.0, float(count + 2)), "xlim")
    vertical = _bounds(ylim, (0.0, 1.0), "ylim")
    axis = _axes(ax, figsize=(7, 4.5))
    if hpd is not None:
        for index, (lower, upper) in enumerate(bounds):
            axis.vlines(x[index], lower, upper, color=color_values[index], linewidth=2)
            axis.hlines(
                (lower, upper),
                x[index] - 0.1,
                x[index] + 0.1,
                color=color_values[index],
                linewidth=2,
            )
    axis.scatter(x, mean, c=color_values, marker="o", label="Posterior mean")
    if observed_mean:
        rates = np.asarray([_native_probability(float(value) + 1e-9) for value in observed])
        axis.scatter(x, rates, c="magenta", marker="x", label="Observed mean")
        axis.legend(loc="best")
    axis.set(
        xlim=horizontal,
        ylim=vertical,
        xlabel="Subgroup ID",
        ylabel="Posterior Response Rates",
        title=(
            "Posterior Probability" if hpd is None else f"Posterior Probability HPD = {coverage:g}"
        ),
    )
    axis.set_xticks(x)
    return axis


def plot_bchm_density(
    fit: BCHMFit,
    *,
    colors: Sequence[str] | None = None,
    line_styles: Sequence[str] | None = None,
    line_width: float = 2.0,
    xlim: tuple[float, float] = (0.0, 1.0),
    ylim: tuple[float, float] | None = (0.0, 20.0),
    points: int = 512,
    max_work: int = 100_000_000,
    ax: Axes | None = None,
) -> Axes:
    """Plot target posterior densities with R ``density`` bandwidth/support.

    Direct bounded Gaussian-kernel summation replaces R's FFT interpolation;
    this is a numerical evaluation choice, not a claim of bitwise plot parity.
    """
    samples = _samples(fit)
    groups = len(samples)
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)):
        raise ValueError("points must be an integer")
    if not 64 <= points <= 4096:
        raise ValueError("points must lie in [64,4096]")
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)):
        raise ValueError("max_work must be an integer")
    work = sum(sample.size for sample in samples) * int(points)
    if not 1 <= max_work <= 500_000_000 or work > max_work:
        raise ValueError("posterior density work exceeds max_work")
    width = float(line_width)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("line_width must be positive and finite")
    line_color = _fit_colors(np.asarray(fit.cluster.result.representative, dtype=int), colors)
    if line_styles is None:
        styles = ["-"] * groups
    else:
        if len(line_styles) != groups:
            raise ValueError("line_styles must contain one value per subgroup")
        styles = list(line_styles)
    horizontal = _bounds(xlim, (0.0, 1.0), "xlim")
    vertical = _bounds(ylim, (0.0, 20.0), "ylim") if ylim is not None else None
    axis = _axes(ax, figsize=(7, 4.5))
    max_density = 0.0
    for i, sample in enumerate(samples):
        grid, density = _density_grid(sample, points=int(points))
        axis.plot(
            grid,
            density,
            color=line_color[i],
            linestyle=styles[i],
            linewidth=width,
            label=f"Subg. {i + 1}",
        )
        max_density = max(max_density, float(np.max(density)))
    axis.set(
        xlim=horizontal,
        ylim=vertical if vertical is not None else (0.0, max_density * 1.1),
        xlabel="Response Rates",
        ylabel="Density",
        title="Posterior Distribution",
    )
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.legend(loc="upper right")
    return axis
