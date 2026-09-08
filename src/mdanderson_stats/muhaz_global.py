"""Global MUHAZ bandwidth selection and the archived single-bandwidth bypass."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, count, finite, scalar
from .muhaz import Boundary, Kernel, MuhazFixed, muhaz_fixed
from .muhaz_mse import MuhazMSE, muhaz_mse


@dataclass(frozen=True)
class MuhazGlobal:
    curve: MuhazFixed
    bandwidths: FloatArray
    scores: FloatArray | None
    selected_index: int
    score: float | None
    diagnostics: MuhazMSE | None
    n_observations: int
    n_events: int
    pilot_bandwidth: float | None = None
    n_min_grid: int | None = None

    @property
    def time(self) -> FloatArray:
        return self.curve.time

    @property
    def hazard(self) -> FloatArray:
        return self.curve.hazard

    @property
    def bandwidth(self) -> float:
        return self.curve.bandwidth


def muhaz_global(
    times: ArrayLike,
    delta: ArrayLike | None = None,
    *,
    bandwidths: ArrayLike | None = None,
    pilot_bandwidth: float | None = None,
    bounds: tuple[float, float] | None = None,
    subset: ArrayLike | None = None,
    n_min_grid: int = 51,
    n_est_grid: int = 101,
    kernel: Kernel = "epanechnikov",
    boundary: Boundary = "both",
    legacy: bool = False,
    rtol: float = 0.001,
    max_refinements: int = 6,
) -> MuhazGlobal:
    """Select a common bandwidth by summed grid MSE, then estimate the hazard.

    A single candidate bypasses MSE computation; scores and diagnostics are None.
    Default upper bound is interpolated at ten subjects at risk. Explicit bounds
    are required when that interpolation is undefined. Bounds above observed
    follow-up are truncated to it, as in the archived interface.
    """
    t, d, bw, pilot, (left, right) = _prepare_selection(
        times,
        delta,
        bandwidths,
        pilot_bandwidth,
        bounds,
        subset,
        n_min_grid,
        n_est_grid,
        legacy,
    )
    diagnostic = None
    scores = None
    selected = 0
    score = None
    if bw.size > 1:
        assert pilot is not None
        diagnostic = muhaz_mse(
            t,
            d,
            bandwidths=bw,
            pilot_bandwidth=pilot,
            grid=np.linspace(left, right, n_min_grid),
            bounds=(left, right),
            kernel=kernel,
            boundary=boundary,
            legacy=legacy,
            rtol=rtol,
            max_refinements=max_refinements,
        )
        scores = diagnostic.mse.sum(axis=1)
        if not np.all(np.isfinite(scores)):
            raise RuntimeError("Summed MSE exceeds numerical range")
        if legacy:
            eligible = (scores > 0) & (scores < 1e30)
            selected = (
                int(np.argmin(np.where(eligible, scores, np.inf)))
                if eligible.any()
                else bw.size - 1
            )
            score = float(scores[selected]) if eligible.any() else 1e30
        else:
            selected = int(np.argmin(scores))
            score = float(scores[selected])
        scores.flags.writeable = False
    curve = muhaz_fixed(
        t,
        d,
        bandwidth=float(bw[selected]),
        grid=np.linspace(left, right, n_est_grid),
        bounds=(left, right),
        kernel=kernel,
        boundary=boundary,
        legacy=legacy,
    )
    bw.flags.writeable = False
    return MuhazGlobal(
        curve,
        bw,
        scores,
        selected,
        score,
        diagnostic,
        t.size,
        int(d.sum()),
        pilot_bandwidth=pilot,
        n_min_grid=int(n_min_grid),
    )


def _prepare_selection(
    times: ArrayLike,
    delta: ArrayLike | None,
    bandwidths: ArrayLike | None,
    pilot_bandwidth: float | None,
    bounds: tuple[float, float] | None,
    subset: ArrayLike | None,
    n_min_grid: int,
    n_est_grid: int,
    legacy: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, float | None, tuple[float, float]]:
    """Shared archived local/global input, default and subset conventions."""
    t = np.asarray(times, dtype=float)
    d = np.ones(t.size) if delta is None else np.asarray(delta, dtype=float)
    if t.ndim != 1 or t.size == 0 or d.shape != t.shape:
        raise ValueError("times and delta must be matching nonempty vectors")
    if subset is not None:
        mask = np.asarray(subset)
        if mask.dtype.kind != "b" or mask.shape != t.shape:
            raise ValueError("subset must be a boolean vector matching times")
        t, d = t[mask], d[mask]
    t, d = finite(t, "times"), count(d, "delta")
    if t.size == 0 or np.any(t < 0) or np.any(d > 1):
        raise ValueError("Selected data must be nonempty, with nonnegative times and binary delta")
    if not isinstance(legacy, (bool, np.bool_)):
        raise ValueError("legacy must be boolean")
    for n, name in [(n_min_grid, "n_min_grid"), (n_est_grid, "n_est_grid")]:
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"{name} must be a positive integer")
    if bounds is None:
        unique, starts = np.unique(np.sort(t), return_index=True)
        risk = t.size - starts
        if not risk[-1] <= 10 <= risk[0]:
            raise ValueError("Ten-at-risk upper bound is undefined; provide explicit bounds")
        interval = np.array([0, float(np.interp(10, risk[::-1], unique[::-1]))])
    else:
        interval = finite(bounds, "bounds").copy()
        if interval.shape != (2,):
            raise ValueError("bounds must contain two values")
        interval[1] = min(interval[1], float(t.max()))
    if not 0 <= interval[0] < interval[1]:
        raise ValueError("Effective bounds must be increasing and nonnegative")
    left, right = map(float, interval)
    bw = None if bandwidths is None else finite(bandwidths, "bandwidths")
    if bw is not None:
        if bw.ndim > 1 or bw.size == 0 or np.any(bw <= 0):
            raise ValueError("bandwidths must be a positive scalar or nonempty vector")
        bw = bw.reshape(-1).copy()
    pilot = None if pilot_bandwidth is None else scalar(pilot_bandwidth, "pilot_bandwidth")
    if pilot is not None and pilot <= 0:
        raise ValueError("pilot_bandwidth must be positive")
    need_mse = bw is None or bw.size > 1
    if need_mse and pilot is None:
        n_events = int(d.sum())
        if n_events == 0:
            raise ValueError("Pilot bandwidth is required for bandwidth selection with no events")
        pilot = (right if legacy else right - left) / (8 * n_events**0.2)
    if bw is None:
        assert pilot is not None
        bw = np.linspace(0.2 * pilot, 20 * pilot, 25)
    return t, d, bw, pilot, (left, right)
