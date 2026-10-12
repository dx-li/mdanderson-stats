"""Source-recovered PRT beta-shaped piecewise hazards with explicit profiles."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betaln

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar


@dataclass(frozen=True)
class PRTHazard:
    grid: FloatArray
    rates: FloatArray
    cdf: FloatArray
    requested_probability: float
    achieved_probability: float
    scale: float
    profile: str
    iterations: int

    def __post_init__(self) -> None:
        grid, rates, cdf = (
            finite(v, n) for v, n in ((self.grid, "grid"), (self.rates, "rates"), (self.cdf, "cdf"))
        )
        if (
            grid.shape != (100,)
            or rates.shape != (100,)
            or cdf.shape != (100,)
            or grid[0] != 0
            or grid[-1] <= 0
            or np.any(np.diff(grid[:99]) <= 0)
            or grid[-1] != grid[-2]
            or np.any(rates < 0)
            or self.profile not in ("safe", "legacy_1_0")
        ):
            raise ValueError("invalid recovered hazard grid, rates or profile")
        expected = np.r_[0.0, -np.expm1(-np.cumsum(np.diff(grid) * rates[:99]))]
        achieved = float(expected[-1])
        expected[-1] = 1.0
        if (
            not np.allclose(cdf, expected, rtol=0, atol=1e-12)
            or not np.isfinite(self.achieved_probability)
            or abs(achieved - self.achieved_probability) > 1e-12
            or not 0 < self.requested_probability < 1
            or not np.isfinite(self.scale)
            or self.scale <= 0
            or isinstance(self.iterations, bool)
            or int(self.iterations) != self.iterations
            or not 1 <= self.iterations <= 100000
        ):
            raise ValueError("invalid hazard calibration metadata")
        tolerance = 1e-12 if self.profile == "safe" else float(np.float32(1e-5))
        if abs(achieved - self.requested_probability) > tolerance:
            raise ValueError("hazard calibration misses its declared probability tolerance")
        for name, value in (("grid", grid), ("rates", rates), ("cdf", cdf)):
            object.__setattr__(self, name, _freeze(value))


def calibrate_prt_hazard(
    probability: float, window: float, alpha: float, beta: float, *, profile: str = "safe"
) -> PRTHazard:
    """Calibrate100 beta-density-shaped rates on the recovered desktop grid.

    Safe calibration solves the cumulative-hazard equation exactly. legacy_1_0
    reproduces its bounded step search; neither is a beta event-time distribution.
    """
    p, w, a, b = (
        scalar(v, n)
        for v, n in zip(
            (probability, window, alpha, beta),
            ("probability", "window", "alpha", "beta"),
            strict=True,
        )
    )
    if (
        not 0 < p < 1
        or min(w, a, b) <= 0
        or max(a, b) > 10000
        or w > 1e6
        or profile not in ("safe", "legacy_1_0")
    ):
        raise ValueError(
            "require probability in (0,1), window in (0,1e6], "
            "shapes in (0,10000], and known profile"
        )
    x = np.arange(1, 101) / 101
    density = np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - betaln(a, b))
    grid = np.r_[np.arange(99) * w / 98, w]
    mass = float(np.dot(np.diff(grid)[:98], density[:98]))
    if not np.isfinite(mass) or mass <= 0 or np.any(~np.isfinite(density)):
        raise ArithmeticError("hazard density is not representable on the recovered grid")
    iterations = 1
    scale = mass / -np.log1p(-p)
    if profile == "legacy_1_0":
        scale, step = 1.0, 10.0
        tolerance = float(np.float32(1e-5))
        for iterations in range(1, 100001):
            if scale <= 0 or not np.isfinite(scale):
                raise ArithmeticError("legacy scale search left its positive domain")
            achieved = -np.expm1(-mass / scale)
            if achieved > p + tolerance:
                scale += step
            if achieved < p - tolerance:
                scale -= step
                step /= 10
            if abs(achieved - p) < tolerance:
                break
        else:
            raise ArithmeticError("legacy hazard calibration exhausted100000 iterations")
    if not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError("hazard scale is not representable")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rates = density / scale
    if np.any(~np.isfinite(rates)) or not np.isfinite(scale) or scale <= 0:
        raise ArithmeticError("hazard scale/rates not representable")
    cdf = np.r_[0.0, -np.expm1(-np.cumsum(np.diff(grid) * rates[:99]))]
    achieved = float(cdf[-1])
    cdf[-1] = 1.0
    return PRTHazard(
        _freeze(grid), _freeze(rates), _freeze(cdf), p, achieved, float(scale), profile, iterations
    )


def prt_hazard_delays(model: PRTHazard, uniforms: ArrayLike) -> FloatArray:
    """Map supplied U[0,1) variates to event delays; safe non-events return infinity.

    Legacy reproduces shifted rates and reuse of the category variate. Native
    code also discards a second variate; callers replaying tapes must consume it.
    """
    if not isinstance(model, PRTHazard):
        raise TypeError("model must be PRTHazard")
    u = finite(uniforms, "uniforms")
    if np.any((u < 0) | (u >= 1)):
        raise ValueError("uniforms must lie in [0,1)")
    if model.profile == "safe":
        hazard = -np.log1p(-u)
        cumulative = np.cumsum(np.diff(model.grid)[:98] * model.rates[:98])
        category = np.searchsorted(cumulative, hazard, side="right")
        result = np.full(u.shape, np.inf)
        event = category < 98
        idx = category[event]
        before = np.r_[0.0, cumulative[:-1]][idx]
        result[event] = model.grid[idx] + (hazard[event] - before) / model.rates[idx]
    else:
        j = np.searchsorted(model.cdf, u, side="right")
        rate = model.rates[j]
        if np.any(rate <= 0):
            raise ArithmeticError("legacy inverse hazard rate is zero")
        conditional = np.where(j == 99, 1.0, -np.expm1(-(model.grid[j] - model.grid[j - 1]) * rate))
        result = model.grid[j - 1] - np.log1p(-u * conditional) / rate
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("legacy inverse event delay is not representable")
    return _freeze(result)


def prt_arrivals_from_uniforms(mean_interarrival: float, uniforms: ArrayLike) -> FloatArray:
    """Recovered finite arrival process: sort800 uniform points on[0,800*mean].

    These are not independent exponential inter-arrival times. Trial runners must
    detect exhaustion of the finite candidate stream explicitly.
    """
    mean = scalar(mean_interarrival, "mean_interarrival")
    u = finite(uniforms, "uniforms")
    if mean <= 0 or u.shape != (800,) or np.any((u < 0) | (u >= 1)):
        raise ValueError("require positive mean and800 uniforms in [0,1)")
    with np.errstate(over="ignore"):
        times = np.sort(u * 800 * mean)
    if np.any(~np.isfinite(times)):
        raise ArithmeticError("arrival times exceed representable range")
    return _freeze(times)
