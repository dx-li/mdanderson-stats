"""SPPCR normal/bootstrap confidence limits with explicit support boundaries."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import finite, scalar
from .sppcr_bootstrap import SPPCRBootstrap
from .sppcr_fit import BoolArray, FloatArray, _freeze, _mask
from .sppcr_frequencies import sppcr_frequencies


@dataclass(frozen=True)
class SPPCRInterval:
    """Estimate, limits, availability and support-clipping diagnostics.

    Unavailable limits are NaN. A reciprocal calibration upper bound may be +inf
    when the corresponding calibration interval includes zero.
    """

    estimate: FloatArray
    lower: FloatArray
    upper: FloatArray
    available: BoolArray
    lower_clipped: BoolArray
    upper_clipped: BoolArray


@dataclass(frozen=True)
class SPPCRIntervals:
    """Intervals use observed estimates and bootstrap spread, not bootstrap means."""

    multiplier: float
    progenitor: tuple[int, int]
    calibration: SPPCRInterval
    inverse_calibration: SPPCRInterval
    frequency: SPPCRInterval
    mutant: SPPCRInterval


def _sd(value: ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    s = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
    if np.any(np.isinf(s)) or np.any(s < 0):
        raise ValueError(f"{name} must be nonnegative and finite, or NaN for unavailable")
    return s


def _interval(
    estimate: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    known: BoolArray,
    lower_clipped: BoolArray | bool,
    upper_clipped: BoolArray | bool,
) -> SPPCRInterval:
    return SPPCRInterval(
        _freeze(estimate),
        _freeze(np.where(known, lower, np.nan)),
        _freeze(np.where(known, upper, np.nan)),
        _mask(known),
        _mask(known & lower_clipped),
        _mask(known & upper_clipped),
    )


def _frequency_interval(
    p: FloatArray, angle: FloatArray, sd: FloatArray, z: float
) -> SPPCRInterval:
    known = np.isfinite(p) & np.isfinite(sd)
    with np.errstate(over="ignore", invalid="ignore"):
        width = z * sd
        lo, hi = angle - width, angle + width
    # Width overflow means the finite-support interval covers all [0, 1].
    a, b = np.clip(lo, 0, np.pi), np.clip(hi, 0, np.pi)
    lower, upper = np.sin(a / 2) ** 2, np.sin(b / 2) ** 2
    # Preserve exact point intervals and inclusion despite transform roundoff.
    lower, upper = np.minimum(lower, p), np.maximum(upper, p)
    lower, upper = np.where(sd == 0, p, lower), np.where(sd == 0, p, upper)
    return _interval(p, lower, upper, known, lo < 0, hi > np.pi)


def sppcr_intervals(
    mu: ArrayLike,
    *,
    progenitor: tuple[int, int],
    calibration_sd: ArrayLike,
    frequency_transformed_sd: ArrayLike,
    mutant_transformed_sd: ArrayLike,
    multiplier: float = 1.959964,
) -> SPPCRIntervals:
    """Build source-style confidence limits from means and bootstrap deviations.

    mu has shape (..., alleles). Calibration and mutant SDs broadcast to (...,);
    frequency transformed SD broadcasts to mu. NaN SD means unavailable. The
    default multiplier is the source's rounded 95% normal critical value; custom
    positive finite multipliers are accepted without asserting a confidence level.
    Calibration limits are clipped at zero; frequency angles at [0, pi].
    """
    m = finite(mu, "mu")
    if m.ndim < 1 or m.shape[-1] == 0 or np.any(m < 0):
        raise ValueError("mu must have a nonempty allele axis and nonnegative values")
    z = scalar(multiplier, "multiplier")
    if z <= 0:
        raise ValueError("multiplier must be positive")
    c_sd = _sd(calibration_sd, m.shape[:-1], "calibration_sd")
    f_sd = _sd(frequency_transformed_sd, m.shape, "frequency_transformed_sd")
    t_sd = _sd(mutant_transformed_sd, m.shape[:-1], "mutant_transformed_sd")
    defined = np.any(m > 0, axis=-1)
    f = sppcr_frequencies(m[defined], progenitor=progenitor)
    p, angle = np.full(m.shape, np.nan), np.full(m.shape, np.nan)
    mutant, mutant_angle = np.full(defined.shape, np.nan), np.full(defined.shape, np.nan)
    p[defined], angle[defined] = f.frequency.value, f.frequency.transformed.value
    mutant[defined], mutant_angle[defined] = f.mutant.value, f.mutant.transformed.value
    with np.errstate(over="ignore", invalid="ignore"):
        total = m.sum(axis=-1)
        width = z * c_sd
        lo, hi = total - width, total + width
    known = np.isfinite(c_sd)
    if np.any(~np.isfinite(total)) or np.any(known & (~np.isfinite(lo) | ~np.isfinite(hi))):
        raise ArithmeticError("SPPCR calibration interval exceeds finite float64 range")
    lower = np.maximum(lo, 0)
    calibration = _interval(total, lower, hi, known, lo < 0, False)
    inverse_known = known & (total > 0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inverse_point = np.where(total > 0, 1 / total, np.nan)
        inverse_lo, inverse_hi = 1 / hi, 1 / lower
    if np.any((total > 0) & ~np.isfinite(inverse_point)) or np.any(
        inverse_known & ((~np.isfinite(inverse_lo)) | ((lower > 0) & ~np.isfinite(inverse_hi)))
    ):
        raise ArithmeticError("SPPCR reciprocal calibration exceeds finite float64 range")
    inverse = _interval(inverse_point, inverse_lo, inverse_hi, inverse_known, False, lo < 0)
    return SPPCRIntervals(
        z,
        f.progenitor,
        calibration,
        inverse,
        _frequency_interval(p, angle, f_sd, z),
        _frequency_interval(mutant, mutant_angle, t_sd, z),
    )


def sppcr_bootstrap_intervals(
    result: SPPCRBootstrap, *, multiplier: float = 1.959964
) -> SPPCRIntervals:
    """Use observed means and corresponding bootstrap standard deviations."""
    s = result.summary
    return sppcr_intervals(
        result.observed_fit.mu,
        progenitor=s.progenitor,
        calibration_sd=s.calibration.standard_deviation,
        frequency_transformed_sd=s.transformed_frequency.standard_deviation,
        mutant_transformed_sd=s.transformed_mutant.standard_deviation,
        multiplier=multiplier,
    )
