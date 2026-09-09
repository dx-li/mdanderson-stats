"""SPPCR frequency ratios and independent-mean delta-method uncertainty."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .sppcr_fit import BoolArray, FloatArray, _freeze, _mask


@dataclass(frozen=True)
class SPPCREstimate:
    """Estimate and standard error; NaN variance explicitly means unavailable."""

    value: FloatArray
    variance: FloatArray
    standard_error: FloatArray

    @property
    def variance_available(self) -> BoolArray:
        return _mask(np.isfinite(self.variance))


@dataclass(frozen=True)
class SPPCRProportion(SPPCREstimate):
    """A frequency, its direct complement and its 2*asin(sqrt(p)) transform."""

    complement: FloatArray
    transformed: SPPCREstimate


@dataclass(frozen=True)
class SPPCRFrequencies:
    """Immutable summaries; the final input axis indexes alleles.

    Progenitor indices are zero-based; repeat the index for a homozygote.
    Frequency arrays retain the allele axis; calibration and mutant arrays omit it.
    """

    mu: FloatArray
    mu_variance: FloatArray
    progenitor: tuple[int, int]
    calibration: SPPCREstimate
    frequency: SPPCRProportion
    mutant: SPPCRProportion


def _others(log_values: FloatArray) -> FloatArray:
    """Log sums excluding each allele, without subtraction or quadratic storage."""
    blank = np.full(log_values.shape[:-1] + (1,), -np.inf)
    left = np.concatenate((blank, np.logaddexp.accumulate(log_values, axis=-1)[..., :-1]), axis=-1)
    right = np.concatenate(
        (np.logaddexp.accumulate(log_values[..., ::-1], axis=-1)[..., -2::-1], blank), axis=-1
    )
    return np.logaddexp(left, right)


def _estimate(value: FloatArray, log_variance: FloatArray, known: BoolArray) -> SPPCREstimate:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        variance = np.where(known, np.exp(log_variance), np.nan)
        # Compute SD separately so a representable SD survives variance underflow.
        sd = np.where(known, np.exp(log_variance / 2), np.nan)
    if np.any(~np.isfinite(value)) or np.any(np.isinf(variance)) or np.any(np.isinf(sd)):
        raise ArithmeticError("SPPCR summary exceeds finite float64 range")
    if np.any(known & (np.isnan(variance) | np.isnan(sd))):
        raise ArithmeticError("invalid SPPCR uncertainty calculation")
    return SPPCREstimate(_freeze(value), _freeze(variance), _freeze(sd))


def _proportion(
    x: FloatArray,
    y: FloatArray,
    vx: FloatArray,
    vy: FloatArray,
    total: FloatArray,
    known: BoolArray,
    *,
    constant: bool = False,
) -> SPPCRProportion:
    # Arguments are logarithms of group means, variances and total mean.
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        lp = -np.logaddexp(0, y - x)
        lq = -np.logaddexp(0, x - y)
        small_p, small_q = np.exp(lp), np.exp(lq)
        p = np.where(lp <= lq, small_p, -np.expm1(lq))
        q = np.where(lq <= lp, small_q, -np.expm1(lp))
        log_var = np.logaddexp(2 * lq + vx, 2 * lp + vy) - 2 * total
        angle = 2 * np.arctan2(np.exp(lp / 2), np.exp(lq / 2))
        log_transformed_var = log_var - lp - lq
    transformed_known = known & np.isfinite(lp) & np.isfinite(lq)
    if constant:
        log_var = np.full_like(p, -np.inf)
        log_transformed_var = np.full_like(p, -np.inf)
        known = np.ones(p.shape, dtype=np.bool_)
        transformed_known = known
    raw = _estimate(p, log_var, known)
    transformed = _estimate(angle, log_transformed_var, transformed_known)
    return SPPCRProportion(raw.value, raw.variance, raw.standard_error, _freeze(q), transformed)


def sppcr_frequencies(
    mu: ArrayLike,
    variance: ArrayLike | None = None,
    *,
    progenitor: tuple[int, int],
) -> SPPCRFrequencies:
    """Summarize nonnegative allele means and independent mean variances.

    mu has shape (..., alleles) and positive total in every experiment. variance
    broadcasts to mu; omitted or NaN entries mean unavailable uncertainty. Negative
    and infinite variances are invalid. All mean variances must be available for
    ordinary delta-method uncertainty; structural constants remain exactly known.
    Pass a fitted result's mu and variance to preserve its boundary diagnostics.
    """
    means = finite(mu, "mu")
    if means.ndim < 1 or means.shape[-1] == 0 or np.any(means < 0):
        raise ValueError("mu must have a nonempty allele axis and nonnegative values")
    parents = count(progenitor, "progenitor")
    if parents.shape != (2,) or np.any(parents >= means.shape[-1]):
        raise ValueError("progenitor must contain two valid zero-based allele indices")
    indices = (int(parents[0]), int(parents[1]))
    if variance is None:
        variances = np.full(means.shape, np.nan)
    else:
        variances = np.broadcast_to(np.asarray(variance, dtype=np.float64), means.shape)
        if np.any(np.isinf(variances)) or np.any(variances < 0):
            raise ValueError("variance must be nonnegative and finite, or NaN for unavailable")
    known = np.all(~np.isnan(variances), axis=-1)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        total = means.sum(axis=-1)
        if np.any(total == 0):
            raise ValueError("allele frequencies are undefined when all means are zero")
        lm = np.log(means)
        lt = np.log(total)
        lv = np.log(np.where(np.isnan(variances), 0, variances))
        lvt = np.logaddexp.reduce(lv, axis=-1)
    calibration = _estimate(total, lvt, known)
    frequency = _proportion(
        lm,
        _others(lm),
        lv,
        _others(lv),
        lt[..., None],
        known[..., None],
        constant=means.shape[-1] == 1,
    )
    mutants = np.ones(means.shape[-1], dtype=np.bool_)
    mutants[list(indices)] = False
    mutant = _proportion(
        np.logaddexp.reduce(lm[..., mutants], axis=-1),
        np.logaddexp.reduce(lm[..., ~mutants], axis=-1),
        np.logaddexp.reduce(lv[..., mutants], axis=-1),
        np.logaddexp.reduce(lv[..., ~mutants], axis=-1),
        lt,
        known,
        constant=not np.any(mutants),
    )
    return SPPCRFrequencies(
        _freeze(means), _freeze(variances), indices, calibration, frequency, mutant
    )
