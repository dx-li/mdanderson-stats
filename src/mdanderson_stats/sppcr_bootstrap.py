"""SPPCR replicate fitting and stable population bootstrap summaries."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import count, finite
from .sppcr_fit import BoolArray, FloatArray, SPPCRMeanFit, _freeze, _mask, sppcr_fit_means
from .sppcr_frequencies import sppcr_frequencies
from .sppcr_generate import SPPCRSamples, sppcr_generate, sppcr_observed_probabilities


@dataclass(frozen=True)
class SPPCRBootstrapSeries:
    """Replicate values, their mean, population variance and standard deviation.

    standard_deviation measures replicate spread, not Monte Carlo error of the
    bootstrap mean. NaN summaries indicate at least one undefined replicate.
    """

    values: FloatArray
    mean: FloatArray
    variance: FloatArray
    standard_deviation: FloatArray


def _series(values: FloatArray) -> SPPCRBootstrapSeries:
    known = np.all(np.isfinite(values), axis=0)
    x = np.where(known, values, 0)
    # Center before squaring, then scale residuals to avoid cancellation and
    # intermediate overflow. Input values are nonnegative, so differences fit.
    delta = x - x[0]
    delta_scale = np.max(np.abs(delta), axis=0)
    scaled_delta = np.divide(delta, delta_scale, out=np.zeros_like(delta), where=delta_scale > 0)
    offset = delta_scale * np.mean(scaled_delta, axis=0)
    mean = x[0] + offset
    residual = delta - offset
    scale = np.max(np.abs(residual), axis=0)
    normalized = np.divide(residual, scale, out=np.zeros_like(residual), where=scale > 0)
    with np.errstate(over="ignore", under="ignore"):
        sd = scale * np.sqrt(np.mean(normalized * normalized, axis=0))
        variance = sd * sd
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(variance)):
        raise ArithmeticError("SPPCR bootstrap summary exceeds finite float64 range")
    return SPPCRBootstrapSeries(
        _freeze(values),
        _freeze(np.where(known, mean, np.nan)),
        _freeze(np.where(known, variance, np.nan)),
        _freeze(np.where(known, sd, np.nan)),
    )


@dataclass(frozen=True)
class SPPCRBootstrapSummary:
    """All replicates retained; defined marks positive-total-mean experiments."""

    progenitor: tuple[int, int]
    defined: BoolArray
    mu: SPPCRBootstrapSeries
    calibration: SPPCRBootstrapSeries
    frequency: SPPCRBootstrapSeries
    transformed_frequency: SPPCRBootstrapSeries
    mutant: SPPCRBootstrapSeries
    transformed_mutant: SPPCRBootstrapSeries


def sppcr_bootstrap_summary(mu: ArrayLike, *, progenitor: tuple[int, int]) -> SPPCRBootstrapSummary:
    """Summarize means shaped (replicates, ..., alleles), using divisor B.

    Replicates must be nonempty, finite and nonnegative. A zero total leaves
    frequencies undefined; their values and experiment summaries are NaN. No
    replicate is omitted. Mean and calibration summaries remain available.
    """
    m = finite(mu, "mu")
    if m.ndim < 2 or m.shape[0] == 0 or m.shape[-1] == 0 or np.any(m < 0):
        raise ValueError("mu must have nonempty replicate and allele axes and nonnegative values")
    defined = np.any(m > 0, axis=-1)
    frequencies = sppcr_frequencies(m[defined], progenitor=progenitor)
    p, t = np.full(m.shape, np.nan), np.full(m.shape, np.nan)
    mutant, tm = np.full(defined.shape, np.nan), np.full(defined.shape, np.nan)
    p[defined], t[defined] = frequencies.frequency.value, frequencies.frequency.transformed.value
    mutant[defined], tm[defined] = frequencies.mutant.value, frequencies.mutant.transformed.value
    with np.errstate(over="ignore"):
        total = m.sum(axis=-1)
    if np.any(~np.isfinite(total)):
        raise ArithmeticError("SPPCR total mean exceeds finite float64 range")
    return SPPCRBootstrapSummary(
        frequencies.progenitor,
        _mask(defined),
        _series(m),
        _series(total),
        _series(p),
        _series(t),
        _series(mutant),
        _series(tm),
    )


@dataclass(frozen=True)
class SPPCRBootstrap:
    """Observed fit, generated counts, replicate fits and population summaries."""

    observed_fit: SPPCRMeanFit
    samples: SPPCRSamples
    fit: SPPCRMeanFit
    summary: SPPCRBootstrapSummary


def sppcr_bootstrap(
    dna: ArrayLike,
    seen: ArrayLike,
    wells: ArrayLike,
    *,
    progenitor: tuple[int, int],
    rng: np.random.Generator,
    replicates: int = 1000,
    probability: ArrayLike | None = None,
    saturation: str = "half",
) -> SPPCRBootstrap:
    """Generate, fit and summarize B independent SPPCR bootstrap replicates.

    Defaults to original observed cell fractions, as in the source. An explicit
    probability array broadcasting to seen selects a caller-specified model.
    saturation applies to both observed and replicate fits. The default half
    policy adjusts only alleles detected in every well at every DNA level.
    Numerical fit failures propagate; RNG state is consumed once sampling starts.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    observed = sppcr_fit_means(dna, seen, wells, saturation=saturation)
    parents = count(progenitor, "progenitor")
    if parents.shape != (2,) or np.any(parents >= observed.mu.shape[-1]):
        raise ValueError("progenitor must contain two valid zero-based allele indices")
    p = (
        sppcr_observed_probabilities(observed.original_seen, observed.wells)
        if probability is None
        else np.broadcast_to(finite(probability, "probability"), observed.original_seen.shape)
    )
    samples = sppcr_generate(p, observed.wells, rng=rng, replicates=replicates)
    fit = sppcr_fit_means(observed.dna, samples.seen, samples.wells, saturation=saturation)
    return SPPCRBootstrap(
        observed, samples, fit, sppcr_bootstrap_summary(fit.mu, progenitor=progenitor)
    )
