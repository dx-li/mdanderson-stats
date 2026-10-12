"""Current UAROET summaries and prior-information diagnostics."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .uaroet import (
    _integer,
    _marginal_log_probabilities,
    uaroet_logits,
    uaroet_parameter_names,
)
from .uaroet_desktop_study import UAROETDesktopStudy
from .uaroet_fit import _draw_prior, _positive_mask


@dataclass(frozen=True)
class UAROETPriorESS:
    """Beta moment matching; this is prior information, not MCMC effective sample size."""

    efficacy: FloatArray
    toxicity: FloatArray
    efficacy_mean: float
    toxicity_mean: float
    mean: float
    minimum: float
    maximum: float
    draws: int


def uaroet_marginal_ess(efficacy: ArrayLike, toxicity: ArrayLike) -> UAROETPriorESS:
    """Match active ordinal marginal means/sample variances to beta shapes.

    Arrays have sample, dose, category axes. Zero-variance marginals do not
    identify finite beta shapes and are rejected. Finite negative estimates
    caused by empirical variance exceeding Bernoulli variance are reported
    unchanged; they are not a valid positive beta-prior information measure.
    """
    arrays = []
    for value in (efficacy, toxicity):
        raw = np.asarray(value)
        if raw.size > 2_000_000 or raw.dtype.kind not in "iuf":
            raise ValueError("marginal samples exceed two million real values")
        array = np.asarray(raw, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("marginal samples must be finite")
        arrays.append(array)
    e, t = arrays
    if (
        e.ndim != 3
        or t.ndim != 3
        or e.shape[:2] != t.shape[:2]
        or not 2 <= e.shape[0] <= 100_000
        or not 1 <= e.shape[1] <= 5
        or not 2 <= e.shape[2] <= 4
        or not 2 <= t.shape[2] <= 4
    ):
        raise ValueError(
            "marginal draws must have matching sample/dose axes and supported dimensions"
        )
    estimates = []
    for a in (e, t):
        if np.any(a < 0) or np.any(a > 1) or not np.allclose(a.sum(-1), 1, atol=1e-12, rtol=0):
            raise ValueError("each marginal draw must be a probability distribution")
        mean = a.mean(axis=0)
        variance = a.var(axis=0, ddof=1)
        if np.any(variance <= 0):
            raise ValueError("zero-variance marginals have no finite beta moment match")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                value = mean * (1 - mean) / variance - 1
            except FloatingPointError as exc:
                raise ArithmeticError("beta moment match exceeds floating-point range") from exc
        estimates.append(_freeze(value))
    combined = np.concatenate([a.ravel() for a in estimates])
    return UAROETPriorESS(
        estimates[0],
        estimates[1],
        float(estimates[0].mean()),
        float(estimates[1].mean()),
        float(combined.mean()),
        float(combined.min()),
        float(combined.max()),
        e.shape[0],
    )


def uaroet_desktop_prior_ess(
    study: UAROETDesktopStudy, *, draws: int = 10_000, rng: np.random.Generator
) -> UAROETPriorESS:
    """Direct independent prior draws using the package's positive-normal increment model.

    Correlation does not affect either marginal, so no copula rectangles or
    association draws are needed. This is not native RNG-stream equivalence.
    """
    if not isinstance(study, UAROETDesktopStudy) or not isinstance(rng, np.random.Generator):
        raise ValueError("require a UAROETDesktopStudy and explicit NumPy Generator")
    n = _integer(draws, "draws", 2, 100_000)
    names = uaroet_parameter_names(study.dose_count, study.efficacy_levels, study.toxicity_levels)
    parameters = _draw_prior(study.prior_mean, study.prior_sd, _positive_mask(names), n, rng)
    e = np.empty((n, study.dose_count, study.efficacy_levels))
    t = np.empty((n, study.dose_count, study.toxicity_levels))
    for i, p in enumerate(parameters):
        el, tl = uaroet_logits(
            p,
            dose_count=study.dose_count,
            efficacy_levels=study.efficacy_levels,
            toxicity_levels=study.toxicity_levels,
        )
        e[i] = np.exp(_marginal_log_probabilities(el))
        t[i] = np.exp(_marginal_log_probabilities(tl))
    return uaroet_marginal_ess(e, t)
