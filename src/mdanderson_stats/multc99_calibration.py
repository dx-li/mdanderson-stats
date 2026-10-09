"""Source-bounded Multc99 prior elicitation and posterior-width planning.

Adapted workflow retains the Multc99 source's noncommercial terms; see
notices/mdanderson-multc99-readme.txt. Searches terminate explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import betainc, betaincc, betaincinv

from ._cdflib import _freeze
from ._validation import FloatArray
from .multc99 import Multc99Design, _integer, _mask, _prior, _real


@dataclass(frozen=True)
class Multc99PriorCalibration:
    dirichlet_parameters: FloatArray
    elementary_means: FloatArray
    selected_event_mean: float
    interval: tuple[float, float]
    requested_coverage: float
    achieved_coverage: float
    concentration: float


def multc99_prior_from_interval(
    elementary_means: ArrayLike,
    event_definition: ArrayLike,
    *,
    width: float,
    coverage: float,
) -> Multc99PriorCalibration:
    """Elicit a Dirichlet prior through one compound-event Beta interval.

    The selected event's mean-centered interval has the requested width and
    prior probability. Search follows the native selected-event alpha bracket
    [1e-7, 10000], additionally respecting the Python concentration cap 1e6.
    Unbracketed targets raise instead of the native unlimited bisection loop.
    """
    means = _prior(elementary_means, "elementary_means", mixture=False)
    if abs(float(means.sum()) - 1) > 1e-12:
        raise ValueError("elementary_means must be positive and sum to one")
    means = means / means.sum()
    definition = _mask(event_definition, "event_definition")
    if len(definition) != means.size or not 0 < sum(definition) < len(definition):
        raise ValueError("event_definition must be a nonempty proper subset matching means")
    interval_width = _real(width, "width", np.finfo(float).eps, 1)
    target = _real(coverage, "coverage", np.finfo(float).eps, 1 - np.finfo(float).eps)
    mu = float(np.array(definition) @ means)
    low, high = max(0.0, mu - interval_width / 2), min(1.0, mu + interval_width / 2)

    def mass(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))
        beta = alpha * ((1 - mu) / mu)
        if mu <= 0.5:
            return float(betainc(alpha, beta, high) - betainc(alpha, beta, low))
        return float(betaincc(alpha, beta, low) - betaincc(alpha, beta, high))

    left, right = float(np.log(1e-7)), float(np.log(min(10_000, 1e6 * mu)))
    fleft, fright = mass(left) - target, mass(right) - target
    if left >= right or fleft * fright > 0:
        raise ValueError(
            "requested interval probability is not bracketed by the source-bounded search"
        )
    root = brentq(lambda a: mass(a) - target, left, right, xtol=1e-12, rtol=1e-13, maxiter=200)
    concentration = float(np.exp(root) / mu)
    achieved = mass(root)
    if not np.isfinite(concentration) or abs(achieved - target) > 1e-9:
        raise ArithmeticError("prior interval calibration did not attain its target")
    return Multc99PriorCalibration(
        _freeze(means * concentration),
        _freeze(means),
        mu,
        (low, high),
        target,
        achieved,
        concentration,
    )


@dataclass(frozen=True)
class Multc99PrecisionSampleSize:
    sample_size: int
    event_count: int
    posterior_mean: float
    equal_tailed_interval: tuple[float, float]
    interval_width: float
    target_posterior_mean: float
    coverage: float


def multc99_precision_sample_size(
    design: Multc99Design,
    event_name: str,
    *,
    target_posterior_mean: float,
    width: float,
    coverage: float,
    search_cap: int = 500,
) -> Multc99PrecisionSampleSize:
    """First source-rounded posterior interval whose width is <= the target.

    The source chooses the floor/ceiling count whose posterior mean is closer
    to 0.5, preferring ceiling on a tie, rather than closer to the target mean.
    Python retains this declared conservative rounding and excludes impossible
    counts. For a conditional event the sample size counts conditioning events,
    not total trial enrollment. The explicit search cap is independent of the
    existing design's cap; reaching it without success raises.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    names = tuple(e.name for e in design.events)
    if event_name not in names:
        raise ValueError("unknown event_name")
    cap = _integer(search_cap, "search_cap", 1, 500)
    target = _real(
        target_posterior_mean, "target_posterior_mean", np.finfo(float).eps, 1 - np.finfo(float).eps
    )
    requested_width = _real(width, "width", np.finfo(float).eps, 1)
    mass = _real(coverage, "coverage", np.finfo(float).eps, 1 - np.finfo(float).eps)
    alpha, beta = design._experimental_beta[names.index(event_name)]
    for n in range(1, cap + 1):
        expected = target * (alpha + beta + n) - alpha
        floor, ceiling = int(np.floor(expected)), int(np.ceil(expected))
        feasible = sorted({x for x in (floor, ceiling) if 0 <= x <= n})
        if not feasible:
            continue
        x = min(
            feasible, key=lambda count: (abs((alpha + count) / (alpha + beta + n) - 0.5), -count)
        )
        posterior_alpha, posterior_beta = alpha + x, beta + n - x
        lo = float(betaincinv(posterior_alpha, posterior_beta, (1 - mass) / 2))
        hi = float(betaincinv(posterior_alpha, posterior_beta, (1 + mass) / 2))
        if not np.isfinite(lo + hi) or not 0 <= lo <= hi <= 1:
            raise ArithmeticError("posterior interval quantiles could not be resolved")
        if hi - lo <= requested_width:
            return Multc99PrecisionSampleSize(
                n,
                x,
                float(posterior_alpha / (posterior_alpha + posterior_beta)),
                (lo, hi),
                hi - lo,
                target,
                mass,
            )
    raise ValueError("posterior precision target is not achieved within search_cap")
