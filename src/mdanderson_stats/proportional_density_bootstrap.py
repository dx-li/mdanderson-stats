"""Failure-only goodness-of-fit bootstrap from Shen et al. (2007), section 3.1."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, scalar
from .proportional_density import _fit_offset, proportional_density


@dataclass(frozen=True)
class ProportionalDensityBootstrap:
    """Conditional observed-failure GOF, distinct from the archive's Pepe area.

    Failed replicates are NaN and prevent a single p-value. Bounds include all
    possible exceedance outcomes of those replicates; no failures are discarded.
    """

    statistic: float
    tau: float
    bootstrap_statistics: FloatArray
    pvalue: float | None
    pvalue_lower: float
    pvalue_upper: float
    monte_carlo_standard_error: float | None
    failed_replicates: int
    failure_reasons: tuple[tuple[str, int], ...]


def _failure_area(
    time: FloatArray, fitted_mass: FloatArray, control_mass: FloatArray, tau: float
) -> float:
    """Exact integral of squared right-continuous CDF difference on [0,tau]."""
    keep = time < tau
    points = time[keep]
    difference = np.cumsum(fitted_mass - control_mass)[keep]
    return float(np.dot(np.diff(np.r_[points, tau]), difference**2))


def proportional_density_bootstrap(
    time: ArrayLike,
    event: ArrayLike,
    treatment: ArrayLike,
    *,
    replicates: int = 999,
    seed: int | None = None,
    tau: float | None = None,
    equal_censoring: bool = False,
) -> ProportionalDensityBootstrap:
    """Bootstrap the paper's alternative observed-failure GOF statistic.

    Draw each arm's fixed number of failures from its fitted observed-failure
    distribution, retain censoring offsets estimated from the original data,
    refit, and compare fitted versus empirical control failure CDFs. Uses unit
    weight and exact step integration to common follow-up (or a smaller tau).
    This does not calibrate treatment-effect tests or the disease-curve Pepe
    statistic. A seed makes the local NumPy generator reproducible.
    """
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or not 1 <= replicates <= 10000
    ):
        raise ValueError("replicates must be an integer in [1,10000]")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")
    fit = proportional_density(time, event, treatment, equal_censoring=equal_censoring)
    t, d, z = (np.asarray(a, dtype=float) for a in (time, event, treatment))
    counts = np.array([np.sum((z == j) & (d == 1)) for j in (0, 1)])
    if int(counts.sum()) * replicates > 50_000_000:
        raise ValueError("bootstrap requires at most 50,000,000 resampled failures")
    common_end = float(min(t[z == 0].max(), t[z == 1].max()))
    end = common_end if tau is None else scalar(tau, "tau")
    if not 0 < end <= common_end:
        raise ValueError("tau must be positive and no greater than common follow-up")
    control = t[(z == 0) & (d == 1)]
    locations = np.searchsorted(fit.time, control)
    empirical = np.bincount(locations, minlength=fit.time.size) / counts[0]
    statistic = _failure_area(fit.time, fit.observed_mass[:, 0], empirical, end)
    if not np.any(fit.time < end):
        raise ValueError("tau must extend beyond at least one failure time")
    # Score convergence makes these sums one to numerical precision; normalize
    # their floating-point summation error for categorical sampling.
    probabilities = fit.observed_mass / fit.observed_mass.sum(axis=0)
    offset = (
        np.log(float(counts[1]) / counts[0])
        + np.log(fit.censoring_survival[:, 1])
        - np.log(fit.censoring_survival[:, 0])
    )
    group = np.repeat([0.0, 1.0], counts)
    rng = np.random.default_rng(seed)
    statistics = np.full(replicates, np.nan)
    reasons: dict[str, int] = {}
    for i in range(replicates):
        chosen = np.r_[
            rng.choice(fit.time.size, size=counts[0], p=probabilities[:, 0]),
            rng.choice(fit.time.size, size=counts[1], p=probabilities[:, 1]),
        ]
        order = np.argsort(chosen, kind="stable")
        chosen, labels = chosen[order], group[order]
        sampled = fit.time[chosen]
        try:
            _, _, eta, _, _ = _fit_offset(sampled, labels, offset[chosen])
        except (ValueError, ArithmeticError) as exc:
            # A discrete bootstrap may produce an unidentified or separated fit.
            # Keep its place in the distribution and expose the calibration gap.
            reason = str(exc)
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        unique, starts = np.unique(sampled, return_index=True)
        fitted = np.add.reduceat(expit(-eta) / counts[0], starts)
        empirical = np.add.reduceat((labels == 0).astype(float) / counts[0], starts)
        statistics[i] = _failure_area(unique, fitted, empirical, end)
    failed = int(np.isnan(statistics).sum())
    exceedances = int(np.sum(statistics >= statistic))
    lower = (1 + exceedances) / (replicates + 1)
    upper = (1 + exceedances + failed) / (replicates + 1)
    pvalue = lower if failed == 0 else None
    return ProportionalDensityBootstrap(
        statistic,
        end,
        _freeze(statistics),
        pvalue,
        lower,
        upper,
        float(np.sqrt(lower * (1 - lower) / replicates)) if failed == 0 else None,
        failed,
        tuple(sorted(reasons.items())),
    )
