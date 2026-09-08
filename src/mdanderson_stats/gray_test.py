"""Gray's weighted, stratified comparison of competing-risk incidence."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincc

from ._validation import FloatArray, count, finite, scalar


@dataclass(frozen=True)
class GrayTest:
    """Scores/covariance refer to all groups except the last (reference) group.

    Singular covariance gives statistic=pvalue=None, not a fabricated test.
    The chi-square approximation has degrees_freedom=len(groups)-1.
    """

    statistic: float | None
    pvalue: float | None
    degrees_freedom: int
    rank: int
    score: FloatArray
    covariance: FloatArray
    groups: tuple[str | float, ...]
    strata: tuple[str | float, ...]
    rho: float


def _labels(
    values: ArrayLike, shape: tuple[int, ...], name: str
) -> tuple[tuple[str | float, ...], NDArray[np.int64]]:
    labels = np.asarray(values, dtype=object)
    if labels.shape != shape:
        raise ValueError(f"{name} must match the time vector")
    strings = all(isinstance(x, str) for x in labels)
    numbers = all(isinstance(x, Real) and not isinstance(x, (bool, np.bool_)) for x in labels)
    if not strings and not numbers:
        raise ValueError(f"{name} must contain all strings or all finite numeric labels")
    if numbers and not np.all(np.isfinite(np.asarray(labels, dtype=float))):
        raise ValueError(f"{name} must not contain missing or nonfinite labels")
    unique, inverse = np.unique(labels, return_inverse=True)
    return tuple(unique.tolist()), inverse.astype(np.int64)


def gray_test(
    time: ArrayLike,
    event: ArrayLike,
    group: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    event_of_interest: int = 1,
    censor: int = 0,
    rho: float = 0,
) -> GrayTest:
    """Compare one cause's incidence across groups, optionally within strata.

    Other noncensoring event codes are competing events. Time/event conventions
    match cumulative_incidence. Group/stratum labels are sorted strings or finite
    numbers, with missing values rejected. Weight is (1-pooled_incidence)**rho.
    Stratum scores and covariance matrices are summed before testing.
    """
    t, code = finite(time, "time"), count(event, "event")
    if t.ndim != 1 or t.size == 0 or code.shape != t.shape or np.any(t < 0):
        raise ValueError("Require nonempty matching time/event vectors and nonnegative times")
    for value, name in [(event_of_interest, "event_of_interest"), (censor, "censor")]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not 0 <= value < 2**53
        ):
            raise ValueError(f"{name} must be a nonnegative integer smaller than 2**53")
    if event_of_interest == censor:
        raise ValueError("event_of_interest must differ from censor")
    power = scalar(rho, "rho")
    groups, g = _labels(group, t.shape, "group")
    levels, st = _labels(np.zeros(t.size) if strata is None else strata, t.shape, "strata")
    ng = len(groups)
    if ng < 2:
        raise ValueError("Gray's test requires at least two groups")
    status = np.where(code == censor, 0, np.where(code == event_of_interest, 1, 2))
    score, covariance = np.zeros(ng - 1), np.zeros((ng - 1, ng - 1))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for k in range(len(levels)):
                selected = st == k
                s, v = _stratum(t[selected], status[selected], g[selected], ng, power)
                score += s
                covariance += v
    except FloatingPointError as exc:
        raise RuntimeError("Gray score/covariance calculation is numerically invalid") from exc
    covariance = covariance / 2 + covariance.T / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = ng * np.finfo(float).eps * float(np.max(np.abs(eigenvalues)))
    if eigenvalues[0] < -128 * tolerance:
        raise RuntimeError("Gray covariance is not positive semidefinite")
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    statistic = pvalue = None
    if rank == ng - 1:
        statistic = float(score @ np.linalg.solve(covariance, score))
        if not np.isfinite(statistic) or statistic < 0:
            raise RuntimeError("Gray test statistic is numerically invalid")
        pvalue = float(gammaincc((ng - 1) / 2, statistic / 2))
    score.flags.writeable = covariance.flags.writeable = False
    return GrayTest(statistic, pvalue, ng - 1, rank, score, covariance, groups, levels, power)


def _stratum(
    time: FloatArray, status: NDArray[np.int64], group: NDArray[np.int64], ng: int, rho: float
) -> tuple[FloatArray, FloatArray]:
    """Aggregate time/group counts, then accumulate Gray score influence moments."""
    order = np.argsort(time, kind="stable")
    _, starts, sizes = np.unique(time[order], return_index=True, return_counts=True)
    status, group = status[order], group[order]
    risk = np.bincount(group, minlength=ng).astype(float)
    survival, incidence = np.ones(ng), np.zeros(ng)
    pooled = 0.0
    score, variance = np.zeros(ng - 1), np.zeros((ng - 1, ng - 1))
    integrated = np.zeros((ng - 1, ng))
    cross, second = np.zeros((ng - 1, ng)), np.zeros(ng)
    for start, size in zip(starts, sizes, strict=True):
        sl = slice(start, start + size)
        counts = np.bincount(status[sl] * ng + group[sl], minlength=3 * ng).reshape(3, ng)
        target, competing = counts[1], counts[2]
        total = int(target.sum())
        active = risk > 0
        if np.any(target + competing):
            h = np.divide(risk, survival, out=np.zeros(ng), where=active)
            h_total = h.sum()
            adjusted = h * (1 - incidence)
            next_survival = survival * (
                1 - np.divide(target + competing, risk, out=np.zeros(ng), where=active)
            )
            next_incidence = incidence + survival * np.divide(
                target, risk, out=np.zeros(ng), where=active
            )
            next_pooled = pooled + total / h_total
            weight = (1 - pooled) ** rho
            influence = weight * (np.diag(h) - np.outer(h, h) / h_total)[:-1]
            integrated += influence * (total / (h_total * (1 - pooled)))
            score += weight * (target - total * adjusted / adjusted.sum())[:-1]
            if total:
                q = np.ones(ng)
                positive = next_survival > 0
                q[positive] -= (1 - next_pooled) / next_survival[positive]
                correction = np.ones(ng)
                if total > 1:
                    correction[active] -= (total - 1) / (h_total * survival[active] - 1)
                mass = np.divide(
                    correction * survival * total,
                    h_total * risk,
                    out=np.zeros(ng),
                    where=active,
                )
                residual = influence - integrated * q
                variance += (residual * mass) @ residual.T
                cross += residual * (q * mass)
                second += q * q * mass
            selected = (next_survival > 0) & (competing > 0)
            if np.any(selected):
                q = np.divide(1 - next_pooled, next_survival, out=np.zeros(ng), where=selected)
                correction = np.ones(ng)
                tied = competing > 1
                correction[tied] -= (competing[tied] - 1) / (risk[tied] - 1)
                mass = np.divide(
                    correction * survival**2 * competing,
                    risk**2,
                    out=np.zeros(ng),
                    where=selected,
                )
                weighted = q * q * mass
                variance += (integrated * weighted) @ integrated.T
                cross -= integrated * weighted
                second += weighted
            survival, incidence, pooled = next_survival, next_incidence, next_pooled
        risk -= counts.sum(axis=0)
    variance += (integrated * second) @ integrated.T
    variance += integrated @ cross.T + cross @ integrated.T
    return score, variance
