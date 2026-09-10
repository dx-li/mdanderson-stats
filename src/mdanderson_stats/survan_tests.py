"""SURVAN multi-group survival comparisons with within-stratum risk sets."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammaincc

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite


@dataclass(frozen=True)
class SurvivalGroupTest:
    groups: tuple[str | int, ...]
    method: str
    patients: FloatArray
    observed: FloatArray
    expected: FloatArray
    score: FloatArray
    covariance: FloatArray
    statistic: float
    degrees_of_freedom: int
    pvalue: float
    strata: tuple[str | int, ...]
    stratum_scores: FloatArray
    stratum_covariances: FloatArray


def _labels(values: ArrayLike, n: int, name: str) -> tuple[tuple[str | int, ...], NDArray[np.intp]]:
    a = np.asarray(values)
    if a.shape != (n,) or a.dtype.kind not in "iuUS":
        raise ValueError(f"{name} must be a matching vector of integer or string labels")
    levels, codes = np.unique(a, return_inverse=True)
    return tuple(levels.tolist()), codes


def survan_group_test(
    time: ArrayLike,
    event: ArrayLike,
    group: ArrayLike,
    *,
    strata: ArrayLike | None = None,
    method: str = "logrank",
) -> SurvivalGroupTest:
    """Log-rank or Gehan–Breslow test; events=1, right censors=0.

    Tied censors remain at risk for tied events. Gehan–Breslow uses the
    within-stratum risk count as its weight, with hypergeometric covariance.
    Only exact time ties are grouped; there is no left truncation or weighting
    of individual records. Singular comparisons use the estimable covariance
    rank; absence of any comparison information raises ValueError.
    """
    if np.iscomplexobj(time) or np.iscomplexobj(event):
        raise ValueError("time and event must be real")
    t, e = finite(time, "time"), count(event, "event")
    if t.ndim != 1 or not 2 <= t.size <= 1_000_000 or e.shape != t.shape:
        raise ValueError("time and event must be matching vectors with 2..1000000 records")
    if np.any(t < 0) or np.any(e > 1):
        raise ValueError("time must be nonnegative and event must be zero or one")
    if method not in ("logrank", "gehan_breslow"):
        raise ValueError("method must be logrank or gehan_breslow")
    groups, g = _labels(group, t.size, "group")
    levels, s = _labels(np.zeros(t.size, dtype=int) if strata is None else strata, t.size, "strata")
    k, ns = len(groups), len(levels)
    if not 2 <= k <= 100 or ns * k * k > 1_000_000 or t.size * k > 20_000_000:
        raise ValueError("require 2..100 groups, strata*groups² <= 1e6 and records*groups <= 2e7")
    scores = np.zeros((ns, k))
    covariances = np.zeros((ns, k, k))
    observed, expected = np.zeros(k), np.zeros(k)
    # Sort once by stratum; avoid a full-data mask for every stratum.
    order = np.argsort(s, kind="stable")
    boundaries = np.r_[0, np.cumsum(np.bincount(s))]
    for j in range(ns):
        ix = order[boundaries[j] : boundaries[j + 1]]
        tt, gg, ee = t[ix], g[ix], e[ix]
        times, ti = np.unique(tt, return_inverse=True)
        flat = ti * k + gg
        exits = np.bincount(flat, minlength=times.size * k).reshape(-1, k)
        deaths = np.bincount(flat, weights=ee, minlength=times.size * k).reshape(-1, k)
        risk = np.cumsum(exits[::-1], axis=0)[::-1].astype(float)
        active = deaths.sum(axis=1) > 0
        risk, deaths = risk[active], deaths[active]
        n, d = risk.sum(axis=1), deaths.sum(axis=1)
        proportion = risk / n[:, None]
        exp = d[:, None] * proportion
        weight = n if method == "gehan_breslow" else np.ones(n.size)
        scores[j] = np.sum(weight[:, None] * (deaths - exp), axis=0)
        observed += deaths.sum(axis=0)
        expected += exp.sum(axis=0)
        factor = np.divide(d * (n - d), n - 1, out=np.zeros_like(n), where=n > 1)
        weighted = weight**2 * factor
        covariances[j] = np.diag(np.sum(weighted[:, None] * proportion, axis=0)) - (
            proportion.T @ (weighted[:, None] * proportion)
        )
    score, covariance = scores.sum(axis=0), covariances.sum(axis=0)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, vectors = np.linalg.eigh(covariance)
    largest = float(eigenvalues[-1])
    if largest <= 0:
        raise ValueError("no information for a between-group survival comparison")
    tolerance = 64 * np.finfo(float).eps * k * largest
    if eigenvalues[0] < -tolerance:
        raise ArithmeticError("survival covariance is not positive semidefinite")
    keep = eigenvalues > tolerance
    df = int(keep.sum())
    projected = vectors.T @ score
    if np.linalg.norm(projected[~keep]) > 1e-8 * max(1, np.linalg.norm(score)):
        raise ArithmeticError("survival score lies outside the resolved covariance space")
    statistic = float(np.sum((projected[keep] / np.sqrt(eigenvalues[keep])) ** 2))
    return SurvivalGroupTest(
        groups,
        method,
        _freeze(np.bincount(g, minlength=k).astype(float)),
        _freeze(observed),
        _freeze(expected),
        _freeze(score),
        _freeze(covariance),
        statistic,
        df,
        float(gammaincc(df / 2, statistic / 2)),
        levels,
        _freeze(scores),
        _freeze(covariances),
    )
