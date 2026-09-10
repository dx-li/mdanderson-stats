"""Two-arm Efron Cox comparison used in IPDfromKM survival analysis."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import expit, ndtr, ndtri

from ._validation import FloatArray, scalar
from .expsurv import ExploratorySurvival, exploratory_survival


@dataclass(frozen=True)
class IPDCoxComparison:
    log_hazard_ratio: float
    standard_error: float
    hazard_ratio: float
    lower: float
    upper: float
    score_statistic: float
    score_pvalue: float
    events: int
    patients: int
    confidence: float


def _risk_events(curve: ExploratorySurvival, times: FloatArray) -> tuple[FloatArray, FloatArray]:
    index = np.searchsorted(curve.time, times)
    bounded = np.minimum(index, len(curve.time) - 1)
    risk = np.where(index < len(curve.time), curve.at_risk[bounded], 0)
    deaths = np.where(
        (index < len(curve.time)) & (curve.time[bounded] == times), curve.events[bounded], 0
    )
    return risk, deaths


def ipd_cox_compare(
    time1: ArrayLike,
    event1: ArrayLike,
    time2: ArrayLike,
    event2: ArrayLike,
    *,
    confidence: float = 0.95,
) -> IPDCoxComparison:
    """Fit a binary-treatment Cox model with Efron ties; HR is arm 2 / arm 1.

    Events are 1 and right censors 0. Tied censors remain at risk for failures
    at that time. Only exact time ties are grouped. The score test is evaluated
    at log-HR=0; it is not the ordinary hypergeometric logrank test with ties.
    Separation and absence of treatment information raise ValueError.
    """
    first, second = exploratory_survival(time1, event1), exploratory_survival(time2, event2)
    patients = first.n_observations + second.n_observations
    if patients > 1_000_000:
        raise ValueError("combined patient count must not exceed 1000000")
    level = scalar(confidence, "confidence")
    if not 1e-6 <= level <= 1 - 1e-12:
        raise ValueError("confidence must be in [1e-6, 1-1e-12]")
    times = np.union1d(first.time[first.events > 0], second.time[second.events > 0])
    if not times.size:
        raise ValueError("no events are available for Cox comparison")
    risk0, d0 = _risk_events(first, times)
    risk1, d1 = _risk_events(second, times)
    deaths = (d0 + d1).astype(int)
    events = int(deaths.sum())
    offsets = np.repeat(np.cumsum(deaths) - deaths, deaths)
    fraction = (np.arange(events) - offsets) / np.repeat(deaths, deaths)
    a0 = np.repeat(risk0, deaths) - np.repeat(d0, deaths) * fraction
    a1 = np.repeat(risk1, deaths) - np.repeat(d1, deaths) * fraction
    observed = float(d1.sum())
    # Exact limiting scores detect a boundary optimum without an arbitrary
    # huge finite coefficient or a false convergence from vanishing curvature.
    lower_score = observed - np.count_nonzero(a0 == 0)
    upper_score = observed - np.count_nonzero(a1 > 0)
    if lower_score <= 0 or upper_score >= 0:
        raise ValueError("no finite Cox estimate: separation or no treatment information")
    with np.errstate(divide="ignore"):
        log_odds = np.log(a1) - np.log(a0)

    def evaluate(beta: float) -> tuple[float, float]:
        positive, negative = expit(log_odds + beta), expit(-log_odds - beta)
        return observed - float(positive.sum()), float(np.sum(positive * negative))

    beta = float(brentq(lambda value: evaluate(value)[0], -64, 64, xtol=1e-12))
    _, information = evaluate(beta)
    if information <= 0:
        raise ArithmeticError("Cox information is not positive at the fitted coefficient")
    se = float(1 / np.sqrt(information))
    score, null_information = evaluate(0)
    statistic = score * score / null_information
    z = float(ndtri((1 + level) / 2))
    with np.errstate(over="ignore", under="ignore"):
        hr, low, high = np.exp([beta, beta - z * se, beta + z * se])
    return IPDCoxComparison(
        beta,
        se,
        float(hr),
        float(low),
        float(high),
        statistic,
        float(2 * ndtr(-np.sqrt(statistic))),
        events,
        patients,
        level,
    )
