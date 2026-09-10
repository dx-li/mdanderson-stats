"""Numerical hazard-peak and admissible-range search for CONFINT survival."""

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from ._validation import scalar
from .bayesian_monitoring import _integer
from .confint_binomial import _assurance
from .confint_survival import CONFINTSurvivalAssurance, confint_survival_probability


@dataclass(frozen=True)
class CONFINTSurvivalHazardRange:
    peak_hazard: float
    peak: CONFINTSurvivalAssurance
    intervals: tuple[tuple[float, float], ...]
    bounds: tuple[float, float]
    lower_clipped: bool
    upper_clipped: bool

    @property
    def peak_at_boundary(self) -> bool:
        return self.peak_hazard in self.bounds


def confint_survival_hazard_range(
    accrual_rate: float,
    accrual_time: float,
    followup_time: float,
    max_length: float,
    *,
    assurance: float = 0.9,
    confidence: float = 0.95,
    target: str = "hazard",
    hazard_bounds: tuple[float, float] = (1e-4, 1000),
    grid_points: int = 65,
    tail_tolerance: float = 1e-14,
) -> CONFINTSurvivalHazardRange:
    """Find a numerical peak and hazard intervals attaining the target assurance.

    Search only within hazard_bounds. Refine grid-local maxima AND minima in
    log-hazard coordinates before bracketing crossings. Intervals touching an
    accepted search endpoint are flagged as clipped, not claimed to end there.
    This finite-grid search is not a certificate that every narrow extremum or
    admissible component has been found. Both hazard and mean CIs are supported.
    """
    low, high = (scalar(value, "hazard_bounds") for value in hazard_bounds)
    if not 0 < low < high:
        raise ValueError("hazard_bounds must be positive and increasing")
    size, requested = _integer(grid_points, "grid_points"), _assurance(assurance)
    if not 17 <= size <= 1025:
        raise ValueError("grid_points must be in 17..1025")
    log_low, log_high = float(np.log(low)), float(np.log(high))
    if log_low >= log_high:
        raise ValueError("hazard_bounds are too close to resolve in log coordinates")

    def hazard_at(value: float) -> float:
        if value <= log_low:
            return low
        if value >= log_high:
            return high
        with np.errstate(over="ignore", under="ignore"):
            return float(np.clip(np.exp(value), low, high))

    @cache
    def evaluate(value: float) -> CONFINTSurvivalAssurance:
        return confint_survival_probability(
            hazard_at(value),
            accrual_rate,
            accrual_time,
            followup_time,
            max_length,
            confidence=confidence,
            target=target,
            tail_tolerance=tail_tolerance,
        )

    grid = np.linspace(log_low, log_high, size)
    values = np.array([evaluate(float(value)).probability for value in grid])
    nodes = set(map(float, grid))
    for index in range(1, size - 1):
        before, current, after = values[index - 1 : index + 2]
        peak = current >= before and current >= after and (current > before or current > after)
        valley = current <= before and current <= after and (current < before or current < after)
        if not peak and not valley:
            continue
        direction = -1 if peak else 1
        optimum = minimize_scalar(
            lambda value: direction * evaluate(float(value)).probability,
            bounds=(grid[index - 1], grid[index + 1]),
            method="bounded",
            options={"xatol": 1e-9},
        )
        if not optimum.success:
            raise ArithmeticError("hazard extremum refinement did not converge")
        nodes.add(float(optimum.x))
    ordered = sorted(nodes)
    best = max(ordered, key=lambda value: evaluate(value).probability)
    intervals: list[tuple[float, float]] = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        left_good = evaluate(left).probability >= requested
        right_good = evaluate(right).probability >= requested
        if not left_good and not right_good:
            continue
        start, end = left, right
        if left_good != right_good:
            root = brentq(
                lambda value: evaluate(float(value)).probability - requested,
                left,
                right,
                xtol=1e-11,
            )
            if left_good:
                end = root
            else:
                start = root
        lower, upper = hazard_at(start), hazard_at(end)
        if intervals and lower <= intervals[-1][1]:
            intervals[-1] = intervals[-1][0], upper
        else:
            intervals.append((lower, upper))
    return CONFINTSurvivalHazardRange(
        hazard_at(best),
        evaluate(best),
        tuple(intervals),
        (low, high),
        evaluate(log_low).probability >= requested,
        evaluate(log_high).probability >= requested,
    )
