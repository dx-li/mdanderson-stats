"""Bracketed inverse calculations for CONFINT survival width assurance."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ._validation import scalar
from .confint_binomial import _assurance
from .confint_survival import CONFINTSurvivalAssurance, confint_survival_probability


@dataclass(frozen=True)
class CONFINTSurvivalSolution:
    parameter: str
    value: float
    assurance: float
    achieved: CONFINTSurvivalAssurance


def confint_survival_solve(
    *,
    hazard: float | None = None,
    accrual_rate: float | None = None,
    accrual_time: float | None = None,
    followup_time: float | None = None,
    max_length: float | None = None,
    assurance: float = 0.9,
    confidence: float = 0.95,
    target: str = "hazard",
    bounds: tuple[float, float] = (1e-4, 1e4),
    tail_tolerance: float = 1e-14,
) -> CONFINTSurvivalSolution:
    """Solve for the one parameter left as None, with all four others supplied.

    Inverting max_length gives its mixture quantile, with zero-event studies
    contributing infinite width. Other parameters can have multiple roots;
    bounds must bracket a crossing. This returns a solution within that bracket,
    not a global minimum or all admissible designs. For hazard, bracket either
    branch deliberately. The default bounds match the native non-hazard range.
    """
    supplied = dict(
        hazard=hazard,
        accrual_rate=accrual_rate,
        accrual_time=accrual_time,
        followup_time=followup_time,
        max_length=max_length,
    )
    missing = [name for name, value in supplied.items() if value is None]
    if len(missing) != 1:
        raise ValueError("supply four study parameters and leave exactly one as None")
    parameter = missing[0]
    fixed = {name: scalar(value, name) for name, value in supplied.items() if value is not None}
    requested = _assurance(assurance)
    low, high = (scalar(value, "bounds") for value in bounds)
    if not 0 <= low < high or (low == 0 and parameter in {"hazard", "accrual_time", "max_length"}):
        raise ValueError(
            "bounds must increase, be nonnegative, and be positive for hazard/time/length"
        )

    def evaluate(value: float) -> CONFINTSurvivalAssurance:
        return confint_survival_probability(
            **(fixed | {parameter: value}),
            confidence=confidence,
            target=target,
            tail_tolerance=tail_tolerance,
        )

    left, right = evaluate(low), evaluate(high)
    f_low, f_high = left.probability - requested, right.probability - requested
    if f_low == 0:
        return CONFINTSurvivalSolution(parameter, low, requested, left)
    if f_high == 0:
        return CONFINTSurvivalSolution(parameter, high, requested, right)
    if (f_low > 0) == (f_high > 0):
        raise ValueError(
            "bounds do not bracket the target; choose another bracket or attainable assurance"
        )
    # Log coordinates retain relative precision across changes of physical units.
    # For a permissible zero endpoint, interpolate from the smallest positive float
    # in the interior and evaluate the exact zero at the endpoint itself.
    log_low = np.log(low) if low > 0 else np.log(np.nextafter(0.0, 1.0))
    log_high = np.log(high)

    def value_at(position: float) -> float:
        if position <= 0:
            return low
        if position >= 1:
            return high
        with np.errstate(over="ignore", under="ignore"):
            return float(np.clip(np.exp(log_low + position * (log_high - log_low)), low, high))

    root = brentq(
        lambda position: evaluate(value_at(position)).probability - requested,
        0,
        1,
        xtol=1e-13,
    )
    answer = value_at(root)
    achieved = evaluate(answer)
    if abs(achieved.probability - requested) > max(1e-9, 2 * achieved.omitted_probability):
        raise ArithmeticError("survival inversion did not resolve the requested assurance")
    return CONFINTSurvivalSolution(parameter, answer, requested, achieved)
