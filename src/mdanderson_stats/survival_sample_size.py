"""Event-driven exponential survival planning for the Nsurvival test families."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri

from ._validation import FloatArray, finite, scalar
from .continuous_sample_size import MeanObjective, _level, _probability


def _times(hazard: ArrayLike, accrual: ArrayLike, followup: ArrayLike) -> tuple[FloatArray, ...]:
    h, a, f = np.broadcast_arrays(
        finite(hazard, "hazard"), finite(accrual, "accrual"), finite(followup, "followup")
    )
    with np.errstate(over="ignore"):
        total = a + f
    if np.any((h <= 0) | (a < 0) | (f < 0) | ~np.isfinite(total)):
        raise ValueError(
            "require positive hazards, nonnegative durations and finite accrual+followup"
        )
    return h, a, f


def exponential_event_probability(
    hazard: ArrayLike, accrual: ArrayLike, followup: ArrayLike
) -> FloatArray:
    """Event probability under uniform accrual and additional follow-up after accrual.

    No dropout or competing risk. Zero accrual means simultaneous enrollment.
    All inputs broadcast; stable series preserve small hazard-duration products.
    """
    h, a, f = _times(hazard, accrual, followup)
    with np.errstate(over="ignore"):
        x, y = h * a, h * f
    small = x < 1e-3
    # 1-(1-exp(-x))/x, expanded where subtraction would lose precision.
    xs = np.where(small, x, 0)
    increment = xs * (0.5 + xs * (-1 / 6 + xs * (1 / 24 + xs * (-1 / 120 + xs / 720))))
    safe_x = np.where(small, 1, x)
    increment = np.where(small, increment, 1 + np.expm1(-x) / safe_x)
    return _probability(-np.expm1(-y) + np.exp(-y) * increment)


def _source_events(
    hc: FloatArray, ht: FloatArray, accrual: FloatArray, followup: FloatArray
) -> tuple[FloatArray, FloatArray]:
    # Source Simpson average of control survival, followed by a hazard-ratio
    # transformation for treatment. Work in effective exposure time to avoid
    # subtracting rounded probabilities or forming an overflowing hazard ratio.
    with np.errstate(over="ignore"):
        x = hc * accrual
    q = -(4 * np.expm1(-x / 2) + np.expm1(-x)) / 6
    small = x < 1e-4
    exposure = np.where(small, accrual * (0.5 - np.where(small, x, 0) / 24), -np.log1p(-q) / hc)
    exposure = followup + exposure
    with np.errstate(over="ignore"):
        return -np.expm1(-hc * exposure), -np.expm1(-ht * exposure)


def _effect(value: ArrayLike, denominator: ArrayLike) -> FloatArray:
    numerator, denominator = np.broadcast_arrays(value, denominator)
    with np.errstate(over="ignore"):
        relative = (numerator - denominator) / denominator
    near = np.abs(relative) < 0.5
    return np.where(
        near, np.log1p(np.where(near, relative, 0)), np.log(numerator) - np.log(denominator)
    )


def _information_fraction(ratio: ArrayLike | None) -> FloatArray:
    if ratio is None:
        return np.asarray(1.0)
    r = finite(ratio, "allocation_ratio")
    if np.any((r < 1e-6) | (r > 1e6)):
        raise ValueError("allocation_ratio must lie in [1e-6, 1e6]")
    return r / (1 + r) ** 2


def survival_event_power(
    events: ArrayLike,
    log_median_ratio: ArrayLike,
    *,
    allocation_ratio: ArrayLike | None = None,
    objective: MeanObjective = "equality",
    margin: ArrayLike = 0,
    alpha: float = 0.05,
    sides: int = 1,
    equivalence: Literal["conservative", "joint"] = "conservative",
) -> FloatArray:
    """Asymptotic planning power at an event count (possibly fractional/expected).

    log_median_ratio = log(treatment median/control median) = log(hc/ht).
    Two-arm information is events*p_control*p_treatment using the planned
    randomization fractions; one-arm information is events. Equality retains
    the dominant normal tail. Joint equivalence evaluates both normal rejection
    conditions; conservative uses twice the worst tail, as in the app example.
    """
    alpha = _level(alpha, sides)
    if objective not in ("equality", "equivalence", "noninferiority", "superiority"):
        raise ValueError("unknown objective")
    if allocation_ratio is None and objective != "equality":
        raise ValueError("margin objectives require a two-arm design")
    if equivalence not in ("conservative", "joint"):
        raise ValueError("equivalence must be conservative or joint")
    d, b, width, fraction = np.broadcast_arrays(
        finite(events, "events"),
        finite(log_median_ratio, "log_median_ratio"),
        finite(margin, "margin"),
        _information_fraction(allocation_ratio),
    )
    if np.any(d < 0):
        raise ValueError("events must be nonnegative")
    if (objective == "equality" and np.any(width != 0)) or (
        objective != "equality" and np.any(width <= 0)
    ):
        raise ValueError("margin must be zero for equality and positive otherwise")
    root_information = np.sqrt(d) * np.sqrt(fraction)
    critical = -ndtri(alpha / sides if objective == "equality" else alpha)
    if objective == "equivalence":
        if equivalence == "conservative":
            value = np.maximum(0, 1 - 2 * ndtr(critical - (width - np.abs(b)) * root_information))
        else:
            lo = (-width - b) * root_information + critical
            hi = (width - b) * root_information - critical
            value = np.where(lo > 0, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))
            value = np.where(hi > lo, value, 0)
    else:
        separation = (
            np.abs(b)
            if objective == "equality"
            else (b + width if objective == "noninferiority" else b - width)
        )
        value = ndtr(separation * root_information - critical)
    return _probability(value)


@dataclass(frozen=True)
class SurvivalSampleSize:
    """Broadcast design results, with group axes last (control/treatment for two arms)."""

    group_sizes: NDArray[np.int64]
    event_probabilities: FloatArray
    required_events: FloatArray
    rounded_events: NDArray[np.int64]
    expected_events: FloatArray
    power: FloatArray
    log_median_ratio: FloatArray
    target_power: float
    objective: str
    event_method: str
    equivalence: str

    @property
    def total_size(self) -> NDArray[np.int64]:
        return self.group_sizes.sum(axis=-1)


def survival_sample_size(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    accrual: ArrayLike,
    followup: ArrayLike,
    parameter: Literal["median", "hazard"] = "median",
    allocation_ratio: ArrayLike | None = None,
    objective: MeanObjective = "equality",
    margin: ArrayLike = 0,
    alpha: float = 0.05,
    sides: int = 1,
    target_power: float = 0.8,
    event_method: Literal["source", "uniform"] = "source",
    equivalence: Literal["conservative", "joint"] = "conservative",
) -> SurvivalSampleSize:
    """Nsurvival planning from medians or exponential hazards, including margin tests.

    followup is ADDITIONAL time after accrual, not maximum duration A+F. Ratio
    means treatment/control; None selects historical-control one-arm equality.
    Source event probabilities use exact uniform accrual for one arm, and the
    documented Simpson/hazard-ratio approximation for two arms. uniform instead
    integrates each arm's exponential event probability exactly.

    Continuous group sizes are rounded up separately. Power uses expected events
    at those sizes and planned allocation fractions. Rounded events are a display
    count; enrollment is based on unrounded required_events, matching the source.
    """
    c, tr = np.broadcast_arrays(finite(control, "control"), finite(treatment, "treatment"))
    if np.any((c <= 0) | (tr <= 0)):
        raise ValueError("medians or hazards must be positive")
    if parameter not in ("median", "hazard") or event_method not in ("source", "uniform"):
        raise ValueError("invalid parameter or event_method")
    with np.errstate(over="ignore"):
        hc, ht = (np.log(2) / c, np.log(2) / tr) if parameter == "median" else (c, tr)
    hc, a, f = _times(hc, accrual, followup)
    ht, a, f = _times(ht, a, f)
    hc, ht, a, f, b, width, fraction = np.broadcast_arrays(
        hc,
        ht,
        a,
        f,
        _effect(tr, c) if parameter == "median" else _effect(c, tr),
        finite(margin, "margin"),
        _information_fraction(allocation_ratio),
    )
    # Validate objective, test levels, and conventions through the power API.
    survival_event_power(
        1,
        b,
        allocation_ratio=allocation_ratio,
        objective=objective,
        margin=width,
        alpha=alpha,
        sides=sides,
        equivalence=equivalence,
    )
    target = scalar(target_power, "target_power")
    if not alpha < target < 1:
        raise ValueError("require alpha < target_power < 1")
    separation = (
        np.abs(b)
        if objective == "equality"
        else (
            width - np.abs(b)
            if objective == "equivalence"
            else b + width
            if objective == "noninferiority"
            else b - width
        )
    )
    if np.any(separation <= 0):
        raise ValueError("expected survival must lie inside the requested alternative")
    critical = -ndtri(alpha / sides if objective == "equality" else alpha)
    # Use a tail probability directly so targets near one do not round to one.
    target_quantile = -ndtri((1 - target) / 2 if objective == "equivalence" else 1 - target)
    with np.errstate(over="ignore", divide="ignore", under="ignore"):
        events = ((critical + target_quantile) / separation) ** 2 / fraction
    if np.any(~np.isfinite(events) | (events <= 0) | (events >= 2**53)):
        raise ArithmeticError("required events are not representable below 2**53")
    if objective == "equivalence" and equivalence == "joint":
        low, high = np.zeros_like(events), events.copy()
        for _ in range(70):
            mid = (low + high) / 2
            passing = (
                survival_event_power(
                    mid,
                    b,
                    allocation_ratio=allocation_ratio,
                    objective=objective,
                    margin=width,
                    alpha=alpha,
                    sides=sides,
                    equivalence="joint",
                )
                >= target
            )
            high = np.where(passing, mid, high)
            low = np.where(passing, low, mid)
        events = high
    if allocation_ratio is None:
        probabilities = exponential_event_probability(ht, a, f)[..., None]
        weights = np.ones_like(probabilities)
    else:
        if event_method == "uniform":
            qc, qt = (
                exponential_event_probability(hc, a, f),
                exponential_event_probability(ht, a, f),
            )
        else:
            qc, qt = _source_events(hc, ht, a, f)
        probabilities = np.stack([qc, qt], axis=-1)
        ratio = np.broadcast_to(np.asarray(allocation_ratio, dtype=float), events.shape)
        weights = np.stack([1 / (1 + ratio), ratio / (1 + ratio)], axis=-1)
    combined = np.sum(weights * probabilities, axis=-1)
    if np.any(combined <= 0):
        raise ValueError("no observable events under the specified follow-up")
    with np.errstate(over="ignore", divide="ignore"):
        raw_sizes = np.ceil((events / combined)[..., None] * weights)
    if np.any(~np.isfinite(raw_sizes) | (raw_sizes < 1)) or np.any(raw_sizes.sum(axis=-1) >= 2**53):
        raise ArithmeticError("required enrollment is not representable below 2**53")
    sizes = raw_sizes.astype(np.int64)
    expected = np.sum(sizes * probabilities, axis=-1)
    power = survival_event_power(
        expected,
        b,
        allocation_ratio=allocation_ratio,
        objective=objective,
        margin=width,
        alpha=alpha,
        sides=sides,
        equivalence=equivalence,
    )
    arrays = [
        np.array(v, copy=True)
        for v in (
            sizes,
            probabilities,
            events,
            np.ceil(events).astype(np.int64),
            expected,
            power,
            b,
        )
    ]
    for value in arrays:
        value.setflags(write=False)
    return SurvivalSampleSize(
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        arrays[5],
        arrays[6],
        target,
        objective,
        event_method,
        equivalence,
    )
