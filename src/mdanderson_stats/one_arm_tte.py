"""One-arm exponential/inverse-gamma time-to-event monitoring."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import finite, scalar
from .inequality import _integrate_ordering, inequality_probability
from .parameter_distribution import ParameterDistribution


@dataclass(frozen=True)
class OneArmTTEDesign:
    standard_prior: np.ndarray
    experimental_prior: np.ndarray
    parameterization: str
    maximize: bool
    delta_inferiority: float | None
    cutoff_inferiority: float | None
    delta_superiority: float | None
    cutoff_superiority: float | None
    max_patients: int
    minimum_patients: int
    periodic_interval: float | None
    monitor_at_accrual: bool
    followup_period: float
    absolute_tolerance: float


@dataclass(frozen=True)
class OneArmTTEMonitor:
    patients: int
    events: int
    total_time: float
    inferiority_probability: float | None
    inferiority_error: float | None
    superiority_probability: float | None
    superiority_error: float | None
    inferior: bool
    superior: bool
    stop_accrual: bool
    reason: str


@dataclass(frozen=True)
class OneArmTTETrial:
    enrollment_time: np.ndarray
    event_time: np.ndarray
    event_observed_time: np.ndarray
    early_monitor: OneArmTTEMonitor | None
    final_monitor: OneArmTTEMonitor
    accrual_stop_time: float
    final_time: float
    monitor_history: tuple[tuple[float, OneArmTTEMonitor], ...]


def _prior(value: ArrayLike, name: str) -> np.ndarray:
    result = finite(value, name)
    if result.shape != (2,) or np.any(result <= 0):
        raise ValueError(f"{name} must be positive shape/scale (2,)")
    return result


def _optional_cutoff(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = scalar(value, name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be in [0,1]")
    return result


def one_arm_tte_design(
    standard_prior: ArrayLike,
    experimental_prior: ArrayLike,
    *,
    parameterization: str = "mean",
    maximize: bool = True,
    delta_inferiority: float | None = 0.0,
    cutoff_inferiority: float | None = None,
    delta_superiority: float | None = 0.0,
    cutoff_superiority: float | None = None,
    max_patients: int,
    minimum_patients: int = 1,
    periodic_interval: float | None = None,
    monitor_at_accrual: bool = True,
    followup_period: float = 0.0,
    absolute_tolerance: float = 1e-9,
) -> OneArmTTEDesign:
    """Construct a design; time units are caller-selected and consistent."""
    if parameterization not in ("mean", "median"):
        raise ValueError("parameterization must be mean or median")
    if not isinstance(maximize, (bool, np.bool_)) or not isinstance(
        monitor_at_accrual, (bool, np.bool_)
    ):
        raise ValueError("maximize and monitor_at_accrual must be boolean")
    s, e = (
        _prior(standard_prior, "standard_prior"),
        _prior(experimental_prior, "experimental_prior"),
    )
    di, ds = (
        (None if delta_inferiority is None else scalar(delta_inferiority, "delta_inferiority")),
        (None if delta_superiority is None else scalar(delta_superiority, "delta_superiority")),
    )
    ci, cs = (
        _optional_cutoff(cutoff_inferiority, "cutoff_inferiority"),
        _optional_cutoff(cutoff_superiority, "cutoff_superiority"),
    )
    if ci is None:
        di = None
    if cs is None:
        ds = None
    if ci is not None and di is None or cs is not None and ds is None:
        raise ValueError("each enabled rule requires a delta")
    if ci is None and cs is None:
        raise ValueError("at least one stopping rule must be enabled")
    if isinstance(max_patients, bool) or np.ndim(max_patients) != 0:
        raise ValueError("max_patients must be an integer in [1,1000]")
    n = scalar(max_patients, "max_patients")
    if isinstance(minimum_patients, (bool, np.bool_)):
        raise ValueError("minimum_patients must be an integer")
    m = scalar(minimum_patients, "minimum_patients")
    if int(n) != n or not 1 <= n <= 1000 or int(m) != m or not 1 <= m <= n:
        raise ValueError("require 1 <= minimum_patients <= max_patients <= 1000")
    interval = None if periodic_interval is None else scalar(periodic_interval, "periodic_interval")
    if interval is not None and interval <= 0:
        raise ValueError("periodic_interval must be positive")
    followup = scalar(followup_period, "followup_period")
    if followup < 0:
        raise ValueError("followup_period must be nonnegative")
    tol = scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tol <= 1e-3:
        raise ValueError("absolute_tolerance must be in [1e-12,1e-3]")
    return OneArmTTEDesign(
        _freeze(s),
        _freeze(e),
        parameterization,
        bool(maximize),
        di,
        ci,
        ds,
        cs,
        int(n),
        int(m),
        interval,
        bool(monitor_at_accrual),
        followup,
        tol,
    )


def _posterior(
    design: OneArmTTEDesign, events: int, total_time: float
) -> tuple[ParameterDistribution, ParameterDistribution]:
    scale = 1.0 if design.parameterization == "mean" else float(np.log(2))
    a, b = design.experimental_prior
    shape, posterior_scale = a + events, b + scale * total_time
    if not np.isfinite(shape) or not np.isfinite(posterior_scale):
        raise ArithmeticError("posterior inverse-gamma parameters overflow")
    return (
        ParameterDistribution(
            "inverse_gamma", float(design.standard_prior[0]), float(design.standard_prior[1])
        ),
        ParameterDistribution("inverse_gamma", float(shape), float(posterior_scale)),
    )


def _probability(
    design: OneArmTTEDesign, events: int, total_time: float, delta: float
) -> tuple[float, float]:
    standard, experimental = _posterior(design, events, total_time)
    if design.maximize:
        x, y = experimental, standard
    else:
        x, y = standard, experimental
        delta = -delta
    if delta == 0:
        result = inequality_probability(x, y, delta=0, absolute_tolerance=design.absolute_tolerance)
        return result.x_greater, result.absolute_error
    return _integrate_ordering(x, y, delta, design.absolute_tolerance)


def one_arm_tte_monitor(
    design: OneArmTTEDesign, patients: int, events: int, total_time: float
) -> OneArmTTEMonitor:
    if not isinstance(design, OneArmTTEDesign):
        raise TypeError("design must be a OneArmTTEDesign")
    if isinstance(patients, bool) or isinstance(events, bool):
        raise ValueError("patients and events must be integers")
    n, d, t = (
        scalar(patients, "patients"),
        scalar(events, "events"),
        scalar(total_time, "total_time"),
    )
    if int(n) != n or not 0 <= n <= design.max_patients or int(d) != d or not 0 <= d <= n:
        raise ValueError("require integer 0 <= events <= patients <= max_patients")
    if t < 0 or not np.isfinite(t) or (d > 0 and t <= 0) or (n == 0 and t > 0):
        raise ValueError("total_time must be nonnegative and positive with events")
    inferior_p = inferior_error = superior_p = superior_error = None
    if design.cutoff_inferiority is not None:
        assert design.delta_inferiority is not None
        inferior_p, inferior_error = _probability(design, int(d), t, design.delta_inferiority)
    if design.cutoff_superiority is not None:
        assert design.delta_superiority is not None
        superior_p, superior_error = _probability(design, int(d), t, design.delta_superiority)
    inferior = (
        design.cutoff_inferiority is not None
        and inferior_p is not None
        and inferior_p < design.cutoff_inferiority
    )
    superior = (
        design.cutoff_superiority is not None
        and superior_p is not None
        and superior_p > design.cutoff_superiority
    )
    maximum = int(n) >= design.max_patients
    reason = (
        "both"
        if inferior and superior
        else "inferior"
        if inferior
        else "superior"
        if superior
        else "maximum"
        if maximum
        else "continue"
    )
    return OneArmTTEMonitor(
        int(n),
        int(d),
        t,
        inferior_p,
        inferior_error,
        superior_p,
        superior_error,
        inferior,
        superior,
        bool(inferior or superior or maximum),
        reason,
    )


def _observed(enrollment: np.ndarray, durations: np.ndarray, now: float) -> tuple[int, float]:
    if not np.isfinite(now):
        raise ArithmeticError("observation time is not finite")
    elapsed = np.maximum(0.0, now - enrollment)
    if not np.all(np.isfinite(elapsed)):
        raise ArithmeticError("patient exposure is not finite")
    observed = durations <= elapsed
    exposure = float(np.sum(np.minimum(elapsed, durations)))
    if not np.isfinite(exposure):
        raise ArithmeticError("total exposure is not finite")
    return int(np.sum(observed)), exposure


def one_arm_tte_trial(
    design: OneArmTTEDesign, enrollment_time: ArrayLike, event_time: ArrayLike
) -> OneArmTTETrial:
    """Run a deterministic calendar trace; ``event_time`` contains durations."""
    enrollment = finite(enrollment_time, "enrollment_time")
    duration = finite(event_time, "event_time")
    if (
        enrollment.ndim != 1
        or duration.shape != enrollment.shape
        or len(enrollment) != design.max_patients
    ):
        raise ValueError("inputs must contain exactly max_patients records")
    if (
        len(enrollment) == 0
        or np.any(np.diff(enrollment) <= 0)
        or np.any(enrollment <= 0)
        or np.any(duration <= 0)
    ):
        raise ValueError("enrollment times must be sorted and event durations positive")
    early = None
    history: list[tuple[float, OneArmTTEMonitor]] = []
    next_period = design.periodic_interval
    period_index = 1
    check_count = 0
    enrolled = 0
    stop_time: float = float(enrollment[-1])
    while enrolled < design.max_patients:
        arrival = float(enrollment[enrolled])
        periodic = next_period is not None and next_period <= arrival
        if periodic:
            assert next_period is not None and design.periodic_interval is not None
            # No calendar checks are allowed before minimum enrollment.  Jump
            # directly to the first grid point strictly after this arrival.
            if enrolled < design.minimum_patients:
                interval = design.periodic_interval
                tick = np.floor(arrival / interval) + 1.0
                if not np.isfinite(tick):
                    raise ArithmeticError("periodic monitoring calendar cannot advance")
                period_index = int(tick)
                candidate = period_index * interval
                if not np.isfinite(candidate) or candidate <= arrival:
                    raise ArithmeticError("periodic monitoring calendar cannot advance")
                next_period = float(candidate)
                continue
            if check_count >= 10_000:
                raise ValueError("monitoring calendar exceeds 10000 checks")
            now = float(next_period)
            d, t = _observed(enrollment[:enrolled], duration[:enrolled], now)
            result = one_arm_tte_monitor(design, enrolled, d, t)
            history.append((now, result))
            check_count += 1
            if result.inferior or result.superior:
                early, stop_time = result, now
                break
            period_index += 1
            candidate = period_index * design.periodic_interval
            if not np.isfinite(candidate) or candidate <= next_period:
                raise ArithmeticError("periodic monitoring calendar cannot advance")
            next_period = candidate
            continue
        if design.monitor_at_accrual and enrolled >= design.minimum_patients:
            if check_count >= 10_000:
                raise ValueError("monitoring calendar exceeds 10000 checks")
            d, t = _observed(enrollment[:enrolled], duration[:enrolled], arrival)
            result = one_arm_tte_monitor(design, enrolled, d, t)
            history.append((arrival, result))
            check_count += 1
            if result.inferior or result.superior:
                early, stop_time = result, arrival
                break
        enrolled += 1
        stop_time = arrival
    if enrolled == design.max_patients and early is None:
        stop_time = float(enrollment[-1])
    end = stop_time + design.followup_period
    if not np.isfinite(end):
        raise ArithmeticError("trial end time is not finite")
    d, t = _observed(enrollment[:enrolled], duration[:enrolled], end)
    final = one_arm_tte_monitor(design, enrolled, d, t)
    observed_time_raw = enrollment[:enrolled] + duration[:enrolled]
    if not np.all(np.isfinite(observed_time_raw)):
        raise ArithmeticError("event observation times are not finite")
    observed_time = _freeze(observed_time_raw)
    return OneArmTTETrial(
        _freeze(enrollment[:enrolled]),
        _freeze(duration[:enrolled]),
        observed_time,
        early,
        final,
        stop_time,
        end,
        tuple(history),
    )
