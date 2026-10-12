"""Supplied-calendar ARAND native controller profile with independent posteriors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .arand_native import (
    ArandNativeArmLimits,
    ArandNativeGates,
    ArandNativeMonitor,
    ArandNativeObservation,
    ArandNativeSelection,
    _arm_units,
    _integer,
    _probabilities,
    _statuses,
    _unit,
    arand_native_allocation,
    arand_native_constraints,
    arand_native_monitor,
    arand_native_select,
    arand_native_statistics,
    arand_native_statuses,
)
from .arand_posterior import arand_binary_posterior, arand_survival_posterior


@dataclass(frozen=True)
class ArandNativeRules:
    """Explicit single-group compatibility rules, not missing-XML defaults.

    Times use months. A zero arm duration disables its start/duration gates, as
    in the original. Contradictory arm minima/maxima retain original precedence.
    """

    arms: tuple[ArandNativeArmLimits, ...]
    minimum_enrollment: int = 0
    maximum_enrollment: int = 100
    minimum_duration: float = 0
    maximum_duration: float = 1e300
    final_followup: float = 0
    stop_when_one_arm: bool = True
    lower: float | tuple[float, ...] = 0.1
    upper: float | tuple[float, ...] = 0.95
    final_cutoff: float | tuple[float, ...] = 0.5
    futility_cutoff: float | tuple[float, ...] = 0.2
    min_action: int = 1
    max_action: int = 2
    power: float = 0.5
    floor: float = 0.1
    runin: int = 0
    force_equal: bool = True
    gates: ArandNativeGates = ArandNativeGates()

    def __post_init__(self) -> None:
        if not 1 <= len(self.arms) <= 10 or any(
            not isinstance(a, ArandNativeArmLimits) for a in self.arms
        ):
            raise ValueError("arms must contain 1 to 10 arm limit records")
        object.__setattr__(self, "arms", tuple(self.arms))
        for name in ("minimum_enrollment", "maximum_enrollment", "runin"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if not isinstance(self.gates, ArandNativeGates):
            raise ValueError("gates must be ArandNativeGates")
        if (
            self.gates.trial_maximum
            and self.gates.trial_minimum
            and self.maximum_enrollment < self.minimum_enrollment
        ):
            raise ValueError("maximum enrollment must be at least the minimum")
        for name in ("minimum_duration", "maximum_duration", "final_followup", "power"):
            if scalar(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            self.gates.duration_maximum
            and self.gates.duration_minimum
            and self.maximum_duration < self.minimum_duration
        ):
            raise ValueError("maximum duration must be at least the minimum")
        for name in ("lower", "upper", "final_cutoff", "futility_cutoff"):
            value = getattr(self, name)
            normalized = _arm_units(value, len(self.arms), name)
            if np.asarray(value).ndim != 0:
                object.__setattr__(self, name, normalized)
        _unit(self.floor, "floor")
        if self.floor * len(self.arms) > 1:
            raise ValueError("infeasible allocation floor")
        if self.min_action not in (0, 1, 2, 3) or self.max_action not in (0, 1, 2, 3):
            raise ValueError("unsupported closure action")
        if not isinstance(self.stop_when_one_arm, bool) or not isinstance(self.force_equal, bool):
            raise ValueError("controller switches must be boolean")


@dataclass(frozen=True)
class ArandNativeEvidence:
    probability_best: FloatArray
    futility_probability: FloatArray | None = None

    def __post_init__(self) -> None:
        p = _probabilities(self.probability_best)
        object.__setattr__(self, "probability_best", _freeze(p))
        if self.futility_probability is not None:
            f = _probabilities(self.futility_probability)
            if len(f) != len(p):
                raise ValueError("futility values must match the arm count")
            object.__setattr__(self, "futility_probability", _freeze(f))


@dataclass(frozen=True)
class ArandNativeState:
    monitor: ArandNativeMonitor
    statuses: tuple[int, ...]
    evidence: ArandNativeEvidence

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "statuses", tuple(_statuses(self.statuses, len(self.evidence.probability_best)))
        )


@dataclass(frozen=True)
class ArandNativeDecision:
    state: ArandNativeState
    time: float
    reason: str
    completed: bool
    selection: ArandNativeSelection | None
    group_selection: ArandNativeSelection | None
    allocation: FloatArray | None
    analysis_times: tuple[float, ...]
    assignment_arms: tuple[int, ...]


def arand_native_decide(
    state: ArandNativeState,
    rules: ArandNativeRules,
    *,
    now: float,
    enrolled: tuple[int, ...],
    observed: int,
    pending: bool,
    analyze: Callable[[float], ArandNativeEvidence],
) -> ArandNativeDecision:
    """One original-order candidate decision with an explicit posterior service.

    ``analyze`` must rank all arms, including closed arms, and use data visible
    at its supplied absolute time. It may be invoked twice at completion.
    """
    k = len(rules.arms)
    now = scalar(now, "now")
    if now < 0 or len(enrolled) != k or len(state.statuses) != k:
        raise ValueError("invalid time or arm dimensions")
    counts = tuple(_integer(n, "enrolled") for n in enrolled)
    total = sum(counts)
    observed = _integer(observed, "observed")
    if observed > total or not isinstance(pending, bool):
        raise ValueError("invalid observed/pending values")
    statuses, evidence, monitor = state.statuses, state.evidence, state.monitor
    times: list[float] = []

    def evaluate(t: float) -> None:
        nonlocal statuses, evidence
        evidence = analyze(t)
        if len(evidence.probability_best) != k:
            raise ValueError("posterior service changed the arm count")
        times.append(t)
        statuses = arand_native_statuses(
            evidence.probability_best,
            statuses,
            lower=rules.lower,
            upper=rules.upper,
            futility=evidence.futility_probability,
            futility_cutoff=rules.futility_cutoff,
            min_action=rules.min_action,
            max_action=rules.max_action,
        )

    gates = rules.gates
    trial_minimum = (not gates.trial_minimum or total >= rules.minimum_enrollment) and (
        not gates.duration_minimum or now >= rules.minimum_duration
    )
    # Native minima use ANY arm and OR between the arm/group conditions.
    arm_minimum = any(n >= a.minimum for n, a in zip(counts, rules.arms, strict=True))
    group_minimum = any(n >= a.group_minimum for n, a in zip(counts, rules.arms, strict=True))
    minimum_met = (
        (arm_minimum or group_minimum)
        if gates.arm_minimum and gates.group_minimum
        else arm_minimum
        if gates.arm_minimum
        else group_minimum
        if gates.group_minimum
        else True
    )
    if trial_minimum and minimum_met:
        look = arand_native_monitor(
            monitor, now=now, enrolled=total, observed=observed, pending=pending
        )
        monitor = look.monitor
        if look.trial_status is not None:
            statuses = (int(look.trial_status),) * k
        if look.analyze:
            evaluate(now)
    completed = trial_minimum and (
        (gates.duration_maximum and now > rules.maximum_duration)
        or (gates.trial_maximum and total >= rules.maximum_enrollment)
    )
    group_selection = None
    if completed:
        reason = (
            "maximum_duration"
            if gates.duration_maximum and now > rules.maximum_duration
            else "maximum_enrollment"
        )
        now += rules.final_followup
        if not np.isfinite(now):
            raise ArithmeticError("final analysis time overflow")
        evaluate(now)
    else:
        statuses = arand_native_constraints(statuses, counts, rules.arms, now=now, gates=gates)
        active = statuses.count(0)
        stop_one = rules.stop_when_one_arm and trial_minimum and active <= 1
        if stop_one:
            group_selection = arand_native_select(
                evidence.probability_best,
                statuses,
                cutoff=rules.final_cutoff,
                upper=rules.upper,
                group=True,
                completed=False,
            )
        reason = (
            "one_or_no_active_arm" if stop_one else "no_active_arms" if active == 0 else "allocate"
        )
    stopped = completed or reason != "allocate"
    selection, allocation = None, None
    assignment_arms: tuple[int, ...] = ()
    if stopped:
        selection = arand_native_select(
            evidence.probability_best,
            statuses,
            cutoff=rules.final_cutoff,
            upper=rules.upper,
            completed=completed,
        )
    else:
        eligible = np.flatnonzero(np.asarray(statuses) == 0)
        assignment_arms = tuple(int(i) for i in eligible)
        allocation = np.zeros(k)
        runin = ((rules.runin + k - 1) // k) * k if rules.force_equal else rules.runin
        if len(eligible) == 1:
            allocation[eligible] = 1
        elif total < runin and rules.force_equal:
            # Original force-equal branch intentionally visits ALL arms.
            assignment_arms = tuple(range(k))
            deficits = np.maximum(runin // k - np.asarray(counts), 0)
            allocation = deficits / (runin - total)
            if not np.isclose(allocation.sum(), 1, atol=1e-12, rtol=0):
                raise ArithmeticError(
                    "native forced run-in weights are not a probability distribution"
                )
        else:
            probabilities = (
                np.ones(len(eligible)) if total < runin else evidence.probability_best[eligible]
            )
            allocation[eligible] = arand_native_allocation(
                probabilities, power=rules.power, floor=rules.floor
            )
        allocation = _freeze(allocation)
    return ArandNativeDecision(
        ArandNativeState(monitor, statuses, evidence),
        now,
        reason,
        completed,
        selection,
        group_selection,
        allocation,
        tuple(times),
        assignment_arms,
    )


@dataclass(frozen=True)
class ArandNativeTrial:
    records: tuple[ArandNativeObservation, ...]
    decisions: tuple[ArandNativeDecision, ...]
    family: str
    parameter: str

    @property
    def final(self) -> ArandNativeDecision:
        return self.decisions[-1]


def arand_native_replay(
    arrival_times: ArrayLike,
    uniforms: ArrayLike,
    observation_delays: ArrayLike,
    indicators: ArrayLike,
    *,
    prior: ArrayLike,
    rules: ArandNativeRules,
    monitor: ArandNativeMonitor,
    family: str = "binary",
    parameter: str = "mean",
    maximize: bool = True,
    futility_threshold: float | tuple[float, ...] | None = None,
    absolute_tolerance: float = 1e-9,
) -> ArandNativeTrial:
    """Replay candidate-by-arm outcome tapes through the native controller.

    Delays are relative to enrollment. In survival mode indicators distinguish
    events from censoring; binary indicators distinguish successes from failures.
    A final candidate is needed to trigger a cap reached by the preceding patient.
    Tape exhaustion raises; it never masquerades as trial completion.
    """
    arrivals, u = finite(arrival_times, "arrival_times"), finite(uniforms, "uniforms")
    delays, values = (
        finite(observation_delays, "observation_delays"),
        finite(indicators, "indicators"),
    )
    p = finite(prior, "prior")
    k = len(rules.arms)
    if (
        arrivals.ndim != 1
        or not 1 <= len(arrivals) <= 10000
        or np.any(arrivals < 0)
        or np.any(np.diff(arrivals) < 0)
        or u.shape != arrivals.shape
        or np.any((u < 0) | (u >= 1))
        or delays.shape != (len(arrivals), k)
        or values.shape != delays.shape
        or np.any(delays < 0)
        or np.any((values != 0) & (values != 1))
        or p.shape != (k, 2)
        or np.any(p <= 0)
    ):
        raise ValueError("invalid candidate tapes or priors")
    if family not in ("binary", "exponential") or parameter not in ("mean", "median"):
        raise ValueError("unsupported family or survival parameter")
    records: list[ArandNativeObservation] = []

    def analyze(now: float) -> ArandNativeEvidence:
        s = arand_native_statistics(tuple(records), now=now, arms=k)
        if family == "binary":
            posterior = arand_binary_posterior(
                s.positive,
                np.asarray(s.observed) - s.positive,
                prior=p,
                maximize=maximize,
                tuning=0,
                threshold=futility_threshold,
                absolute_tolerance=absolute_tolerance,
            )
        else:
            posterior = arand_survival_posterior(
                s.positive,
                s.exposure,
                parameter=parameter,
                prior=p,
                maximize=maximize,
                tuning=0,
                threshold=futility_threshold,
                absolute_tolerance=absolute_tolerance,
            )
        return ArandNativeEvidence(posterior.best.probability, posterior.exceedance_probability)

    state = ArandNativeState(monitor, (0,) * k, analyze(0))
    decisions: list[ArandNativeDecision] = []
    for i, now in enumerate(arrivals):
        stats = arand_native_statistics(tuple(records), now=float(now), arms=k)
        decision = arand_native_decide(
            state,
            rules,
            now=float(now),
            enrolled=stats.enrolled,
            observed=sum(stats.observed),
            pending=sum(stats.observed) < len(records),
            analyze=analyze,
        )
        decisions.append(decision)
        state = decision.state
        if decision.selection is not None:
            return ArandNativeTrial(tuple(records), tuple(decisions), family, parameter)
        assert decision.allocation is not None
        # Native comparison is <=, including at zero cumulative probability.
        population = decision.assignment_arms
        weights = decision.allocation[list(population)]
        selected = int(np.searchsorted(np.cumsum(weights), u[i], side="left"))
        if selected >= len(population):
            raise ArithmeticError("allocation cumulative probability does not cover uniform")
        arm = population[selected]
        end = float(now + delays[i, arm])
        if not np.isfinite(end):
            raise ArithmeticError("observation time overflow")
        records.append(ArandNativeObservation(arm, float(now), end, int(values[i, arm])))
    raise ValueError("candidate tape exhausted before the native controller stopped")
