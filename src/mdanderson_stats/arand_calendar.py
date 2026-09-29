"""Adaptive Randomization replay with explicit controller policies.

The posterior kernels are source-backed; floor transforms, futility ranking,
trigger ordering, and a few calendar edge choices are Python policies because
the version 5.2 guide does not specify those details.
"""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import finite, scalar
from .arand_posterior import (
    arand_best_probability,
    arand_binary_posterior,
    arand_survival_posterior,
)

Rule = Literal["futility", "suspension", "early_winner"]


@dataclass(frozen=True)
class ArandControllerPolicy:
    """Required choices for controller behavior the native guide leaves open.

    Suspended arms remain in posterior ranking, so they can reactivate. Choose
    `all_arms` or `nonfutile_arms` to decide whether permanently futile arms
    remain in ranking. Final winner selection includes reversibly suspended arms
    but excludes permanently futile arms. `mixture` guarantees the allocation floor when feasible;
    clamp-and-renormalize can fall below it. Tied arrivals follow tape order,
    so a maximum-enrollment cap may admit only the earlier tied candidates.
    """

    floor_transform: Literal["clamp_renormalize", "mixture"]
    ranking_scope: Literal["all_arms", "nonfutile_arms"]
    trigger_order: tuple[Rule, Rule, Rule]
    duration_minimum_precedence: Literal["duration_wins", "minimum_enrollment_wins"]
    multiple_winner: Literal["highest_probability", "lowest_arm_index"]
    all_suspended_action: Literal["stop", "skip_arrivals_until_next_analysis"]
    same_time_order: Literal["analysis_before_arrival", "arrival_before_analysis"]


@dataclass(frozen=True)
class ArandCalendarLook:
    """Compact controller state recorded at an arrival or explicit analysis time."""

    time: float
    enrolled: int
    observed: int
    posterior_probability: NDArray[np.float64] | None
    exceedance_probability: NDArray[np.float64] | None
    allocation_probability: NDArray[np.float64] | None
    suspended: NDArray[np.bool_]
    futile: NDArray[np.bool_]
    assigned_patient: int | None
    assigned_arm: int | None
    status: str


@dataclass(frozen=True)
class ArandCalendarTrial:
    """Replay result; completed trials separate enrollment-close and follow-up time."""

    family: str
    assignments: NDArray[np.int64]
    enrollment_times: NDArray[np.float64]
    looks: tuple[ArandCalendarLook, ...]
    suspended: NDArray[np.bool_]
    futile: NDArray[np.bool_]
    early_winner: int | None
    final_winner: int | None
    reason: str
    stopped_early: bool
    stop_time: float | None
    final_analysis_time: float | None
    decision_duration: float
    completed_duration: float
    attempted_arrivals: int
    blocked_arrivals: int
    data_seed: int | None


def _readonly(values: ArrayLike, dtype: type | np.dtype | None = None) -> NDArray:
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _validate_policy(policy: ArandControllerPolicy) -> None:
    if not isinstance(policy, ArandControllerPolicy):
        raise TypeError("policy must be an explicit ArandControllerPolicy")
    if policy.floor_transform not in {"clamp_renormalize", "mixture"}:
        raise ValueError("invalid floor_transform")
    if policy.ranking_scope not in {"all_arms", "nonfutile_arms"}:
        raise ValueError("invalid ranking_scope")
    if len(policy.trigger_order) != 3 or set(policy.trigger_order) != {
        "futility",
        "suspension",
        "early_winner",
    }:
        raise ValueError("trigger_order must list futility, suspension, and early_winner once")
    if policy.duration_minimum_precedence not in {"duration_wins", "minimum_enrollment_wins"}:
        raise ValueError("invalid duration_minimum_precedence")
    if policy.multiple_winner not in {"highest_probability", "lowest_arm_index"}:
        raise ValueError("invalid multiple_winner")
    if policy.all_suspended_action not in {"stop", "skip_arrivals_until_next_analysis"}:
        raise ValueError("invalid all_suspended_action")
    if policy.same_time_order not in {"analysis_before_arrival", "arrival_before_analysis"}:
        raise ValueError("invalid same_time_order")


def _array(value: ArrayLike, name: str, *, binary: bool = False) -> NDArray:
    result = finite(value, name)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must be a nonempty matrix")
    if binary and np.any(~np.isin(result, [0, 1])):
        raise ValueError(f"{name} must contain only 0/1 values")
    return result


def _integer(value: int, name: str, low: int, high: int) -> int:
    number = scalar(value, name)
    if number != np.floor(number) or not low <= number <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return int(number)


def _floor(raw: np.ndarray, minimum: float, method: str) -> np.ndarray:
    if minimum == 0:
        return raw / raw.sum()
    arms = raw.size
    if minimum * arms > 1:
        raise ValueError("minimum_allocation is infeasible for the active arms")
    if method == "clamp_renormalize":
        result = np.maximum(raw, minimum)
        return result / result.sum()
    return (1.0 - arms * minimum) * raw / raw.sum() + minimum


def _weights(probability: np.ndarray, error: np.ndarray, tuning: float) -> np.ndarray:
    if tuning == 0:
        return np.full(probability.size, 1.0 / probability.size)
    if np.any(probability <= error):
        raise ArithmeticError("best-arm allocation tail unresolved; tighten tolerance")
    with np.errstate(divide="raise", over="ignore", under="ignore"):
        log_probability = np.log(probability)
        logits = tuning * (log_probability - np.max(log_probability))
        weights = np.exp(logits)
    return weights / weights.sum()


def _rank_state(
    parameters: np.ndarray,
    futile: np.ndarray,
    scope: str,
    family: str,
    maximize: bool,
    tolerance: float,
    base_probability: np.ndarray | None = None,
    base_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if scope == "all_arms" and base_probability is not None and base_error is not None:
        return base_probability.copy(), base_error.copy()
    rank_arms = np.arange(len(parameters)) if scope == "all_arms" else np.flatnonzero(~futile)
    probability = np.full(len(parameters), np.nan)
    error = np.full(len(parameters), np.nan)
    if rank_arms.size:
        result = arand_best_probability(
            parameters[rank_arms],
            family=family,
            maximize=maximize,
            absolute_tolerance=tolerance,
        )
        probability[rank_arms], error[rank_arms] = result.probability, result.absolute_error
    return probability, error


def _posterior_state(
    now: float,
    prior: np.ndarray,
    family: str,
    parameter: str,
    maximize: bool,
    futility_threshold: float | None,
    assigned_patients: list[int],
    assigned_arms: list[int],
    enrollment_times: list[float],
    outcomes: np.ndarray | None,
    delays: np.ndarray | None,
    event_times: np.ndarray | None,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    arms = len(prior)
    threshold = futility_threshold if maximize else None
    if family == "binary":
        successes = np.zeros(arms, dtype=int)
        failures = np.zeros(arms, dtype=int)
        observed = 0
        assert outcomes is not None and delays is not None
        for patient, arm, enrolled_at in zip(assigned_patients, assigned_arms, enrollment_times):
            if now >= enrolled_at + delays[patient, arm]:
                observed += 1
                if outcomes[patient, arm] == 1:
                    successes[arm] += 1
                else:
                    failures[arm] += 1
        posterior = arand_binary_posterior(
            successes,
            failures,
            prior=prior,
            maximize=maximize,
            tuning=0,
            threshold=threshold,
            absolute_tolerance=tolerance,
        )
    else:
        observed_events = np.zeros(arms, dtype=int)
        exposure = np.zeros(arms, dtype=float)
        observed = 0
        assert event_times is not None
        for patient, arm, enrolled_at in zip(assigned_patients, assigned_arms, enrollment_times):
            age = max(0.0, now - enrolled_at)
            event_time = event_times[patient, arm]
            observed_event = now >= enrolled_at + event_time
            observed += int(observed_event)
            observed_events[arm] += int(observed_event)
            exposure[arm] += event_time if observed_event else age
        posterior = arand_survival_posterior(
            observed_events,
            exposure,
            prior=prior,
            parameter=parameter,
            maximize=maximize,
            tuning=0,
            threshold=threshold,
            absolute_tolerance=tolerance,
        )
    return (
        posterior.parameters,
        posterior.exceedance_probability,
        np.array([observed]),
        posterior.best.probability,
        posterior.best.absolute_error,
    )


def arand_calendar_replay(
    arrival_times: ArrayLike,
    assignment_uniforms: ArrayLike,
    *,
    prior: ArrayLike,
    family: Literal["binary", "exponential"],
    analysis_times: ArrayLike,
    max_enrollment: int | None,
    max_duration: float | None,
    final_followup: float,
    minimum_enrollment: int,
    binary_outcomes: ArrayLike | None = None,
    binary_delays: ArrayLike | None = None,
    binary_window: float | None = None,
    event_times: ArrayLike | None = None,
    parameter: Literal["mean", "median"] = "mean",
    maximize: bool = True,
    initial_equal_randomization: int = 0,
    tuning: float = 0.5,
    minimum_allocation: float = 0.0,
    loser_cutoff: float | None = None,
    early_winner_cutoff: float | None = None,
    final_winner_cutoff: float | None = None,
    futility_threshold: float | None = None,
    futility_cutoff: float | None = None,
    policy: ArandControllerPolicy,
    absolute_tolerance: float = 1e-9,
    max_work: int = 100_000_000,
    data_seed: int | None = None,
) -> ArandCalendarTrial:
    """Replay one adaptive trial from arrival, assignment, and potential-outcome tapes.

    `analysis_times` supplements arrival-time looks. At a look, outcomes due at
    that instant and survival exposure through that instant are included before
    controller rules run. At an arrival, the rules run before assigning that patient.
    Binary delay matrices are time-to-availability per patient and potential arm;
    omitting them requires `binary_window` and delays every endpoint to that time.
    Exponential `event_times` are event delays from assignment; observations beyond
    the trial's final analysis time are right-censored at that time.

    The minimum-enrollment gate defers futility classification, loser suspension,
    and early stopping until the gate is met. If no active arm remains before the
    gate, arrivals are blocked regardless of the all-suspended stop policy; the
    replay does not assign to a suspended or permanently futile arm to reach the
    minimum. An exhausted tape below the minimum has no final winner. Under
    `minimum_enrollment_wins`, a duration reached below the minimum permits
    enrollment only until the first accepted patient reaches the minimum; if the
    tape ends first, the trial is reported as incomplete at tape exhaustion.
    A duration cutoff blocks an arrival exactly at its boundary under
    `duration_wins`; `minimum_enrollment_wins` accepts it if the minimum has not
    yet been reached. Tied arrivals are processed in tape order.
    """
    _validate_policy(policy)
    arrivals = finite(arrival_times, "arrival_times")
    uniforms = finite(assignment_uniforms, "assignment_uniforms")
    prior_array = finite(prior, "prior")
    if (
        arrivals.ndim != 1
        or not arrivals.size
        or np.any(arrivals < 0)
        or np.any(np.diff(arrivals) < 0)
    ):
        raise ValueError("arrival_times must be nonempty, nonnegative, and nondecreasing")
    if uniforms.shape != arrivals.shape or np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError("assignment_uniforms must match arrivals and lie in [0,1)")
    if prior_array.ndim != 2 or prior_array.shape[1] != 2 or not 1 <= len(prior_array) <= 10:
        raise ValueError("prior must contain 1 to 10 rows of two parameters")
    if np.any(prior_array <= 0):
        raise ValueError("prior parameters must be positive")
    arms = len(prior_array)
    if family not in {"binary", "exponential"}:
        raise ValueError("family must be binary or exponential")
    if len(arrivals) > 10_000 or len(arrivals) * arms > 2_000_000:
        raise ValueError("arrival tape exceeds controller resource limits")
    try:
        requested_look_count = len(cast(Sized, analysis_times))
    except TypeError as error:
        raise ValueError("analysis_times must be a one-dimensional vector") from error
    if requested_look_count > 10_000:
        raise ValueError("analysis_times exceeds controller resource limits")
    looks = finite(analysis_times, "analysis_times")
    if looks.ndim != 1 or np.any(looks < 0) or np.any(np.diff(looks) <= 0):
        raise ValueError("analysis_times must be a strictly increasing nonnegative vector")
    max_work_value = _integer(max_work, "max_work", 1, 2_000_000_000)
    nlooks_bound = len(arrivals) + len(looks) + 1
    estimated_work = (
        nlooks_bound * len(arrivals) * arms + nlooks_bound * arms * max(1, arms - 1) * 500
    )
    estimated_storage = nlooks_bound * (arms * 48 + 1_024)
    if nlooks_bound > 10_000 or estimated_storage > 10_000_000 or estimated_work > max_work_value:
        raise ValueError("arrival/outcome tapes exceed controller resource limits")
    if family == "binary":
        if binary_outcomes is None or event_times is not None:
            raise ValueError("binary replay requires binary_outcomes and forbids event_times")
        outcomes = _array(binary_outcomes, "binary_outcomes", binary=True)
        if outcomes.shape != (len(arrivals), arms):
            raise ValueError("binary_outcomes must have shape (patients, arms)")
        if binary_delays is None:
            if binary_window is None:
                raise ValueError("binary_window is required when binary_delays is omitted")
            delay = scalar(binary_window, "binary_window")
            delays = np.full(outcomes.shape, delay)
        else:
            if binary_window is not None:
                raise ValueError("supply binary_delays or binary_window, not both")
            delays = _array(binary_delays, "binary_delays")
            if delays.shape != outcomes.shape:
                raise ValueError("binary_delays must match binary_outcomes")
            if np.any(delays < 0):
                raise ValueError("binary_delays must be nonnegative")
        latent_events = None
    else:
        if (
            event_times is None
            or binary_outcomes is not None
            or binary_delays is not None
            or binary_window is not None
        ):
            raise ValueError("exponential replay requires event_times and no binary inputs")
        latent_events = np.asarray(event_times, dtype=float)
        if (
            latent_events.ndim != 2
            or latent_events.shape != (len(arrivals), arms)
            or np.any(np.isnan(latent_events))
            or np.any(latent_events < 0)
        ):
            raise ValueError("event_times must be nonnegative with shape (patients, arms)")
        event_deadlines = arrivals[:, None] + latent_events
        if np.any(np.isfinite(latent_events) & ~np.isfinite(event_deadlines)) or np.any(
            np.isfinite(latent_events)
            & (latent_events > 0)
            & (event_deadlines <= arrivals[:, None])
        ):
            raise ValueError("finite exponential event deadlines are not representable")
        delays = None
        outcomes = None
    if family == "binary" and binary_window is not None and binary_window <= 0:
        raise ValueError("binary_window must be positive")
    if parameter not in {"mean", "median"}:
        raise ValueError("parameter must be mean or median")
    if not isinstance(maximize, (bool, np.bool_)):
        raise ValueError("maximize must be boolean")
    followup = scalar(final_followup, "final_followup")
    if followup < 0:
        raise ValueError("final_followup must be nonnegative")
    duration = None if max_duration is None else scalar(max_duration, "max_duration")
    if duration is not None and duration <= 0:
        raise ValueError("max_duration must be positive")
    enrollment_cap = (
        None
        if max_enrollment is None
        else _integer(max_enrollment, "max_enrollment", 1, len(arrivals))
    )
    if enrollment_cap is not None and not 1 <= enrollment_cap <= len(arrivals):
        raise ValueError("max_enrollment must lie within the arrival tape")
    if enrollment_cap is None and duration is None:
        raise ValueError("max_enrollment or max_duration must be specified")
    initial_n = _integer(
        initial_equal_randomization, "initial_equal_randomization", 0, len(arrivals)
    )
    minimum_n = _integer(minimum_enrollment, "minimum_enrollment", 1, 100_000)
    if enrollment_cap is not None and minimum_n > enrollment_cap:
        raise ValueError("minimum_enrollment cannot exceed max_enrollment")
    if initial_n > len(arrivals):
        raise ValueError("initial_equal_randomization cannot exceed the arrival tape")
    tune = scalar(tuning, "tuning")
    floor = scalar(minimum_allocation, "minimum_allocation")
    if tune < 0 or not 0 <= floor < 1:
        raise ValueError("tuning must be nonnegative and minimum_allocation in [0,1)")
    tolerance = scalar(absolute_tolerance, "absolute_tolerance")
    if not 1e-12 <= tolerance <= 1e-3:
        raise ValueError("absolute_tolerance must lie in [1e-12, 1e-3]")
    loser = None if loser_cutoff is None else scalar(loser_cutoff, "loser_cutoff")
    early = (
        None if early_winner_cutoff is None else scalar(early_winner_cutoff, "early_winner_cutoff")
    )
    final_cut = (
        None if final_winner_cutoff is None else scalar(final_winner_cutoff, "final_winner_cutoff")
    )
    futile_cut = None if futility_cutoff is None else scalar(futility_cutoff, "futility_cutoff")
    futile_threshold = (
        None if futility_threshold is None else scalar(futility_threshold, "futility_threshold")
    )
    for name, value in (
        ("loser_cutoff", loser),
        ("early_winner_cutoff", early),
        ("final_winner_cutoff", final_cut),
        ("futility_cutoff", futile_cut),
    ):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0,1]")
    if (futile_cut is None) != (futile_threshold is None):
        raise ValueError("futility_threshold and futility_cutoff must be supplied together")
    if futile_threshold is not None and (
        (family == "binary" and not 0 <= futile_threshold <= 1)
        or (family == "exponential" and futile_threshold < 0)
    ):
        raise ValueError("futility_threshold is outside the outcome parameter domain")
    if initial_n > (enrollment_cap if enrollment_cap is not None else len(arrivals)):
        raise ValueError("initial_equal_randomization cannot exceed max_enrollment")
    if family == "binary":
        assert delays is not None
        if binary_window is not None and scalar(binary_window, "binary_window") < 0:
            raise ValueError("binary_window must be nonnegative")
        if np.any(~np.isfinite(arrivals[:, None] + delays)):
            raise ValueError("binary observation deadlines must be finite")
    if data_seed is not None and (
        isinstance(data_seed, bool) or not isinstance(data_seed, (int, np.integer)) or data_seed < 0
    ):
        raise ValueError("data_seed must be a nonnegative integer or None")
    if family == "binary":
        assert delays is not None
        deadlines = arrivals[:, None] + delays
        if np.any(~np.isfinite(deadlines)) or np.any(
            (delays > 0) & (deadlines <= arrivals[:, None])
        ):
            raise ValueError("binary observation deadlines are not representable")
    if duration is not None and not np.isfinite(duration + followup):
        raise ValueError("maximum duration plus final follow-up is not finite")
    if not np.isfinite(arrivals[-1] + followup):
        raise ValueError("latest arrival plus final follow-up is not finite")
    if not maximize and futile_cut is not None:
        raise ValueError("futility is available only when maximizing")

    assigned_patients: list[int] = []
    assigned_arms: list[int] = []
    enrolled_at: list[float] = []
    records: list[ArandCalendarLook] = []
    suspended = np.zeros(arms, dtype=bool)
    futile = np.zeros(arms, dtype=bool)
    early_winner: int | None = None
    reason = "arrival_tape_exhausted"
    stop_time: float | None = None
    enrollment_closed = False
    anchor_time = 0.0
    attempted = blocked = 0
    last_enrollment_time: float | None = None
    analysis_index = 0
    stopped_early = False
    incomplete = False

    def model_state(
        now: float,
    ) -> tuple[np.ndarray, np.ndarray | None, int, np.ndarray, np.ndarray]:
        parameters, exceedance, observed, base_probability, base_error = _posterior_state(
            now,
            prior_array,
            family,
            parameter,
            bool(maximize),
            futile_threshold,
            assigned_patients,
            assigned_arms,
            enrolled_at,
            outcomes,
            delays,
            latent_events,
            tolerance,
        )
        return parameters, exceedance, int(observed[0]), base_probability, base_error

    def ranked(
        parameters: np.ndarray, base_probability: np.ndarray, base_error: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return _rank_state(
            parameters,
            futile,
            policy.ranking_scope,
            "beta" if family == "binary" else "inverse_gamma",
            bool(maximize),
            tolerance,
            base_probability,
            base_error,
        )

    def choose_winner(probability: np.ndarray, eligible: np.ndarray, cutoff: float) -> int | None:
        winners = eligible[probability[eligible] > cutoff]
        if not winners.size:
            return None
        if policy.multiple_winner == "lowest_arm_index":
            return int(np.min(winners))
        return int(winners[np.argmax(probability[winners])])

    def inspect(
        now: float,
        *,
        run_rules: bool,
        final: bool = False,
        assigned_patient: int | None = None,
        assigned_arm: int | None = None,
    ) -> tuple[ArandCalendarLook, int | None, str | None]:
        nonlocal suspended, futile
        n = len(assigned_arms)
        if n < initial_n and not final:
            probability = exceedance = None
            allocation = np.full(arms, 1.0 / arms)
            status = "initial_equal_randomization"
            stop_winner = None
            stop_reason = None
            if outcomes is not None:
                assert delays is not None
                observed = sum(
                    now >= enrolled_at[i] + delays[patient, arm]
                    for i, (patient, arm) in enumerate(zip(assigned_patients, assigned_arms))
                )
            else:
                assert latent_events is not None
                observed = sum(
                    now >= enrolled_at[i] + latent_events[patient, arm]
                    for i, (patient, arm) in enumerate(zip(assigned_patients, assigned_arms))
                )
        else:
            parameters, exceedance, observed, base_probability, base_error = model_state(now)
            probability, errors = ranked(parameters, base_probability, base_error)
            stop_winner = None
            stop_reason = None
            if run_rules and n >= minimum_n:
                for rule in policy.trigger_order:
                    if rule == "futility" and maximize and futile_cut is not None:
                        assert exceedance is not None
                        newly_futile = (~futile) & (exceedance < futile_cut)
                        futile |= newly_futile
                        suspended[futile] = False
                        probability, errors = ranked(parameters, base_probability, base_error)
                        if np.all(futile):
                            stop_reason = "all_arms_futile"
                            break
                    elif rule == "suspension" and loser is not None:
                        comparable = np.flatnonzero(~futile)
                        for arm in comparable:
                            if probability[arm] < loser:
                                suspended[arm] = True
                            else:
                                suspended[arm] = False
                    elif (
                        rule == "early_winner"
                        and not final
                        and early is not None
                        and n >= minimum_n
                    ):
                        active = np.flatnonzero(~futile & ~suspended)
                        stop_winner = choose_winner(probability, active, early)
                        if stop_winner is not None:
                            stop_reason = "early_winner"
                            break
            active = np.flatnonzero(~futile & ~suspended)
            if stop_reason is None and not active.size:
                if np.all(futile):
                    stop_reason = "all_arms_futile"
                elif policy.all_suspended_action == "stop" and n >= minimum_n:
                    stop_reason = "all_arms_suspended"
            if final or stop_reason is not None:
                allocation = None
                status = "final_analysis" if final else stop_reason or "stopped"
            elif n < initial_n:
                allocation = np.full(arms, 1 / arms)
                status = "initial_equal_randomization"
            elif active.size:
                local = _weights(probability[active], errors[active], tune)
                allocation = np.zeros(arms)
                allocation[active] = _floor(local, floor, policy.floor_transform)
                status = "analyzed"
            else:
                allocation = None
                status = "all_arms_futile" if np.all(futile) else "all_arms_suspended"
        record = ArandCalendarLook(
            float(now),
            n,
            observed,
            None if probability is None else _readonly(probability, dtype=float),
            None if exceedance is None else _readonly(exceedance, dtype=float),
            None if allocation is None else _readonly(allocation, dtype=float),
            _readonly(suspended, dtype=bool),
            _readonly(futile, dtype=bool),
            assigned_patient,
            assigned_arm,
            status,
        )
        records.append(record)
        return record, stop_winner, stop_reason

    def scheduled(at: float) -> bool:
        nonlocal analysis_index
        return analysis_index < len(looks) and looks[analysis_index] == at

    def process_analysis(at: float) -> bool:
        nonlocal analysis_index, early_winner, reason, stop_time, stopped_early
        analysis_index += 1
        _, winner, stop_reason = inspect(at, run_rules=True)
        if stop_reason is not None:
            reason = stop_reason
            stop_time = at
            stopped_early = True
            early_winner = winner
            return True
        return False

    # Walk the union in event time order. Arrival-time looks precede assignment;
    # a scheduled look at the same instant follows the selected policy ordering.
    arrival_index = 0
    while arrival_index < len(arrivals) and not stopped_early:
        at = float(arrivals[arrival_index])
        if enrollment_closed and at > anchor_time + followup:
            break
        while analysis_index < len(looks) and looks[analysis_index] < at and not stopped_early:
            if enrollment_closed and looks[analysis_index] >= anchor_time + followup:
                break
            if not enrollment_closed:
                if (
                    duration is not None
                    and looks[analysis_index] >= duration
                    and (
                        policy.duration_minimum_precedence == "duration_wins"
                        or len(assigned_arms) >= minimum_n
                    )
                ):
                    enrollment_closed = True
                    anchor_time = duration
                    reason = "maximum_duration"
            if enrollment_closed and looks[analysis_index] >= anchor_time + followup:
                break
            if process_analysis(float(looks[analysis_index])):
                break
        if stopped_early:
            break
        if enrollment_closed and at > anchor_time + followup:
            break
        if (
            not enrollment_closed
            and duration is not None
            and at >= duration
            and (
                policy.duration_minimum_precedence == "duration_wins"
                or len(assigned_arms) >= minimum_n
            )
        ):
            enrollment_closed = True
            anchor_time = duration
            reason = "maximum_duration"
        if enrollment_closed and at > anchor_time + followup:
            break
        if (
            scheduled(at)
            and policy.same_time_order == "analysis_before_arrival"
            and (not enrollment_closed or at < anchor_time + followup)
        ):
            if not enrollment_closed and process_analysis(at):
                break
            if enrollment_closed:
                process_analysis(at)
        while arrival_index < len(arrivals) and arrivals[arrival_index] == at:
            patient = arrival_index
            arrival_index += 1
            attempted += 1
            if enrollment_closed:
                blocked += 1
                continue
            if enrollment_cap is not None and len(assigned_arms) >= enrollment_cap:
                enrollment_closed = True
                anchor_time = float(last_enrollment_time or 0.0)
                reason = "maximum_enrollment"
                blocked += 1
                continue
            _, winner, stop_reason = inspect(at, run_rules=True)
            if stop_reason is not None:
                reason, stop_time, early_winner, stopped_early = stop_reason, at, winner, True
                break
            active = np.flatnonzero(~futile & ~suspended)
            if not active.size:
                if np.all(futile):
                    reason, stop_time, stopped_early = "all_arms_futile", at, True
                    break
                blocked += 1
                if policy.all_suspended_action == "stop" and len(assigned_arms) >= minimum_n:
                    reason, stop_time, stopped_early = "all_arms_suspended", at, True
                    break
                continue
            last_record = records[-1]
            assert last_record.allocation_probability is not None
            allocation = last_record.allocation_probability
            arm = int(np.searchsorted(np.cumsum(allocation), uniforms[patient], side="right"))
            arm = min(arm, arms - 1)
            assigned_patients.append(patient)
            assigned_arms.append(arm)
            enrolled_at.append(at)
            last_enrollment_time = at
            records[-1] = ArandCalendarLook(
                last_record.time,
                last_record.enrolled,
                last_record.observed,
                last_record.posterior_probability,
                last_record.exceedance_probability,
                last_record.allocation_probability,
                last_record.suspended,
                last_record.futile,
                patient,
                arm,
                "assigned",
            )
            if enrollment_cap is not None and len(assigned_arms) >= enrollment_cap:
                enrollment_closed = True
                anchor_time = at
                reason = "maximum_enrollment"
            elif (
                duration is not None
                and at >= duration
                and policy.duration_minimum_precedence == "minimum_enrollment_wins"
                and len(assigned_arms) >= minimum_n
            ):
                enrollment_closed = True
                anchor_time = at
                reason = "minimum_enrollment_after_duration"
        if stopped_early:
            break
        if (
            scheduled(at)
            and policy.same_time_order == "arrival_before_analysis"
            and (not enrollment_closed or at < anchor_time + followup)
        ):
            if process_analysis(at):
                break

    if not stopped_early and not enrollment_closed:
        enrollment_closed = True
        if (
            duration is not None
            and policy.duration_minimum_precedence == "minimum_enrollment_wins"
            and len(assigned_arms) < minimum_n
        ):
            anchor_time = float(arrivals[-1])
            reason = "minimum_not_reached_input_exhausted"
            incomplete = True
        elif duration is not None:
            anchor_time = duration
            reason = "maximum_duration"
        else:
            anchor_time = float(last_enrollment_time or 0.0)
            reason = "arrival_tape_exhausted"
    if not stopped_early and enrollment_closed and reason == "arrival_tape_exhausted":
        reason = (
            "maximum_enrollment"
            if enrollment_cap is not None and len(assigned_arms) >= enrollment_cap
            else reason
        )
    if not stopped_early and not incomplete:
        final_time = float(anchor_time + followup)
        if not np.isfinite(final_time):
            raise ValueError("final analysis time is not finite")
        while analysis_index < len(looks) and looks[analysis_index] < final_time:
            at = float(looks[analysis_index])
            if process_analysis(at):
                break
        if stopped_early:
            final_time = float(stop_time)
        else:
            final_record, _, final_stop_reason = inspect(final_time, run_rules=True, final=True)
            if final_stop_reason == "all_arms_futile":
                reason = final_stop_reason
            probabilities = (
                np.full(arms, np.nan)
                if final_record.posterior_probability is None
                else final_record.posterior_probability
            )
            rank_arms = (
                np.flatnonzero(np.isfinite(probabilities) & ~futile)
                if final_record.enrolled >= minimum_n and not np.all(futile)
                else np.array([], dtype=int)
            )
            final_winner = (
                None
                if final_cut is None or not rank_arms.size
                else choose_winner(probabilities, rank_arms, final_cut)
            )
            final_winner = final_winner if final_winner is not None else None
            if (
                family == "binary"
                and final_record.observed < final_record.enrolled
                and final_stop_reason != "all_arms_futile"
            ):
                final_winner = None
                reason = "final_followup_incomplete"
    if stopped_early:
        assert stop_time is not None
        final_time = stop_time
        reported_final_time = None
        final_winner = None
    elif not incomplete:
        reported_final_time = final_time
    else:
        reported_final_time = None
        final_winner = None
        final_time = anchor_time
    assignments = _readonly(assigned_arms, dtype=np.int64)
    enrollment_array = _readonly(enrolled_at, dtype=float)
    decision_anchor = stop_time if stop_time is not None else anchor_time
    decision_duration = float(decision_anchor)
    completed_duration = float(final_time)
    return ArandCalendarTrial(
        family,
        assignments,
        enrollment_array,
        tuple(records),
        _readonly(suspended, dtype=bool),
        _readonly(futile, dtype=bool),
        early_winner,
        final_winner,
        reason,
        stopped_early,
        stop_time,
        reported_final_time,
        decision_duration,
        completed_duration,
        attempted,
        blocked,
        data_seed,
    )
