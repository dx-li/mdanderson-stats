"""Completed-outcome, no-control PLBARPO platform trial replay.

The platform-entry rules in this module are explicit Python conventions, not
claims of native scheduler parity. Arms enter once in caller-supplied order;
closed arms are never reopened. Each entrant receives its own equal-randomized
burn-in, and all outcomes are available before the next allocation update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import count, finite, scalar
from .barpo import barpo_monitor
from .barpo_trial import _categorical, _integer, _uniform_tape
from .plbarpo_allocation import PLBarpoActiveAllocation, plbarpo_active_allocation

_MAX_ARMS = 100
_MAX_ACTIVE = 10
_MAX_PATIENTS = 2_000
_MAX_LOOKS = 200
_MAX_TRIAL_WORK = 20_000_000


def _bool_vector(value: ArrayLike, name: str, length: int) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != (length,):
        raise ValueError(f"{name} must be a boolean vector with length {length}")
    result = np.array(raw, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _ledger_counts(value: ArrayLike, name: str, length: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.shape != (length,):
        raise ValueError(f"{name} must be a real vector with one value per arm")
    result = count(raw, name)
    if np.any(result >= 2**53):
        raise ValueError(f"{name} values must be below 2**53")
    return np.asarray(result, dtype=np.int64)


def _ledger_real(value: ArrayLike, name: str, length: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.shape != (length,):
        raise ValueError(f"{name} must be a real vector with one value per arm")
    return finite(raw, name)


def _prior(value: ArrayLike, arms: int) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (arms, 2):
        raise ValueError(f"prior must have shape ({arms}, 2)")
    result = finite(raw, "prior")
    if np.any(result <= 0) or np.any(~np.isfinite(result.sum(axis=1))):
        raise ValueError("prior must contain positive beta shapes with finite sums")
    return result


def _readonly(value: ArrayLike, dtype: type = np.float64) -> NDArray:
    result: NDArray = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _streams(
    rng: int | np.integer | np.random.Generator | None,
) -> tuple[np.random.Generator, np.random.Generator, tuple[int, int]]:
    if isinstance(rng, np.random.Generator):
        entropy = rng.integers(0, 2**32, size=4, dtype=np.uint32)
        root = np.random.SeedSequence(entropy)
    else:
        if rng is not None and (
            isinstance(rng, (bool, np.bool_))
            or not isinstance(rng, (int, np.integer))
            or int(rng) < 0
        ):
            raise ValueError("rng must be a nonnegative integer, Generator or None")
        root = np.random.SeedSequence(None if rng is None else int(rng))
    child = root.spawn(2)
    seeds = (int(child[0].generate_state(1, dtype=np.uint64)[0]),
             int(child[1].generate_state(1, dtype=np.uint64)[0]))
    return np.random.default_rng(seeds[0]), np.random.default_rng(seeds[1]), seeds


@dataclass(frozen=True)
class PLBarpoTrialLook:
    enrolled: int
    active_before: NDArray[np.bool_]
    active_after: NDArray[np.bool_]
    best_probability: NDArray[np.float64]
    best_probability_error: NDArray[np.float64]
    futility_probability: NDArray[np.float64] | None
    efficacy_probability: NDArray[np.float64] | None
    final_probability: NDArray[np.float64] | None
    futile: NDArray[np.bool_]
    efficacious: NDArray[np.bool_]
    final_efficacious: NDArray[np.bool_]


@dataclass(frozen=True)
class PLBarpoTrialResult:
    assignments: NDArray[np.int64]
    responses: NDArray[np.int8]
    assignment_probability: NDArray[np.float64]
    assigned: NDArray[np.int64]
    successes: NDArray[np.int64]
    failures: NDArray[np.int64]
    entered: NDArray[np.bool_]
    active: NDArray[np.bool_]
    stopped: NDArray[np.bool_]
    cap_closed: NDArray[np.bool_]
    stop_kind: tuple[str, ...]
    arm_stop_enrollment: NDArray[np.int64]
    early_futility: NDArray[np.bool_]
    early_efficacy: NDArray[np.bool_]
    final_efficacy: NDArray[np.bool_]
    final_assessed: NDArray[np.bool_]
    final_efficacy_probability: NDArray[np.float64]
    looks: tuple[PLBarpoTrialLook, ...]
    enrolled: int
    max_total_n: int
    stop_reason: str
    work_units: int
    rng_seeds: tuple[int, int]


@dataclass(frozen=True)
class _Config:
    truth: NDArray[np.float64]
    prior: NDArray[np.float64]
    initial_active: NDArray[np.bool_]
    candidate_order: NDArray[np.int64]
    min_n: NDArray[np.int64]
    max_n: NDArray[np.int64]
    max_total_n: int
    look_sizes: NDArray[np.int64]
    burn_in: int
    method: str
    tau: float
    tau1: float
    target_probability: NDArray[np.float64] | None
    minimum_probability: NDArray[np.float64] | None
    control: bool
    early_monitoring: bool
    theta_fut: float | None
    pfut: float | None
    theta_eff: float | None
    peff: float | None
    theta_final: float | None
    pfinal: float | None
    absolute_tolerance: float
    floor_policy: str
    work_limit: int
    active_capacity: int


def _prepare(
    true_response: ArrayLike,
    *,
    prior: ArrayLike,
    initial_active: ArrayLike,
    candidate_order: ArrayLike,
    min_n_per_arm: ArrayLike,
    max_n_per_arm: ArrayLike,
    max_total_n: int,
    look_sizes: ArrayLike,
    burn_in_per_arm: int,
    method: str,
    tau: float,
    tau1: float,
    target_probability: ArrayLike | None,
    minimum_probability: ArrayLike | None,
    early_monitoring: bool,
    theta_fut: float | None,
    pfut: float | None,
    theta_eff: float | None,
    peff: float | None,
    theta_final: float | None,
    pfinal: float | None,
    absolute_tolerance: float,
    max_work: int,
) -> _Config:
    truth_raw = np.asarray(true_response)
    if (
        truth_raw.dtype.kind not in "iuf"
        or truth_raw.ndim != 1
        or not 1 <= truth_raw.size <= _MAX_ARMS
    ):
        raise ValueError(f"true_response must be a real vector of 1..{_MAX_ARMS} arms")
    truth = finite(truth_raw, "true_response")
    arms = int(truth.size)
    if np.any((truth < 0) | (truth > 1)):
        raise ValueError("true_response values must lie in [0,1]")
    prior_array = _prior(prior, arms)
    active = _bool_vector(initial_active, "initial_active", arms)
    active_count = int(active.sum())
    if not 1 <= active_count <= _MAX_ACTIVE:
        raise ValueError(f"initial_active must contain 1..{_MAX_ACTIVE} active arms")
    raw_candidates = np.asarray(candidate_order)
    if (
        raw_candidates.dtype.kind not in "iuf"
        or raw_candidates.ndim != 1
        or raw_candidates.size > arms
    ):
        raise ValueError("candidate_order must be a bounded integer vector of ledger indices")
    candidate_counts = count(raw_candidates, "candidate_order")
    if np.any(candidate_counts >= arms):
        raise ValueError("candidate_order contains an out-of-range arm index")
    candidates = np.asarray(candidate_counts, dtype=np.int64)
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_order entries must be unique")
    if np.any(active[candidates]):
        raise ValueError("candidate_order must exclude initially active arms")

    maximum = _integer(max_total_n, "max_total_n", 1, _MAX_PATIENTS)
    minimum = _ledger_counts(min_n_per_arm, "min_n_per_arm", arms)
    per_arm_max = _ledger_counts(max_n_per_arm, "max_n_per_arm", arms)
    if np.any(minimum < 1) or np.any(per_arm_max < minimum) or np.any(per_arm_max > maximum):
        raise ValueError("per-arm bounds must satisfy 1 <= min_n <= max_n <= max_total_n")
    looks_raw = np.asarray(look_sizes)
    if (
        looks_raw.dtype.kind not in "iuf"
        or looks_raw.ndim != 1
        or not 1 <= looks_raw.size <= _MAX_LOOKS
    ):
        raise ValueError(f"look_sizes must contain 1..{_MAX_LOOKS} global enrollment counts")
    look_count = count(looks_raw, "look_sizes")
    if np.any(look_count < 1) or np.any(look_count > maximum):
        raise ValueError("look_sizes must be in [1,max_total_n]")
    looks = np.asarray(look_count, dtype=np.int64)
    if np.any(np.diff(looks) <= 0) or int(looks[-1]) != maximum:
        raise ValueError("look_sizes must increase strictly and end at max_total_n")
    burn = _integer(burn_in_per_arm, "burn_in_per_arm", 0, maximum)
    possible_entrants = np.concatenate((np.flatnonzero(active), candidates))
    if np.any(per_arm_max[possible_entrants] < burn):
        raise ValueError("every entrant must have enough capacity for burn-in")
    if burn * active_count > maximum:
        raise ValueError("initial active set cannot complete per-arm burn-in before max_total_n")
    if method not in ("barcp", "barn2n", "barmtv", "dbcd"):
        raise ValueError("method must be barcp, barn2n, barmtv, or dbcd")
    if method == "dbcd" and burn < 1:
        raise ValueError("dbcd requires burn_in_per_arm >= 1")
    if not isinstance(early_monitoring, (bool, np.bool_)):
        raise ValueError("early_monitoring must be boolean")
    tau_value, tau1_value = scalar(tau, "tau"), scalar(tau1, "tau1")
    if tau_value < 0 or tau1_value < 0:
        raise ValueError("tau and tau1 must be nonnegative")
    tolerance = scalar(absolute_tolerance, "absolute_tolerance")
    if tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    work_limit = _integer(max_work, "max_work", 1, _MAX_TRIAL_WORK)

    target: NDArray[np.float64] | None
    if method == "dbcd":
        if target_probability is None:
            raise ValueError("target_probability is required for dbcd")
        target = _ledger_real(target_probability, "target_probability", arms)
        if np.any(target <= 0) or not np.isclose(target.sum(), 1.0, rtol=0, atol=2e-12):
            raise ValueError("DBCD targets must be positive and sum to one across the ledger")
    else:
        target = None if target_probability is None else _ledger_real(
            target_probability, "target_probability", arms
        )
    if minimum_probability is None:
        floors = None
    else:
        floors = _ledger_real(minimum_probability, "minimum_probability", arms)
        if np.any(floors < 0):
            raise ValueError("minimum_probability must be nonnegative")
        largest = np.sort(floors)[-active_count:]
        if float(largest.sum()) > 1.0:
            raise ValueError("floors must be feasible for every active subset of capacity")

    # Validate monitoring gates before random streams are derived.
    active_idx = np.flatnonzero(active)
    zero = np.zeros(active_count, dtype=np.int64)
    barpo_monitor(
        zero,
        zero,
        prior=prior_array[active_idx],
        theta_fut=theta_fut,
        pfut=pfut,
        theta_eff=theta_eff,
        peff=peff,
        theta_final=theta_final,
        pfinal=pfinal,
        absolute_tolerance=tolerance,
    )
    maxn = np.zeros(arms, dtype=np.int64)
    maxn[active] = burn
    dry_target = None if target is None else np.where(active, target, 0.0)
    dry_floor = None if floors is None else np.where(active, floors, 0.0)
    if burn > 0:
        plbarpo_active_allocation(
            np.zeros(arms, dtype=np.int64),
            np.zeros(arms, dtype=np.int64),
            maxn,
            active,
            prior=prior_array,
            method=method,
            tau=tau_value,
            tau1=tau1_value,
            max_n=maximum if method == "barn2n" else None,
            target_probability=dry_target,
            minimum_probability=dry_floor,
            absolute_tolerance=tolerance,
        )
    work_bound = (2 * maximum + 2 * len(looks) + 1) * active_count**2
    if work_bound > work_limit:
        raise ValueError("per-trial active-allocation work exceeds max_work")
    return _Config(
        truth,
        prior_array,
        active,
        candidates,
        minimum,
        per_arm_max,
        maximum,
        looks,
        burn,
        method,
        tau_value,
        tau1_value,
        target,
        floors,
        False,
        bool(early_monitoring),
        theta_fut,
        pfut,
        theta_eff,
        peff,
        theta_final,
        pfinal,
        tolerance,
        "barpo_water_filling",
        work_limit,
        active_count,
    )


def _run_prepared(
    config: _Config,
    assignment_tape: NDArray[np.float64] | None,
    outcome_tape: NDArray[np.float64] | None,
    assignment_rng: np.random.Generator,
    outcome_rng: np.random.Generator,
    seeds: tuple[int, int],
) -> PLBarpoTrialResult:
    arms = len(config.truth)
    maximum = config.max_total_n
    assigned = np.zeros(arms, dtype=np.int64)
    successes = np.zeros(arms, dtype=np.int64)
    failures = np.zeros(arms, dtype=np.int64)
    entered = np.zeros(arms, dtype=bool)
    entered[:] = config.initial_active
    active = config.initial_active.copy()
    stopped = np.zeros(arms, dtype=bool)
    cap_closed = np.zeros(arms, dtype=bool)
    early_futility = np.zeros(arms, dtype=bool)
    early_efficacy = np.zeros(arms, dtype=bool)
    final_efficacy = np.zeros(arms, dtype=bool)
    final_assessed = np.zeros(arms, dtype=bool)
    final_efficacy_probability = np.full(arms, np.nan)
    stop_kind = np.full(arms, "not_entered", dtype=object)
    stop_kind[active] = "active"
    stop_enrollment = np.zeros(arms, dtype=np.int64)
    assignment = np.empty(maximum, dtype=np.int64)
    response = np.empty(maximum, dtype=np.int8)
    allocation_history = np.zeros((maximum, arms), dtype=float)
    candidate_position = 0
    n = 0
    work = 0
    reason = "maximum_enrollment"
    looks: list[PLBarpoTrialLook] = []
    look_set = set(int(x) for x in config.look_sizes)
    allocation = np.zeros(arms)
    allocation_valid = False

    def fill_slots() -> bool:
        nonlocal candidate_position
        changed = False
        while (
            active.sum() < config.active_capacity
            and candidate_position < len(config.candidate_order)
        ):
            idx = int(config.candidate_order[candidate_position])
            candidate_position += 1
            if n >= maximum:
                break
            entered[idx] = True
            active[idx] = True
            stop_kind[idx] = "active"
            changed = True
        return changed

    def current_weights() -> PLBarpoActiveAllocation:
        nonlocal work
        active_n = int(active.sum())
        work += active_n**2
        if work > config.work_limit:
            raise RuntimeError("PLBARPO trial allocation work exceeded max_work")
        return plbarpo_active_allocation(
            successes,
            failures,
            assigned,
            active,
            prior=config.prior,
            method=config.method,
            tau=config.tau,
            tau1=config.tau1,
            max_n=maximum if config.method == "barn2n" else None,
            target_probability=(
                None
                if config.target_probability is None
                else np.where(active, config.target_probability, 0.0)
            ),
            minimum_probability=(
                None
                if config.minimum_probability is None
                else np.where(active, config.minimum_probability, 0.0)
            ),
            absolute_tolerance=config.absolute_tolerance,
        )

    def draw(tape: NDArray[np.float64] | None, generator: np.random.Generator, index: int) -> float:
        return float(tape[index]) if tape is not None else float(generator.random())

    if config.burn_in == 0:
        allocation = current_weights().allocation_probability.copy()
        allocation_valid = True

    while n < maximum and np.any(active):
        under_burn = active & (assigned < config.burn_in)
        if np.any(under_burn):
            allocation = np.zeros(arms)
            allocation[under_burn] = 1.0 / int(under_burn.sum())
            allocation_valid = False
        elif not allocation_valid:
            allocation = current_weights().allocation_probability.copy()
            allocation_valid = True

        arm_u = draw(assignment_tape, assignment_rng, n)
        arm = _categorical(allocation, arm_u)
        if not active[arm]:
            raise ArithmeticError("PLBARPO assignment selected an inactive arm")
        assignment[n] = arm
        allocation_history[n] = allocation
        response[n] = int(draw(outcome_tape, outcome_rng, n) < config.truth[arm])
        assigned[arm] += 1
        if response[n]:
            successes[arm] += 1
        else:
            failures[arm] += 1
        n += 1

        cap_hit = assigned[arm] >= config.max_n[arm]
        if n not in look_set:
            if cap_hit:
                if config.pfinal is not None:
                    work += 1
                    if work > config.work_limit:
                        raise RuntimeError("PLBARPO trial monitoring work exceeded max_work")
                    cap_monitor = barpo_monitor(
                        successes[arm : arm + 1],
                        failures[arm : arm + 1],
                        prior=config.prior[arm : arm + 1],
                        theta_final=config.theta_final,
                        pfinal=config.pfinal,
                        absolute_tolerance=config.absolute_tolerance,
                    )
                    if cap_monitor.final_efficacy_probability is not None:
                        assert cap_monitor.final_efficacious is not None
                        final_assessed[arm] = True
                        final_efficacy_probability[arm] = float(
                            cap_monitor.final_efficacy_probability[0]
                        )
                        final_efficacy[arm] = bool(cap_monitor.final_efficacious[0])
                cap_closed[arm] = True
                active[arm] = False
                stopped[arm] = True
                if not final_efficacy[arm]:
                    stop_kind[arm] = "arm_maximum"
                else:
                    stop_kind[arm] = "final_efficacy_at_arm_cap"
                stop_enrollment[arm] = n
                if n < maximum:
                    fill_slots()
                allocation.fill(0.0)
                allocation_valid = False
                if active.any() and not np.any(active & (assigned < config.burn_in)):
                    allocation = current_weights().allocation_probability.copy()
                    allocation_valid = True
                elif not np.any(active):
                    reason = "all_arms_closed_no_candidates"
                    break
            continue

        active_before = active.copy()
        best = np.zeros(arms)
        best_error = np.zeros(arms)
        futility = None
        efficacy = None
        final_probability = None
        futile = np.zeros(arms, dtype=bool)
        efficacious = np.zeros(arms, dtype=bool)
        final_flags = np.zeros(arms, dtype=bool)
        idx = np.flatnonzero(active)
        is_final = n == maximum
        if idx.size:
            work += len(idx) ** 2
            if work > config.work_limit:
                raise RuntimeError("PLBARPO trial monitoring work exceeded max_work")
            assess_final = is_final or (cap_hit and config.pfinal is not None)
            monitoring = barpo_monitor(
                successes[idx],
                failures[idx],
                prior=config.prior[idx],
                theta_fut=None if is_final or not config.early_monitoring else config.theta_fut,
                pfut=None if is_final or not config.early_monitoring else config.pfut,
                theta_eff=None if is_final or not config.early_monitoring else config.theta_eff,
                peff=None if is_final or not config.early_monitoring else config.peff,
                theta_final=config.theta_final if assess_final else None,
                pfinal=config.pfinal if assess_final else None,
                absolute_tolerance=config.absolute_tolerance,
            )
            best[idx] = monitoring.posterior.best_probability
            best_error[idx] = monitoring.posterior.best_probability_error
            if monitoring.futility_probability is not None:
                futility = np.zeros(arms)
                futility[idx] = monitoring.futility_probability
            if monitoring.efficacy_probability is not None:
                efficacy = np.zeros(arms)
                efficacy[idx] = monitoring.efficacy_probability
            if monitoring.final_efficacy_probability is not None:
                assert monitoring.final_efficacious is not None
                final_probability = np.zeros(arms)
                final_probability[idx] = monitoring.final_efficacy_probability
                if is_final:
                    final_assessed[idx] = assigned[idx] >= config.min_n[idx]
                    final_efficacy_probability[idx[final_assessed[idx]]] = (
                        monitoring.final_efficacy_probability[final_assessed[idx]]
                    )
                    final_flags[idx[final_assessed[idx]]] = monitoring.final_efficacious[
                        final_assessed[idx]
                    ]
                    final_efficacy |= final_flags
                    newly_final = idx[final_flags[idx]]
                    stopped[newly_final] = True
                    active[newly_final] = False
                    stop_kind[newly_final] = "final_efficacy"
                    stop_enrollment[newly_final] = n
                elif cap_hit:
                    cap_position = int(np.searchsorted(idx, arm))
                    if assigned[arm] >= config.min_n[arm]:
                        final_assessed[arm] = True
                        final_efficacy_probability[arm] = float(
                            monitoring.final_efficacy_probability[cap_position]
                        )
                        final_efficacy[arm] = bool(monitoring.final_efficacious[cap_position])
                        final_flags[arm] = final_efficacy[arm]

            if not is_final and config.early_monitoring:
                eligible = assigned[idx] >= config.min_n[idx]
                if monitoring.futile is not None:
                    futile[idx[eligible]] = monitoring.futile[eligible]
                if monitoring.efficacious is not None:
                    efficacious[idx[eligible]] = monitoring.efficacious[eligible]
                if np.any(futile & efficacious):
                    raise ValueError("an active arm cannot be futile and efficacious at one look")
                early_futility |= futile
                early_efficacy |= efficacious

        # Apply same-look declarations and cap closure together, only after all
        # probabilities have been evaluated on the pre-transition active set.
        closing = futile | efficacious
        if np.any(closing):
            stopped[closing] = True
            stop_enrollment[closing & (stop_enrollment == 0)] = n
            stop_kind[closing & futile] = "futility"
            stop_kind[closing & efficacious] = "efficacy"
            active[closing] = False
        if cap_hit:
            cap_closed[arm] = True
            stopped[arm] = True
            active[arm] = False
            if not closing[arm] and not final_efficacy[arm]:
                stop_kind[arm] = "trial_maximum" if is_final else "arm_maximum"
                stop_enrollment[arm] = n
            elif final_efficacy[arm] and not closing[arm]:
                stop_kind[arm] = "final_efficacy" if is_final else "final_efficacy_at_arm_cap"
                stop_enrollment[arm] = n

        if not is_final and n < maximum and active.sum() < config.active_capacity:
            fill_slots()
        if is_final:
            remaining_active = active.copy()
            stopped[remaining_active] = True
            stop_kind[remaining_active] = "trial_maximum"
            stop_enrollment[remaining_active] = n
            active[remaining_active] = False
        active_after = active.copy()
        looks.append(
            PLBarpoTrialLook(
                n,
                _readonly(active_before, bool),
                _readonly(active_after, bool),
                _freeze(best),
                _freeze(best_error),
                None if futility is None else _freeze(futility),
                None if efficacy is None else _freeze(efficacy),
                None if final_probability is None else _freeze(final_probability),
                _readonly(futile, bool),
                _readonly(efficacious, bool),
                _readonly(final_flags, bool),
            )
        )
        if is_final:
            reason = "maximum_enrollment"
            break
        if not np.any(active):
            reason = "all_arms_closed_no_candidates"
            break
        allocation.fill(0.0)
        allocation_valid = False
        if not np.any(active & (assigned < config.burn_in)):
            allocation = current_weights().allocation_probability.copy()
            allocation_valid = True

    if n == maximum:
        reason = "maximum_enrollment"
    return PLBarpoTrialResult(
        _readonly(assignment[:n], np.int64),
        _readonly(response[:n], np.int8),
        _freeze(allocation_history[:n]),
        _readonly(assigned, np.int64),
        _readonly(successes, np.int64),
        _readonly(failures, np.int64),
        _readonly(entered, bool),
        _readonly(active, bool),
        _readonly(stopped, bool),
        _readonly(cap_closed, bool),
        tuple(str(x) for x in stop_kind),
        _readonly(stop_enrollment, np.int64),
        _readonly(early_futility, bool),
        _readonly(early_efficacy, bool),
        _readonly(final_efficacy, bool),
        _readonly(final_assessed, bool),
        _freeze(final_efficacy_probability),
        tuple(looks),
        n,
        maximum,
        reason,
        work,
        seeds,
    )


def run_plbarpo_trial(
    true_response: ArrayLike,
    *,
    prior: ArrayLike,
    initial_active: ArrayLike,
    candidate_order: ArrayLike,
    min_n_per_arm: ArrayLike,
    max_n_per_arm: ArrayLike,
    max_total_n: int,
    look_sizes: ArrayLike,
    burn_in_per_arm: int,
    method: str = "barcp",
    tau: float = 0.5,
    tau1: float = 1.0,
    target_probability: ArrayLike | None = None,
    minimum_probability: ArrayLike | None = None,
    early_monitoring: bool = True,
    theta_fut: float | None = None,
    pfut: float | None = None,
    theta_eff: float | None = None,
    peff: float | None = None,
    theta_final: float | None = None,
    pfinal: float | None = None,
    rng: int | np.integer | np.random.Generator | None = None,
    assignment_uniforms: ArrayLike | None = None,
    outcome_uniforms: ArrayLike | None = None,
    absolute_tolerance: float = 1e-9,
    max_work: int = _MAX_TRIAL_WORK,
) -> PLBarpoTrialResult:
    """Replay one no-control PLBARPO platform trial under explicit Python rules.

    Global looks are the only statistical monitoring times and the last look
    must equal ``max_total_n``. At a look, all eligible closure flags are
    evaluated together, then replacement follows caller ``candidate_order``.
    Reaching an arm cap closes it immediately and replaces it before the next
    patient. Each entrant gets its own equal-randomized burn-in; other arms are
    not reset. Active posterior weights otherwise stay frozen until a look or
    active-set change. Initial and candidate indices are zero-based. The two
    exact-length uniform tapes make patient-level assignment and binary
    outcome replay deterministic.
    """
    config = _prepare(
        true_response,
        prior=prior,
        initial_active=initial_active,
        candidate_order=candidate_order,
        min_n_per_arm=min_n_per_arm,
        max_n_per_arm=max_n_per_arm,
        max_total_n=max_total_n,
        look_sizes=look_sizes,
        burn_in_per_arm=burn_in_per_arm,
        method=method,
        tau=tau,
        tau1=tau1,
        target_probability=target_probability,
        minimum_probability=minimum_probability,
        early_monitoring=early_monitoring,
        theta_fut=theta_fut,
        pfut=pfut,
        theta_eff=theta_eff,
        peff=peff,
        theta_final=theta_final,
        pfinal=pfinal,
        absolute_tolerance=absolute_tolerance,
        max_work=max_work,
    )
    assign_tape = _uniform_tape(assignment_uniforms, config.max_total_n, "assignment_uniforms")
    outcome_tape = _uniform_tape(outcome_uniforms, config.max_total_n, "outcome_uniforms")
    assignment_rng, outcome_rng, seeds = _streams(rng)
    return _run_prepared(config, assign_tape, outcome_tape, assignment_rng, outcome_rng, seeds)
