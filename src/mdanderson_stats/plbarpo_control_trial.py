"""Completed-outcome control-enabled PLBARPO platform trial replay.

The scheduler is an explicit Python convention. Arm 0 is a persistent control;
new arms enter once in the supplied order, and outcomes are available before
allocation or monitoring updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import scalar
from .plbarpo_allocation import plbarpo_active_allocation
from .plbarpo_control import plbarpo_control_monitor
from .plbarpo_trial import (
    _MAX_ARMS,
    _MAX_PATIENTS,
    _MAX_TRIAL_WORK,
    _bool_vector,
    _categorical,
    _Config,
    _integer,
    _ledger_counts,
    _prepare,
    _readonly,
    _streams,
    _uniform_tape,
)


@dataclass(frozen=True)
class PLBarpoControlLook:
    enrolled: int
    active_before: NDArray[np.bool_]
    active_after: NDArray[np.bool_]
    futility_probability: NDArray[np.float64] | None
    efficacy_probability: NDArray[np.float64] | None
    final_probability: NDArray[np.float64] | None
    control_counts: NDArray[np.float64]
    absolute_error: NDArray[np.float64]
    window_start: NDArray[np.int64]
    window_end: NDArray[np.int64]
    futile: NDArray[np.bool_]
    efficacious: NDArray[np.bool_]
    final_efficacious: NDArray[np.bool_]


@dataclass(frozen=True)
class PLBarpoControlTrialResult:
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
    entry_index: NDArray[np.int64]
    stop_index: NDArray[np.int64]
    early_futility: NDArray[np.bool_]
    early_efficacy: NDArray[np.bool_]
    final_assessed: NDArray[np.bool_]
    final_efficacy: NDArray[np.bool_]
    final_efficacy_probability: NDArray[np.float64]
    final_control_counts: NDArray[np.float64]
    final_absolute_error: NDArray[np.float64]
    control_mode: str
    looks: tuple[PLBarpoControlLook, ...]
    enrolled: int
    max_total_n: int
    stop_reason: str
    work_units: int
    rng_seeds: tuple[int, int]


def _control_counts(
    mode: str,
    start: NDArray[np.int64],
    n: int,
    assignment: NDArray[np.int64],
    response: NDArray[np.int8],
    experimental: NDArray[np.int64],
    control_success: int,
    control_failure: int,
) -> NDArray[np.float64]:
    result = np.zeros((len(experimental), 2), dtype=float)
    if mode == "entire":
        result[:, 0] = control_success
        result[:, 1] = control_failure
        return result
    for row, arm in enumerate(experimental):
        window = (assignment[:n] == 0) & (np.arange(n) >= start[arm])
        result[row, 0] = np.count_nonzero(window & (response[:n] == 1))
        result[row, 1] = np.count_nonzero(window & (response[:n] == 0))
    return result


def _validate_control_mode(value: str) -> str:
    if value not in ("entire", "concurrent"):
        raise ValueError("control_mode must be 'entire' or 'concurrent'")
    return value


def run_plbarpo_control_trial(
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
    control_mode: str,
    method: str = "barcp",
    tau: float = 0.5,
    tau1: float = 1.0,
    target_probability: ArrayLike | None = None,
    minimum_probability: ArrayLike | None = None,
    early_monitoring: bool = True,
    pfut: float | None = None,
    peff: float | None = None,
    pfinal: float | None = None,
    rng: int | np.integer | np.random.Generator | None = None,
    assignment_uniforms: ArrayLike | None = None,
    outcome_uniforms: ArrayLike | None = None,
    absolute_tolerance: float = 1e-9,
    max_work: int = _MAX_TRIAL_WORK,
) -> PLBarpoControlTrialResult:
    """Replay a no-control-replacement PLBARPO trial with a persistent control.

    Arm 0 is the control and must be initially active, cannot enter the
    candidate queue, and must have ``max_n_per_arm[0] == max_total_n``. The
    initial active set defines the active-arm capacity. All entrants, including
    the control, participate in the equal-randomized burn-in; during burn-in,
    already satisfied arms wait while under-quota entrants are randomized.
    Floors apply only after burn-in and are not per-patient guarantees.

    In concurrent mode an experimental arm's control window is the half-open
    patient-index interval ``[entry_index, current_enrollment)``. The start is
    the zero-based index of its first eligible patient and the close is the
    number enrolled after its last eligible assignment. At a look this includes
    the control observations accrued through that look. Entire mode uses all
    control patients observed so far. Allocation ranking still uses the joint
    active-arm posterior; control-relative probabilities are used only for
    futility/efficacy monitoring. These are explicit Python policies, not a
    claim of native scheduler parity.
    """
    mode = _validate_control_mode(control_mode)
    truth = np.asarray(true_response)
    if truth.ndim != 1 or not 2 <= truth.size <= _MAX_ARMS:
        raise ValueError(f"true_response must describe 2..{_MAX_ARMS} ledger arms")
    arms = int(truth.size)
    initial = _bool_vector(initial_active, "initial_active", arms)
    if not initial[0] or int(initial.sum()) < 2:
        raise ValueError("the persistent control and at least one treatment must start active")
    candidates = np.asarray(candidate_order)
    if candidates.ndim != 1 or np.any(candidates == 0):
        raise ValueError("candidate_order must exclude persistent control arm 0")
    maximum = _integer(max_total_n, "max_total_n", 1, _MAX_PATIENTS)
    max_counts = _ledger_counts(max_n_per_arm, "max_n_per_arm", arms)
    if max_counts[0] != maximum:
        raise ValueError("control max_n_per_arm[0] must equal max_total_n")
    if not isinstance(early_monitoring, (bool, np.bool_)):
        raise ValueError("early_monitoring must be boolean")

    # Reuse the common ledger/allocation preflight while leaving comparison
    # thresholds to the control-relative monitor.
    config: _Config = _prepare(
        true_response,
        prior=prior,
        initial_active=initial_active,
        candidate_order=candidate_order,
        min_n_per_arm=min_n_per_arm,
        max_n_per_arm=max_n_per_arm,
        max_total_n=maximum,
        look_sizes=look_sizes,
        burn_in_per_arm=burn_in_per_arm,
        method=method,
        tau=tau,
        tau1=tau1,
        target_probability=target_probability,
        minimum_probability=minimum_probability,
        early_monitoring=False,
        theta_fut=None,
        pfut=None,
        theta_eff=None,
        peff=None,
        theta_final=None,
        pfinal=None,
        absolute_tolerance=absolute_tolerance,
        max_work=max_work,
    )
    if np.any(config.candidate_order == 0):
        raise ValueError("candidate_order must exclude persistent control arm 0")
    experimental_active = config.initial_active[1:]
    if not np.any(experimental_active):
        raise ValueError("at least one experimental arm must start active")
    # Validate monitor thresholds without performing a numerical comparison.
    for name, threshold in (("pfut", pfut), ("peff", peff), ("pfinal", pfinal)):
        if threshold is not None:
            parsed = scalar(threshold, name)
            if not 0 <= parsed <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
    assignment_tape = _uniform_tape(assignment_uniforms, maximum, "assignment_uniforms")
    outcome_tape = _uniform_tape(outcome_uniforms, maximum, "outcome_uniforms")
    assignment_rng, outcome_rng, seeds = _streams(rng)

    return _run_control_prepared(
        config,
        mode,
        bool(early_monitoring),
        pfut,
        peff,
        pfinal,
        assignment_tape,
        outcome_tape,
        assignment_rng,
        outcome_rng,
        seeds,
    )


def _run_control_prepared(
    config: _Config,
    mode: str,
    early_monitoring: bool,
    pfut: float | None,
    peff: float | None,
    pfinal: float | None,
    assignment_tape: NDArray[np.float64] | None,
    outcome_tape: NDArray[np.float64] | None,
    assignment_rng: np.random.Generator,
    outcome_rng: np.random.Generator,
    seeds: tuple[int, int],
) -> PLBarpoControlTrialResult:
    arms = len(config.truth)
    maximum = config.max_total_n
    active = np.array(config.initial_active, copy=True)
    entered = active.copy()
    entry_index = np.full(arms, -1, dtype=np.int64)
    entry_index[active] = 0
    stopped = np.zeros(arms, dtype=bool)
    cap_closed = np.zeros(arms, dtype=bool)
    stop_kind = ["not_entered"] * arms
    for arm in np.flatnonzero(active):
        stop_kind[arm] = "active"
    stop_index = np.full(arms, -1, dtype=np.int64)
    early_futility = np.zeros(arms, dtype=bool)
    early_efficacy = np.zeros(arms, dtype=bool)
    final_assessed = np.zeros(arms, dtype=bool)
    final_efficacy = np.zeros(arms, dtype=bool)
    final_probability = np.full(arms, np.nan)
    final_control_counts = np.full((arms, 2), np.nan)
    final_absolute_error = np.full(arms, np.nan)
    assigned = np.zeros(arms, dtype=np.int64)
    successes = np.zeros(arms, dtype=np.int64)
    failures = np.zeros(arms, dtype=np.int64)
    assignment = np.empty(maximum, dtype=np.int64)
    response = np.empty(maximum, dtype=np.int8)
    allocation_history = np.zeros((maximum, arms))
    allocation = np.zeros(arms)
    allocation_valid = False
    looks: list[PLBarpoControlLook] = []
    candidate_position = 0
    n = 0
    work = 0
    reason = "maximum_enrollment"
    look_set = set(int(x) for x in config.look_sizes)

    def current_weights() -> NDArray[np.float64]:
        nonlocal work
        work += int(active.sum()) ** 2
        if work > config.work_limit:
            raise RuntimeError("PLBARPO control trial allocation work exceeded max_work")
        result = plbarpo_active_allocation(
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
        return result.allocation_probability.copy()

    def fill_slots() -> None:
        nonlocal candidate_position, allocation_valid
        while active.sum() < config.active_capacity and candidate_position < len(
            config.candidate_order
        ):
            arm = int(config.candidate_order[candidate_position])
            candidate_position += 1
            if arm == 0 or entered[arm]:
                continue
            active[arm] = True
            entered[arm] = True
            entry_index[arm] = n
            stop_kind[arm] = "active"
            allocation.fill(0.0)
            allocation_valid = False

    if config.burn_in == 0:
        allocation = current_weights()
        allocation_valid = True

    while n < maximum and np.any(active[1:]):
        under_burn = active & (assigned < config.burn_in)
        if np.any(under_burn):
            allocation = np.zeros(arms)
            allocation[under_burn] = 1.0 / int(under_burn.sum())
            allocation_valid = False
        elif not allocation_valid:
            allocation = current_weights()
            allocation_valid = True
        uniform = (
            float(assignment_tape[n])
            if assignment_tape is not None
            else float(assignment_rng.random())
        )
        arm_index = int(_categorical(allocation, uniform))
        if not active[arm_index]:
            raise ArithmeticError("PLBARPO control assignment selected an inactive arm")
        assignment[n] = arm_index
        allocation_history[n] = allocation
        outcome_u = (
            float(outcome_tape[n]) if outcome_tape is not None else float(outcome_rng.random())
        )
        response[n] = int(outcome_u < config.truth[arm_index])
        assigned[arm_index] += 1
        if response[n]:
            successes[arm_index] += 1
        else:
            failures[arm_index] += 1
        n += 1

        cap_hit = arm_index != 0 and assigned[arm_index] >= config.max_n[arm_index]
        is_look = n in look_set
        if not is_look and not cap_hit:
            continue
        active_before = active.copy()
        exp_idx = np.flatnonzero(active[1:]) + 1
        fprob = eprob = finalprob = None
        futile = np.zeros(arms, dtype=bool)
        efficacious = np.zeros(arms, dtype=bool)
        final_flags = np.zeros(arms, dtype=bool)
        used_control = np.full((arms, 2), np.nan)
        look_error = np.full(arms, np.nan)
        is_final = n == maximum
        assess_final = is_final or (cap_hit and pfinal is not None)
        if exp_idx.size:
            ctrl = _control_counts(
                mode,
                entry_index,
                n,
                assignment,
                response,
                exp_idx,
                int(successes[0]),
                int(failures[0]),
            )
            used_control[exp_idx] = ctrl
            work += exp_idx.size**2
            if work > config.work_limit:
                raise RuntimeError("PLBARPO control trial monitoring work exceeded max_work")
            monitored = plbarpo_control_monitor(
                successes[exp_idx],
                failures[exp_idx],
                prior=config.prior[exp_idx],
                control_prior=config.prior[0],
                control_counts=ctrl[0] if mode == "entire" else ctrl,
                control_mode=mode,
                pfut=pfut if is_look and not is_final and early_monitoring else None,
                peff=peff if is_look and not is_final and early_monitoring else None,
                pfinal=pfinal if assess_final else None,
                absolute_tolerance=config.absolute_tolerance,
            )
            look_error[exp_idx] = monitored.absolute_error
            if monitored.futility_probability is not None:
                fprob = np.zeros(arms)
                fprob[exp_idx] = monitored.futility_probability
            if monitored.efficacy_probability is not None:
                eprob = np.zeros(arms)
                eprob[exp_idx] = monitored.efficacy_probability
            if monitored.final_efficacy_probability is not None:
                finalprob = np.full(arms, np.nan)
                final_ok = assigned[exp_idx] >= config.min_n[exp_idx]
                if cap_hit and not is_final:
                    final_ok &= exp_idx == arm_index
                selected = exp_idx[final_ok]
                final_assessed[selected] = True
                finalprob[selected] = monitored.final_efficacy_probability[final_ok]
                final_probability[selected] = monitored.final_efficacy_probability[final_ok]
                final_control_counts[selected] = ctrl[final_ok]
                final_absolute_error[selected] = monitored.absolute_error[final_ok]
                assert monitored.final_efficacious is not None
                final_flags[selected] = monitored.final_efficacious[final_ok]
                final_efficacy[selected] |= final_flags[selected]
            if is_look and not is_final and early_monitoring:
                eligible = assigned[exp_idx] >= config.min_n[exp_idx]
                if monitored.futile is not None:
                    futile[exp_idx[eligible]] = monitored.futile[eligible]
                if monitored.efficacious is not None:
                    efficacious[exp_idx[eligible]] = monitored.efficacious[eligible]
                if np.any(futile & (efficacious | final_flags)):
                    raise ValueError(
                        "an experimental arm cannot be declared futile and efficacious at one look"
                    )
                early_futility |= futile
                early_efficacy |= efficacious

        closing = futile | efficacious | final_flags
        for idx in np.flatnonzero(closing):
            stopped[idx] = True
            active[idx] = False
            stop_index[idx] = n
            if final_flags[idx]:
                stop_kind[idx] = "final_efficacy" if is_final else "final_efficacy_at_arm_cap"
            elif efficacious[idx]:
                stop_kind[idx] = "efficacy"
            else:
                stop_kind[idx] = "futility"
        if cap_hit:
            cap_closed[arm_index] = True
            stopped[arm_index] = True
            active[arm_index] = False
            stop_index[arm_index] = n
            if not final_efficacy[arm_index] and not closing[arm_index]:
                stop_kind[arm_index] = "arm_maximum"
            elif final_efficacy[arm_index] and not closing[arm_index]:
                stop_kind[arm_index] = "final_efficacy_at_arm_cap"
        if not is_final and active.sum() < config.active_capacity:
            fill_slots()
        terminal_no_experimental = (
            not is_final
            and not np.any(active[1:])
            and candidate_position >= len(config.candidate_order)
        )
        if is_final:
            remaining = active.copy()
            stopped[remaining] = True
            for idx in np.flatnonzero(remaining):
                stop_kind[idx] = "trial_maximum"
                stop_index[idx] = n
            active[remaining] = False
            reason = "maximum_enrollment"
            # Arm zero is only ended by trial termination, never by a rule.
            if active[0]:
                active[0] = False
                stopped[0] = True
                stop_kind[0] = "trial_maximum"
                stop_index[0] = n
        elif terminal_no_experimental:
            reason = "all_experimental_arms_closed_no_candidates"
            active[0] = False
            stopped[0] = True
            stop_kind[0] = "trial_ended"
            stop_index[0] = n
        after = active.copy()
        if is_look:
            window_start = np.full(arms, -1, dtype=np.int64)
            window_end = np.full(arms, -1, dtype=np.int64)
            window_start[exp_idx] = 0 if mode == "entire" else entry_index[exp_idx]
            window_end[exp_idx] = n
            looks.append(
                PLBarpoControlLook(
                    n,
                    _readonly(active_before, bool),
                    _readonly(after, bool),
                    None if fprob is None else _freeze(fprob),
                    None if eprob is None else _freeze(eprob),
                    None if finalprob is None else _freeze(finalprob),
                    _freeze(used_control),
                    _freeze(look_error),
                    _readonly(window_start, np.int64),
                    _readonly(window_end, np.int64),
                    _readonly(futile, bool),
                    _readonly(efficacious, bool),
                    _readonly(final_flags, bool),
                )
            )
        if is_final:
            break
        if terminal_no_experimental:
            break
        allocation.fill(0.0)
        allocation_valid = False
        if not np.any(active & (assigned < config.burn_in)) and np.any(active[1:]):
            allocation = current_weights()
            allocation_valid = True

    if n == maximum:
        reason = "maximum_enrollment"
    return PLBarpoControlTrialResult(
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
        tuple(stop_kind),
        _readonly(entry_index, np.int64),
        _readonly(stop_index, np.int64),
        _readonly(early_futility, bool),
        _readonly(early_efficacy, bool),
        _readonly(final_assessed, bool),
        _readonly(final_efficacy, bool),
        _freeze(final_probability),
        _freeze(final_control_counts),
        _freeze(final_absolute_error),
        mode,
        tuple(looks),
        n,
        maximum,
        reason,
        work,
        seeds,
    )
