"""Sequential completed-outcome BARPO trial conduct.

The guide specifies equal-randomization burn-in followed by adaptive
randomization and scheduled monitoring, but not block construction or conflict
priority. This module uses replayable balanced random-permutation blocks, holds
adaptive probabilities fixed within each cohort, and requires callers to choose
arm-level or whole-trial stopping. Outcomes are available before the next
allocation update; delayed outcomes are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .barpo import BarpoMonitoring, barpo_allocation, barpo_monitor, barpo_posterior

_MAX_ARMS = 10
_MAX_PATIENTS = 2_000
_MAX_TRIAL_WORK = 10_000_000


def _bool_vector(value: ArrayLike, name: str, size: int) -> NDArray[np.bool_]:
    result = np.asarray(value)
    if result.dtype.kind != "b" or result.shape != (size,):
        raise ValueError(f"{name} must be a boolean vector with one entry per arm")
    copied = np.array(result, dtype=bool, copy=True)
    copied.flags.writeable = False
    return copied


def _frozen_typed(value: ArrayLike, dtype: np.dtype | type) -> NDArray:
    copied = np.array(value, dtype=dtype, copy=True)
    copied.flags.writeable = False
    return copied


def _frozen_bool(value: ArrayLike) -> NDArray[np.bool_]:
    copied = np.array(value, dtype=bool, copy=True)
    copied.flags.writeable = False
    return copied


def _integer(value: object, name: str, low: int, high: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu f" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    number = float(raw)
    if not np.isfinite(number) or number != np.floor(number) or not low <= number <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(number)


def _uniform_tape(value: ArrayLike | None, n: int, name: str) -> FloatArray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.shape != (n,) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must have shape ({n},)")
    tape = finite(raw, name)
    if np.any((tape < 0.0) | (tape >= 1.0)):
        raise ValueError(f"{name} values must lie in [0,1)")
    return tape


def _generator(value: int | np.random.Generator | None, name: str) -> np.random.Generator:
    if isinstance(value, np.random.Generator):
        return value
    if value is None or (isinstance(value, (int, np.integer)) and not isinstance(value, bool)):
        return np.random.default_rng(value)
    raise ValueError(f"{name} must be an integer seed, Generator, or None")


def _categorical(probability: FloatArray, uniform: float) -> int:
    cumulative = np.cumsum(probability)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise ArithmeticError("BARPO randomization probabilities are invalid")
    return min(
        int(np.searchsorted(cumulative, uniform * cumulative[-1], side="right")),
        len(probability) - 1,
    )


@dataclass(frozen=True)
class BarpoTrialLook:
    """Compact posterior-monitoring record at one completed-outcome look."""

    patients: int
    successes: NDArray[np.int64]
    failures: NDArray[np.int64]
    futility_probability: FloatArray | None
    efficacy_probability: FloatArray | None
    final_probability: FloatArray | None
    futile: NDArray[np.bool_] | None
    efficacious: NDArray[np.bool_] | None
    final_efficacious: NDArray[np.bool_] | None


@dataclass(frozen=True)
class BarpoTrialResult:
    """One completed-outcome BARPO trial; arm identifiers are one-based."""

    assignments: NDArray[np.int64]
    responses: NDArray[np.int8]
    assignment_probability: FloatArray
    equal_randomization_probability: FloatArray
    assigned: NDArray[np.int64]
    successes: NDArray[np.int64]
    failures: NDArray[np.int64]
    early_futility: NDArray[np.bool_]
    early_efficacy: NDArray[np.bool_]
    final_efficacy: NDArray[np.bool_]
    stopped_arms: NDArray[np.bool_]
    arm_stop_patient: NDArray[np.int64]
    looks: tuple[BarpoTrialLook, ...]
    enrolled: int
    final_assessed: bool
    stop_reason: str


def _prepare(
    true_response: ArrayLike,
    *,
    prior: ArrayLike,
    max_n: int,
    burn_in: int,
    er_block_size: int,
    cohort_size: int,
    looks: ArrayLike,
    min_n: int,
    method: str,
    control: bool,
    early_monitoring: bool,
    theta_fut: float | None,
    pfut: float | None,
    theta_eff: float | None,
    peff: float | None,
    theta_final: float | None,
    pfinal: float | None,
    tau: float,
    tau1: float,
    target_probability: ArrayLike | None,
    minimum_probability: ArrayLike | None,
    absolute_tolerance: float,
    stopping_policy: str,
) -> dict[str, Any]:
    truth = finite(true_response, "true_response")
    if truth.ndim != 1 or not 1 <= truth.size <= _MAX_ARMS or np.any((truth < 0) | (truth > 1)):
        raise ValueError("true_response must contain 1..10 probabilities in [0,1]")
    arms = int(truth.size)
    maximum = _integer(max_n, "max_n", 1, _MAX_PATIENTS)
    burn = _integer(burn_in, "burn_in", 0, maximum)
    block = _integer(er_block_size, "er_block_size", max(1, arms), _MAX_PATIENTS)
    if block % arms:
        raise ValueError("er_block_size must be a multiple of the number of arms")
    cohort = _integer(cohort_size, "cohort_size", 1, maximum)
    minimum = _integer(min_n, "min_n", 1, maximum)
    if stopping_policy not in ("arm", "trial"):
        raise ValueError("stopping_policy must be 'arm' or 'trial'")
    if not isinstance(control, (bool, np.bool_)):
        raise ValueError("control must be boolean")
    if not isinstance(early_monitoring, (bool, np.bool_)):
        raise ValueError("early_monitoring must be boolean")
    if control and arms < 2:
        raise ValueError("control monitoring requires at least two arms")
    if method not in ("barcp", "barn2n", "barmtv", "dbcd"):
        raise ValueError("method must be barcp, barn2n, barmtv, or dbcd")
    if burn > 0 and burn < arms and burn < maximum:
        # A partial ER block is still a valid replayable random subset, but a
        # DBCD update would have a zero assigned proportion for some arms.
        if method == "dbcd":
            raise ValueError("dbcd requires burn_in to assign at least one patient per arm")
    if method == "dbcd" and maximum < arms:
        raise ValueError("dbcd requires enough patients to assign every arm")
    if method == "dbcd" and burn < maximum and burn < block:
        raise ValueError("dbcd requires a complete balanced ER block before adaptive allocation")
    look_counts = count(looks, "looks")
    if look_counts.ndim != 1 or np.any(look_counts < 1) or np.any(look_counts > maximum):
        raise ValueError("looks must be an increasing vector of enrollment counts in [1,max_n]")
    look_values = np.asarray(look_counts, dtype=np.int64)
    if np.any(np.diff(look_values) <= 0):
        raise ValueError("looks must be strictly increasing")
    if stopping_policy == "trial" and maximum > _MAX_PATIENTS:
        raise ValueError("trial work exceeds its hard limit")

    # A zero-count dry call validates all threshold combinations and obtains a
    # posterior for checking allocation settings before any random draws.
    zeros = np.zeros(arms, dtype=np.int64)
    monitoring = barpo_monitor(
        zeros,
        zeros,
        prior=prior,
        control=control,
        theta_fut=theta_fut,
        pfut=pfut,
        theta_eff=theta_eff,
        peff=peff,
        theta_final=theta_final,
        pfinal=pfinal,
        absolute_tolerance=absolute_tolerance,
    )
    tau_value, tau1_value = scalar(tau, "tau"), scalar(tau1, "tau1")
    if tau_value < 0 or tau1_value < 0:
        raise ValueError("tau and tau1 must be nonnegative")
    target = (
        None if target_probability is None else finite(target_probability, "target_probability")
    )
    floor = (
        None if minimum_probability is None else finite(minimum_probability, "minimum_probability")
    )
    dry_assigned = np.ones(arms, dtype=np.int64)
    if method == "barn2n":
        dry_assigned[:] = 0
    dry_probability = barpo_allocation(
        monitoring.posterior,
        dry_assigned,
        method=method,
        tau=tau_value,
        tau1=tau1_value,
        max_n=maximum if method == "barn2n" else None,
        target_probability=target,
        minimum_probability=floor,
    )
    tolerance = scalar(absolute_tolerance, "absolute_tolerance")
    if tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    work = maximum * arms
    if work > _MAX_TRIAL_WORK:
        raise ValueError("trial arm-patient work exceeds the hard limit")
    return {
        "truth": truth.copy(),
        "prior": np.asarray(prior, dtype=float).copy(),
        "arms": arms,
        "max_n": maximum,
        "burn_in": burn,
        "er_block_size": block,
        "cohort_size": cohort,
        "looks": tuple(int(x) for x in look_values),
        "min_n": minimum,
        "method": method,
        "control": bool(control),
        "early_monitoring": bool(early_monitoring),
        "theta_fut": theta_fut,
        "pfut": pfut,
        "theta_eff": theta_eff,
        "peff": peff,
        "theta_final": theta_final,
        "pfinal": pfinal,
        "tau": tau_value,
        "tau1": tau1_value,
        "target_probability": target,
        "minimum_probability": floor,
        "absolute_tolerance": tolerance,
        "stopping_policy": stopping_policy,
        "dry_probability": dry_probability,
    }


def _run_prepared(
    config: dict[str, Any],
    assignment_tape: FloatArray | None,
    outcome_tape: FloatArray | None,
    assignment_rng: np.random.Generator,
    outcome_rng: np.random.Generator,
) -> BarpoTrialResult:
    arms = int(config["arms"])
    maximum = int(config["max_n"])
    burn = int(config["burn_in"])
    block_size = int(config["er_block_size"])
    cohort_size = int(config["cohort_size"])
    truth = np.asarray(config["truth"], dtype=float)
    assigned = np.zeros(arms, dtype=np.int64)
    successes = np.zeros(arms, dtype=np.int64)
    failures = np.zeros(arms, dtype=np.int64)
    stopped = np.zeros(arms, dtype=bool)
    early_futility = np.zeros(arms, dtype=bool)
    early_efficacy = np.zeros(arms, dtype=bool)
    final_efficacy = np.zeros(arms, dtype=bool)
    arm_stop_patient = np.zeros(arms, dtype=np.int64)
    assignments = np.empty(maximum, dtype=np.int64)
    responses = np.empty(maximum, dtype=np.int8)
    assignment_probability = np.zeros((maximum, arms), dtype=float)
    equal_probability = np.zeros((maximum, arms), dtype=float)
    block_remaining = np.zeros(arms, dtype=np.int64)
    block_remaining_total = 0
    n = 0
    stop_reason = "maximum"
    look_records: list[BarpoTrialLook] = []
    look_set = set(config["looks"])
    final_assessed = False
    max_n_within = maximum

    def draw(tape: FloatArray | None, generator: np.random.Generator, index: int) -> float:
        return float(tape[index]) if tape is not None else float(generator.random())

    while n < maximum:
        boundary = max_n_within
        upcoming_looks = [x for x in config["looks"] if int(x) > n]
        if upcoming_looks:
            boundary = min(boundary, int(upcoming_looks[0]))
        if n < burn:
            boundary = min(boundary, burn)
        batch = min(cohort_size, boundary - n)
        if batch <= 0:
            raise ArithmeticError("BARPO scheduling failed to advance enrollment")

        if n >= burn:
            active_experiments = ~stopped[1:] if bool(config["control"]) else ~stopped
            if not np.any(active_experiments):
                stop_reason = "all_arms_closed"
                break
            if n == burn or n == 0:
                posterior = barpo_posterior(
                    successes,
                    failures,
                    prior=config["prior"],
                    absolute_tolerance=float(config["absolute_tolerance"]),
                )
            allocation = barpo_allocation(
                posterior,
                assigned,
                method=str(config["method"]),
                tau=float(config["tau"]),
                tau1=float(config["tau1"]),
                max_n=maximum if config["method"] == "barn2n" else None,
                target_probability=config["target_probability"],
                stopped=stopped,
                minimum_probability=(
                    None
                    if config["minimum_probability"] is None
                    else np.where(stopped, 0.0, config["minimum_probability"])
                ),
            )
        for _ in range(batch):
            if n < burn:
                if block_remaining_total == 0:
                    block_remaining[:] = block_size // arms
                    block_remaining_total = block_size
                probability = block_remaining / block_remaining_total
                equal_probability[n].fill(1.0 / arms)
                arm = _categorical(probability, draw(assignment_tape, assignment_rng, n))
                block_remaining[arm] -= 1
                block_remaining_total -= 1
                assignment_probability[n] = probability
            else:
                probability = allocation
                arm = _categorical(probability, draw(assignment_tape, assignment_rng, n))
                assignment_probability[n] = probability
            response = int(draw(outcome_tape, outcome_rng, n) < truth[arm])
            assignments[n] = arm + 1
            responses[n] = response
            assigned[arm] += 1
            if response:
                successes[arm] += 1
            else:
                failures[arm] += 1
            n += 1

        if n in look_set and n < maximum and n >= int(config["min_n"]) and n >= burn:
            monitoring = barpo_monitor(
                successes,
                failures,
                prior=config["prior"],
                control=bool(config["control"]),
                theta_fut=config["theta_fut"],
                pfut=config["pfut"],
                theta_eff=config["theta_eff"],
                peff=config["peff"],
                theta_final=config["theta_final"],
                pfinal=config["pfinal"],
                absolute_tolerance=float(config["absolute_tolerance"]),
            )
            fut = (
                np.zeros(arms, dtype=bool)
                if not bool(config["early_monitoring"]) or monitoring.futile is None
                else monitoring.futile.copy()
            )
            eff = (
                np.zeros(arms, dtype=bool)
                if not bool(config["early_monitoring"]) or monitoring.efficacious is None
                else monitoring.efficacious.copy()
            )
            eligible = ~stopped
            if bool(config["control"]):
                eligible[0] = False
            fut &= eligible
            eff &= eligible
            if np.any(fut & eff):
                raise ValueError(
                    "an arm cannot be declared futile and efficacious at the same look"
                )
            new_terminal = fut | eff
            early_futility |= fut
            early_efficacy |= eff
            look_records.append(
                _look_record(
                    n,
                    successes,
                    failures,
                    monitoring,
                    final=False,
                    early_flags=(fut, eff),
                )
            )
            if np.any(new_terminal):
                stopped |= new_terminal
                arm_stop_patient[new_terminal & (arm_stop_patient == 0)] = n
                if config["stopping_policy"] == "trial":
                    stop_reason = (
                        "trial_stop_futility"
                        if np.any(fut) and not np.any(eff)
                        else (
                            "trial_stop_efficacy"
                            if np.any(eff) and not np.any(fut)
                            else "trial_stop_mixed"
                        )
                    )
                    break
                active_experiments = ~stopped[1:] if bool(config["control"]) else ~stopped
                if not np.any(active_experiments):
                    stop_reason = "all_arms_closed"
                    break
            posterior = monitoring.posterior
        else:
            posterior = barpo_posterior(
                successes,
                failures,
                prior=config["prior"],
                absolute_tolerance=float(config["absolute_tolerance"]),
            )

    else:
        n = maximum

    if n == maximum:
        final_assessed = config["theta_final"] is not None or config["pfinal"] is not None
        if final_assessed:
            final_monitoring = barpo_monitor(
                successes,
                failures,
                prior=config["prior"],
                control=bool(config["control"]),
                theta_fut=config["theta_fut"],
                pfut=config["pfut"],
                theta_eff=config["theta_eff"],
                peff=config["peff"],
                theta_final=config["theta_final"],
                pfinal=config["pfinal"],
                absolute_tolerance=float(config["absolute_tolerance"]),
            )
            final_flags = (
                np.zeros(arms, dtype=bool)
                if final_monitoring.final_efficacious is None
                else final_monitoring.final_efficacious.copy()
            )
            final_flags[stopped] = False
            final_efficacy = final_flags
            look_records.append(
                _look_record(
                    n,
                    successes,
                    failures,
                    final_monitoring,
                    final=True,
                    final_flags=final_flags,
                )
            )
        stop_reason = "maximum"
    return BarpoTrialResult(
        _frozen_typed(assignments[:n], np.int64),
        _frozen_typed(responses[:n], np.int8),
        _freeze(assignment_probability[:n]),
        _freeze(equal_probability[:n]),
        _frozen_typed(assigned, np.int64),
        _frozen_typed(successes, np.int64),
        _frozen_typed(failures, np.int64),
        _frozen_bool(early_futility),
        _frozen_bool(early_efficacy),
        _frozen_bool(final_efficacy),
        _frozen_bool(stopped),
        _frozen_typed(arm_stop_patient, np.int64),
        tuple(look_records),
        n,
        final_assessed,
        stop_reason,
    )


def _look_record(
    n: int,
    successes: NDArray[np.int64],
    failures: NDArray[np.int64],
    monitoring: BarpoMonitoring,
    *,
    final: bool,
    final_flags: NDArray[np.bool_] | None = None,
    early_flags: tuple[NDArray[np.bool_], NDArray[np.bool_]] | None = None,
) -> BarpoTrialLook:
    return BarpoTrialLook(
        n,
        _frozen_typed(successes, np.int64),
        _frozen_typed(failures, np.int64),
        monitoring.futility_probability,
        monitoring.efficacy_probability,
        monitoring.final_efficacy_probability,
        (
            _frozen_bool(early_flags[0])
            if early_flags is not None
            else (None if final else monitoring.futile)
        ),
        (
            _frozen_bool(early_flags[1])
            if early_flags is not None
            else (None if final else monitoring.efficacious)
        ),
        _frozen_bool(final_flags)
        if final_flags is not None
        else (monitoring.final_efficacious if final else None),
    )


def run_barpo_trial(
    true_response: ArrayLike,
    *,
    prior: ArrayLike,
    max_n: int,
    burn_in: int,
    er_block_size: int,
    cohort_size: int,
    looks: ArrayLike = (),
    min_n: int = 1,
    method: str = "barcp",
    control: bool = False,
    early_monitoring: bool = True,
    theta_fut: float | None = None,
    pfut: float | None = None,
    theta_eff: float | None = None,
    peff: float | None = None,
    theta_final: float | None = None,
    pfinal: float | None = None,
    tau: float = 0.5,
    tau1: float = 1.0,
    target_probability: ArrayLike | None = None,
    minimum_probability: ArrayLike | None = None,
    stopping_policy: Literal["arm", "trial"] = "arm",
    assignment_uniforms: ArrayLike | None = None,
    outcome_uniforms: ArrayLike | None = None,
    assignment_rng: int | np.random.Generator | None = None,
    outcome_rng: int | np.random.Generator | None = None,
    absolute_tolerance: float = 1e-9,
) -> BarpoTrialResult:
    """Run one adaptive BARPO trial with complete outcomes at cohort updates.

    Equal-randomization blocks are balanced and sampled without replacement;
    ``assignment_probability`` records the conditional probability vector used
    at each patient. ``equal_randomization_probability`` records the block's
    unconditional 1/K marginal. Interim looks are only actionable after both
    ``burn_in`` and ``min_n``. At ``max_n`` only final efficacy is evaluated.
    """
    config = _prepare(
        true_response,
        prior=prior,
        max_n=max_n,
        burn_in=burn_in,
        er_block_size=er_block_size,
        cohort_size=cohort_size,
        looks=looks,
        min_n=min_n,
        method=method,
        control=control,
        early_monitoring=early_monitoring,
        theta_fut=theta_fut,
        pfut=pfut,
        theta_eff=theta_eff,
        peff=peff,
        theta_final=theta_final,
        pfinal=pfinal,
        tau=tau,
        tau1=tau1,
        target_probability=target_probability,
        minimum_probability=minimum_probability,
        absolute_tolerance=absolute_tolerance,
        stopping_policy=stopping_policy,
    )
    maximum = int(config["max_n"])
    assignments = _uniform_tape(assignment_uniforms, maximum, "assignment_uniforms")
    outcomes = _uniform_tape(outcome_uniforms, maximum, "outcome_uniforms")
    if assignments is not None and assignment_rng is not None:
        raise ValueError("supply assignment_uniforms or assignment_rng, not both")
    if outcomes is not None and outcome_rng is not None:
        raise ValueError("supply outcome_uniforms or outcome_rng, not both")
    if (
        isinstance(assignment_rng, (int, np.integer))
        and not isinstance(assignment_rng, (bool, np.bool_))
        and isinstance(outcome_rng, (int, np.integer))
        and not isinstance(outcome_rng, (bool, np.bool_))
        and int(assignment_rng) == int(outcome_rng)
    ):
        raise ValueError("assignment_rng and outcome_rng seeds must differ")
    arng = _generator(assignment_rng, "assignment_rng")
    orng = _generator(outcome_rng, "outcome_rng")
    if arng.bit_generator is orng.bit_generator:
        raise ValueError("assignment_rng and outcome_rng must be independent generators")
    return _run_prepared(config, assignments, outcomes, arng, orng)
