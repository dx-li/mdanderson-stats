"""Bounded serial operating-characteristic simulation for BARPO trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .barpo_trial import _generator, _prepare, _run_prepared

_MAX_TRIALS = 10_000
_MAX_RESULT_CELLS = 2_000_000
_MAX_TAPE_CELLS = 2_000_000
_MAX_SIMULATION_WORK = 50_000_000


def _owned(value: ArrayLike, dtype: np.dtype | type | None = None) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _integer(value: object, name: str, low: int, high: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    number = float(raw)
    if not np.isfinite(number) or number != np.floor(number) or not low <= number <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(number)


def _tape(value: ArrayLike | None, shape: tuple[int, int], name: str) -> FloatArray | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.shape != shape or np.iscomplexobj(raw):
        raise ValueError(f"{name} must have shape {shape}")
    result = finite(raw, name)
    if np.any((result < 0.0) | (result >= 1.0)):
        raise ValueError(f"{name} must contain values in [0,1)")
    return result


def _stream_pair(
    rng: int | np.random.Generator | np.random.SeedSequence | None,
    assignment_rng: int | np.random.Generator | None,
    outcome_rng: int | np.random.Generator | None,
) -> tuple[np.random.Generator, np.random.Generator]:
    if rng is not None and (assignment_rng is not None or outcome_rng is not None):
        raise ValueError("use rng or the separate assignment_rng/outcome_rng pair, not both")
    if rng is None and assignment_rng is None and outcome_rng is None:
        children = np.random.SeedSequence().spawn(2)
        return np.random.default_rng(children[0]), np.random.default_rng(children[1])
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            seeds = rng.integers(0, 2**63, size=2, dtype=np.int64)
            return np.random.default_rng(int(seeds[0])), np.random.default_rng(int(seeds[1]))
        if isinstance(rng, np.random.SeedSequence):
            children = rng.spawn(2)
        elif isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
            children = np.random.SeedSequence(int(rng)).spawn(2)
        else:
            raise ValueError("rng must be a seed, SeedSequence, Generator, or None")
        return np.random.default_rng(children[0]), np.random.default_rng(children[1])
    if assignment_rng is None:
        arng = np.random.default_rng()
    else:
        arng = _generator(assignment_rng, "assignment_rng")
    if outcome_rng is None:
        orng = np.random.default_rng()
    else:
        orng = _generator(outcome_rng, "outcome_rng")
    if arng.bit_generator is orng.bit_generator:
        raise ValueError("assignment_rng and outcome_rng must be independent generators")
    if (
        isinstance(assignment_rng, (int, np.integer))
        and isinstance(outcome_rng, (int, np.integer))
        and int(assignment_rng) == int(outcome_rng)
    ):
        raise ValueError("assignment_rng and outcome_rng seeds must differ")
    return arng, orng


def _rate(values: NDArray[np.bool_], trials: int) -> tuple[FloatArray, FloatArray]:
    probability = values.mean(axis=0, dtype=float)
    mcse = np.sqrt(probability * (1.0 - probability) / trials)
    return _freeze(probability), _freeze(mcse)


@dataclass(frozen=True)
class BarpoSimulation:
    """Compact replicated BARPO outcomes and operating characteristics.

    Selection/declaration probabilities use all simulated trials as the
    denominator. ``allocation_probability_by_enrollment`` averages the
    conditional randomization vectors used at each patient index; rows after
    trial stopping or non-enrollment are zero. During a balanced ER block this
    is the conditional remaining-block probability, while
    ``equal_randomization_probability`` in a single-trial result records the
    unconditional 1/K marginal.
    """

    patients_by_arm: NDArray[np.int64]
    successes_by_arm: NDArray[np.int64]
    failures_by_arm: NDArray[np.int64]
    enrolled: NDArray[np.int64]
    early_futility: NDArray[np.bool_]
    early_efficacy: NDArray[np.bool_]
    final_efficacy: NDArray[np.bool_]
    stop_reason: tuple[str, ...]
    early_futility_probability: FloatArray
    early_futility_mcse: FloatArray
    early_efficacy_probability: FloatArray
    early_efficacy_mcse: FloatArray
    final_efficacy_probability: FloatArray
    final_efficacy_mcse: FloatArray
    cumulative_efficacy_probability: FloatArray
    cumulative_efficacy_mcse: FloatArray
    any_early_futility_probability: float
    any_early_futility_mcse: float
    any_early_efficacy_probability: float
    any_early_efficacy_mcse: float
    any_final_efficacy_probability: float
    any_final_efficacy_mcse: float
    any_cumulative_efficacy_probability: float
    any_cumulative_efficacy_mcse: float
    false_early_efficacy_probability: float | None
    false_early_efficacy_mcse: float | None
    false_final_efficacy_probability: float | None
    false_final_efficacy_mcse: float | None
    false_cumulative_efficacy_probability: float | None
    false_cumulative_efficacy_mcse: float | None
    allocation_probability_by_enrollment: FloatArray
    mean_patients_by_arm: FloatArray
    mean_successes_by_arm: FloatArray
    mean_enrolled: float
    enrolled_mcse: float
    early_stop_probability: float
    null_arms: NDArray[np.bool_] | None
    trials: int
    max_n: int
    method: str


def simulate_barpo(
    true_response: ArrayLike,
    *,
    prior: ArrayLike,
    max_n: int,
    trials: int = 1_000,
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
    null_arms: ArrayLike | None = None,
    rng: int | np.random.Generator | np.random.SeedSequence | None = None,
    assignment_rng: int | np.random.Generator | None = None,
    outcome_rng: int | np.random.Generator | None = None,
    assignment_uniforms: ArrayLike | None = None,
    outcome_uniforms: ArrayLike | None = None,
    absolute_tolerance: float = 1e-9,
    max_work: int = 50_000_000,
) -> BarpoSimulation:
    """Run serial independent BARPO trials and summarize operating characteristics.

    ``rng`` spawns independent assignment and outcome streams. Alternatively,
    supply the two named streams or replay both ``(trials,max_n)`` uniform tapes.
    Outcomes are immediately available before each subsequent allocation update.
    A ``null_arms`` mask is required to interpret false efficacy declarations;
    without it the result reports only declaration probabilities.
    """
    n_trials = _integer(trials, "trials", 1, _MAX_TRIALS)
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
    arms = int(config["arms"])
    maximum = int(config["max_n"])
    if n_trials * arms > _MAX_RESULT_CELLS or maximum * arms > _MAX_RESULT_CELLS:
        raise ValueError("BARPO simulation result arrays exceed the cell limit")
    estimated_work = n_trials * maximum * arms * arms
    work_limit = _integer(max_work, "max_work", 1, _MAX_SIMULATION_WORK)
    if estimated_work > work_limit:
        raise ValueError("BARPO simulation work bound exceeds max_work")
    if null_arms is None:
        null_mask = None
    else:
        null_mask = np.asarray(null_arms)
        if null_mask.dtype.kind != "b" or null_mask.shape != (arms,):
            raise ValueError("null_arms must be a boolean mask with one entry per arm")
        null_mask = np.array(null_mask, dtype=bool, copy=True)
        null_mask.flags.writeable = False

    tape_cells = n_trials * maximum
    if tape_cells > _MAX_TAPE_CELLS and (
        assignment_uniforms is not None or outcome_uniforms is not None
    ):
        raise ValueError("supplied uniform tapes exceed the two-million-cell limit")
    allocation_tape = _tape(assignment_uniforms, (n_trials, maximum), "assignment_uniforms")
    outcome_tape = _tape(outcome_uniforms, (n_trials, maximum), "outcome_uniforms")
    arng, orng = _stream_pair(rng, assignment_rng, outcome_rng)

    patients = np.zeros((n_trials, arms), dtype=np.int64)
    successes = np.zeros_like(patients)
    failures = np.zeros_like(patients)
    enrolled = np.zeros(n_trials, dtype=np.int64)
    early_futility = np.zeros((n_trials, arms), dtype=bool)
    early_efficacy = np.zeros_like(early_futility)
    final_efficacy = np.zeros_like(early_futility)
    allocation_sum = np.zeros((maximum, arms), dtype=float)
    reasons: list[str] = []
    for trial in range(n_trials):
        result = _run_prepared(
            config,
            None if allocation_tape is None else allocation_tape[trial],
            None if outcome_tape is None else outcome_tape[trial],
            arng,
            orng,
        )
        patients[trial] = result.assigned
        successes[trial] = result.successes
        failures[trial] = result.failures
        enrolled[trial] = result.enrolled
        early_futility[trial] = result.early_futility
        early_efficacy[trial] = result.early_efficacy
        final_efficacy[trial] = result.final_efficacy
        allocation_sum[: result.enrolled] += result.assignment_probability
        reasons.append(result.stop_reason)

    early_fut_prob, early_fut_mcse = _rate(early_futility, n_trials)
    early_eff_prob, early_eff_mcse = _rate(early_efficacy, n_trials)
    final_eff_prob, final_eff_mcse = _rate(final_efficacy, n_trials)
    any_fut = np.any(early_futility, axis=1)
    any_early_eff = np.any(early_efficacy, axis=1)
    any_final_eff = np.any(final_efficacy, axis=1)
    cumulative_efficacy = early_efficacy | final_efficacy
    cumulative_eff_prob, cumulative_eff_mcse = _rate(cumulative_efficacy, n_trials)
    any_cumulative_eff = np.any(cumulative_efficacy, axis=1)
    any_fut_prob, any_fut_mcse = _rate(any_fut[:, None], n_trials)
    any_early_eff_prob, any_early_eff_mcse = _rate(any_early_eff[:, None], n_trials)
    any_final_eff_prob, any_final_eff_mcse = _rate(any_final_eff[:, None], n_trials)
    any_cumulative_eff_prob, any_cumulative_eff_mcse = _rate(any_cumulative_eff[:, None], n_trials)
    if null_mask is None:
        false_early_prob = false_early_mcse = false_final_prob = false_final_mcse = None
        false_cumulative_prob = false_cumulative_mcse = None
    else:
        false_early = np.any(early_efficacy[:, null_mask], axis=1)
        false_final = np.any(final_efficacy[:, null_mask], axis=1)
        false_cumulative = np.any(cumulative_efficacy[:, null_mask], axis=1)
        a, b = _rate(false_early[:, None], n_trials)
        false_early_prob, false_early_mcse = float(a[0]), float(b[0])
        a, b = _rate(false_final[:, None], n_trials)
        false_final_prob, false_final_mcse = float(a[0]), float(b[0])
        a, b = _rate(false_cumulative[:, None], n_trials)
        false_cumulative_prob, false_cumulative_mcse = float(a[0]), float(b[0])
    mean_enrolled = float(enrolled.mean())
    if n_trials == 1:
        enrolled_mcse = 0.0
    else:
        enrolled_mcse = float(enrolled.std(ddof=1) / np.sqrt(n_trials))
    return BarpoSimulation(
        _owned(patients, np.int64),
        _owned(successes, np.int64),
        _owned(failures, np.int64),
        _owned(enrolled, np.int64),
        _owned(early_futility, bool),
        _owned(early_efficacy, bool),
        _owned(final_efficacy, bool),
        tuple(reasons),
        early_fut_prob,
        early_fut_mcse,
        early_eff_prob,
        early_eff_mcse,
        final_eff_prob,
        final_eff_mcse,
        cumulative_eff_prob,
        cumulative_eff_mcse,
        float(any_fut_prob[0]),
        float(any_fut_mcse[0]),
        float(any_early_eff_prob[0]),
        float(any_early_eff_mcse[0]),
        float(any_final_eff_prob[0]),
        float(any_final_eff_mcse[0]),
        float(any_cumulative_eff_prob[0]),
        float(any_cumulative_eff_mcse[0]),
        false_early_prob,
        false_early_mcse,
        false_final_prob,
        false_final_mcse,
        false_cumulative_prob,
        false_cumulative_mcse,
        _freeze(allocation_sum / n_trials),
        _freeze(patients.mean(axis=0)),
        _freeze(successes.mean(axis=0)),
        mean_enrolled,
        enrolled_mcse,
        float(np.mean(enrolled < maximum)),
        None if null_mask is None else _owned(null_mask, bool),
        n_trials,
        maximum,
        str(config["method"]),
    )
