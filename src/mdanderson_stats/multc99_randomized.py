"""Corrected Multc99 balanced randomized-arm replay and simulation.

Adapted workflow retains the upstream noncommercial source terms; see
notices/mdanderson-multc99-readme.txt. Legacy indexing, omitted-final-outcome,
conditional-denominator and biased-tie defects are corrected explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .multc99 import Multc99Design, _immutable, _integer


@dataclass(frozen=True)
class Multc99RandomizedTrial:
    arm_sample_sizes: NDArray[np.int64]
    elementary_counts: NDArray[np.int64]
    terminated_arms: NDArray[np.bool_]
    lower_hits: NDArray[np.bool_]
    upper_hits: NDArray[np.bool_]
    selected_arm: int | None
    target_posterior_means: FloatArray
    allocation_sequence: NDArray[np.int64]


@dataclass(frozen=True)
class Multc99RandomizedSimulation:
    design: Multc99Design
    arm_probabilities: FloatArray
    trials: int
    seed: int
    target_event: str
    maximize: bool
    reassign: bool
    arm_sample_sizes: NDArray[np.int64]
    elementary_counts: NDArray[np.int64]
    terminated_arms: NDArray[np.bool_]
    selected_arms: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    termination_probability: FloatArray


def _seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _target(design: Multc99Design, name: str, maximize: bool | None) -> tuple[int, bool]:
    names = tuple(e.name for e in design.events)
    if name not in names:
        raise ValueError("unknown target_event")
    index = names.index(name)
    if maximize is None:
        if design.events[index].event_type == "other":
            raise ValueError("target event of type other requires explicit maximize")
        maximize = design.events[index].event_type == "efficacy"
    if not isinstance(maximize, bool):
        raise ValueError("maximize must be bool")
    return index, maximize


def run_multc99_randomized_trial(
    design: Multc99Design,
    assignments: ArrayLike,
    arm_outcomes: ArrayLike,
    *,
    target_event: str,
    maximize: bool | None = None,
    reassign: bool = False,
    seed: int = 0,
) -> Multc99RandomizedTrial:
    """Replay a randomized potential tape through arm stops and final selection.

    design.max_subjects is the TOTAL assignment-slot budget. Monitoring occurs
    after each global cohort on the arm receiving that slot, using its own
    observed event counts. Without reassignment, later slots for a stopped arm
    are skipped. Reassignment balances and shuffles only its remaining slots.
    Selection uses correct conditional posterior means and uniform ties among
    surviving arms. The final slot's patient is fully observed, without an
    interim stop at the total cap. Arm indices are zero-based.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    if not isinstance(reassign, bool):
        raise ValueError("reassign must be bool")
    index, maximize = _target(design, target_event, maximize)
    raw = np.asarray(arm_outcomes)
    if (
        raw.ndim != 2
        or raw.shape[1] != design.max_subjects
        or not 2 <= raw.shape[0] <= 10
        or raw.dtype.kind not in "iu"
    ):
        raise ValueError("arm_outcomes must contain 2--10 full integer potential tapes")
    arms = raw.shape[0]
    if (
        arms > design.max_subjects
        or np.any(raw < 0)
        or np.any(raw >= design.experimental_prior.size)
    ):
        raise ValueError("arm outcomes must be valid categories, with at least one slot per arm")
    schedule = np.asarray(assignments)
    if (
        schedule.shape != (design.max_subjects,)
        or schedule.dtype.kind not in "iu"
        or np.any(schedule >= arms)
        or np.any(schedule < 0)
    ):
        raise ValueError("assignments must contain max_subjects valid integer arm indices")
    schedule = np.array(schedule, dtype=np.int64, copy=True)
    rng = np.random.default_rng(_seed(seed))
    counts = np.zeros((arms, design.experimental_prior.size), dtype=np.int64)
    sizes = np.zeros(arms, dtype=np.int64)
    stopped = np.zeros(arms, dtype=bool)
    lower = np.zeros((arms, len(design.events)), dtype=bool)
    upper = np.zeros_like(lower)
    allocations: list[int] = []
    for slot, arm_raw in enumerate(schedule):
        arm = int(arm_raw)
        if stopped[arm]:
            continue
        outcome = int(raw[arm, sizes[arm]])
        counts[arm, outcome] += 1
        sizes[arm] += 1
        allocations.append(arm)
        if slot + 1 < design.max_subjects and (slot + 1) % design.cohort_size == 0:
            state = design.monitor_counts(counts[arm])
            if state.decision == "stop":
                stopped[arm] = True
                lower[arm], upper[arm] = state.lower_hits, state.upper_hits
                surviving = np.flatnonzero(~stopped)
                if surviving.size == 0:
                    break
                if reassign:
                    remaining = np.flatnonzero(schedule[slot + 1 :] == arm) + slot + 1
                    replacements = surviving[np.arange(remaining.size) % surviving.size]
                    schedule[remaining] = rng.permutation(replacements)
    alpha, beta = design._experimental_beta[index]
    successes = counts @ design._numerators[index]
    observed = counts @ design._denominators[index]
    means = (alpha + successes) / (alpha + beta + observed)
    active = np.flatnonzero(~stopped)
    selected = None
    if active.size:
        best = np.max(means[active]) if maximize else np.min(means[active])
        selected = int(rng.choice(active[means[active] == best]))
    return Multc99RandomizedTrial(
        _immutable(sizes),
        _immutable(counts),
        _immutable(stopped),
        _immutable(lower),
        _immutable(upper),
        selected,
        _freeze(means),
        _immutable(np.array(allocations, dtype=np.int64)),
    )


def _balanced_assignments(arms: int, maximum: int, rng: np.random.Generator) -> NDArray[np.int64]:
    split = int(np.ceil(maximum / (2 * arms))) * arms
    schedule = np.arange(maximum, dtype=np.int64) % arms
    schedule[:split] = rng.permutation(schedule[:split])
    schedule[split:] = rng.permutation(schedule[split:])
    return schedule


def simulate_multc99_randomized(
    design: Multc99Design,
    arm_probabilities: ArrayLike,
    *,
    target_event: str,
    maximize: bool | None = None,
    reassign: bool = False,
    trials: int = 10_000,
    seed: int = 0,
) -> Multc99RandomizedSimulation:
    """Serial arm-selection OCs with the recovered two-stage balanced allocation.

    Output selection vectors start with None followed by arms 0,...,K-1.
    One total slot budget is used; there is no undisclosed per-arm cap or time
    scheduler. Native random streams and the legacy sequential coin-flip tie
    selection are not reproduced.
    """
    if not isinstance(design, Multc99Design):
        raise TypeError("design must be a Multc99Design")
    if not isinstance(reassign, bool):
        raise ValueError("reassign must be bool")
    _, maximize = _target(design, target_event, maximize)
    repetitions = _integer(trials, "trials", 1, 10_000)
    raw = np.asarray(arm_probabilities)
    if (
        raw.ndim != 2
        or not 2 <= raw.shape[0] <= 10
        or raw.shape[1] != design.experimental_prior.size
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError("arm_probabilities must have shape (2--10 arms, elementary categories)")
    arms = raw.shape[0]
    probabilities = np.asarray(raw, dtype=float)
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(probabilities < 0)
        or np.any(np.abs(probabilities.sum(axis=1) - 1) > 1e-12)
    ):
        raise ValueError("each arm probability vector must be nonnegative and sum to one")
    if arms > design.max_subjects:
        raise ValueError("total slot budget must be at least the arm count")
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    if (
        repetitions * design.max_subjects * (arms + probabilities.shape[1] + len(design.events))
        > 20_000_000
    ):
        raise ValueError("randomized simulation exceeds the 20,000,000-work budget")
    rng = np.random.default_rng(_seed(seed))
    sizes = np.empty((repetitions, arms), dtype=np.int64)
    counts = np.empty((repetitions, arms, probabilities.shape[1]), dtype=np.int64)
    stopped = np.empty((repetitions, arms), dtype=bool)
    selected = np.empty(repetitions, dtype=np.int64)
    for j in range(repetitions):
        schedule = _balanced_assignments(arms, design.max_subjects, rng)
        tapes = np.array([rng.choice(p.size, size=design.max_subjects, p=p) for p in probabilities])
        trial = run_multc99_randomized_trial(
            design,
            schedule,
            tapes,
            target_event=target_event,
            maximize=maximize,
            reassign=reassign,
            seed=int(rng.integers(0, 2**64, dtype=np.uint64)),
        )
        sizes[j], counts[j], stopped[j] = (
            trial.arm_sample_sizes,
            trial.elementary_counts,
            trial.terminated_arms,
        )
        selected[j] = -1 if trial.selected_arm is None else trial.selected_arm
    selection = np.bincount(selected + 1, minlength=arms + 1) / repetitions
    return Multc99RandomizedSimulation(
        design,
        _freeze(probabilities),
        repetitions,
        int(seed),
        target_event,
        maximize,
        reassign,
        _immutable(sizes),
        _immutable(counts),
        _immutable(stopped),
        _immutable(selected),
        _freeze(selection),
        _freeze(np.sqrt(selection * (1 - selection) / repetitions)),
        _freeze(np.mean(stopped, axis=0)),
    )
