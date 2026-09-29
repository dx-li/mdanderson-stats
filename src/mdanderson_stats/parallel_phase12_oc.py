"""Bounded serial operating-characteristic summaries for the four-arm C design."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .parallel_phase12 import simulate_parallel_phase12

_MAX_OC_TRIALS = 10_000
_MAX_PATIENT_WORK = 1_000_000
_STOP_REASONS = ("efficacy", "futility", "no admissible arms", "maximum sample size")


def _bernoulli_mcse(probability: FloatArray, count: int) -> FloatArray:
    if count < 2:
        return np.full(probability.shape, np.nan)
    return np.sqrt(probability * (1 - probability) / count)


def _freeze_int(value: ArrayLike, *, dtype: np.dtype = np.dtype(np.int64)) -> NDArray:
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _mean_mcse(total: FloatArray, total_squared: FloatArray, count: int) -> FloatArray:
    if count < 2:
        return np.full(total.shape, np.nan)
    variance = (total_squared - total * total / count) / (count - 1)
    return np.sqrt(np.maximum(variance, 0.0) / count)


def _ratio_mcse(
    events: FloatArray,
    exposures: FloatArray,
    event_squares: FloatArray,
    exposure_squares: FloatArray,
    cross_products: FloatArray,
    count: int,
) -> tuple[FloatArray, FloatArray]:
    totals = np.sum(exposures)
    if totals == 0:
        undefined = np.full(exposures.shape, np.nan)
        return undefined, undefined.copy()
    rates = np.divide(events, exposures, out=np.full(events.shape, np.nan), where=exposures > 0)
    if count < 2:
        return rates, np.full(rates.shape, np.nan)
    numerator = event_squares - 2 * rates * cross_products + rates**2 * exposure_squares
    denominator = exposures**2
    mcse = np.sqrt(np.maximum(numerator, 0.0) * count / ((count - 1) * denominator))
    mcse = np.where(exposures > 0, mcse, np.nan)
    return rates, mcse


@dataclass(frozen=True)
class ParallelPhase12OC:
    """Aggregate trial outcomes; no patient histories are retained.

    Arm arrays follow native zero-based order. Selection arrays contain one
    entry per arm; no-selection probability is reported separately. ``*_total``
    event/exposure fields are across simulated trials. Event rates are pooled
    across enrolled patients, with MCSE calculated by treating each trial as
    the independent sampling unit. One-trial MCSEs are undefined (NaN); rates
    for never-enrolled arms are undefined (NaN).
    """

    n_trials: int
    per_trial_seeds: NDArray[np.uint64]
    selected_counts: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_count: int
    no_selection_probability: float
    no_selection_mcse: float
    stopping_reasons: tuple[str, ...]
    stopping_counts: NDArray[np.int64]
    stopping_probability: FloatArray
    stopping_mcse: FloatArray
    admissible_counts: NDArray[np.int64]
    admissibility_probability: FloatArray
    admissibility_mcse: FloatArray
    treated_total: NDArray[np.int64]
    mean_treated: FloatArray
    mcse_treated: FloatArray
    toxicity_total: NDArray[np.int64]
    mean_toxicities: FloatArray
    mcse_toxicities: FloatArray
    response_total: NDArray[np.int64]
    mean_responses: FloatArray
    mcse_responses: FloatArray
    toxicity_rate: FloatArray
    toxicity_rate_mcse: FloatArray
    response_rate: FloatArray
    response_rate_mcse: FloatArray
    mean_total_enrollment: float
    mcse_total_enrollment: float
    total_enrollment: int
    mean_phase_one_enrollment: float
    mcse_phase_one_enrollment: float
    phase_one_enrollment_total: int
    optimal_arms: tuple[int, ...] | None
    optimal_selection_probability: float | None
    optimal_selection_mcse: float | None


def simulate_parallel_phase12_oc(
    toxicity_probability: ArrayLike,
    efficacy_probability: ArrayLike,
    *,
    n_trials: int,
    seed: int,
    optimal_arms: Iterable[int] | None = None,
) -> ParallelPhase12OC:
    """Run bounded serial trials and summarize source-defined operating characteristics.

    A `SeedSequence(seed, spawn_key=(trial_index,))` supplies one uint64 seed
    per trial. Reproduce trial ``i`` by calling ``simulate_parallel_phase12``
    with ``rng=np.random.default_rng(result.per_trial_seeds[i])``. No histories
    or per-patient outcomes are retained by this aggregate wrapper.

    ``optimal_arms`` is an optional, user-specified set of arms. If supplied,
    its probability is only the event that the design selected one of those
    arms; no source-defined optimal-arm success rule is inferred.
    """
    if isinstance(n_trials, (bool, np.bool_)) or not isinstance(n_trials, (int, np.integer)):
        raise ValueError("n_trials must be an integer")
    count = int(n_trials)
    if not 1 <= count <= _MAX_OC_TRIALS:
        raise ValueError(f"n_trials must be in [1,{_MAX_OC_TRIALS}]")
    if count * 100 > _MAX_PATIENT_WORK:
        raise ValueError("operating-characteristic workload exceeds the bounded patient-work limit")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    toxicity = finite(toxicity_probability, "toxicity_probability")
    efficacy = finite(efficacy_probability, "efficacy_probability")
    if (
        toxicity.shape != (4,)
        or efficacy.shape != (4,)
        or np.any((toxicity < 0) | (toxicity > 1))
        or np.any((efficacy < 0) | (efficacy > 1))
    ):
        raise ValueError("toxicity and efficacy probabilities must be four-vectors in [0,1]")

    selected_arms: tuple[int, ...] | None = None
    if optimal_arms is not None:
        try:
            raw_optimal = np.asarray(tuple(optimal_arms))
        except TypeError as exc:
            raise ValueError(
                "optimal_arms must be an iterable set of integer indices 0..3"
            ) from exc
        if (
            raw_optimal.ndim != 1
            or raw_optimal.size == 0
            or raw_optimal.dtype.kind not in "iu"
            or raw_optimal.dtype.kind == "b"
            or np.any((raw_optimal < 0) | (raw_optimal > 3))
            or np.unique(raw_optimal).size != raw_optimal.size
        ):
            raise ValueError("optimal_arms must be a nonempty set of unique integer indices 0..3")
        selected_arms = tuple(sorted(int(arm) for arm in raw_optimal))

    seeds = np.empty(count, dtype=np.uint64)
    selected_counts = np.zeros(4, dtype=np.int64)
    stopping_counts = np.zeros(len(_STOP_REASONS), dtype=np.int64)
    admissible_counts = np.zeros(4, dtype=np.int64)
    totals = np.zeros((6, 4), dtype=float)
    squared_totals = np.zeros_like(totals)
    tox_squares = np.zeros(4, dtype=float)
    response_squares = np.zeros(4, dtype=float)
    tox_exposure_squares = np.zeros(4, dtype=float)
    response_exposure_squares = np.zeros(4, dtype=float)
    tox_cross_products = np.zeros(4, dtype=float)
    response_cross_products = np.zeros(4, dtype=float)
    no_selection_count = 0
    optimal_selection_count = 0

    for index in range(count):
        trial_seed = int(
            np.random.SeedSequence(int(seed), spawn_key=(index,)).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds[index] = trial_seed
        result = simulate_parallel_phase12(
            toxicity, efficacy, rng=np.random.default_rng(trial_seed)
        )
        if result.selected is None:
            no_selection_count += 1
        else:
            selected_counts[result.selected] += 1
            if selected_arms is not None and result.selected in selected_arms:
                optimal_selection_count += 1
        stopping_counts[_STOP_REASONS.index(result.reason)] += 1
        admissible_counts += np.asarray(result.admissible, dtype=np.int64)

        treated = np.asarray(result.treated)
        toxicities = np.asarray(result.toxicities)
        responses = np.asarray(result.responses)
        total_enrollment = float(np.sum(treated))
        phase_one = (
            total_enrollment
            if result.phase_one_enrollment is None
            else float(result.phase_one_enrollment)
        )
        observations = np.vstack(
            (treated, toxicities, responses, [total_enrollment] * 4, [phase_one] * 4, [1.0] * 4)
        )
        totals += observations
        squared_totals += observations**2
        tox_squares += toxicities**2
        response_squares += responses**2
        tox_exposure_squares += treated**2
        response_exposure_squares += treated**2
        tox_cross_products += toxicities * treated
        response_cross_products += responses * treated

    selection_probability = selected_counts / count
    selection_mcse = _bernoulli_mcse(selection_probability, count)
    no_probability = no_selection_count / count
    no_mcse = float(_bernoulli_mcse(np.asarray(no_probability), count))
    stopping_probability = stopping_counts / count
    stopping_mcse = _bernoulli_mcse(stopping_probability, count)
    admissibility_probability = admissible_counts / count
    admissibility_mcse = _bernoulli_mcse(admissibility_probability, count)
    mean_treated = totals[0] / count
    mean_toxicities = totals[1] / count
    mean_responses = totals[2] / count
    mcse_treated = _mean_mcse(totals[0], squared_totals[0], count)
    mcse_toxicities = _mean_mcse(totals[1], squared_totals[1], count)
    mcse_responses = _mean_mcse(totals[2], squared_totals[2], count)
    total_enrollments = totals[3, 0]
    total_phase_one = totals[4, 0]
    mean_total = total_enrollments / count
    mean_phase = total_phase_one / count
    total_enroll_mcse = float(_mean_mcse(totals[3, :1], squared_totals[3, :1], count)[0])
    phase_one_mcse = float(_mean_mcse(totals[4, :1], squared_totals[4, :1], count)[0])
    toxicity_rate, toxicity_rate_mcse = _ratio_mcse(
        totals[1], totals[0], tox_squares, tox_exposure_squares, tox_cross_products, count
    )
    response_rate, response_rate_mcse = _ratio_mcse(
        totals[2],
        totals[0],
        response_squares,
        response_exposure_squares,
        response_cross_products,
        count,
    )
    optimal_probability = None if selected_arms is None else optimal_selection_count / count
    optimal_mcse = (
        None
        if optimal_probability is None
        else float(_bernoulli_mcse(np.asarray(optimal_probability), count))
    )
    return ParallelPhase12OC(
        n_trials=count,
        per_trial_seeds=_freeze_int(seeds, dtype=np.dtype(np.uint64)),
        selected_counts=_freeze_int(selected_counts),
        selection_probability=_freeze(selection_probability),
        selection_mcse=_freeze(selection_mcse),
        no_selection_count=no_selection_count,
        no_selection_probability=no_probability,
        no_selection_mcse=no_mcse,
        stopping_reasons=_STOP_REASONS,
        stopping_counts=_freeze_int(stopping_counts),
        stopping_probability=_freeze(stopping_probability),
        stopping_mcse=_freeze(stopping_mcse),
        admissible_counts=_freeze_int(admissible_counts),
        admissibility_probability=_freeze(admissibility_probability),
        admissibility_mcse=_freeze(admissibility_mcse),
        treated_total=_freeze_int(totals[0].astype(np.int64)),
        mean_treated=_freeze(mean_treated),
        mcse_treated=_freeze(mcse_treated),
        toxicity_total=_freeze_int(totals[1].astype(np.int64)),
        mean_toxicities=_freeze(mean_toxicities),
        mcse_toxicities=_freeze(mcse_toxicities),
        response_total=_freeze_int(totals[2].astype(np.int64)),
        mean_responses=_freeze(mean_responses),
        mcse_responses=_freeze(mcse_responses),
        toxicity_rate=_freeze(toxicity_rate),
        toxicity_rate_mcse=_freeze(toxicity_rate_mcse),
        response_rate=_freeze(response_rate),
        response_rate_mcse=_freeze(response_rate_mcse),
        mean_total_enrollment=mean_total,
        mcse_total_enrollment=total_enroll_mcse,
        total_enrollment=int(total_enrollments),
        mean_phase_one_enrollment=mean_phase,
        mcse_phase_one_enrollment=phase_one_mcse,
        phase_one_enrollment_total=int(total_phase_one),
        optimal_arms=selected_arms,
        optimal_selection_probability=optimal_probability,
        optimal_selection_mcse=optimal_mcse,
    )
