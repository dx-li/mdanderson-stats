"""Bounded serial operating characteristics for completed-outcome PLBARPO.

Every replicate uses the explicit no-control controller in
:mod:`plbarpo_trial`; this module adds no new conduct rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .plbarpo_trial import (
    _MAX_ARMS,
    _MAX_TRIAL_WORK,
    _integer,
    _prepare,
    _run_prepared,
    _streams,
)

_MAX_TRIALS = 5_000
_MAX_TOTAL_WORK = 50_000_000
_MAX_SUMMARY_CELLS = 2_000_000
_METRICS = (
    "entry",
    "futility",
    "early_efficacy",
    "final_assessed",
    "final_efficacy",
    "any_efficacy",
    "cap_closure",
)


def _readonly(value: ArrayLike, dtype: type = np.float64) -> NDArray:
    result: NDArray = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _null_mask(value: ArrayLike | None, arms: int) -> NDArray[np.bool_] | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != (arms,):
        raise ValueError("null_arms must be a boolean vector with one entry per ledger arm")
    result = np.array(raw, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _mcse(probability: NDArray[np.float64] | float, n: int) -> NDArray[np.float64] | float:
    result = np.sqrt(np.asarray(probability) * (1.0 - np.asarray(probability)) / n)
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class PLBarpoSimulation:
    """Aggregate arm and trial OCs; no replicate histories are retained.

    ``metric_probability`` uses all trials. ``metric_given_entry`` conditions
    on the arm having entered and is NaN when the entry count is zero. Metrics
    are ordered by ``metric_names``. False-efficacy fields are present only
    when callers supply an explicit null-arm mask.
    """

    trials: int
    arm_count: int
    metric_names: tuple[str, ...]
    metric_counts: NDArray[np.int64]
    metric_probability: NDArray[np.float64]
    metric_mcse: NDArray[np.float64]
    metric_given_entry: NDArray[np.float64]
    assigned_mean: NDArray[np.float64]
    assigned_mcse: NDArray[np.float64]
    responses_mean: NDArray[np.float64]
    responses_mcse: NDArray[np.float64]
    total_enrollment_mean: float
    total_enrollment_mcse: float
    total_responses_mean: float
    total_responses_mcse: float
    no_efficacy_count: int
    no_efficacy_probability: float
    no_efficacy_mcse: float
    early_stop_count: int
    early_stop_probability: float
    early_stop_mcse: float
    null_arms: NDArray[np.bool_] | None
    false_efficacy_counts: NDArray[np.int64] | None
    false_efficacy_probability: NDArray[np.float64] | None
    false_efficacy_mcse: NDArray[np.float64] | None
    familywise_false_efficacy_count: int | None
    familywise_false_efficacy_probability: float | None
    familywise_false_efficacy_mcse: float | None
    trial_seeds: NDArray[np.uint64]
    stream_seeds: NDArray[np.uint64]
    total_work_units: int

    def metric_index(self, name: str) -> int:
        """Return the row index for one named arm-level metric."""
        try:
            return self.metric_names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown PLBARPO OC metric {name!r}") from exc


def simulate_plbarpo(
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
    trials: int = 1000,
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
    null_arms: ArrayLike | None = None,
    rng: int | np.integer | np.random.Generator | None = None,
    absolute_tolerance: float = 1e-9,
    max_work: int = _MAX_TRIAL_WORK,
    max_total_work: int = _MAX_TOTAL_WORK,
) -> PLBarpoSimulation:
    """Run independent sequential replicates and return compact OCs.

    One RNG is split into recorded per-trial seeds, then into independent
    assignment and outcome streams. Supplying ``trial_seeds[i]`` as ``rng`` to
    :func:`run_plbarpo_trial` reproduces replicate ``i``. No patient histories
    survive aggregation. Error rates are calculated only against the explicit
    ``null_arms`` mask; response truth alone does not label a null hypothesis.
    """
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    trials = trial_count
    truth_shape = np.asarray(true_response).shape
    if len(truth_shape) != 1 or not 1 <= truth_shape[0] <= _MAX_ARMS:
        raise ValueError("true_response must be a bounded one-dimensional arm vector")
    arms = int(truth_shape[0])
    null_mask = _null_mask(null_arms, arms)
    work_limit = _integer(max_work, "max_work", 1, _MAX_TRIAL_WORK)
    total_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)

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
        max_work=work_limit,
    )
    if trials * arms * len(_METRICS) > _MAX_SUMMARY_CELLS:
        raise ValueError("PLBARPO trial-arm summary exceeds the hard allocation bound")
    per_trial_bound = (2 * config.max_total_n + 2 * len(config.look_sizes) + 1) * (
        config.active_capacity**2
    )
    if trials * per_trial_bound > total_limit:
        raise ValueError("worst-case PLBARPO simulation work exceeds max_total_work")

    if isinstance(rng, np.random.Generator):
        entropy: int | list[int] | None = [
            int(v) for v in rng.integers(0, 2**32, size=4, dtype=np.uint32)
        ]
    elif rng is None:
        entropy = None
    elif (
        isinstance(rng, (bool, np.bool_))
        or not isinstance(rng, (int, np.integer))
        or int(rng) < 0
    ):
        raise ValueError("rng must be a nonnegative integer, Generator or None")
    else:
        entropy = int(rng)
    trial_sequences = np.random.SeedSequence(entropy).spawn(trials)
    trial_seeds = np.empty(trials, dtype=np.uint64)
    stream_seeds = np.empty((trials, 2), dtype=np.uint64)
    metric_counts = np.zeros((len(_METRICS), arms), dtype=np.int64)
    assigned_sum = np.zeros(arms)
    assigned_square_sum = np.zeros(arms)
    response_sum = np.zeros(arms)
    response_square_sum = np.zeros(arms)
    total_n_sum = total_n_square_sum = 0.0
    total_y_sum = total_y_square_sum = 0.0
    no_efficacy_count = early_stop_count = total_work = 0
    false_counts = np.zeros(arms, dtype=np.int64) if null_mask is not None else None
    familywise_false_count = 0

    for i, child in enumerate(trial_sequences):
        trial_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        trial_seeds[i] = trial_seed
        assignment_rng, outcome_rng, stream_pair = _streams(trial_seed)
        stream_seeds[i] = stream_pair
        trial = _run_prepared(
            config, None, None, assignment_rng, outcome_rng, stream_pair
        )
        total_work += trial.work_units
        if total_work > total_limit:
            raise RuntimeError("observed PLBARPO work exceeded max_total_work")
        any_eff = trial.early_efficacy | trial.final_efficacy
        metrics = (
            trial.entered,
            trial.early_futility,
            trial.early_efficacy,
            trial.final_assessed,
            trial.final_efficacy,
            any_eff,
            trial.cap_closed,
        )
        for row, flags in enumerate(metrics):
            metric_counts[row] += flags
        assigned = trial.assigned.astype(float)
        responses = trial.successes.astype(float)
        assigned_sum += assigned
        assigned_square_sum += assigned * assigned
        response_sum += responses
        response_square_sum += responses * responses
        total_n = float(trial.enrolled)
        total_y = float(trial.successes.sum())
        total_n_sum += total_n
        total_n_square_sum += total_n * total_n
        total_y_sum += total_y
        total_y_square_sum += total_y * total_y
        no_efficacy_count += int(not np.any(any_eff))
        early_stop_count += int(trial.enrolled < config.max_total_n)
        if null_mask is not None and false_counts is not None:
            false_declarations = any_eff & null_mask
            false_counts += false_declarations
            familywise_false_count += int(np.any(false_declarations))
        del trial

    probabilities = metric_counts.astype(float) / trials
    mcse = np.sqrt(probabilities * (1.0 - probabilities) / trials)
    entered = metric_counts[0]
    given_entry = np.full_like(probabilities, np.nan)
    np.divide(metric_counts, entered[None, :], out=given_entry, where=entered[None, :] > 0)

    def mean_and_mcse(
        total: NDArray[np.float64], squares: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mean = total / trials
        if trials == 1:
            error = np.full_like(mean, np.nan)
        else:
            variance = np.maximum((squares - total * total / trials) / (trials - 1), 0.0)
            error = np.sqrt(variance / trials)
        return mean, error

    assigned_mean, assigned_mcse = mean_and_mcse(assigned_sum, assigned_square_sum)
    response_mean, response_mcse = mean_and_mcse(response_sum, response_square_sum)

    def scalar_mean_error(total: float, squares: float) -> tuple[float, float]:
        mean = total / trials
        if trials == 1:
            return mean, float("nan")
        variance = max((squares - total * total / trials) / (trials - 1), 0.0)
        return mean, float(np.sqrt(variance / trials))

    total_n_mean, total_n_mcse = scalar_mean_error(total_n_sum, total_n_square_sum)
    total_y_mean, total_y_mcse = scalar_mean_error(total_y_sum, total_y_square_sum)
    no_efficacy_probability = no_efficacy_count / trials
    early_stop_probability = early_stop_count / trials
    false_probability = None if false_counts is None else false_counts.astype(float) / trials
    return PLBarpoSimulation(
        trials,
        arms,
        _METRICS,
        _readonly(metric_counts, np.int64),
        _readonly(probabilities),
        _readonly(mcse),
        _readonly(given_entry),
        _readonly(assigned_mean),
        _readonly(assigned_mcse),
        _readonly(response_mean),
        _readonly(response_mcse),
        total_n_mean,
        total_n_mcse,
        total_y_mean,
        total_y_mcse,
        no_efficacy_count,
        no_efficacy_probability,
        float(_mcse(no_efficacy_probability, trials)),
        early_stop_count,
        early_stop_probability,
        float(_mcse(early_stop_probability, trials)),
        None if null_mask is None else _readonly(null_mask, bool),
        None if false_counts is None else _readonly(false_counts, np.int64),
        None if false_probability is None else _readonly(false_probability),
        None if false_probability is None else _readonly(_mcse(false_probability, trials)),
        None if null_mask is None else familywise_false_count,
        None if null_mask is None else familywise_false_count / trials,
        None if null_mask is None else float(_mcse(familywise_false_count / trials, trials)),
        _readonly(trial_seeds, np.uint64),
        _readonly(stream_seeds, np.uint64),
        total_work,
    )
