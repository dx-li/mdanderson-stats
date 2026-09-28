"""Bounded serial operating characteristics for control-enabled PLBARPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar
from .plbarpo_control_trial import (
    PLBarpoControlTrialResult,
    _run_control_prepared,
    _validate_control_mode,
)
from .plbarpo_trial import (
    _MAX_ARMS,
    _MAX_PATIENTS,
    _MAX_TRIAL_WORK,
    _bool_vector,
    _Config,
    _integer,
    _ledger_counts,
    _prepare,
    _readonly,
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


def _mcse(probability: NDArray[np.float64] | float, n: int) -> NDArray[np.float64] | float:
    result = np.sqrt(np.asarray(probability) * (1.0 - np.asarray(probability)) / n)
    return float(result) if result.ndim == 0 else result


def _null_mask(value: ArrayLike | None, arms: int) -> NDArray[np.bool_] | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != (arms,):
        raise ValueError("null_arms must be a boolean vector with one entry per ledger arm")
    if raw[0]:
        raise ValueError("null_arms[0] cannot mark the persistent control as an efficacy null")
    result = np.array(raw, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class PLBarpoControlSimulation:
    """Compact control-enabled arm/trial operating characteristics.

    Metric rows follow ``metric_names`` and all-trial denominators. Conditional
    rates use the number of trials in which each arm entered, with NaN for an
    arm that never entered. Efficacy and false-efficacy metrics exclude arm 0.
    """

    trials: int
    arm_count: int
    control_mode: str
    metric_names: tuple[str, ...]
    metric_counts: NDArray[np.int64]
    metric_probability: NDArray[np.float64]
    metric_mcse: NDArray[np.float64]
    metric_given_entry: NDArray[np.float64]
    metric_given_entry_mcse: NDArray[np.float64]
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
            raise ValueError(f"unknown PLBARPO control OC metric {name!r}") from exc


def simulate_plbarpo_control(
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
    trials: int = 1000,
    method: str = "barcp",
    tau: float = 0.5,
    tau1: float = 1.0,
    target_probability: ArrayLike | None = None,
    minimum_probability: ArrayLike | None = None,
    early_monitoring: bool = True,
    pfut: float | None = None,
    peff: float | None = None,
    pfinal: float | None = None,
    null_arms: ArrayLike | None = None,
    rng: int | np.integer | np.random.Generator | None = None,
    absolute_tolerance: float = 1e-9,
    max_work: int = _MAX_TRIAL_WORK,
    max_total_work: int = _MAX_TOTAL_WORK,
) -> PLBarpoControlSimulation:
    """Run independent control-enabled trials and aggregate their OCs.

    ``null_arms`` is an optional explicit full-ledger efficacy-null mask; the
    persistent control must be false in this mask. Trial seeds can be passed to
    ``run_plbarpo_control_trial`` to reproduce individual replicates. The
    simulation is serial and retains no patient- or posterior-level histories.
    """
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    truth_raw = np.asarray(true_response)
    if truth_raw.ndim != 1 or not 2 <= truth_raw.size <= _MAX_ARMS:
        raise ValueError("true_response must be a bounded control-plus-treatment vector")
    arms = int(truth_raw.size)
    mode = _validate_control_mode(control_mode)
    null = _null_mask(null_arms, arms)
    work_limit = _integer(max_work, "max_work", 1, _MAX_TRIAL_WORK)
    total_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    initial = _bool_vector(initial_active, "initial_active", arms)
    if not initial[0] or int(initial.sum()) < 2:
        raise ValueError("control arm 0 and at least one treatment must initially be active")
    maximum = _integer(max_total_n, "max_total_n", 1, _MAX_PATIENTS)
    per_arm_max = _ledger_counts(max_n_per_arm, "max_n_per_arm", arms)
    if per_arm_max[0] != maximum:
        raise ValueError("control max_n_per_arm[0] must equal max_total_n")
    if not isinstance(early_monitoring, (bool, np.bool_)):
        raise ValueError("early_monitoring must be boolean")
    for name, threshold in (("pfut", pfut), ("peff", peff), ("pfinal", pfinal)):
        if threshold is not None:
            parsed = scalar(threshold, name)
            if not 0 <= parsed <= 1:
                raise ValueError(f"{name} must be in [0, 1]")

    # Shared validation/allocation preflight; control-relative gates are checked
    # above without numerical integration before caller randomness is consumed.
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
        max_work=work_limit,
    )
    if np.any(config.candidate_order == 0):
        raise ValueError("candidate_order must exclude persistent control arm 0")
    per_trial_bound = (2 * config.max_total_n + 2 * len(config.look_sizes) + 1) * (
        config.active_capacity**2
    )
    if trial_count * arms * len(_METRICS) > _MAX_SUMMARY_CELLS:
        raise ValueError("control PLBARPO trial-arm summary exceeds the hard allocation bound")
    if trial_count * per_trial_bound > total_limit:
        raise ValueError("worst-case control PLBARPO work exceeds max_total_work")

    if isinstance(rng, np.random.Generator):
        entropy: int | list[int] | None = [
            int(value) for value in rng.integers(0, 2**32, size=4, dtype=np.uint32)
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
    children = np.random.SeedSequence(entropy).spawn(trial_count)
    trial_seeds = np.empty(trial_count, dtype=np.uint64)
    stream_seeds = np.empty((trial_count, 2), dtype=np.uint64)
    metric_counts = np.zeros((len(_METRICS), arms), dtype=np.int64)
    assigned_sum = np.zeros(arms)
    assigned_square_sum = np.zeros(arms)
    response_sum = np.zeros(arms)
    response_square_sum = np.zeros(arms)
    total_n_sum = total_n_squares = 0.0
    total_y_sum = total_y_squares = 0.0
    no_efficacy_count = early_stop_count = total_work = 0
    false_counts = np.zeros(arms, dtype=np.int64) if null is not None else None
    familywise_false_count = 0

    for i, child in enumerate(children):
        trial_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        trial_seeds[i] = trial_seed
        assignment_rng, outcome_rng, stream_pair = _streams(trial_seed)
        stream_seeds[i] = stream_pair
        trial: PLBarpoControlTrialResult = _run_control_prepared(
            config,
            mode,
            bool(early_monitoring),
            pfut,
            peff,
            pfinal,
            None,
            None,
            assignment_rng,
            outcome_rng,
            stream_pair,
        )
        total_work += trial.work_units
        if total_work > total_limit:
            raise RuntimeError("observed control PLBARPO work exceeded max_total_work")
        any_eff = trial.early_efficacy | trial.final_efficacy
        any_eff[0] = False
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
        total_n_squares += total_n * total_n
        total_y_sum += total_y
        total_y_squares += total_y * total_y
        no_efficacy_count += int(not np.any(any_eff[1:]))
        early_stop_count += int(trial.enrolled < config.max_total_n)
        if null is not None and false_counts is not None:
            declarations = any_eff & null
            false_counts += declarations
            familywise_false_count += int(np.any(declarations))
        del trial

    probabilities = metric_counts.astype(float) / trial_count
    metric_mcse = np.sqrt(probabilities * (1.0 - probabilities) / trial_count)
    entry_count = metric_counts[0]
    conditional = np.full_like(probabilities, np.nan)
    np.divide(metric_counts, entry_count[None, :], out=conditional, where=entry_count[None, :] > 0)
    conditional_mcse = np.full_like(probabilities, np.nan)
    np.divide(
        np.sqrt(conditional * (1.0 - conditional)),
        np.sqrt(entry_count[None, :]),
        out=conditional_mcse,
        where=entry_count[None, :] > 0,
    )

    def mean_and_mcse(
        total: NDArray[np.float64], squares: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mean = total / trial_count
        if trial_count == 1:
            return mean, np.full_like(mean, np.nan)
        variance = np.maximum((squares - total * total / trial_count) / (trial_count - 1), 0.0)
        return mean, np.sqrt(variance / trial_count)

    def scalar_mean_and_mcse(total: float, squares: float) -> tuple[float, float]:
        mean = total / trial_count
        if trial_count == 1:
            return mean, float("nan")
        variance = max((squares - total * total / trial_count) / (trial_count - 1), 0.0)
        return mean, float(np.sqrt(variance / trial_count))

    assigned_mean, assigned_mcse = mean_and_mcse(assigned_sum, assigned_square_sum)
    responses_mean, responses_mcse = mean_and_mcse(response_sum, response_square_sum)
    enrollment_mean, enrollment_mcse = scalar_mean_and_mcse(total_n_sum, total_n_squares)
    response_total_mean, response_total_mcse = scalar_mean_and_mcse(total_y_sum, total_y_squares)
    no_efficacy_probability = no_efficacy_count / trial_count
    early_stop_probability = early_stop_count / trial_count
    false_probability = None if false_counts is None else false_counts.astype(float) / trial_count
    return PLBarpoControlSimulation(
        trial_count,
        arms,
        mode,
        _METRICS,
        _readonly(metric_counts, np.int64),
        _readonly(probabilities),
        _readonly(metric_mcse),
        _readonly(conditional),
        _readonly(conditional_mcse),
        _readonly(assigned_mean),
        _readonly(assigned_mcse),
        _readonly(responses_mean),
        _readonly(responses_mcse),
        enrollment_mean,
        enrollment_mcse,
        response_total_mean,
        response_total_mcse,
        no_efficacy_count,
        no_efficacy_probability,
        float(_mcse(no_efficacy_probability, trial_count)),
        early_stop_count,
        early_stop_probability,
        float(_mcse(early_stop_probability, trial_count)),
        None if null is None else _readonly(null, bool),
        None if false_counts is None else _readonly(false_counts, np.int64),
        None if false_probability is None else _readonly(false_probability),
        None if false_probability is None else _readonly(_mcse(false_probability, trial_count)),
        None if null is None else familywise_false_count,
        None if null is None else familywise_false_count / trial_count,
        None if null is None else float(_mcse(familywise_false_count / trial_count, trial_count)),
        _readonly(trial_seeds, np.uint64),
        _readonly(stream_seeds, np.uint64),
        total_work,
    )
