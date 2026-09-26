"""Exact bounded look-ahead over pending binary BMA-CRM outcomes."""

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .bmacrm import BMACRMPosterior, _raw_numeric, _scalar_value, fit_bmacrm
from .bmacrm_decision import BMACRMDecision, _dose_index, bmacrm_decision

_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_MAX_COMPLETIONS = 1024
_MAX_TOTAL_EVALUATIONS = 2_000_000
_MAX_FIT_EVALUATIONS = 200_000


@dataclass(frozen=True)
class BMACRMLookAhead:
    """Decision plus exact completion witnesses evaluated by look-ahead."""

    action: Literal["start", "treat", "wait", "stop", "select_mtd"]
    dose: int | None
    reason: Literal["complete", "invariant", "outcome_dependent", "work_limit"]
    total_completions: int
    evaluated_completions: int
    evaluations: int
    decisions: tuple[BMACRMDecision, ...]
    completion_events: NDArray[np.int64]


def _freeze_counts(counts: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.frombuffer(counts.tobytes(), dtype=np.int64).reshape(counts.shape)


def _setting(value: int, name: str, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer from 1 to {upper}")
    number = int(value)
    if not 1 <= number <= upper:
        raise ValueError(f"{name} must be an integer from 1 to {upper}")
    return number


def _validate(
    posterior: BMACRMPosterior,
    pending: ArrayLike,
    *,
    current_dose: int | None,
    starting_dose: int,
    safety_cutoff: object,
    final: bool,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int, int, float]:
    if not isinstance(posterior, BMACRMPosterior):
        raise TypeError("posterior must be a BMACRMPosterior")
    skeletons = _raw_numeric(posterior.skeletons, "posterior.skeletons", 5 * _MAX_DOSES)
    if skeletons.ndim == 1:
        skeletons = skeletons[None, :]
    events = _raw_numeric(posterior.events, "posterior.events", _MAX_DOSES)
    subjects = _raw_numeric(posterior.subjects, "posterior.subjects", _MAX_DOSES)
    weights = _raw_numeric(posterior.prior_model_weights, "posterior.prior_model_weights", 5)
    input_weights = (
        weights
        if posterior.input_model_prior is None
        else _raw_numeric(posterior.input_model_prior, "posterior.input_model_prior", 5)
    )
    if (
        skeletons.ndim != 2
        or not 1 <= skeletons.shape[0] <= 5
        or not 1 <= skeletons.shape[1] <= _MAX_DOSES
        or events.shape != (skeletons.shape[1],)
        or subjects.shape != events.shape
        or weights.shape != (skeletons.shape[0],)
        or input_weights.shape != (skeletons.shape[0],)
    ):
        raise ValueError("posterior has inconsistent skeleton, count, or model-prior shapes")
    if (
        np.any(~np.isfinite(skeletons))
        or np.any((skeletons <= 0) | (skeletons >= 1))
        or np.any(np.diff(skeletons, axis=1) < 0)
        or np.any(~np.isfinite(events) | ~np.isfinite(subjects))
        or np.any((events < 0) | (subjects < events))
        or np.any(events != np.floor(events))
        or np.any(subjects != np.floor(subjects))
        or np.sum(subjects) > _MAX_SUBJECTS
        or np.any(~np.isfinite(weights) | (weights < 0))
        or not np.any(weights > 0)
        or np.any(~np.isfinite(input_weights) | (input_weights < 0))
        or not np.any(input_weights > 0)
        or not 0 < _scalar_value(posterior.target, "posterior.target") < 1
        or not 1e-3 <= _scalar_value(posterior.prior_sd, "posterior.prior_sd") <= 10
    ):
        raise ValueError("posterior contains invalid CRM inputs")

    raw_pending = np.asarray(pending)
    if (
        raw_pending.ndim != 1
        or raw_pending.shape != events.shape
        or raw_pending.dtype.kind not in "iuf"
    ):
        raise ValueError("pending must be a nonnegative integer count per dose")
    if (
        np.any(~np.isfinite(raw_pending))
        or np.any(raw_pending < 0)
        or np.any(raw_pending > _MAX_SUBJECTS)
        or np.any(raw_pending != np.floor(raw_pending))
    ):
        raise ValueError("pending counts per dose must not exceed 10000")
    pending_counts = np.asarray(raw_pending, dtype=np.int64)
    if np.any(pending_counts < 0) or np.sum(subjects) + np.sum(pending_counts) > _MAX_SUBJECTS:
        raise ValueError("observed plus pending subjects must not exceed 10000")
    if isinstance(final, np.bool_):
        final = bool(final)
    if not isinstance(final, bool):
        raise ValueError("final must be boolean")
    start = _dose_index(starting_dose, "starting_dose", skeletons.shape[1])
    current = -1
    if current_dose is not None:
        current = _dose_index(current_dose, "current_dose", skeletons.shape[1])
    if (
        not final
        and np.sum(subjects) + np.sum(pending_counts) > 0
        and (current < 0 or subjects[current] + pending_counts[current] < 1)
    ):
        raise ValueError("current_dose must be supplied and have a treated subject")
    cutoff = _scalar_value(np.asarray(safety_cutoff), "safety_cutoff")
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("safety_cutoff must be finite and lie in [0,1]")
    return events.astype(np.int64), pending_counts, start, current, cutoff


def bmacrm_lookahead(
    posterior: BMACRMPosterior,
    pending: ArrayLike,
    *,
    current_dose: int | None = None,
    starting_dose: int = 0,
    safety_cutoff: float = 0.9,
    final: bool = False,
    max_completions: int = 128,
    max_evaluations: int = 200_000,
) -> BMACRMLookAhead:
    """Act only when every completion of currently pending binary outcomes
    yields the same BMA-CRM action and dose.

    The exhaustive grid groups counts by dose, so its size is
    ``product(pending[j] + 1)``. It is preflighted before any refits. Each
    hypothetical posterior uses the original model prior weights, not the
    observed-data posterior model weights. If the bounded enumeration is too
    large, the result is an explicit wait with reason ``work_limit``.
    """
    completion_limit = _setting(max_completions, "max_completions", _MAX_COMPLETIONS)
    evaluation_limit = _setting(max_evaluations, "max_evaluations", _MAX_TOTAL_EVALUATIONS)
    events, pending_counts, start, current, cutoff = _validate(
        posterior,
        pending,
        current_dose=current_dose,
        starting_dose=starting_dose,
        safety_cutoff=safety_cutoff,
        final=final,
    )
    total_completions = prod(int(value) + 1 for value in pending_counts)
    if total_completions > completion_limit:
        return BMACRMLookAhead(
            "wait",
            None,
            "work_limit",
            total_completions,
            0,
            0,
            (),
            _freeze_counts(np.empty((0, events.size), dtype=np.int64)),
        )

    def direct_decision() -> BMACRMDecision:
        return bmacrm_decision(
            posterior,
            current_dose=None if current < 0 else current,
            starting_dose=start,
            safety_cutoff=cutoff,
            final=final,
        )

    if not np.any(pending_counts):
        decision = direct_decision()
        return BMACRMLookAhead(
            decision.action,
            decision.dose,
            "complete",
            1,
            1,
            0,
            (decision,),
            _freeze_counts(events[None, :]),
        )

    completion_rows: list[NDArray[np.int64]] = []
    decision_rows: list[BMACRMDecision] = []
    used_evaluations = 0
    model_prior = (
        posterior.prior_model_weights
        if posterior.input_model_prior is None
        else posterior.input_model_prior
    )

    def evaluate(completed_events: NDArray[np.int64]) -> BMACRMDecision:
        nonlocal used_evaluations
        remaining = evaluation_limit - used_evaluations
        if remaining < 1:
            raise RuntimeError("BMA-CRM look-ahead exhausted its integration evaluation budget")
        fitted = fit_bmacrm(
            posterior.skeletons,
            completed_events,
            posterior.subjects + pending_counts,
            target=posterior.target,
            prior_sd=posterior.prior_sd,
            model_prior=model_prior,
            max_evaluations=min(_MAX_FIT_EVALUATIONS, remaining),
        )
        used_evaluations += fitted.evaluations
        decision = bmacrm_decision(
            fitted,
            current_dose=None if current < 0 else current,
            starting_dose=start,
            safety_cutoff=cutoff,
            final=final,
        )
        completion_rows.append(completed_events.copy())
        decision_rows.append(decision)
        return decision

    all_zero = np.zeros(events.size, dtype=np.int64)
    all_positive = pending_counts.copy()
    first = evaluate(events + all_zero)
    second = evaluate(events + all_positive)
    if (first.action, first.dose) != (second.action, second.dose):
        return BMACRMLookAhead(
            "wait",
            None,
            "outcome_dependent",
            total_completions,
            len(decision_rows),
            used_evaluations,
            tuple(decision_rows),
            _freeze_counts(np.stack(completion_rows)),
        )

    for candidate in product(*(range(int(value) + 1) for value in pending_counts)):
        row = np.asarray(candidate, dtype=np.int64)
        if np.array_equal(row, all_zero) or np.array_equal(row, all_positive):
            continue
        decision = evaluate(events + row)
        if (first.action, first.dose) != (decision.action, decision.dose):
            return BMACRMLookAhead(
                "wait",
                None,
                "outcome_dependent",
                total_completions,
                len(decision_rows),
                used_evaluations,
                tuple(decision_rows),
                _freeze_counts(np.stack(completion_rows)),
            )
    return BMACRMLookAhead(
        first.action,
        first.dose,
        "invariant",
        total_completions,
        len(decision_rows),
        used_evaluations,
        tuple(decision_rows),
        _freeze_counts(np.stack(completion_rows)),
    )
