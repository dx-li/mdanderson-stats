"""Serial operating-characteristic simulation for the MTADF dose rule."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .mtadf import MTADFPrior, _scalar, mtadf_decision, mtadf_toxicity_prior

_MAX_DOSES = 20
_MAX_TRIALS = 10_000
_MAX_PATIENTS = 1_000
_MAX_DECISIONS = 1_000_000


def _freeze_int(values: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=np.int64).reshape(
        values.shape
    )


def _setting(value: object, name: str, lower: int, upper: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    number = int(raw)
    if not lower <= number <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return number


def _probabilities(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or not 1 <= raw.size <= _MAX_DOSES or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with 1..{_MAX_DOSES} entries")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return result


@dataclass(frozen=True)
class MTADFSimulation:
    """Compact operating-characteristic summaries and per-trial dose counts."""

    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_probability: float
    no_selection_mcse: float
    early_stop_probability: float
    early_stop_mcse: float
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_responses: FloatArray
    stop_reason: tuple[str, ...]


def simulate_mtadf(
    true_toxicity: ArrayLike,
    true_efficacy: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 100,
    starting_dose: int = 0,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    margin: float = 0.05,
    concentration: float = 0.5,
    prior: MTADFPrior | None = None,
    rng: int | np.integer | np.random.Generator | None = None,
    max_total_decisions: int = 200_000,
) -> MTADFSimulation:
    """Simulate complete-outcome cohorts and summarize dose selection.

    Toxicity and efficacy counts are generated independently from their
    dose-specific binomial marginals. The initial decision and every cohort
    review count against the bounded decision-work budget. Early stopping
    includes trials for which no starting dose is admissible.
    """
    tox = _probabilities(true_toxicity, "true_toxicity")
    eff = _probabilities(true_efficacy, "true_efficacy")
    if tox.shape != eff.shape:
        raise ValueError("true_toxicity and true_efficacy must have matching dose counts")
    cohort_count = _setting(cohorts, "cohorts", 1, _MAX_PATIENTS)
    size = _setting(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    trial_count = _setting(trials, "trials", 1, _MAX_TRIALS)
    planned = cohort_count * size
    if planned > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    work_limit = _setting(max_total_decisions, "max_total_decisions", 1, _MAX_DECISIONS)
    work = trial_count * (cohort_count + 1)
    if work > work_limit:
        raise ValueError("requested simulation exceeds max_total_decisions")
    start = _setting(starting_dose, "starting_dose", 0, tox.size - 1)
    phi = _scalar(toxicity_limit, "toxicity_limit")
    cutoff = _scalar(safety_cutoff, "safety_cutoff")
    delta = _scalar(margin, "margin")
    concentration_value = _scalar(concentration, "concentration")
    beta_prior = (
        prior
        if prior is not None
        else mtadf_toxicity_prior(phi, cutoff, delta, concentration_value)
    )
    if not isinstance(beta_prior, MTADFPrior):
        raise ValueError("prior must be an MTADFPrior")
    if isinstance(rng, np.random.Generator):
        generator = rng
    elif rng is None or (
        isinstance(rng, (int, np.integer)) and not isinstance(rng, (bool, np.bool_))
    ):
        generator = np.random.default_rng(rng)
    else:
        raise ValueError("rng must be an integer seed, Generator, or None")

    n_by_trial = np.zeros((trial_count, tox.size), dtype=np.int64)
    y_by_trial = np.zeros_like(n_by_trial)
    r_by_trial = np.zeros_like(n_by_trial)
    selected = np.full(trial_count, -1, dtype=np.int64)
    reasons: list[str] = []
    for trial in range(trial_count):
        initial = mtadf_decision(
            n_by_trial[trial],
            y_by_trial[trial],
            r_by_trial[trial],
            starting_dose=start,
            toxicity_limit=phi,
            safety_cutoff=cutoff,
            prior=beta_prior,
        )
        if initial.action == "stop":
            reasons.append("no_admissible_start")
            continue
        assert initial.dose is not None
        dose = initial.dose
        reason = "maximum_enrollment_no_safe_tried_dose"
        for cohort_index in range(cohort_count):
            y_by_trial[trial, dose] += generator.binomial(size, tox[dose])
            r_by_trial[trial, dose] += generator.binomial(size, eff[dose])
            n_by_trial[trial, dose] += size
            if cohort_index == cohort_count - 1:
                final = mtadf_decision(
                    n_by_trial[trial],
                    y_by_trial[trial],
                    r_by_trial[trial],
                    final=True,
                    toxicity_limit=phi,
                    safety_cutoff=cutoff,
                    prior=beta_prior,
                )
                if final.action == "select_obd":
                    assert final.dose is not None
                    selected[trial] = final.dose
                    reason = "maximum_enrollment"
                else:
                    reason = "maximum_enrollment_no_safe_tried_dose"
                break
            interim = mtadf_decision(
                n_by_trial[trial],
                y_by_trial[trial],
                r_by_trial[trial],
                current_dose=dose,
                starting_dose=start,
                toxicity_limit=phi,
                safety_cutoff=cutoff,
                prior=beta_prior,
            )
            if interim.action == "stop":
                reason = "early_safety_stop"
                break
            assert interim.dose is not None
            dose = interim.dose
        reasons.append(reason)

    frequencies = np.bincount(selected[selected >= 0], minlength=tox.size).astype(float)
    selected_frequency = frequencies / trial_count
    selection_mcse = np.sqrt(selected_frequency * (1 - selected_frequency) / trial_count)
    no_select = float(np.count_nonzero(selected < 0) / trial_count)
    early = float(np.count_nonzero(n_by_trial.sum(axis=1) < planned) / trial_count)
    return MTADFSimulation(
        _freeze_int(n_by_trial),
        _freeze_int(y_by_trial),
        _freeze_int(r_by_trial),
        _freeze_int(selected),
        _freeze(selected_frequency),
        _freeze(selection_mcse),
        no_select,
        sqrt(no_select * (1 - no_select) / trial_count),
        early,
        sqrt(early * (1 - early) / trial_count),
        _freeze(np.mean(n_by_trial, axis=0)),
        _freeze(np.mean(y_by_trial, axis=0)),
        _freeze(np.mean(r_by_trial, axis=0)),
        tuple(reasons),
    )
