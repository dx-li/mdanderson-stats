"""Operating-characteristic summaries for independent U2OET trial replicates."""

import json
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet_decision import U2OETCriteria
from .u2oet_simulation import U2OETTrial


def _mean_error(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    if len(x) == 0:
        return np.asarray(np.nan), np.asarray(np.nan)
    scale = np.max(np.abs(x), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    normalized = x / scale
    mean = np.mean(normalized, axis=0) * scale
    error = (
        np.std(normalized, axis=0, ddof=1) / np.sqrt(len(x)) * scale
        if len(x) > 1
        else np.full_like(mean, np.nan)
    )
    return mean, error


@dataclass(frozen=True)
class U2OETOperatingCharacteristics:
    trials: int
    selected_trials: int
    selection_probability: FloatArray
    selection_mcse: FloatArray
    none_probability: float
    none_mcse: float
    early_stop_probability: float
    early_stop_mcse: float
    mean_enrollment: float
    enrollment_mcse: float
    mean_treated: FloatArray
    treated_mcse: FloatArray
    true_utility: FloatArray
    true_acceptable: np.ndarray
    true_best: np.ndarray
    best_probability: float
    best_mcse: float
    rselect: FloatArray
    rtreat: FloatArray
    mean_rselect: float
    rselect_mcse: float
    mean_rtreat: float
    rtreat_mcse: float
    mean_duration: float
    duration_mcse: float
    final_max_split_rhat: FloatArray
    design_json: str


def summarize_u2oet_trials(trials: Iterable[U2OETTrial]) -> U2OETOperatingCharacteristics:
    """Summarize like-for-like replicates, without retaining full trial histories.

    Rselect is conditional on selection; no-selection entries remain NaN.
    Rtreat uses each trial's actual final enrollment N, as in the paper.
    A constant true utility surface makes both normalized scores undefined.
    """
    design = ""
    counts: list[FloatArray] = []
    selections: list[FloatArray] = []
    early: list[bool] = []
    duration: list[float] = []
    diagnostic: list[float] = []
    used_seeds: set[int] = set()
    for trial in trials:
        if not isinstance(trial, U2OETTrial) or not trial.design_json:
            raise ValueError("require trial results with recorded design metadata")
        if not design:
            design = trial.design_json
        elif design != trial.design_json:
            raise ValueError("cannot pool different designs, priors, scenarios or MCMC settings")
        seeds = {trial.data_seed, trial.posterior_seed}
        if used_seeds & seeds or len(seeds) != 2:
            raise ValueError("replicates must not reuse data or posterior RNG seeds")
        used_seeds.update(seeds)
        if len(counts) >= 100_000:
            raise ValueError("at most 100000 replicates are supported")
        n = trial.patients.treated
        if (
            np.any(n < 0)
            or not np.all(np.isfinite(n))
            or np.any(n != np.floor(n))
            or n.sum() != len(trial.patients.records)
            or n.sum() == 0
        ):
            raise ValueError("invalid enrollment counts in trial")
        selected = np.zeros_like(n)
        if trial.selected is not None:
            i, j = trial.selected
            if trial.stopped_early or not (0 <= i < n.shape[0] and 0 <= j < n.shape[1]):
                raise ValueError("invalid selected dose in trial")
            selected[i, j] = 1
        counts.append(n)
        selections.append(selected)
        early.append(trial.stopped_early)
        duration.append(trial.analysis_time)
        diagnostic.append(trial.final_max_split_rhat)
    if not counts:
        raise ValueError("at least one trial is required")
    recipe = json.loads(design)
    if recipe.get("format_version") != 1:
        raise ValueError("unsupported trial metadata version")
    joint = np.asarray(recipe["scenario_joint"], dtype=float)
    utility = np.asarray(recipe["utility"], dtype=float)
    criteria = U2OETCriteria(**recipe["criteria"])
    baseline = float(utility.min())
    truth = baseline + np.sum(joint * (utility - baseline), axis=(-2, -1))
    if not np.all(np.isfinite(truth)):
        raise ArithmeticError("true expected utility exceeds floating-point range")
    treated, selected = np.array(counts), np.array(selections)
    if treated.shape[1:] != truth.shape:
        raise ValueError("trial enrollment and scenario grids differ")
    repetitions = len(treated)
    enrollment = treated.sum(axis=(1, 2))
    has_selection = selected.sum(axis=(1, 2)) > 0
    acceptable = (
        joint[:, :, criteria.efficacy_level :, :].sum(axis=(-2, -1)) >= criteria.min_efficacy
    ) & (joint[:, :, :, criteria.toxicity_level :].sum(axis=(-2, -1)) <= criteria.max_toxicity)
    best = (
        acceptable & (truth == np.max(truth[acceptable]))
        if np.any(acceptable)
        else np.zeros_like(acceptable)
    )
    selection_probability = selected.mean(axis=0)
    none_probability = float(np.mean(~has_selection))
    stop_probability = float(np.mean(early))
    best_probability = float(np.mean(selected[:, best].sum(axis=1))) if np.any(best) else np.nan

    def binary_mcse(p: float | FloatArray) -> FloatArray:
        return np.asarray(np.sqrt(p * (1 - p) / repetitions))

    low, high = float(truth.min()), float(truth.max())
    rselect = np.full(repetitions, np.nan)
    rtreat = np.full(repetitions, np.nan)
    if high > low:
        score = 100 * ((truth - low) / (high - low))
        rselect[has_selection] = (selected[has_selection] * score).sum(axis=(1, 2))
        rtreat = ((treated / enrollment[:, None, None]) * score).sum(axis=(1, 2))
    mean_n, error_n = _mean_error(enrollment)
    mean_treated, error_treated = _mean_error(treated)
    mean_rs, error_rs = _mean_error(rselect[has_selection])
    mean_rt, error_rt = _mean_error(rtreat)
    mean_duration, error_duration = _mean_error(np.asarray(duration))
    masks = [np.frombuffer(x.tobytes(), dtype=bool).reshape(x.shape) for x in (acceptable, best)]
    return U2OETOperatingCharacteristics(
        repetitions,
        int(has_selection.sum()),
        _freeze(selection_probability),
        _freeze(binary_mcse(selection_probability)),
        none_probability,
        float(binary_mcse(none_probability)),
        stop_probability,
        float(binary_mcse(stop_probability)),
        float(mean_n),
        float(error_n),
        _freeze(mean_treated),
        _freeze(error_treated),
        _freeze(truth),
        masks[0],
        masks[1],
        best_probability,
        float(binary_mcse(best_probability)),
        _freeze(rselect),
        _freeze(rtreat),
        float(mean_rs),
        float(error_rs),
        float(mean_rt),
        float(error_rt),
        float(mean_duration),
        float(error_duration),
        _freeze(diagnostic),
        design,
    )
