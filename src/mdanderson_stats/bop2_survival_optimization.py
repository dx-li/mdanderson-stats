"""Common-path Monte Carlo calibration for single-arm BOP2 survival designs."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .bayesian_monitoring import _owned
from .bop2_binary import BOP2InfeasibleError, _objective_constraints
from .bop2_survival import BOP2SurvivalDesign, bop2_survival_design
from .bop2_survival_trial import _observe, _survival_paths, _trial_count, simulate_bop2_survival


@dataclass(frozen=True)
class BOP2SurvivalOperatingCharacteristics:
    """Monte Carlo estimates ordered as null, alternative."""

    success_probability: FloatArray
    success_mcse: FloatArray
    expected_sample_size: FloatArray
    n_trials: int


@dataclass(frozen=True)
class BOP2SurvivalOptimization:
    cutoff_scale: float
    gamma: float
    calibration_design: BOP2SurvivalDesign
    analysis_design: BOP2SurvivalDesign
    calibration_oc: BOP2SurvivalOperatingCharacteristics
    validation_oc: BOP2SurvivalOperatingCharacteristics
    analysis_oc: BOP2SurvivalOperatingCharacteristics
    parameter_pairs: int
    objective: str
    minimum_power: float | None


def _oc(
    probability: FloatArray, en: FloatArray, trials: int
) -> BOP2SurvivalOperatingCharacteristics:
    return BOP2SurvivalOperatingCharacteristics(
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / trials)),
        _owned(en),
        trials,
    )


def _grid_oc(
    probabilities: FloatArray, looks: NDArray[np.int64], scales: FloatArray, gammas: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Evaluate all candidates with the monitor's exact floating-point comparisons."""
    trials = probabilities.shape[-1]
    success = np.empty((scales.size, gammas.size))
    en = np.empty_like(success)
    for j, gamma in enumerate(gammas):
        factors = (looks / looks[-1]) ** gamma
        for start in range(0, scales.size, 128):
            stop = min(start + 128, scales.size)
            cutoffs = scales[start:stop, None] * factors
            if np.any(cutoffs == 0):
                raise ArithmeticError("posterior cutoff underflows; increase cutoff_scales")
            alive = np.ones((stop - start, trials), dtype=bool)
            enrollment = np.full(stop - start, int(looks[0]) * trials, dtype=np.int64)
            for k in range(looks.size):
                alive &= probabilities[k] > cutoffs[:, k, None]
                survivors = alive.sum(axis=-1)
                if k + 1 < looks.size:
                    enrollment += (looks[k + 1] - looks[k]) * survivors
            success[start:stop, j] = survivors / trials
            en[start:stop, j] = enrollment / trials
    return success, en


def optimize_bop2_survival(
    max_subjects: int,
    null_median: float,
    alternative_median: float,
    *,
    accrual_rate: float,
    final_followup: float,
    type1_error: float = 0.1,
    looks: ArrayLike | None = None,
    cutoff_scales: ArrayLike | None = None,
    gammas: ArrayLike | None = None,
    analysis_prior: ArrayLike | None = None,
    error_control: str = "strict",
    objective: str = "power",
    minimum_power: float | None = None,
    n_trials: int = 10000,
    n_validation: int = 10000,
    arrival: str = "fixed",
    rng: np.random.Generator | int | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2SurvivalOptimization:
    """Select on empirical error/power, then independently validate without reselection.

    Strict constrains estimated error, not true error. Closest can exceed the target.
    Uses the default null-centered weak prior for calibration; an informative analysis
    prior is evaluated separately after selection. No native optimizer parity is claimed.
    """
    floor = _objective_constraints(objective, minimum_power, error_control)
    p0, p1, alpha = (
        scalar(null_median, "null_median"),
        scalar(alternative_median, "alternative_median"),
        scalar(type1_error, "type1_error"),
    )
    if not 0 < p0 < p1 or not 0 < alpha < 1:
        raise ValueError("require 0<null_median<alternative_median and type1_error in (0,1)")
    if error_control not in ("strict", "closest"):
        raise ValueError("error_control must be strict or closest")
    scales = finite(
        np.arange(50, 100) / 100 if cutoff_scales is None else cutoff_scales, "cutoff_scales"
    )
    powers = finite(np.arange(21) / 20 if gammas is None else gammas, "gammas")
    if (
        scales.ndim != 1
        or powers.ndim != 1
        or not scales.size
        or not powers.size
        or scales.size * powers.size > 10000
        or np.any((scales <= 0) | (scales >= 1))
        or np.any((powers < 0) | (powers > 1))
    ):
        raise ValueError(
            "require nonempty 1D grids, scales in (0,1), gammas in [0,1], <=10000 pairs"
        )
    trials, validation_trials = _trial_count(n_trials), _trial_count(n_validation)
    baseline = bop2_survival_design(
        max_subjects,
        p0,
        cutoff_scale=float(scales[0]),
        gamma=float(powers[0]),
        looks=looks,
        min_subjects=min_subjects,
        cohort_size=cohort_size,
    )
    # Validate the optional prior before simulation; it must never affect selection.
    if analysis_prior is not None:
        bop2_survival_design(
            max_subjects,
            p0,
            cutoff_scale=float(scales[0]),
            gamma=float(powers[0]),
            looks=baseline.looks,
            prior=analysis_prior,
        )
    if baseline.looks.size * trials > 10000000:
        raise ValueError(
            "calibration supports at most 10000000 trial/look combinations per scenario"
        )
    generator = np.random.default_rng(rng)
    successes, enrollments = [], []
    for median in (p0, p1):
        probabilities = np.empty((baseline.looks.size, trials))
        for start, enrolled, times, fup in _survival_paths(
            baseline.max_subjects, median, accrual_rate, final_followup, trials, arrival, generator
        ):
            for k, n in enumerate(baseline.looks):
                _, events, total = _observe(enrolled, times, int(n), baseline.max_subjects, fup)
                probabilities[k, start : start + enrolled.shape[0]] = baseline.monitor(
                    events, total, int(n)
                ).success_probability
        success, en = _grid_oc(probabilities, baseline.looks, scales, powers)
        successes.append(success)
        enrollments.append(en)
    error, power = successes
    best_key: tuple[float, ...] | None = None
    best: tuple[int, int] | None = None
    for i in range(scales.size):
        for j in range(powers.size):
            if (error_control == "strict" and error[i, j] > alpha) or (
                floor is not None and power[i, j] < floor
            ):
                continue
            ranking = (
                (-power[i, j], enrollments[0][i, j])
                if objective == "power"
                else (enrollments[0][i, j], -power[i, j])
            )
            key = ((abs(error[i, j] - alpha),) if error_control == "closest" else ()) + ranking
            if best_key is None or key < best_key:
                best_key, best = key, (i, j)
    if best is None:
        raise BOP2InfeasibleError(
            "no grid candidate satisfies the empirical error and power constraints"
        )
    i, j = best
    scale, gamma = float(scales[i]), float(powers[j])
    calibration = bop2_survival_design(
        max_subjects, p0, cutoff_scale=scale, gamma=gamma, looks=baseline.looks
    )
    analysis = (
        calibration
        if analysis_prior is None
        else bop2_survival_design(
            max_subjects,
            p0,
            cutoff_scale=scale,
            gamma=gamma,
            looks=baseline.looks,
            prior=analysis_prior,
        )
    )

    def validate(design: BOP2SurvivalDesign) -> BOP2SurvivalOperatingCharacteristics:
        results = [
            simulate_bop2_survival(
                design,
                median,
                accrual_rate=accrual_rate,
                final_followup=final_followup,
                n_trials=validation_trials,
                arrival=arrival,
                rng=generator,
            )
            for median in (p0, p1)
        ]
        return _oc(
            np.array([r.success_probability for r in results]),
            np.array([r.expected_sample_size for r in results]),
            validation_trials,
        )

    validation = validate(calibration)
    return BOP2SurvivalOptimization(
        scale,
        gamma,
        calibration,
        analysis,
        _oc(
            np.array([s[i, j] for s in successes]), np.array([e[i, j] for e in enrollments]), trials
        ),
        validation,
        validation if analysis is calibration else validate(analysis),
        int(scales.size * powers.size),
        objective,
        floor,
    )
