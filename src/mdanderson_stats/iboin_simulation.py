"""Bounded serial complete-outcome simulation for iBOIN."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, finite, scalar
from .iboin import IBOINDesign
from .iboin_final import (
    IBOINSelection,
    _validate_selection_options,
    select_iboin_trial_mtd,
)
from .iboin_trial import (
    IBOINTrialReplay,
    _conduct_settings,
    _IBOINConductState,
)

_MAX_REPETITIONS = 100_000
_DEFAULT_TOTAL_WORK = 1_000_000
_MAX_TOTAL_WORK = 10_000_000
_MAX_TOTAL_STORAGE_BYTES = 128 * 1024 * 1024
_MAX_HISTORY_CELLS = 500_000


@dataclass(frozen=True)
class IBOINSimulatedTrial:
    seed: int
    replay: IBOINTrialReplay
    selection: IBOINSelection


@dataclass(frozen=True)
class IBOINOperatingCharacteristics:
    repetitions: int
    selection_probability: FloatArray
    selection_mcse: FloatArray
    stop_reason_labels: tuple[str, ...]
    stop_reason_probability: FloatArray
    stop_reason_mcse: FloatArray
    mean_patients_by_dose: FloatArray
    mean_patients_mcse: FloatArray
    mean_dlt_by_dose: FloatArray
    mean_dlt_mcse: FloatArray
    mean_grade2_by_dose: FloatArray
    mean_grade2_mcse: FloatArray
    mean_total_patients: float
    total_patients_mcse: float
    total_patient_quantiles: FloatArray
    selected_dose: NDArray[np.int64]
    stop_reason: NDArray[np.str_]
    total_patients: NDArray[np.int64]
    trial_seeds: NDArray[np.uint64]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _probabilities(
    design: IBOINDesign, grade2_probability: ArrayLike, dlt_probability: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(design, IBOINDesign):
        raise TypeError("design must be an IBOINDesign")
    if np.iscomplexobj(grade2_probability) or np.iscomplexobj(dlt_probability):
        raise ValueError("severity probabilities must be real")
    grade2 = finite(grade2_probability, "grade2_probability")
    dlt = finite(dlt_probability, "dlt_probability")
    expected = (np.asarray(design.skeleton).size,)
    if (
        grade2.shape != expected
        or dlt.shape != expected
        or np.any(grade2 < 0)
        or np.any(dlt < 0)
        or np.any(grade2 + dlt > 1)
    ):
        raise ValueError(
            "grade2_probability and dlt_probability must match doses, be nonnegative, "
            "and sum to at most one at every dose"
        )
    return grade2, dlt


def _run_trial(
    design: IBOINDesign,
    grade2_probability: FloatArray,
    dlt_probability: FloatArray,
    *,
    cohort_size: int,
    starting_dose: int,
    titration: bool,
    titration_cap: int | None,
    max_patients: int,
    seed: int,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    eligible_doses: ArrayLike | None,
    tie_policy: Literal["lowest", "highest"],
    enforce_deescalation_boundary: bool,
    record_history: bool,
) -> IBOINSimulatedTrial:
    settings = _conduct_settings(
        design, cohort_size, starting_dose, titration, titration_cap, max_patients
    )
    n_doses, cohort_count, start_count, use_titration, cap, max_count = settings
    if max_count * n_doses > _MAX_HISTORY_CELLS:
        raise ValueError("maximum simulated outcome-by-dose history exceeds its bounded limit")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    generator = np.random.default_rng(seed_value)
    state = _IBOINConductState(
        design,
        n_doses,
        cohort_count,
        start_count,
        use_titration,
        cap,
        max_count,
        record_history=record_history,
    )
    for _ in range(max_count):
        if state.next_dose is None:
            break
        dose = state.next_dose - 1
        value = generator.random()
        dlt = int(value < dlt_probability[dose])
        grade2 = int(
            dlt_probability[dose] <= value < dlt_probability[dose] + grade2_probability[dose]
        )
        state.observe(dlt, grade2)
    replay = state.result()
    selection = select_iboin_trial_mtd(
        design,
        replay,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
    )
    return IBOINSimulatedTrial(seed_value, replay, selection)


def simulate_iboin_trial(
    design: IBOINDesign,
    grade2_probability: ArrayLike,
    dlt_probability: ArrayLike,
    *,
    cohort_size: int,
    max_patients: int,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    seed: int,
    starting_dose: int = 1,
    titration: bool = True,
    titration_cap: int | None = None,
    eligible_doses: ArrayLike | None = None,
    tie_policy: Literal["lowest", "highest"] = "lowest",
    enforce_deescalation_boundary: bool = False,
) -> IBOINSimulatedTrial:
    """Simulate one trial with mutually exclusive grade-2/DLT outcome classes."""
    grade2, dlt = _probabilities(design, grade2_probability, dlt_probability)
    _validate_selection_options(
        design,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
        require_all_custom_weights=True,
    )
    return _run_trial(
        design,
        grade2,
        dlt,
        cohort_size=cohort_size,
        starting_dose=starting_dose,
        titration=titration,
        titration_cap=titration_cap,
        max_patients=max_patients,
        seed=seed,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
        record_history=True,
    )


def simulate_iboin(
    design: IBOINDesign,
    grade2_probability: ArrayLike,
    dlt_probability: ArrayLike,
    *,
    cohort_size: int,
    max_patients: int,
    repetitions: int,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    seed: int | None = None,
    trial_seeds: ArrayLike | None = None,
    starting_dose: int = 1,
    titration: bool = True,
    titration_cap: int | None = None,
    eligible_doses: ArrayLike | None = None,
    tie_policy: Literal["lowest", "highest"] = "lowest",
    enforce_deescalation_boundary: bool = False,
    max_total_work: int = _DEFAULT_TOTAL_WORK,
    max_total_storage_bytes: int = _MAX_TOTAL_STORAGE_BYTES,
) -> IBOINOperatingCharacteristics:
    """Run bounded serial trials; seed arrays replay trials independently.

    Work is conservatively bounded by repetitions times maximum enrollment
    times the number of doses plus one, covering conduct and boundary scans.
    Only aggregate arrays survive each trial iteration.
    """
    grade2, dlt = _probabilities(design, grade2_probability, dlt_probability)
    _validate_selection_options(
        design,
        prior_mode=prior_mode,
        isotonic_weights=isotonic_weights,
        eligible_doses=eligible_doses,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
        require_all_custom_weights=True,
    )
    n_doses, cohort_count, start_count, use_titration, cap, max_count = _conduct_settings(
        design, cohort_size, starting_dose, titration, titration_cap, max_patients
    )
    reps = scalar(repetitions, "repetitions")
    work_limit = scalar(max_total_work, "max_total_work")
    storage_limit = scalar(max_total_storage_bytes, "max_total_storage_bytes")
    if (
        isinstance(repetitions, (bool, np.bool_))
        or reps != int(reps)
        or not 1 <= reps <= _MAX_REPETITIONS
        or max_count * n_doses > _MAX_HISTORY_CELLS
        or isinstance(max_total_work, (bool, np.bool_))
        or work_limit != int(work_limit)
        or not 1 <= work_limit <= _MAX_TOTAL_WORK
        or isinstance(max_total_storage_bytes, (bool, np.bool_))
        or storage_limit != int(storage_limit)
        or not 1 <= storage_limit <= _MAX_TOTAL_STORAGE_BYTES
    ):
        raise ValueError("invalid repetitions, history size, or simulation resource limits")
    reps_int = int(reps)
    total_work = reps_int * max_count * (n_doses + 1)
    if total_work > work_limit:
        raise ValueError("worst-case simulation work exceeds max_total_work")
    storage = reps_int * 256 + max_count * n_doses * 160 + n_doses * 512 + 4096
    if storage > storage_limit:
        raise ValueError("simulated peak storage estimate exceeds max_total_storage_bytes")
    if (seed is None) == (trial_seeds is None):
        raise ValueError("provide exactly one of seed or trial_seeds")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be a nonnegative integer")
        root_seed = int(seed)
        if root_seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        seeds = np.empty(reps_int, dtype=np.uint64)
    else:
        raw_seeds = np.asarray(trial_seeds)
        if (
            raw_seeds.shape != (reps_int,)
            or raw_seeds.dtype.kind not in "ui"
            or (raw_seeds.dtype.kind == "i" and np.any(raw_seeds < 0))
        ):
            raise ValueError(
                "trial_seeds must be a nonnegative integer vector matching repetitions"
            )
        seeds = np.array(raw_seeds, dtype=np.uint64, copy=True)

    selected = np.zeros(reps_int, dtype=np.int64)
    stop_labels = ("stop_safety", "stop_precision", "stop_max_patients")
    stop_reasons = np.empty(reps_int, dtype="U20")
    patient_totals = np.empty(reps_int, dtype=np.int64)
    patients_sum = np.zeros(n_doses, dtype=np.float64)
    dlt_sum = np.zeros(n_doses, dtype=np.float64)
    grade2_sum = np.zeros(n_doses, dtype=np.float64)
    patients_sumsq = np.zeros(n_doses, dtype=np.float64)
    dlt_sumsq = np.zeros(n_doses, dtype=np.float64)
    grade2_sumsq = np.zeros(n_doses, dtype=np.float64)
    patient_total_sum = 0.0
    patient_total_sumsq = 0.0
    seed_sequence = np.random.SeedSequence(root_seed) if seed is not None else None
    for index in range(reps_int):
        if seed is not None:
            assert seed_sequence is not None
            seeds[index] = seed_sequence.spawn(1)[0].generate_state(1, dtype=np.uint64)[0]
        result = _run_trial(
            design,
            grade2,
            dlt,
            cohort_size=cohort_count,
            starting_dose=start_count,
            titration=use_titration,
            titration_cap=cap if use_titration else None,
            max_patients=max_count,
            seed=int(seeds[index]),
            prior_mode=prior_mode,
            isotonic_weights=isotonic_weights,
            eligible_doses=eligible_doses,
            tie_policy=tie_policy,
            enforce_deescalation_boundary=enforce_deescalation_boundary,
            record_history=False,
        )
        replay = result.replay
        selected[index] = (
            0 if result.selection.selected_dose is None else result.selection.selected_dose
        )
        if replay.stop_reason not in stop_labels:
            raise RuntimeError(f"unexpected terminal iBOIN stop reason: {replay.stop_reason!r}")
        stop_reasons[index] = replay.stop_reason
        patient_totals[index] = replay.dlt.size
        patients_sum += replay.patients
        dlt_sum += replay.toxicities
        grade2_sum += replay.grade2_toxicities
        patients_sumsq += replay.patients.astype(np.float64) ** 2
        dlt_sumsq += replay.toxicities.astype(np.float64) ** 2
        grade2_sumsq += replay.grade2_toxicities.astype(np.float64) ** 2
        patient_total_sum += replay.dlt.size
        patient_total_sumsq += float(replay.dlt.size) ** 2
        del result, replay

    select_prob = np.bincount(selected, minlength=n_doses + 1) / reps_int
    select_mcse = (
        np.full(select_prob.shape, np.nan)
        if reps_int == 1
        else np.sqrt(select_prob * (1 - select_prob) / reps_int)
    )
    stop_prob = np.asarray([np.mean(stop_reasons == label) for label in stop_labels])
    stop_mcse = (
        np.full(stop_prob.shape, np.nan)
        if reps_int == 1
        else np.sqrt(stop_prob * (1 - stop_prob) / reps_int)
    )

    def mean_mcse(total: np.ndarray, square_total: np.ndarray) -> FloatArray:
        if reps_int == 1:
            return np.full(total.shape, np.nan)
        centered = np.maximum(0.0, square_total - total * total / reps_int)
        return np.sqrt(centered / (reps_int * (reps_int - 1)))

    patient_mcse = mean_mcse(patients_sum, patients_sumsq)
    dlt_mcse = mean_mcse(dlt_sum, dlt_sumsq)
    grade2_mcse = mean_mcse(grade2_sum, grade2_sumsq)
    total_patient_mcse = float(
        mean_mcse(np.asarray([patient_total_sum]), np.asarray([patient_total_sumsq]))[0]
    )
    return IBOINOperatingCharacteristics(
        reps_int,
        _readonly(select_prob),
        _readonly(select_mcse),
        stop_labels,
        _readonly(stop_prob),
        _readonly(stop_mcse),
        _readonly(patients_sum / reps_int),
        _readonly(patient_mcse),
        _readonly(dlt_sum / reps_int),
        _readonly(dlt_mcse),
        _readonly(grade2_sum / reps_int),
        _readonly(grade2_mcse),
        float(np.mean(patient_totals)),
        total_patient_mcse,
        _readonly(np.quantile(patient_totals, [0.1, 0.5, 0.9])),
        _readonly(selected),
        _readonly(stop_reasons),
        _readonly(patient_totals),
        _readonly(seeds),
    )
