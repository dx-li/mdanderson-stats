"""Bounded serial operating-characteristic simulation for TITE-BOIN12."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .boin12 import BOIN12Design
from .tite_boin12 import _readonly
from .tite_boin12_bda import _prior as _bda_prior
from .tite_boin12_calendar import run_tite_boin12_calendar_trial
from .uboin_simulation import uboin_gumbel_probabilities

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

_MAX_DOSES = 100
_MAX_PATIENTS = 1000
_MAX_TRIALS = 10_000
_MAX_WORK = 100_000_000
_MAX_RETAINED_VALUES = 5_000_000
_MAX_SCRATCH_VALUES = 5_000_000
_MAX_BDA_FIT_WORK = 20_000_000
_MAX_BDA_RETAINED_VALUES = 2_000_000


@dataclass(frozen=True)
class TITEBOIN12Simulation:
    """Compact trial outcomes and across-trial operating characteristics.

    ``selected_obd`` uses zero for no selection. Selection probabilities and
    standard errors have indices ``0..n_doses``. Count means and MCSEs are
    computed across independent trials, not pooled across patients. The three
    returned replay seeds are ordered arrival, outcome/timing, and sampler;
    passing them back as ``trial_seed_triplets`` replays those same streams.
    Durations and accrual-stop times are measured from calendar origin zero.
    MCSEs are NaN for a single simulated trial.
    """

    selected_obd: IntArray
    stop_reason: tuple[str, ...]
    trial_seed_triplets: NDArray[np.uint64]
    patients_by_dose: IntArray
    toxicities_by_dose: IntArray
    efficacies_by_dose: IntArray
    final_duration: FloatArray
    accrual_stop_time: FloatArray
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    patients_mcse: FloatArray
    mean_toxicities: FloatArray
    toxicities_mcse: FloatArray
    mean_efficacies: FloatArray
    efficacies_mcse: FloatArray
    mean_duration: float
    duration_mcse: float
    mean_accrual_stop_time: float
    accrual_stop_time_mcse: float
    early_stop_probability: float
    early_stop_mcse: float

    @property
    def no_selection_probability(self) -> float:
        return float(self.selection_probability[0])

    @property
    def no_selection_mcse(self) -> float:
        return float(self.selection_mcse[0])


def tite_boin12_gumbel_probabilities(
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    *,
    association: float,
) -> FloatArray:
    """Convert Gumbel marginals to TITE-BOIN12's explicit four-cell order.

    The U-BOIN helper returns ``(dose, efficacy 0/1, toxicity 0/1)``. This
    function returns ``(noT/E, noT/noE, T/E, T/noE)`` to match the TITE BDA
    cell convention.
    """
    source = uboin_gumbel_probabilities(toxicity, efficacy, association=association)
    result = np.column_stack((source[:, 1, 0], source[:, 0, 0], source[:, 1, 1], source[:, 0, 1]))
    return _readonly(result)


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must lie in [{low},{high}]")
    return int(value)


def _probabilities(value: ArrayLike, doses: int | None = None) -> FloatArray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 4 or not 1 <= raw.shape[0] <= _MAX_DOSES:
        raise ValueError("joint_probabilities must have shape (1..100 doses,4 cells)")
    if doses is not None and raw.shape[0] != doses:
        raise ValueError("joint_probabilities dose count must match the design")
    if np.iscomplexobj(raw):
        raise ValueError("joint_probabilities must be real-valued")
    probabilities = np.asarray(raw, dtype=float)
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("joint_probabilities must be finite and nonnegative")
    totals = np.sum(probabilities, axis=1)
    if np.any(~np.isfinite(totals)) or np.any(np.abs(totals - 1.0) > 1e-12):
        raise ValueError("each joint-probability row must sum to one within 1e-12")
    return probabilities / totals[:, None]


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if not 0 <= int(value) <= np.iinfo(np.uint64).max:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    return int(value)


def _triplets(value: ArrayLike, trials: int) -> NDArray[np.uint64]:
    raw = np.asarray(value)
    if raw.shape != (trials, 3) or raw.dtype.kind not in "iu":
        raise ValueError("trial_seed_triplets must be an integer array with shape (trials,3)")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError("trial_seed_triplets must be nonnegative")
    if raw.dtype.itemsize > 8 and np.any(raw > np.iinfo(np.uint64).max):
        raise ValueError("trial_seed_triplets must fit in unsigned 64-bit integers")
    return np.asarray(raw, dtype=np.uint64).copy()


def _mean_mcse(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = np.mean(values, axis=0)
    if values.shape[0] == 1:
        error = np.full_like(mean, np.nan, dtype=float)
    else:
        error = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    if np.any(~np.isfinite(mean)):
        raise ArithmeticError("trial summary mean or MCSE is not representable")
    return mean, error


def _time_mean_mcse(values: FloatArray) -> tuple[float, float]:
    scale = float(np.max(values))
    if scale == 0:
        return 0.0, float("nan") if values.size == 1 else 0.0
    scaled = values / scale
    mean = scale * float(np.mean(scaled))
    error = (
        float("nan")
        if values.size == 1
        else scale * float(np.std(scaled, ddof=1) / np.sqrt(values.size))
    )
    if not np.isfinite(mean) or (values.size > 1 and not np.isfinite(error)):
        raise ArithmeticError("time mean or MCSE is not representable")
    return mean, error


def _trial_seeds(seed: int, trials: int) -> NDArray[np.uint64]:
    result = np.empty((trials, 3), dtype=np.uint64)
    for trial in range(trials):
        for stream in range(3):
            sequence = np.random.SeedSequence(seed, spawn_key=(trial, stream))
            result[trial, stream] = sequence.generate_state(1, dtype=np.uint64)[0]
    return result


def _potential_tapes(
    rng: np.random.Generator,
    probabilities: FloatArray,
    patients: int,
    toxicity_window: float,
    efficacy_window: float,
) -> tuple[FloatArray, FloatArray]:
    doses = probabilities.shape[0]
    tox_delays = np.full((patients, doses), np.inf, dtype=float)
    eff_delays = np.full((patients, doses), np.inf, dtype=float)
    for dose in range(doses):
        cells = rng.choice(4, size=patients, p=probabilities[dose])
        tox_event = cells >= 2
        eff_event = (cells == 0) | (cells == 2)
        if np.any(tox_event):
            tox_delays[tox_event, dose] = rng.random(int(np.sum(tox_event))) * toxicity_window
        if np.any(eff_event):
            eff_delays[eff_event, dose] = rng.random(int(np.sum(eff_event))) * efficacy_window
    return tox_delays, eff_delays


def simulate_tite_boin12(
    design: BOIN12Design,
    joint_probabilities: ArrayLike,
    accrual_rate: float,
    *,
    toxicity_window: float,
    efficacy_window: float,
    cohorts: int = 6,
    cohort_size: int = 3,
    trials: int = 100,
    start_dose: int = 1,
    arrival: Literal["fixed", "exponential"] = "fixed",
    event_time_policy: Literal["uniform"] = "uniform",
    decision_lag: float = 0.0,
    method: Literal["al", "bda"] = "al",
    prior_concentrations: ArrayLike | None = None,
    draws: int | None = None,
    warmup: int | None = None,
    chains: int | None = None,
    bda_max_work: int = _MAX_BDA_FIT_WORK,
    max_pending_toxicity: float = 0.5,
    max_pending_efficacy: float = 0.5,
    run_in_3plus3: bool = False,
    seed: int | None = None,
    trial_seed_triplets: ArrayLike | None = None,
    max_work: int = _MAX_WORK,
) -> TITEBOIN12Simulation:
    """Simulate TITE-BOIN12 OC serially under explicit joint outcome truths.

    ``joint_probabilities[d]`` is ordered ``(noT/E, noT/noE, T/E, T/noE)``.
    ``arrival='fixed'`` uses gaps ``1/accrual_rate`` for every patient,
    including the first; exponential arrival samples every gap independently
    with that mean. Event delays are independent uniforms over each endpoint's
    assessment window conditional on its event indicator. These are explicit
    Python timing policies, not native defaults. Each trial uses independent
    arrival, outcome/timing and sampler seeds. Supply either ``seed`` or
    ``trial_seed_triplets`` to make the run reproducible; the returned triplets
    can be passed back for exact trial-stream replay.

    The simulation retains only per-trial dose counts, selection/stop,
    durations and replay seeds. Mean count MCSEs use trial-level standard
    errors. Duration is final endpoint ascertainment time from calendar origin
    zero; accrual-stop time is returned separately.
    """
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    grid = _probabilities(joint_probabilities)
    doses = grid.shape[0]
    if arrival not in ("fixed", "exponential"):
        raise ValueError("arrival must be 'fixed' or 'exponential'")
    if event_time_policy != "uniform":
        raise ValueError("event_time_policy currently supports only 'uniform'")
    if method not in ("al", "bda"):
        raise ValueError("method must be 'al' or 'bda'")
    rate = float(accrual_rate)
    tox_window = float(toxicity_window)
    eff_window = float(efficacy_window)
    lag = float(decision_lag)
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(1.0 / rate) or 1.0 / rate <= 0:
        raise ValueError("accrual_rate must be positive with a finite mean interarrival time")
    fixed_gap = 1.0 / rate
    if (
        not np.isfinite(tox_window)
        or tox_window <= 0
        or not np.isfinite(eff_window)
        or eff_window <= 0
    ):
        raise ValueError("assessment windows must be finite and positive")
    if not np.isfinite(lag) or lag < 0:
        raise ValueError("decision_lag must be finite and nonnegative")
    n_cohorts = _integer(cohorts, "cohorts", 1, _MAX_PATIENTS)
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    repetitions = _integer(trials, "trials", 1, _MAX_TRIALS)
    start = _integer(start_dose, "start_dose", 1, doses)
    maximum = n_cohorts * size
    if maximum > _MAX_PATIENTS:
        raise ValueError("cohorts * cohort_size must be at most 1000")
    if start > doses:
        raise ValueError("start_dose must not exceed the number of truth doses")
    if maximum % size:
        raise ValueError("the planned patient count must contain complete cohorts")
    pending_tox = float(max_pending_toxicity)
    pending_eff = float(max_pending_efficacy)
    if (
        not np.isfinite(pending_tox)
        or not 0 <= pending_tox <= 1
        or not np.isfinite(pending_eff)
        or not 0 <= pending_eff <= 1
    ):
        raise ValueError("pending thresholds must be finite values in [0,1]")
    if not isinstance(run_in_3plus3, (bool, np.bool_)):
        raise ValueError("run_in_3plus3 must be a boolean")
    if run_in_3plus3 and design.toxicity_limit != 0.25:
        raise ValueError("the 3+3 run-in is available only when toxicity_limit is 0.25")
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)):
        raise ValueError("max_work must be an integer")
    if not 1 <= max_work <= _MAX_WORK:
        raise ValueError(f"max_work must lie in [1,{_MAX_WORK}]")

    if (seed is None) == (trial_seed_triplets is None):
        raise ValueError("supply exactly one of seed or trial_seed_triplets")
    master_seed = None if seed is None else _seed(seed)
    replay_seeds = (
        None if trial_seed_triplets is None else _triplets(trial_seed_triplets, repetitions)
    )

    if method == "bda":
        if prior_concentrations is None or draws is None or warmup is None or chains is None:
            raise ValueError(
                "BDA requires an explicit prior_concentrations, draws, warmup, and chains"
            )
        prior = _bda_prior(prior_concentrations, doses)
        draws_n = _integer(draws, "draws", 20, 1_000_000)
        warmup_n = _integer(warmup, "warmup", 0, 1_000_000)
        chains_n = _integer(chains, "chains", 2, 1_000_000)
        bda_limit = _integer(bda_max_work, "bda_max_work", 1, _MAX_BDA_FIT_WORK)
        fit_work = chains_n * (warmup_n + draws_n + 1) * (maximum + doses)
        if fit_work > bda_limit:
            raise ValueError(
                f"per-fit BDA work estimate {fit_work} exceeds bda_max_work={bda_limit}"
            )
        fit_values = chains_n * draws_n * doses * 40
        if fit_values > _MAX_BDA_RETAINED_VALUES:
            raise ValueError(
                f"BDA fit requires {fit_values} retained/scratch values; "
                f"limit is {_MAX_BDA_RETAINED_VALUES}"
            )
    elif (
        prior_concentrations is not None
        or draws is not None
        or warmup is not None
        or chains is not None
    ):
        raise ValueError("BDA prior and sampler settings are only used with method='bda'")
    else:
        prior = None
        draws_n = warmup_n = chains_n = 0
        bda_limit = _MAX_BDA_FIT_WORK

    look_bound = 2 * maximum + n_cohorts
    per_trial_calendar_work = 2 * look_bound * maximum
    per_trial_fit_work = max(0, n_cohorts - 1) * fit_work if method == "bda" else 0
    total_work = repetitions * (per_trial_calendar_work + per_trial_fit_work + maximum * doses * 4)
    if total_work > max_work:
        raise ValueError(f"total work estimate {total_work} exceeds max_work={max_work}")
    retained_values = repetitions * (10 * doses + 12)
    scratch_values = (
        2 * maximum * doses
        + 20 * maximum
        + 12 * look_bound * doses
        + (fit_values if method == "bda" else 0)
    )
    if retained_values > _MAX_RETAINED_VALUES:
        raise ValueError(
            f"retained trial results require {retained_values} values; "
            f"limit is {_MAX_RETAINED_VALUES}"
        )
    if scratch_values > _MAX_SCRATCH_VALUES:
        raise ValueError(
            f"single-trial scratch requires {scratch_values} values; limit is {_MAX_SCRATCH_VALUES}"
        )
    if not 1 <= per_trial_calendar_work + per_trial_fit_work <= _MAX_WORK:
        raise ValueError("single-trial calendar/sampler work exceeds the supported limit")

    if arrival == "fixed":
        # Check the deterministic, no-suspension schedule and every endpoint
        # deadline before generating trial seeds. Random exponential calendars
        # receive the same checks inside the replay after their gaps are drawn.
        clock = 0.0
        for patient_index in range(maximum):
            previous = clock
            clock += fixed_gap
            if not np.isfinite(clock) or clock <= previous:
                raise ArithmeticError("fixed accrual calendar is not representable")
            if patient_index and patient_index % size == 0:
                previous = clock
                clock += lag
                if not np.isfinite(clock) or (lag > 0 and clock <= previous):
                    raise ArithmeticError("fixed decision-lag calendar is not representable")
            for window in (tox_window, eff_window):
                deadline = clock + window
                if not np.isfinite(deadline) or deadline <= clock:
                    raise ArithmeticError("fixed endpoint deadline is not representable")

    # Seed creation follows all workload, storage, truth and sampler preflight.
    if replay_seeds is None:
        assert master_seed is not None
        replay_seeds = _trial_seeds(master_seed, repetitions)
    patients = np.zeros((repetitions, doses), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    efficacies = np.zeros_like(patients)
    selected = np.zeros(repetitions, dtype=np.int64)
    durations = np.empty(repetitions, dtype=float)
    accrual_stop = np.empty(repetitions, dtype=float)
    reasons: list[str] = []

    for trial in range(repetitions):
        arrival_rng = np.random.default_rng(int(replay_seeds[trial, 0]))
        outcome_rng = np.random.default_rng(int(replay_seeds[trial, 1]))
        sampler_rng = np.random.default_rng(int(replay_seeds[trial, 2]))
        gaps = (
            np.full(maximum, fixed_gap, dtype=float)
            if arrival == "fixed"
            else np.asarray(arrival_rng.exponential(fixed_gap, size=maximum), dtype=float)
        )
        tox_tape, eff_tape = _potential_tapes(outcome_rng, grid, maximum, tox_window, eff_window)
        trial_result = run_tite_boin12_calendar_trial(
            design,
            gaps,
            tox_tape,
            eff_tape,
            toxicity_window=tox_window,
            efficacy_window=eff_window,
            cohort_size=size,
            start_dose=start,
            method=method,
            decision_lag=lag,
            prior_concentrations=prior,
            rng=sampler_rng if method == "bda" else None,
            draws=draws_n if method == "bda" else None,
            warmup=warmup_n if method == "bda" else None,
            chains=chains_n if method == "bda" else None,
            bda_max_work=bda_limit,
            max_calendar_work=per_trial_calendar_work + per_trial_fit_work,
            max_pending_toxicity=pending_tox,
            max_pending_efficacy=pending_eff,
            run_in_3plus3=run_in_3plus3,
        )
        patients[trial] = trial_result.patients
        toxicities[trial] = trial_result.observed_toxicities
        efficacies[trial] = trial_result.observed_efficacies
        selected[trial] = 0 if trial_result.selected_obd is None else trial_result.selected_obd
        durations[trial] = trial_result.final_time
        accrual_stop[trial] = trial_result.accrual_stop_time
        reasons.append(trial_result.stop_reason)
        del trial_result, tox_tape, eff_tape, gaps, arrival_rng, outcome_rng, sampler_rng

    probabilities = np.bincount(selected, minlength=doses + 1).astype(float) / repetitions
    selection_mcse = np.sqrt(probabilities * (1.0 - probabilities) / repetitions)
    if repetitions == 1:
        selection_mcse.fill(np.nan)
    mean_patients, patients_mcse = _mean_mcse(patients.astype(float))
    mean_tox, tox_mcse = _mean_mcse(toxicities.astype(float))
    mean_eff, eff_mcse = _mean_mcse(efficacies.astype(float))
    mean_duration, duration_mcse = _time_mean_mcse(durations)
    mean_accrual, accrual_mcse = _time_mean_mcse(accrual_stop)
    return TITEBOIN12Simulation(
        _readonly(selected, np.int64),
        tuple(reasons),
        _readonly(replay_seeds, np.uint64),
        _readonly(patients, np.int64),
        _readonly(toxicities, np.int64),
        _readonly(efficacies, np.int64),
        _readonly(durations),
        _readonly(accrual_stop),
        _readonly(probabilities),
        _readonly(selection_mcse),
        _readonly(mean_patients),
        _readonly(patients_mcse),
        _readonly(mean_tox),
        _readonly(tox_mcse),
        _readonly(mean_eff),
        _readonly(eff_mcse),
        mean_duration,
        duration_mcse,
        mean_accrual,
        accrual_mcse,
        float(np.mean(np.asarray(reasons) != "maximum_cohorts")),
        (
            float("nan")
            if repetitions == 1
            else float(
                np.sqrt(
                    np.mean(np.asarray(reasons) != "maximum_cohorts")
                    * (1.0 - np.mean(np.asarray(reasons) != "maximum_cohorts"))
                    / repetitions
                )
            )
        ),
    )
