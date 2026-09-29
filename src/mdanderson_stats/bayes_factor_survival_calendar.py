"""Explicit-calendar replay and bounded simulation for BayesFactorTTE."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bayes_factor_survival import BayesFactorSurvivalState, bayes_factor_survival
from .beta_binomial import _owned

_MAX_TRIAL_CHECKS = 1_000
_DEFAULT_TOTAL_WORK = 20_000
_MAX_TOTAL_WORK = 1_000_000
_DEFAULT_TOTAL_QUADRATURES = 1_000
_MAX_TOTAL_QUADRATURES = 100_000
_MAX_TOTAL_STORAGE_BYTES = 128 * 1024 * 1024
_KERNEL_SCRATCH_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class BayesFactorSurvivalTrial:
    """Calendar replay; times use one caller-selected unit."""

    enrollment_times: FloatArray
    event_durations: FloatArray
    censor_durations: FloatArray
    event_observed: np.ndarray
    followup_times: FloatArray
    monitor_history: tuple[tuple[float, BayesFactorSurvivalState], ...]
    early_monitor: BayesFactorSurvivalState | None
    stop_reason: str
    accrual_stop_time: float
    final_time: float
    final_monitor: BayesFactorSurvivalState


def _vector(value: ArrayLike, name: str) -> FloatArray:
    result = finite(value, name)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _censor_vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or np.any(np.isnan(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must be one-dimensional, positive, and may contain +inf")
    return result


def _duration_vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or np.any(np.isnan(result)) or np.any(result <= 0):
        raise ValueError(f"{name} must be one-dimensional, positive, and may contain +inf")
    return result


def _ledger(
    arrivals: np.ndarray,
    events: np.ndarray,
    censors: np.ndarray,
    now: float,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    elapsed = np.maximum(0.0, now - arrivals)
    followup = np.minimum(elapsed, np.minimum(events, censors))
    with np.errstate(over="ignore"):
        event_calendar_time = arrivals + events
        censor_calendar_time = arrivals + censors
    if np.any(
        np.isfinite(events)
        & (~np.isfinite(event_calendar_time) | (event_calendar_time <= arrivals))
    ):
        raise ArithmeticError("positive event duration is not representable on the calendar")
    if np.any(
        np.isfinite(censors)
        & (~np.isfinite(censor_calendar_time) | (censor_calendar_time <= arrivals))
    ):
        raise ArithmeticError("positive censor duration is not representable on the calendar")
    # Arrival/event ties are decided on absolute calendar values. An event tied
    # with its independent censor duration is counted as observed.
    observed = (events <= censors) & (event_calendar_time <= now)
    followup[observed] = events[observed]
    censor_observed = (censors < events) & (censor_calendar_time <= now)
    followup[censor_observed] = censors[censor_observed]
    exposure = float(np.sum(followup))
    if not np.isfinite(exposure):
        raise ArithmeticError("total time-on-test is not representable")
    return int(observed.sum()), exposure, observed, followup


def bayes_factor_survival_trial(
    enrollment_times: ArrayLike,
    event_durations: ArrayLike,
    *,
    check_times: ArrayLike,
    final_time: float,
    null_median: float,
    alternative_median_mode: float,
    censor_durations: ArrayLike | None = None,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
) -> BayesFactorSurvivalTrial:
    """Replay a supplied enrollment/event tape at explicit interim checks.

    At each check, arrivals at that exact time enroll first; events tied with
    the check are observed, and the Bayes-factor rule is then evaluated. An
    event tied with a patient's censor duration is counted as an event. The
    first interim decision stops accrual; later arrivals/checks are ignored.
    ``final_time`` remains prespecified after an early stop, allowing the
    enrolled patients additional follow-up. A check exactly at final_time is
    reserved for the final monitor and is not evaluated twice.
    """
    arrivals = _vector(enrollment_times, "enrollment_times")
    durations = _duration_vector(event_durations, "event_durations")
    checks = _vector(check_times, "check_times")
    final = scalar(final_time, "final_time")
    if arrivals.size == 0 or arrivals.size > 500 or durations.shape != arrivals.shape:
        raise ValueError("require matching nonempty enrollment/event tapes of at most 500")
    if np.any(np.diff(arrivals) <= 0) or np.any(arrivals < 0) or np.any(durations <= 0):
        raise ValueError(
            "arrivals must increase from nonnegative time; event durations must be positive"
        )
    if checks.size > _MAX_TRIAL_CHECKS or np.any(np.diff(checks) <= 0) or np.any(checks < 0):
        raise ValueError("check_times must be strictly increasing, nonnegative, and at most 1000")
    if not np.isfinite(final) or final < arrivals[-1] or (checks.size and checks[-1] > final):
        raise ValueError("final_time must cover the planned arrivals and all checks")
    if censor_durations is None:
        censors = np.full(arrivals.shape, np.inf)
    else:
        censors = _censor_vector(censor_durations, "censor_durations")
        if censors.shape != arrivals.shape:
            raise ValueError("censor_durations must be positive and match the enrollment tape")

    history: list[tuple[float, BayesFactorSurvivalState]] = []
    enrolled = 0
    early: BayesFactorSurvivalState | None = None
    stop_reason = "maximum_enrollment"
    stop_time = float(arrivals[-1])
    for check in checks:
        check_time = float(check)
        if check_time == final:
            break
        while enrolled < arrivals.size and arrivals[enrolled] <= check_time:
            enrolled += 1
        if enrolled == 0:
            continue
        d, exposure, _, _ = _ledger(
            arrivals[:enrolled], durations[:enrolled], censors[:enrolled], check_time
        )
        state = bayes_factor_survival(
            d,
            exposure,
            null_median=null_median,
            alternative_median_mode=alternative_median_mode,
            inferiority_cutoff=inferiority_cutoff,
            superiority_cutoff=superiority_cutoff,
        )
        history.append((check_time, state))
        decision = str(state.decision)
        if decision in ("inferiority", "superiority"):
            early = state
            stop_reason = decision
            stop_time = min(check_time, float(arrivals[-1]))
            break

    if early is None:
        enrolled = arrivals.size
    final_arrivals = arrivals[:enrolled]
    final_durations = durations[:enrolled]
    final_censors = censors[:enrolled]
    final_events, final_exposure, observed, followup = _ledger(
        final_arrivals, final_durations, final_censors, final
    )
    final_state = bayes_factor_survival(
        final_events,
        final_exposure,
        null_median=null_median,
        alternative_median_mode=alternative_median_mode,
        inferiority_cutoff=inferiority_cutoff,
        superiority_cutoff=superiority_cutoff,
        final=True,
    )
    return BayesFactorSurvivalTrial(
        _owned(final_arrivals),
        _owned(final_durations),
        _owned(final_censors),
        _readonly(observed),
        _owned(followup),
        tuple(history),
        early,
        stop_reason,
        stop_time,
        final,
        final_state,
    )


@dataclass(frozen=True)
class BayesFactorSurvivalSimulation:
    repetitions: int
    early_inferiority_probability: float
    early_superiority_probability: float
    final_inferiority_probability: float
    final_superiority_probability: float
    early_inferiority_mcse: float
    early_superiority_mcse: float
    final_inferiority_mcse: float
    final_superiority_mcse: float
    mean_patients_enrolled: float
    patient_count_quantiles: FloatArray
    early_inferiority: np.ndarray
    early_superiority: np.ndarray
    final_inferiority: np.ndarray
    final_superiority: np.ndarray
    patients_enrolled: np.ndarray
    trial_seed_pairs: np.ndarray


def simulate_bayes_factor_survival(
    *,
    null_median: float,
    alternative_median_mode: float,
    true_median: float,
    accrual_rate: float,
    max_patients: int,
    repetitions: int,
    check_times: ArrayLike,
    final_followup: float,
    seed: int | None = None,
    trial_seed_pairs: ArrayLike | None = None,
    inferiority_cutoff: float = 0.15,
    superiority_cutoff: float = 0.8,
    max_total_work: int = _DEFAULT_TOTAL_WORK,
    max_total_quadratures: int = _DEFAULT_TOTAL_QUADRATURES,
    max_total_storage_bytes: int = _MAX_TOTAL_STORAGE_BYTES,
) -> BayesFactorSurvivalSimulation:
    """Serial simulation under an explicit Python timing convention.

    The first patient arrives at time zero, subsequent gaps are exponential
    with the requested accrual rate, and ``check_times`` are absolute calendar
    times shared across trials. Each trial's final time is the later of its
    last planned arrival and the last check, plus ``final_followup``. Event
    durations are exponential with median ``true_median``. These conventions
    make no claim of native accrual/check-schedule parity.
    """
    maximum = scalar(max_patients, "max_patients")
    reps = scalar(repetitions, "repetitions")
    median = scalar(true_median, "true_median")
    rate = scalar(accrual_rate, "accrual_rate")
    followup = scalar(final_followup, "final_followup")
    work_limit = scalar(max_total_work, "max_total_work")
    quadrature_limit = scalar(max_total_quadratures, "max_total_quadratures")
    storage_limit = scalar(max_total_storage_bytes, "max_total_storage_bytes")
    if (
        isinstance(max_patients, (bool, np.bool_))
        or isinstance(repetitions, (bool, np.bool_))
        or maximum != int(maximum)
        or reps != int(reps)
        or not 1 <= maximum <= 500
        or not 1 <= reps <= 20_000
        or median <= 0
        or rate <= 0
        or followup < 0
        or isinstance(max_total_work, (bool, np.bool_))
        or work_limit != int(work_limit)
        or not 1 <= work_limit <= _MAX_TOTAL_WORK
        or isinstance(max_total_quadratures, (bool, np.bool_))
        or quadrature_limit != int(quadrature_limit)
        or not 1 <= quadrature_limit <= _MAX_TOTAL_QUADRATURES
        or isinstance(max_total_storage_bytes, (bool, np.bool_))
        or storage_limit != int(storage_limit)
        or not 1 <= storage_limit <= _MAX_TOTAL_STORAGE_BYTES
    ):
        raise ValueError("invalid simulation sizes, rates, times, or resource limits")
    checks = _vector(check_times, "check_times")
    if checks.size > _MAX_TRIAL_CHECKS or np.any(np.diff(checks) <= 0) or np.any(checks < 0):
        raise ValueError("check_times must be strictly increasing and nonnegative")
    n, check_count = int(maximum), int(checks.size)
    reps_int = int(reps)
    work = reps_int * (n + check_count + 1)
    storage_bytes = (
        reps_int * 64  # retained indicators, counts, and replayable seed pairs
        + n * 160  # current arrival, duration, ledger work, and returned trial arrays
        + check_count * 1024  # monitor states plus Python history records
        + 4096  # quantiles and scalar/array headers
        + _KERNEL_SCRATCH_BYTES  # worst allowed adaptive iMOM quadrature scratch
    )
    if work > work_limit:
        raise ValueError("worst-case simulated patient/check work exceeds max_total_work")
    if reps_int * (check_count + 1) > quadrature_limit:
        raise ValueError("worst-case Bayes-factor evaluations exceed max_total_quadratures")
    if storage_bytes > storage_limit:
        raise ValueError("simulated peak storage estimate exceeds max_total_storage_bytes")
    if (seed is None) == (trial_seed_pairs is None):
        raise ValueError("provide exactly one of seed or trial_seed_pairs")
    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be a nonnegative integer")
        seed_value = int(seed)
        if seed_value < 0:
            raise ValueError("seed must be a nonnegative integer")
        seed_pairs = np.empty((reps_int, 2), dtype=np.uint64)
    else:
        raw_seeds = np.asarray(trial_seed_pairs)
        if (
            raw_seeds.shape != (reps_int, 2)
            or raw_seeds.dtype.kind not in "ui"
            or (raw_seeds.dtype.kind == "i" and np.any(raw_seeds < 0))
        ):
            raise ValueError("trial_seed_pairs must be an integer array of shape (repetitions, 2)")
        seed_pairs = np.array(raw_seeds, dtype=np.uint64, copy=True)
    event_mean = median / np.log(2)
    gap_mean = 1.0 / rate
    if not np.isfinite(event_mean) or not np.isfinite(gap_mean):
        raise ArithmeticError("simulation time scale is not representable")
    # Validate the statistical design before advancing either RNG stream.
    bayes_factor_survival(
        0,
        0.0,
        null_median=null_median,
        alternative_median_mode=alternative_median_mode,
        inferiority_cutoff=inferiority_cutoff,
        superiority_cutoff=superiority_cutoff,
    )

    early_i = np.zeros(reps_int, dtype=bool)
    early_s = np.zeros(reps_int, dtype=bool)
    final_i = np.zeros(reps_int, dtype=bool)
    final_s = np.zeros(reps_int, dtype=bool)
    patient_counts = np.empty(reps_int, dtype=np.int64)
    seed_root = np.random.SeedSequence(int(seed_value)) if seed is not None else None
    for index in range(reps_int):
        if seed is not None:
            assert seed_root is not None
            child = seed_root.spawn(1)[0]
            pair = child.generate_state(2, dtype=np.uint64)
            seed_pairs[index] = pair
        accrual_rng = np.random.default_rng(int(seed_pairs[index, 0]))
        outcome_rng = np.random.default_rng(int(seed_pairs[index, 1]))
        gaps = accrual_rng.exponential(gap_mean, n - 1)
        arrivals = np.empty(n, dtype=np.float64)
        arrivals[0] = 0.0
        if n > 1:
            arrivals[1:] = np.cumsum(gaps)
        durations = outcome_rng.exponential(event_mean, n)
        if (
            np.any(durations <= 0)
            or not np.isfinite(arrivals[-1])
            or not np.all(np.isfinite(durations))
        ):
            raise ArithmeticError("simulated time tape is not representable")
        horizon = max(float(arrivals[-1]), float(checks[-1]) if check_count else 0.0)
        final_time = horizon + followup
        if not np.isfinite(final_time) or (followup > 0 and final_time <= horizon):
            raise ArithmeticError("simulated final time is not representable")
        # A check at the final time belongs to final monitoring only.
        trial_checks = checks[checks < final_time]
        trial = bayes_factor_survival_trial(
            arrivals,
            durations,
            check_times=trial_checks,
            final_time=final_time,
            null_median=null_median,
            alternative_median_mode=alternative_median_mode,
            inferiority_cutoff=inferiority_cutoff,
            superiority_cutoff=superiority_cutoff,
        )
        if trial.early_monitor is not None:
            early_i[index] = str(trial.early_monitor.decision) == "inferiority"
            early_s[index] = str(trial.early_monitor.decision) == "superiority"
        final_i[index] = str(trial.final_monitor.decision) == "inferiority"
        final_s[index] = str(trial.final_monitor.decision) == "superiority"
        patient_counts[index] = len(trial.enrollment_times)
        del trial, arrivals, durations, gaps, accrual_rng, outcome_rng
    q = np.quantile(patient_counts, [0.1, 0.5, 0.9])

    def estimate(values: np.ndarray) -> tuple[float, float]:
        p = float(np.mean(values))
        error = float("nan") if reps_int == 1 else float(np.sqrt(p * (1.0 - p) / reps_int))
        return p, error

    ei, ei_se = estimate(early_i)
    es, es_se = estimate(early_s)
    fi, fi_se = estimate(final_i)
    fs, fs_se = estimate(final_s)
    return BayesFactorSurvivalSimulation(
        reps_int,
        ei,
        es,
        fi,
        fs,
        ei_se,
        es_se,
        fi_se,
        fs_se,
        float(np.mean(patient_counts)),
        _owned(q),
        _readonly(early_i),
        _readonly(early_s),
        _readonly(final_i),
        _readonly(final_s),
        _readonly(patient_counts),
        _readonly(seed_pairs),
    )
