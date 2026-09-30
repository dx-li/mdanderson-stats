"""Bounded serial operating characteristics for the six-dose Phase I/II calendar."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray, finite, scalar
from .parallel_phase12_calendar import simulate_phase12_calendar
from .parallel_phase12_importance import _settings as _importance_settings
from .parallel_phase12_model import _MEAN, _SD, phase12_snapshot

_MAX_TRIALS = 10_000
_MAX_TOTAL_PATIENT_ASSIGNMENTS = 5_000_000
_MAX_TOTAL_CALENDAR_ATTEMPTS = 50_000_000
_MAX_TOTAL_IMPORTANCE_COMPONENT_EVALUATIONS = 2_000_000_000
_MAX_TOTAL_IMPORTANCE_MODE_ITERATIONS = 100_000_000
_MAX_TOTAL_MCMC_TRANSITIONS = 1_000_000_000
_MAX_ESTIMATED_TRIAL_ALLOCATION_BYTES = 512 * 1024**2
_PER_TRIAL_IMPORTANCE_COMPONENT_LIMIT = 100_000_000
_PER_TRIAL_IMPORTANCE_MODE_LIMIT = 2_100_000
_STOP_REASONS = (
    "phase-I toxicity",
    "at most one admissible dose",
    "futility",
    "source efficacy selection",
    "no open unsuspended doses",
    "maximum enrollment",
    "maximum duration",
)


def _positive_int(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f"{name} must be in [{low},{high}]")
    return number


def _nonnegative_int(value: int, name: str, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    number = int(value)
    if not 0 <= number <= high:
        raise ValueError(f"{name} must be in [0,{high}]")
    return number


def _seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    number = int(value)
    if number < 0:
        raise ValueError("seed must be a nonnegative integer")
    return number


def _immutable_int(values: ArrayLike, dtype: np.dtype = np.dtype(np.int64)) -> NDArray:
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bernoulli_mcse(probability: FloatArray, count: int) -> FloatArray:
    if count < 2:
        return np.full(probability.shape, np.nan)
    return np.sqrt(probability * (1.0 - probability) / count)


def _mean_mcse(mean: FloatArray, m2: FloatArray, count: int) -> tuple[FloatArray, FloatArray]:
    if count < 2:
        return mean.copy(), np.full(mean.shape, np.nan)
    error = np.sqrt(np.maximum(m2, 0.0) / ((count - 1) * count))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(error)):
        raise ArithmeticError("aggregate count mean or MCSE is not representable")
    return mean.copy(), error


def _scaled_time_summary(values: FloatArray) -> tuple[float, float]:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, float("nan") if values.size == 1 else 0.0
    normalized = values / scale
    mean = scale * float(np.mean(normalized))
    mcse = (
        float("nan")
        if values.size == 1
        else scale * float(np.std(normalized, ddof=1) / np.sqrt(values.size))
    )
    if not np.isfinite(mean) or (values.size > 1 and not np.isfinite(mcse)):
        raise ArithmeticError("calendar-time mean or MCSE is not representable")
    return mean, mcse


def _pooled_rate(
    events: NDArray[np.int64],
    exposure: NDArray[np.int64],
    event_squares: FloatArray,
    exposure_squares: FloatArray,
    cross_products: FloatArray,
    count: int,
) -> tuple[FloatArray, FloatArray]:
    rates = np.divide(
        events,
        exposure,
        out=np.full(6, np.nan, dtype=float),
        where=exposure > 0,
    )
    errors = np.full(6, np.nan)
    if count < 2:
        return rates, errors
    for dose in range(6):
        denominator = int(exposure[dose])
        if denominator == 0:
            continue
        rate = float(rates[dose])
        centered_sum_squares = (
            event_squares[dose]
            - 2.0 * rate * cross_products[dose]
            + rate * rate * exposure_squares[dose]
        )
        errors[dose] = np.sqrt(
            max(centered_sum_squares, 0.0) * count / ((count - 1) * denominator**2)
        )
    return rates, errors


def _optimal_doses(value: Iterable[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        selected = []
        for dose in value:
            selected.append(dose)
            if len(selected) > 6:
                break
        raw = np.asarray(selected)
    except TypeError as exc:
        raise ValueError(
            "optimal_doses must be an iterable of unique integer indices 0..5"
        ) from exc
    if (
        raw.ndim != 1
        or raw.size == 0
        or raw.dtype.kind not in "iu"
        or raw.dtype.kind == "b"
        or np.any((raw < 0) | (raw > 5))
        or np.unique(raw).size != raw.size
    ):
        raise ValueError("optimal_doses must be a nonempty set of unique integer indices 0..5")
    return tuple(sorted(int(dose) for dose in raw))


def _diagnostic_update(current: float | None, values: ArrayLike) -> float | None:
    array = np.asarray(values, dtype=float)
    if np.any(np.isinf(array)):
        return float("inf")
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return current
    maximum = float(np.max(finite_values))
    return maximum if current is None else max(current, maximum)


@dataclass(frozen=True)
class Phase12CalendarOC:
    """Compact trial-level operating characteristics for the six-dose calendar.

    All event probabilities use all completed trials as their denominator.
    ``generated_*`` summaries count complete simulated outcome truth, including
    outcomes that were not yet available at the analysis cutoff. ``observed_*``
    summaries use outcomes available at each trial's final analysis time, or at
    its stop time when phase I stopped before a posterior analysis. Rates pool
    patients within arms; their MCSEs treat whole trials as independent units.
    """

    n_trials: int
    per_trial_seeds: NDArray[np.uint64]
    early_selection_counts: NDArray[np.int64]
    early_selection_probability: FloatArray
    early_selection_mcse: FloatArray
    future_selection_counts: NDArray[np.int64]
    future_selection_probability: FloatArray
    future_selection_mcse: FloatArray
    selected_counts: NDArray[np.int64]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    no_selection_count: int
    no_selection_probability: float
    no_selection_mcse: float
    early_ineligible_selection_count: int
    early_ineligible_selection_probability: float
    early_ineligible_given_early_selection: float | None
    stopping_reasons: tuple[str, ...]
    stopping_counts: NDArray[np.int64]
    stopping_probability: FloatArray
    stopping_mcse: FloatArray
    phase_one_admissible_counts: NDArray[np.int64]
    phase_one_admissibility_probability: FloatArray
    phase_one_admissibility_mcse: FloatArray
    treated_total: NDArray[np.int64]
    mean_treated: FloatArray
    mcse_treated: FloatArray
    generated_toxicity_total: NDArray[np.int64]
    mean_generated_toxicities: FloatArray
    mcse_generated_toxicities: FloatArray
    generated_response_total: NDArray[np.int64]
    mean_generated_responses: FloatArray
    mcse_generated_responses: FloatArray
    generated_toxicity_rate: FloatArray
    generated_toxicity_rate_mcse: FloatArray
    generated_response_rate: FloatArray
    generated_response_rate_mcse: FloatArray
    observed_toxicity_total: NDArray[np.int64]
    observed_toxicity_denominator: NDArray[np.int64]
    mean_observed_toxicities: FloatArray
    mcse_observed_toxicities: FloatArray
    observed_toxicity_rate: FloatArray
    observed_toxicity_rate_mcse: FloatArray
    observed_response_total: NDArray[np.int64]
    observed_response_denominator: NDArray[np.int64]
    mean_observed_responses: FloatArray
    mcse_observed_responses: FloatArray
    observed_response_rate: FloatArray
    observed_response_rate_mcse: FloatArray
    total_enrollment: int
    mean_total_enrollment: float
    mcse_total_enrollment: float
    phase_one_enrollment_total: int
    mean_phase_one_enrollment: float
    mcse_phase_one_enrollment: float
    mean_enrollment_stop_time_days: float
    mcse_enrollment_stop_time_days: float
    final_analysis_trial_count: int
    mean_final_analysis_time_days: float | None
    mcse_final_analysis_time_days: float | None
    optimal_doses: tuple[int, ...] | None
    optimal_set_selected_count: int | None
    optimal_set_selection_probability: float | None
    optimal_set_selection_mcse: float | None
    mcmc_fit_count: int
    max_mcmc_split_rhat: float | None
    nonfinite_mcmc_rhat_fit_count: int
    importance_fit_count: int
    importance_nonconverged_fit_count: int
    importance_nonconverged_trial_count: int
    importance_nonconverged_trial_probability: float | None
    max_importance_ratio_mcse: float | None
    nonfinite_importance_ratio_mcse_fit_count: int
    importance_component_evaluations: int
    importance_mode_iterations: int
    mcmc_configured_transition_slots: int


def simulate_phase12_calendar_oc(
    toxicity_probability: ArrayLike,
    efficacy_probability: ArrayLike,
    *,
    n_trials: int,
    seed: int,
    optimal_doses: Iterable[int] | None = None,
    accrual_per_year: float = 72,
    efficacy_window: float = 84,
    toxicity_window: float = 28,
    max_patients: int = 80,
    max_duration: float = float("inf"),
    max_attempts: int = 10_000,
    complete_followup: bool = False,
    posterior_backend: Literal["mcmc", "importance"] = "mcmc",
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    importance_relative_error: float = 0.001,
    importance_max_integrations: int = 10_000,
    importance_multivariate_weight: float = 0.99,
    importance_max_mode_iterations: int = 1_000,
    max_total_patient_assignments: int = 1_000_000,
    max_total_calendar_attempts: int = 10_000_000,
    max_total_importance_component_evaluations: int = 1_000_000_000,
    max_total_importance_mode_iterations: int = 20_000_000,
    max_total_mcmc_transitions: int = 250_000_000,
    max_estimated_trial_allocation_bytes: int = 128 * 1024**2,
) -> Phase12CalendarOC:
    """Run six-dose calendar trials serially and retain only aggregate summaries.

    ``optimal_doses`` is a caller-defined set. Its hit probability means only
    that one of those dose indices was selected, whether early or at final
    selection; the software does not infer an optimal-dose success rule.
    Trial ``i`` is replayed by passing ``default_rng(per_trial_seeds[i])`` and
    the same scenario/calendar/backend settings to ``simulate_phase12_calendar``.
    """
    trial_count = _positive_int(n_trials, "n_trials", 1, _MAX_TRIALS)
    base_seed = _seed(seed)
    toxicity = finite(toxicity_probability, "toxicity_probability")
    efficacy = finite(efficacy_probability, "efficacy_probability")
    if (
        toxicity.shape != (6,)
        or efficacy.shape != (6,)
        or np.any((toxicity < 0.0) | (toxicity > 1.0))
        or np.any((efficacy < 0.0) | (efficacy > 1.0))
    ):
        raise ValueError("toxicity and efficacy probabilities must be six-vectors in [0,1]")
    selected_optimal = _optimal_doses(optimal_doses)
    rate = scalar(accrual_per_year, "accrual_per_year")
    response_window = scalar(efficacy_window, "efficacy_window")
    toxicity_window_value = scalar(toxicity_window, "toxicity_window")
    if (
        rate <= 0.0
        or not np.isfinite(365.0 / rate)
        or response_window < 1.0
        or toxicity_window_value < 1.0
    ):
        raise ValueError("accrual must be positive; event windows must be at least one day")
    patient_cap = _positive_int(max_patients, "max_patients", 1, 10_000)
    attempt_cap = _positive_int(max_attempts, "max_attempts", 1, 1_000_000)
    duration = float(max_duration)
    if np.isnan(duration) or duration <= 0.0:
        raise ValueError("max_duration must be positive (infinity is allowed)")
    if not isinstance(complete_followup, (bool, np.bool_)):
        raise ValueError("complete_followup must be boolean")
    if posterior_backend not in ("mcmc", "importance"):
        raise ValueError("posterior_backend must be 'mcmc' or 'importance'")
    draw_count = _positive_int(draws, "draws", 8, 100_000)
    warmup_count = _nonnegative_int(warmup, "warmup", 100_000)
    chain_count = _positive_int(chains, "chains", 2, 16)
    integration_cap = _positive_int(
        importance_max_integrations, "importance_max_integrations", 100, 1_000_000
    )
    if integration_cap % 100:
        raise ValueError("importance_max_integrations must be a multiple of 100")
    mode_cap = _positive_int(
        importance_max_mode_iterations, "importance_max_mode_iterations", 1, 10_000
    )
    relative_error = scalar(importance_relative_error, "importance_relative_error")
    mixture_weight = scalar(importance_multivariate_weight, "importance_multivariate_weight")
    if posterior_backend == "importance":
        _importance_settings(
            _MEAN,
            _SD,
            0.30,
            0.10,
            0.33,
            (0.1, 0.9),
            relative_error,
            integration_cap,
            mixture_weight,
            mode_cap,
        )

    patient_work_cap = _positive_int(
        max_total_patient_assignments,
        "max_total_patient_assignments",
        1,
        _MAX_TOTAL_PATIENT_ASSIGNMENTS,
    )
    attempt_work_cap = _positive_int(
        max_total_calendar_attempts,
        "max_total_calendar_attempts",
        1,
        _MAX_TOTAL_CALENDAR_ATTEMPTS,
    )
    importance_work_cap = _positive_int(
        max_total_importance_component_evaluations,
        "max_total_importance_component_evaluations",
        1,
        _MAX_TOTAL_IMPORTANCE_COMPONENT_EVALUATIONS,
    )
    mode_work_cap = _positive_int(
        max_total_importance_mode_iterations,
        "max_total_importance_mode_iterations",
        1,
        _MAX_TOTAL_IMPORTANCE_MODE_ITERATIONS,
    )
    mcmc_work_cap = _positive_int(
        max_total_mcmc_transitions,
        "max_total_mcmc_transitions",
        1,
        _MAX_TOTAL_MCMC_TRANSITIONS,
    )
    allocation_cap = _positive_int(
        max_estimated_trial_allocation_bytes,
        "max_estimated_trial_allocation_bytes",
        1,
        _MAX_ESTIMATED_TRIAL_ALLOCATION_BYTES,
    )

    if trial_count * patient_cap > patient_work_cap:
        raise ValueError("worst-case patient assignments exceed max_total_patient_assignments")
    if trial_count * attempt_cap > attempt_work_cap:
        raise ValueError("worst-case calendar attempts exceed max_total_calendar_attempts")
    max_fits = patient_cap // 5 + 1
    per_trial_component_work = max_fits * integration_cap * 67
    per_trial_mode_work = max_fits * mode_cap
    if posterior_backend == "importance":
        if per_trial_component_work > _PER_TRIAL_IMPORTANCE_COMPONENT_LIMIT:
            raise ValueError("per-trial importance work exceeds the calendar component limit")
        if per_trial_mode_work > _PER_TRIAL_IMPORTANCE_MODE_LIMIT:
            raise ValueError("per-trial importance mode work exceeds the calendar limit")
        if trial_count * per_trial_component_work > importance_work_cap:
            raise ValueError(
                "worst-case importance component evaluations exceed their aggregate limit"
            )
        if trial_count * per_trial_mode_work > mode_work_cap:
            raise ValueError("worst-case importance mode iterations exceed their aggregate limit")
    else:
        transition_work = trial_count * max_fits * chain_count * (draw_count + warmup_count)
        if transition_work > mcmc_work_cap:
            raise ValueError("worst-case MCMC chain transitions exceed max_total_mcmc_transitions")

    # Conservative allocation estimate for one active trial: Python records
    # and attempts plus their frozen arrays, compact analysis snapshots/errors,
    # the largest fit's returned/scratch arrays, and aggregate output vectors.
    if posterior_backend == "mcmc":
        # A cached fit can coexist with its replacement and the latter's
        # sorted, split-chain, and variance/quantile summary scratch arrays.
        fit_bytes = chain_count * draw_count * 512
    else:
        fit_bytes = 8 * 1024**2
    estimated_bytes = (
        patient_cap * 512
        + attempt_cap * 256
        + max_fits * 4096
        + fit_bytes
        # Seed copies, duration array, Python final-time list and its temporary
        # numeric conversion can coexist when constructing the result.
        + trial_count * 80
        + 128 * 1024
    )
    if estimated_bytes > allocation_cap:
        raise ValueError(
            "estimated one-trial allocation exceeds max_estimated_trial_allocation_bytes"
        )

    # All input and worst-case work/storage checks are complete before seeds
    # are materialized or trial-level random generators are created.
    seeds = np.empty(trial_count, dtype=np.uint64)
    early_counts = np.zeros(6, dtype=np.int64)
    future_counts = np.zeros(6, dtype=np.int64)
    stopping_counts = np.zeros(len(_STOP_REASONS), dtype=np.int64)
    admissible_counts = np.zeros(6, dtype=np.int64)
    no_selection_count = early_ineligible_count = 0
    optimal_selection_count = 0
    treated_total = np.zeros(6, dtype=np.int64)
    generated_tox_total = np.zeros(6, dtype=np.int64)
    generated_response_total = np.zeros(6, dtype=np.int64)
    observed_tox_total = np.zeros(6, dtype=np.int64)
    observed_tox_denominator = np.zeros(6, dtype=np.int64)
    observed_response_total = np.zeros(6, dtype=np.int64)
    observed_response_denominator = np.zeros(6, dtype=np.int64)
    event_squares = {"generated_tox": np.zeros(6), "generated_response": np.zeros(6)}
    exposure_squares = {"generated_tox": np.zeros(6), "generated_response": np.zeros(6)}
    cross_products = {"generated_tox": np.zeros(6), "generated_response": np.zeros(6)}
    observed_event_squares = {"tox": np.zeros(6), "response": np.zeros(6)}
    observed_exposure_squares = {"tox": np.zeros(6), "response": np.zeros(6)}
    observed_cross_products = {"tox": np.zeros(6), "response": np.zeros(6)}

    means = {
        name: np.zeros(6)
        for name in (
            "treated",
            "generated_tox",
            "generated_response",
            "observed_tox",
            "observed_response",
        )
    }
    moments2 = {name: np.zeros(6) for name in means}
    total_enrollment_mean = total_enrollment_m2 = 0.0
    phase_one_mean = phase_one_m2 = 0.0
    phase_one_total = 0
    durations = np.empty(trial_count)
    final_times: list[float] = []
    max_mcmc_rhat: float | None = None
    max_importance_ratio_error: float | None = None
    mcmc_fit_count = importance_fit_count = 0
    nonfinite_rhat_count = nonfinite_ratio_count = 0
    importance_nonconverged_fit_count = importance_nonconverged_trial_count = 0
    importance_component_evaluations = importance_mode_iterations = 0
    mcmc_transition_slots = 0

    for trial_index in range(trial_count):
        trial_seed = int(
            np.random.SeedSequence(base_seed, spawn_key=(trial_index,)).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        seeds[trial_index] = trial_seed
        per_trial_component_budget = (
            max(67, min(_PER_TRIAL_IMPORTANCE_COMPONENT_LIMIT, per_trial_component_work))
            if posterior_backend == "importance"
            else 67
        )
        result = simulate_phase12_calendar(
            toxicity,
            efficacy,
            accrual_per_year=rate,
            efficacy_window=response_window,
            toxicity_window=toxicity_window_value,
            max_patients=patient_cap,
            max_duration=duration,
            max_attempts=attempt_cap,
            complete_followup=bool(complete_followup),
            posterior_backend=posterior_backend,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            importance_relative_error=relative_error,
            importance_max_integrations=integration_cap,
            importance_multivariate_weight=mixture_weight,
            importance_max_mode_iterations=mode_cap,
            max_total_posterior_component_evaluations=per_trial_component_budget,
            rng=np.random.default_rng(trial_seed),
        )
        if result.reason not in _STOP_REASONS:
            raise RuntimeError(
                f"calendar returned an unrecognized stopping reason: {result.reason}"
            )
        stopping_counts[_STOP_REASONS.index(result.reason)] += 1
        admissible_counts += np.asarray(result.phase_one_admissible, dtype=np.int64)

        if result.early_selected is not None and result.future_selected is not None:
            raise RuntimeError("calendar returned both early and future selections")
        selected = result.early_selected
        if selected is not None:
            early_counts[selected] += 1
            if result.selected_eligible is False:
                early_ineligible_count += 1
        elif result.future_selected is not None:
            selected = result.future_selected
            future_counts[selected] += 1
        else:
            no_selection_count += 1
        if selected is not None and selected_optimal is not None and selected in selected_optimal:
            optimal_selection_count += 1

        records = np.asarray(result.records)
        doses = records[:, 0].astype(np.intp) if records.size else np.empty(0, dtype=np.intp)
        treated = np.bincount(doses, minlength=6).astype(np.int64)
        generated_tox = (
            np.bincount(doses, weights=records[:, 4], minlength=6).astype(np.int64)
            if records.size
            else np.zeros(6, dtype=np.int64)
        )
        generated_response = (
            np.bincount(doses, weights=records[:, 2], minlength=6).astype(np.int64)
            if records.size
            else np.zeros(6, dtype=np.int64)
        )
        cutoff = (
            result.final_analysis_time
            if result.final_analysis_time is not None
            else result.stop_time
        )
        observed = phase12_snapshot(records, time=cutoff).tally
        observed_response = observed[:, 1].astype(np.int64)
        observed_response_n = observed[:, :2].sum(axis=1).astype(np.int64)
        observed_tox = observed[:, 3].astype(np.int64)
        observed_tox_n = observed[:, 2:].sum(axis=1).astype(np.int64)
        phase_one_count = (
            len(records) if result.phase_two_start is None else int(result.phase_two_start)
        )
        phase_one_total += phase_one_count

        for name, values in (
            ("treated", treated),
            ("generated_tox", generated_tox),
            ("generated_response", generated_response),
            ("observed_tox", observed_tox),
            ("observed_response", observed_response),
        ):
            delta = values - means[name]
            means[name] += delta / (trial_index + 1)
            moments2[name] += delta * (values - means[name])
        total_enrollment = int(treated.sum())
        delta_total = total_enrollment - total_enrollment_mean
        total_enrollment_mean += delta_total / (trial_index + 1)
        total_enrollment_m2 += delta_total * (total_enrollment - total_enrollment_mean)
        delta_phase = phase_one_count - phase_one_mean
        phase_one_mean += delta_phase / (trial_index + 1)
        phase_one_m2 += delta_phase * (phase_one_count - phase_one_mean)

        treated_total += treated
        generated_tox_total += generated_tox
        generated_response_total += generated_response
        observed_tox_total += observed_tox
        observed_tox_denominator += observed_tox_n
        observed_response_total += observed_response
        observed_response_denominator += observed_response_n
        for key, events in (
            ("generated_tox", generated_tox),
            ("generated_response", generated_response),
        ):
            event_squares[key] += events.astype(float) ** 2
            exposure_squares[key] += treated.astype(float) ** 2
            cross_products[key] += events * treated
        for key, events, exposure in (
            ("tox", observed_tox, observed_tox_n),
            ("response", observed_response, observed_response_n),
        ):
            observed_event_squares[key] += events.astype(float) ** 2
            observed_exposure_squares[key] += exposure.astype(float) ** 2
            observed_cross_products[key] += events * exposure

        durations[trial_index] = float(result.stop_time)
        if result.final_analysis_time is not None:
            final_times.append(float(result.final_analysis_time))

        last_tally: FloatArray | None = None
        trial_nonconverged = False
        for analysis in result.analyses:
            if last_tally is not None and np.array_equal(analysis.snapshot.tally, last_tally):
                continue
            last_tally = analysis.snapshot.tally
            if analysis.posterior_backend == "mcmc":
                mcmc_fit_count += 1
                assert analysis.max_split_rhat is not None
                max_mcmc_rhat = _diagnostic_update(max_mcmc_rhat, [analysis.max_split_rhat])
                nonfinite_rhat_count += int(not np.isfinite(analysis.max_split_rhat))
                mcmc_transition_slots += chain_count * (draw_count + warmup_count)
            else:
                importance_fit_count += 1
                if analysis.posterior_converged is False:
                    importance_nonconverged_fit_count += 1
                    trial_nonconverged = True
                assert analysis.ratio_mc_se is not None
                errors = np.asarray(analysis.ratio_mc_se)
                nonfinite_ratio_count += int(not np.all(np.isfinite(errors)))
                max_importance_ratio_error = _diagnostic_update(max_importance_ratio_error, errors)
        importance_nonconverged_trial_count += int(trial_nonconverged)
        importance_component_evaluations += result.posterior_component_evaluations
        importance_mode_iterations += result.posterior_mode_iterations
        del result

    early_probability = early_counts / trial_count
    future_probability = future_counts / trial_count
    selected_counts = early_counts + future_counts
    selection_probability = selected_counts / trial_count
    no_probability = no_selection_count / trial_count
    stopping_probability = stopping_counts / trial_count
    admissibility_probability = admissible_counts / trial_count
    if not np.isclose(
        float(early_probability.sum() + future_probability.sum() + no_probability), 1.0
    ):
        raise ArithmeticError("early/future/no-selection categories do not sum to one")

    mean_treated, mcse_treated = _mean_mcse(means["treated"], moments2["treated"], trial_count)
    mean_generated_tox, mcse_generated_tox = _mean_mcse(
        means["generated_tox"], moments2["generated_tox"], trial_count
    )
    mean_generated_response, mcse_generated_response = _mean_mcse(
        means["generated_response"], moments2["generated_response"], trial_count
    )
    mean_observed_tox, mcse_observed_tox = _mean_mcse(
        means["observed_tox"], moments2["observed_tox"], trial_count
    )
    mean_observed_response, mcse_observed_response = _mean_mcse(
        means["observed_response"], moments2["observed_response"], trial_count
    )
    generated_tox_rate, generated_tox_rate_mcse = _pooled_rate(
        generated_tox_total,
        treated_total,
        event_squares["generated_tox"],
        exposure_squares["generated_tox"],
        cross_products["generated_tox"],
        trial_count,
    )
    generated_response_rate, generated_response_rate_mcse = _pooled_rate(
        generated_response_total,
        treated_total,
        event_squares["generated_response"],
        exposure_squares["generated_response"],
        cross_products["generated_response"],
        trial_count,
    )
    observed_tox_rate, observed_tox_rate_mcse = _pooled_rate(
        observed_tox_total,
        observed_tox_denominator,
        observed_event_squares["tox"],
        observed_exposure_squares["tox"],
        observed_cross_products["tox"],
        trial_count,
    )
    observed_response_rate, observed_response_rate_mcse = _pooled_rate(
        observed_response_total,
        observed_response_denominator,
        observed_event_squares["response"],
        observed_exposure_squares["response"],
        observed_cross_products["response"],
        trial_count,
    )
    mean_duration, duration_mcse = _scaled_time_summary(durations)
    if final_times:
        mean_final_time: float | None
        final_time_mcse: float | None
        mean_final_time, final_time_mcse = _scaled_time_summary(np.asarray(final_times))
    else:
        mean_final_time = final_time_mcse = None
    optimal_probability = (
        None if selected_optimal is None else optimal_selection_count / trial_count
    )
    optimal_mcse = (
        None
        if optimal_probability is None
        else float(_bernoulli_mcse(np.asarray(optimal_probability), trial_count))
    )
    conditional_early_ineligible = (
        None if early_counts.sum() == 0 else float(early_ineligible_count / early_counts.sum())
    )
    nonconverged_probability = (
        None
        if posterior_backend != "importance"
        else float(importance_nonconverged_trial_count / trial_count)
    )

    return Phase12CalendarOC(
        trial_count,
        _immutable_int(seeds, dtype=np.dtype(np.uint64)),
        _immutable_int(early_counts),
        _freeze(early_probability),
        _freeze(_bernoulli_mcse(early_probability, trial_count)),
        _immutable_int(future_counts),
        _freeze(future_probability),
        _freeze(_bernoulli_mcse(future_probability, trial_count)),
        _immutable_int(selected_counts),
        _freeze(selection_probability),
        _freeze(_bernoulli_mcse(selection_probability, trial_count)),
        no_selection_count,
        no_probability,
        float(_bernoulli_mcse(np.asarray(no_probability), trial_count)),
        early_ineligible_count,
        early_ineligible_count / trial_count,
        conditional_early_ineligible,
        _STOP_REASONS,
        _immutable_int(stopping_counts),
        _freeze(stopping_probability),
        _freeze(_bernoulli_mcse(stopping_probability, trial_count)),
        _immutable_int(admissible_counts),
        _freeze(admissibility_probability),
        _freeze(_bernoulli_mcse(admissibility_probability, trial_count)),
        _immutable_int(treated_total),
        _freeze(mean_treated),
        _freeze(mcse_treated),
        _immutable_int(generated_tox_total),
        _freeze(mean_generated_tox),
        _freeze(mcse_generated_tox),
        _immutable_int(generated_response_total),
        _freeze(mean_generated_response),
        _freeze(mcse_generated_response),
        _freeze(generated_tox_rate),
        _freeze(generated_tox_rate_mcse),
        _freeze(generated_response_rate),
        _freeze(generated_response_rate_mcse),
        _immutable_int(observed_tox_total),
        _immutable_int(observed_tox_denominator),
        _freeze(mean_observed_tox),
        _freeze(mcse_observed_tox),
        _freeze(observed_tox_rate),
        _freeze(observed_tox_rate_mcse),
        _immutable_int(observed_response_total),
        _immutable_int(observed_response_denominator),
        _freeze(mean_observed_response),
        _freeze(mcse_observed_response),
        _freeze(observed_response_rate),
        _freeze(observed_response_rate_mcse),
        int(treated_total.sum()),
        float(total_enrollment_mean),
        float(
            _mean_mcse(
                np.asarray([total_enrollment_mean]), np.asarray([total_enrollment_m2]), trial_count
            )[1][0]
        ),
        phase_one_total,
        float(phase_one_mean),
        float(
            _mean_mcse(np.asarray([phase_one_mean]), np.asarray([phase_one_m2]), trial_count)[1][0]
        ),
        mean_duration,
        duration_mcse,
        len(final_times),
        mean_final_time,
        final_time_mcse,
        selected_optimal,
        None if selected_optimal is None else optimal_selection_count,
        optimal_probability,
        optimal_mcse,
        mcmc_fit_count,
        max_mcmc_rhat,
        nonfinite_rhat_count,
        importance_fit_count,
        importance_nonconverged_fit_count,
        importance_nonconverged_trial_count,
        nonconverged_probability,
        max_importance_ratio_error,
        nonfinite_ratio_count,
        importance_component_evaluations,
        importance_mode_iterations,
        mcmc_transition_slots,
    )
