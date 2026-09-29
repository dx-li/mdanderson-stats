"""Explicit CiBolus prior elicitation by balanced pseudo-data simulation.

The paper's pseudo-data design treats caller-supplied response/toxicity
probabilities as the state of nature. This module does not infer an
interpolation between response endpoints or optimize prior variances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .cibolus import (
    CiBolusObservation,
    CiBolusPrior,
    _prediction_inputs,
    cibolus_response,
    cibolus_toxicity,
)
from .cibolus_fit import fit_cibolus
from .uaroet import _integer

_MAX_REPETITIONS = 1_000
_MAX_PATIENTS_PER_REGIMEN = 50
_MAX_PSEUDO_PATIENTS = 400
_MAX_TOTAL_EVALUATIONS = 2_000_000
_MAX_TOTAL_WORK = 50_000_000
_MAX_RETAINED_CELLS = 2_000_000


def _freeze_dtype(value: ArrayLike, dtype: np.dtype) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _real_bounded(value: ArrayLike, name: str, maximum: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.size > maximum:
        raise ValueError(f"{name} must be bounded real numeric data")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _grid(
    concentrations: ArrayLike, bolus_fractions: ArrayLike, endpoints: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    endpoint_raw = np.asarray(endpoints)
    if endpoint_raw.size > 20:
        raise ValueError("endpoints must contain at most 20 values")
    endpoint_count = endpoint_raw.size
    utility = np.zeros((endpoint_count + 2, 2), dtype=float)
    c, q, e, _ = _prediction_inputs(concentrations, bolus_fractions, endpoints, utility)
    return c, q, e


def _joint_grid(value: ArrayLike, expected: tuple[int, ...]) -> FloatArray:
    raw = np.asarray(value)
    if raw.size > _MAX_RETAINED_CELLS:
        raise ValueError("joint probability grid exceeds the retained-cell limit")
    if raw.dtype.kind not in "iuf" or raw.shape != expected:
        raise ValueError(f"joint probabilities must have shape {expected} and real values")
    result = np.array(raw, dtype=float, copy=True)
    if np.any(~np.isfinite(result)) or np.any(result < 0):
        raise ValueError("joint probabilities must be finite and nonnegative")
    totals = result.sum(axis=(-2, -1))
    if np.any(np.abs(totals - 1.0) > 1e-12):
        raise ValueError("joint probabilities must sum to one for every regimen")
    result /= totals[..., None, None]
    return _freeze_dtype(result, np.dtype(float))


def _moment_triplet(samples: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = np.asarray(samples, dtype=float)
    tolerance = 64 * np.finfo(float).eps
    if (
        not np.all(np.isfinite(values))
        or np.any(values < -tolerance)
        or np.any(values > 1 + tolerance)
    ):
        raise ArithmeticError("prior-predictive probability is outside [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    mean = np.mean(values, axis=0)
    # The empirical prior-predictive distribution uses its population moment.
    # This also keeps the plug-in beta-moment variance within m*(1-m).
    variance = np.var(values, axis=0, ddof=0)
    constant = np.all(values == values[0:1], axis=0)
    mean = np.where(constant, values[0], mean)
    variance = np.where(constant, 0.0, variance)
    ess = np.full_like(mean, np.nan)
    numerator = mean * (1.0 - mean)
    positive_variance = variance > 0
    np.divide(numerator, variance, out=ess, where=positive_variance)
    ess[positive_variance] -= 1.0
    if np.any(ess[positive_variance] < -1e-12):
        raise ArithmeticError("empirical probability moments imply a negative beta ESS")
    ess[positive_variance] = np.maximum(ess[positive_variance], 0.0)
    constant_interior = (~positive_variance) & (mean > 0) & (mean < 1)
    ess[constant_interior] = np.inf
    return (
        _freeze_dtype(mean, np.dtype(float)),
        _freeze_dtype(variance, np.dtype(float)),
        _freeze_dtype(ess, np.dtype(float)),
    )


@dataclass(frozen=True)
class CiBolusPriorPredictiveMoments:
    """Prior probability moments on a concentration/bolus regimen grid.

    Probability variances are empirical population moments (ddof=0). The
    beta-moment ESS is ``mean*(1-mean)/variance - 1``: a constant interior
    probability has limiting ESS +infinity, while a constant endpoint has
    undefined (NaN) ESS. ``source_probability_*`` is the four-metric subset
    used for the paper's variance calibration.
    """

    prior: CiBolusPrior
    concentrations: FloatArray
    bolus_fractions: FloatArray
    endpoints: FloatArray
    draws: int
    chains: int
    joint_probability_mean: FloatArray
    joint_probability_variance: FloatArray
    bolus_response_mean: FloatArray
    bolus_response_variance: FloatArray
    bolus_response_ess: FloatArray
    cumulative_response_mean: FloatArray
    cumulative_response_variance: FloatArray
    cumulative_response_ess: FloatArray
    toxicity_at_bolus_mean: FloatArray
    toxicity_at_bolus_variance: FloatArray
    toxicity_at_bolus_ess: FloatArray
    toxicity_at_response_one_mean: FloatArray
    toxicity_at_response_one_variance: FloatArray
    toxicity_at_response_one_ess: FloatArray
    toxicity_at_failure_mean: FloatArray
    toxicity_at_failure_variance: FloatArray
    toxicity_at_failure_ess: FloatArray
    source_probability_names: tuple[str, ...]
    source_probability_mean: FloatArray
    source_probability_variance: FloatArray
    source_probability_ess: FloatArray


def cibolus_prior_predictive_moments(
    prior: CiBolusPrior,
    concentrations: ArrayLike,
    bolus_fractions: ArrayLike,
    endpoints: ArrayLike,
    *,
    draws: int,
    chains: int,
    rng: np.random.Generator,
    max_work: int = 20_000_000,
    max_retained_cells: int = _MAX_RETAINED_CELLS,
) -> CiBolusPriorPredictiveMoments:
    """Estimate elicited probability means, variances and beta ESS from a prior.

    Variances are empirical population moments (ddof=0) across direct prior
    draws, so plug-in beta ESS remains nonnegative when numerically defined.

    The returned metrics are p0, cumulative response probabilities at every
    supplied endpoint, toxicity conditional on bolus response, toxicity at
    response time one, and toxicity after failure. Sampling is direct from the
    Gaussian log-parameter prior; it does not fit a posterior.
    """
    if not isinstance(prior, CiBolusPrior) or not isinstance(rng, np.random.Generator):
        raise ValueError("prior and explicit NumPy rng are required")
    draw_count = _integer(draws, "draws", 8, 100_000)
    chain_count = _integer(chains, "chains", 2, 8)
    work_limit = _integer(max_work, "max_work", 1, _MAX_TOTAL_WORK)
    retained_limit = _integer(max_retained_cells, "max_retained_cells", 1, _MAX_RETAINED_CELLS)
    c, q, e = _grid(concentrations, bolus_fractions, endpoints)
    regimens = c.size * q.size
    categories = e.size + 2
    response_times = np.unique(np.concatenate((e, np.array([1.0]))))
    response_indices = np.searchsorted(response_times, e)
    response_one_index = int(np.searchsorted(response_times, 1.0))
    sample_count = draw_count * chain_count
    grid_work = sample_count * regimens * categories
    extra_work = sample_count * regimens * 3
    response_work = sample_count * regimens * response_times.size
    if grid_work + extra_work + response_work > work_limit:
        raise ValueError("prior predictive work exceeds max_work")
    fit_retained_per_draw = 12 + regimens * (categories * 2 + 5)
    fit_retained = sample_count * fit_retained_per_draw
    temporary_cells = sample_count * regimens * (categories + 2 * e.size + 2)
    output_cells = regimens * (categories * 4 + e.size * 3 + 30)
    fit_summary_scratch = 14 * sample_count * (11 + regimens)
    if 2 * fit_retained + fit_summary_scratch + temporary_cells + 2 * output_cells > retained_limit:
        raise ValueError("prior predictive live arrays exceed max_retained_cells")
    utility = np.zeros((categories, 2))
    fit = fit_cibolus(
        [],
        prior,
        c,
        q,
        e,
        utility=utility,
        draws=draw_count,
        warmup=0,
        chains=chain_count,
        rng=rng,
        max_evaluations=1,
        max_work=work_limit,
    )
    joint = np.asarray(fit.joint).reshape((sample_count, c.size, q.size, categories, 2))
    bolus_response = np.empty((sample_count, c.size, q.size))
    cumulative = np.empty((sample_count, c.size, q.size, e.size))
    response_at_one = np.empty((sample_count, c.size, q.size))
    tox_bolus = np.empty((sample_count, c.size, q.size))
    for draw_index, theta in enumerate(fit.log_parameters.reshape((sample_count, 11))):
        for ci, concentration in enumerate(c):
            for qi, bolus in enumerate(q):
                response = cibolus_response(
                    response_times, float(concentration), float(bolus), theta
                )
                bolus_response[draw_index, ci, qi] = response.bolus_probability
                cumulative[draw_index, ci, qi] = response.cdf[response_indices]
                response_at_one[draw_index, ci, qi] = response.cdf[response_one_index]
                tox_bolus[draw_index, ci, qi] = cibolus_toxicity(
                    0.0, float(concentration), float(bolus), theta
                )
    joint_mean = np.mean(joint, axis=0)
    joint_var = np.var(joint, axis=0, ddof=0)
    joint_var = np.where(np.all(joint == joint[:1], axis=0), 0.0, joint_var)
    p0 = _moment_triplet(bolus_response)
    f_e = _moment_triplet(cumulative)
    t0 = _moment_triplet(tox_bolus)
    t1 = _moment_triplet(fit.toxicity_at_one_response.reshape((sample_count, c.size, q.size)))
    tf = _moment_triplet(fit.toxicity_at_one_failure.reshape((sample_count, c.size, q.size)))
    f_one = _moment_triplet(response_at_one)
    source_mean = np.stack((p0[0], f_one[0], t0[0], t1[0]), axis=-1)
    source_var = np.stack((p0[1], f_one[1], t0[1], t1[1]), axis=-1)
    source_ess = np.stack((p0[2], f_one[2], t0[2], t1[2]), axis=-1)
    return CiBolusPriorPredictiveMoments(
        prior,
        _freeze_dtype(c, np.dtype(float)),
        _freeze_dtype(q, np.dtype(float)),
        _freeze_dtype(e, np.dtype(float)),
        draw_count,
        chain_count,
        _freeze_dtype(joint_mean, np.dtype(float)),
        _freeze_dtype(joint_var, np.dtype(float)),
        *p0,
        *f_e,
        *t0,
        *t1,
        *tf,
        ("p0", "response_at_one", "toxicity_at_zero_response", "toxicity_at_one_response"),
        _freeze_dtype(source_mean, np.dtype(float)),
        _freeze_dtype(source_var, np.dtype(float)),
        _freeze_dtype(source_ess, np.dtype(float)),
    )


@dataclass(frozen=True)
class CiBolusPriorCalibration:
    """Pseudo-data estimate of log prior means and its between-replicate error."""

    prior: CiBolusPrior
    pseudo_prior: CiBolusPrior
    concentrations: FloatArray
    bolus_fractions: FloatArray
    endpoints: FloatArray
    repetitions: int
    patients_per_regimen: int
    joint_probabilities: FloatArray
    joint_counts: np.ndarray
    replicate_seeds: np.ndarray
    replicate_posterior_mean: FloatArray
    replicate_between_sd: FloatArray
    replicate_mean_mcse: FloatArray
    replicate_parameter_rhat: FloatArray
    replicate_sampler_mcse: FloatArray
    likelihood_evaluations: np.ndarray
    work_units: np.ndarray


def _observations_from_counts(
    counts: np.ndarray, c: FloatArray, q: FloatArray, endpoints: FloatArray
) -> list[CiBolusObservation]:
    rows: list[CiBolusObservation] = []
    for ci, concentration in enumerate(c):
        for qi, bolus in enumerate(q):
            for category in range(endpoints.size + 2):
                if category == 0:
                    kind, lower, upper = "bolus", None, None
                elif category == endpoints.size + 1:
                    kind, lower, upper = "failure", None, None
                else:
                    kind = "interval"
                    lower = 0.0 if category == 1 else float(endpoints[category - 2])
                    upper = float(endpoints[category - 1])
                for toxicity in (0, 1):
                    for _ in range(int(counts[ci, qi, category, toxicity])):
                        rows.append(
                            CiBolusObservation(
                                float(concentration),
                                float(bolus),
                                kind,
                                bool(toxicity),
                                lower=lower,
                                upper=upper,
                            )
                        )
    return rows


def counts_shape_cells(repetitions: int, concentrations: int, bolus: int, categories: int) -> int:
    return repetitions * concentrations * bolus * categories * 2


def calibrate_cibolus_prior(
    concentrations: ArrayLike,
    bolus_fractions: ArrayLike,
    endpoints: ArrayLike,
    joint_probabilities: ArrayLike,
    *,
    repetitions: int,
    patients_per_regimen: int,
    pseudo_prior: CiBolusPrior,
    resulting_sd: ArrayLike,
    pseudo_draws: int,
    pseudo_warmup: int,
    pseudo_chains: int,
    rng: np.random.Generator,
    max_total_evaluations: int = _MAX_TOTAL_EVALUATIONS,
    max_total_work: int = _MAX_TOTAL_WORK,
    max_retained_cells: int = _MAX_RETAINED_CELLS,
) -> CiBolusPriorCalibration:
    """Estimate log-prior means from balanced multinomial pseudo-datasets.

    ``joint_probabilities`` explicitly supplies the full response-category ×
    toxicity distribution for each regimen; no interpolation or truth-model
    parameters are inferred. The returned prior uses caller-specified
    ``resulting_sd``. Variances are not optimized by this routine.
    """
    if not isinstance(pseudo_prior, CiBolusPrior) or not isinstance(rng, np.random.Generator):
        raise ValueError("pseudo_prior and explicit NumPy rng are required")
    reps = _integer(repetitions, "repetitions", 1, _MAX_REPETITIONS)
    per_regimen = _integer(
        patients_per_regimen, "patients_per_regimen", 1, _MAX_PATIENTS_PER_REGIMEN
    )
    draws = _integer(pseudo_draws, "pseudo_draws", 8, 100_000)
    warmup = _integer(pseudo_warmup, "pseudo_warmup", 0, 100_000)
    chains = _integer(pseudo_chains, "pseudo_chains", 2, 8)
    total_eval_limit = _integer(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_TOTAL_EVALUATIONS
    )
    total_work_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    retained_limit = _integer(max_retained_cells, "max_retained_cells", 1, _MAX_RETAINED_CELLS)
    c, q, e = _grid(concentrations, bolus_fractions, endpoints)
    regimen_count = c.size * q.size
    pseudo_patients = regimen_count * per_regimen
    if pseudo_patients > _MAX_PSEUDO_PATIENTS:
        raise ValueError("balanced pseudo-dataset exceeds the 400-patient cap")
    category_count = e.size + 2
    truth = _joint_grid(joint_probabilities, (c.size, q.size, category_count, 2))
    fit_retained_per_draw = 12 + c.size * q.size * (category_count * 2 + 5)
    fit_retained = chains * draws * fit_retained_per_draw
    count_cells = counts_shape_cells(reps, c.size, q.size, category_count)
    output_cells = reps * (11 * 5 + 4)
    fit_summary_scratch = 14 * chains * draws * (11 + regimen_count)
    live_cells = (
        truth.size + 2 * fit_retained + fit_summary_scratch + 2 * output_cells + 2 * count_cells
    )
    if live_cells > retained_limit:
        raise ValueError("calibration live arrays exceed max_retained_cells")
    minimum_evaluations_per_fit = chains * (1 + warmup + draws)
    minimum_total_evaluations = reps * minimum_evaluations_per_fit
    per_fit_min_work = minimum_evaluations_per_fit * pseudo_patients + (
        chains * draws * regimen_count * category_count
    )
    if minimum_total_evaluations > total_eval_limit:
        raise ValueError("minimum calibration likelihood calls exceed max_total_evaluations")
    if reps * per_fit_min_work > total_work_limit:
        raise ValueError("minimum calibration work exceeds max_total_work")
    sd = _real_bounded(resulting_sd, "resulting_sd", 11)
    if sd.shape != (11,) or np.any(sd < 0):
        raise ValueError("resulting_sd must contain eleven nonnegative values")

    fit_means = np.empty((reps, 11))
    rhat = np.empty((reps, 11))
    sampler_mcse = np.empty((reps, 11))
    counts = np.empty((reps, c.size, q.size, category_count, 2), dtype=np.int64)
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=reps, dtype=np.uint64)
    evaluations = np.empty(reps, dtype=np.int64)
    work = np.empty(reps, dtype=np.int64)
    total_evaluations = 0
    total_work = 0
    utility = np.zeros((category_count, 2))
    minimum_calls_per_fit = minimum_evaluations_per_fit
    minimum_work_per_fit = per_fit_min_work
    for replicate, seed in enumerate(seeds):
        if total_evaluations + minimum_calls_per_fit > total_eval_limit:
            raise RuntimeError(
                f"calibration exhausted max_total_evaluations before replicate {replicate}"
            )
        if total_work + minimum_work_per_fit > total_work_limit:
            raise RuntimeError(f"calibration exhausted max_total_work before replicate {replicate}")
        replicate_rng = np.random.default_rng(int(seed))
        for ci in range(c.size):
            for qi in range(q.size):
                counts[replicate, ci, qi] = replicate_rng.multinomial(
                    per_regimen, truth[ci, qi].reshape(-1)
                ).reshape((category_count, 2))
        observations = _observations_from_counts(counts[replicate], c, q, e)
        remaining_evaluations = total_eval_limit - total_evaluations
        remaining_work = total_work_limit - total_work
        fit = fit_cibolus(
            observations,
            pseudo_prior,
            c,
            q,
            e,
            utility=utility,
            draws=draws,
            warmup=warmup,
            chains=chains,
            rng=replicate_rng,
            max_evaluations=min(remaining_evaluations, _MAX_TOTAL_EVALUATIONS),
            max_work=min(remaining_work, _MAX_TOTAL_WORK),
        )
        fit_means[replicate] = np.mean(fit.log_parameters, axis=(0, 1))
        rhat[replicate] = fit.parameter_summary.split_rhat
        sampler_mcse[replicate] = fit.parameter_summary.batch_mean_mcse
        evaluations[replicate] = fit.likelihood_evaluations
        work[replicate] = fit.work_units
        total_evaluations += fit.likelihood_evaluations
        total_work += fit.work_units
        del fit, observations
    mean = np.mean(fit_means, axis=0)
    if reps > 1:
        between_sd = np.std(fit_means, axis=0, ddof=1)
        mean_mcse = between_sd / np.sqrt(reps)
    else:
        between_sd = np.full(11, np.nan)
        mean_mcse = np.full(11, np.nan)
    return CiBolusPriorCalibration(
        CiBolusPrior(mean, sd),
        pseudo_prior,
        _freeze_dtype(c, np.dtype(float)),
        _freeze_dtype(q, np.dtype(float)),
        _freeze_dtype(e, np.dtype(float)),
        reps,
        per_regimen,
        truth,
        _freeze_dtype(counts, np.dtype(np.int64)),
        _freeze_dtype(seeds, np.dtype(np.uint64)),
        _freeze_dtype(fit_means, np.dtype(float)),
        _freeze_dtype(between_sd, np.dtype(float)),
        _freeze_dtype(mean_mcse, np.dtype(float)),
        _freeze_dtype(rhat, np.dtype(float)),
        _freeze_dtype(sampler_mcse, np.dtype(float)),
        _freeze_dtype(evaluations, np.dtype(np.int64)),
        _freeze_dtype(work, np.dtype(np.int64)),
    )
