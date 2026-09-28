"""Empirical CRM trial simulation and prior-information ESS paths.

The dose model is ``p(d, beta) = d**exp(beta)`` with a Gaussian prior on beta.
One complete adaptive, unit-cohort trial is simulated per replication. The
source ``essCRM`` resamples uniform subsets of that path for every subset size;
this module Rao--Blackwellizes those subsets, whose expected information is
``m / max_patients`` times the complete-path information. This is a Python
implementation, not an assertion of native random-stream or finite-grid
parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad_vec
from scipy.optimize import brentq

from ._validation import FloatArray, scalar

_MAX_DOSES = 20
_MAX_PATIENTS = 200
_MAX_REPLICATIONS = 1_000
_QUAD_LIMIT = 100
_MAX_EVALS_PER_INTEGRAL = 4_200
_MAX_WORK = 1_000_000_000
_MAX_TAPE_CELLS = 200_000
_LOG_MAX = log(np.finfo(float).max)
_LOG_MIN_SUBNORMAL = log(np.nextafter(0.0, 1.0))


def _freeze(values: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    candidate = scalar(value, name)
    if candidate != np.floor(candidate) or not minimum <= candidate <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return int(candidate)


def _positive_logit_parameter(value: float, name: str) -> float:
    result = scalar(value, name)
    if result <= 0.0 or result > 100.0:
        raise ValueError(f"{name} must lie in (0, 100]")
    return result


def _log_failure_from_log_t(log_t: float) -> float:
    """Return log(1-exp(-t)) from log(t), retaining tiny-tail log mass."""
    if log_t < -36.0:
        return log_t
    if log_t > _LOG_MAX:
        return 0.0
    t = exp(log_t)
    if t < log(2.0):
        return log(-np.expm1(-t))
    return log1p(-exp(-t))


def log1p(value: float) -> float:
    # Local wrapper keeps the import list small while preserving small terms.
    return float(np.log1p(value))


def _t_and_ratio(log_t: float) -> tuple[float, float]:
    if log_t < _LOG_MIN_SUBNORMAL:
        return 0.0, 1.0
    if log_t > _LOG_MAX:
        return float("inf"), 0.0
    t = exp(log_t)
    if t < 1e-5:
        # t / expm1(t), through terms needed at this scale.
        ratio = 1.0 - t / 2.0 + t * t / 12.0
    elif t > 350.0:
        ratio = 0.0
    else:
        ratio = t / np.expm1(t)
    return t, ratio


def _no_event_loglik_curvature(log_t: float) -> float:
    """Second derivative of log(1-exp(-t)) with respect to log(t)."""
    if log_t < -36.0:
        if log_t < _LOG_MIN_SUBNORMAL:
            return 0.0
        t = exp(log_t)
        return -0.5 * t + t * t / 6.0
    if log_t > _LOG_MAX:
        return 0.0
    t = exp(log_t)
    if t < 1e-4:
        return -0.5 * t + t * t / 6.0 - t**4 / 180.0
    ratio = 0.0 if t > 350.0 else t / np.expm1(t)
    return ratio * (1.0 - t - ratio)


def _log_posterior(
    beta: float,
    log_a: FloatArray,
    successes: FloatArray,
    failures: FloatArray,
    beta_sd: float,
) -> float:
    value = -0.5 * (beta / beta_sd) ** 2
    for log_ai, y, n0 in zip(log_a, successes, failures, strict=True):
        if y:
            log_t = float(log_ai + beta)
            if log_t > _LOG_MAX:
                return float("-inf")
            if log_t < _LOG_MIN_SUBNORMAL:
                t = 0.0
            else:
                t = exp(log_t)
            value -= float(y) * t
        if n0:
            value += float(n0) * _log_failure_from_log_t(float(log_ai + beta))
    return float(value)


def _posterior_score(
    beta: float,
    log_a: FloatArray,
    successes: FloatArray,
    failures: FloatArray,
    beta_sd: float,
) -> float:
    score = -beta / (beta_sd * beta_sd)
    for log_ai, y, n0 in zip(log_a, successes, failures, strict=True):
        log_t = float(log_ai + beta)
        t, ratio = _t_and_ratio(log_t)
        if y:
            if not np.isfinite(t):
                return float("-inf")
            score -= float(y) * t
        if n0:
            score += float(n0) * ratio
    return float(score)


def _posterior_curvature(
    beta: float,
    log_a: FloatArray,
    successes: FloatArray,
    failures: FloatArray,
    beta_sd: float,
) -> float:
    curvature = -1.0 / (beta_sd * beta_sd)
    for log_ai, y, n0 in zip(log_a, successes, failures, strict=True):
        log_t = float(log_ai + beta)
        t, _ = _t_and_ratio(log_t)
        if y:
            if not np.isfinite(t):
                return float("-inf")
            curvature -= float(y) * t
        if n0:
            curvature += float(n0) * _no_event_loglik_curvature(log_t)
    return float(curvature)


def _integral(
    function,
    lower: float,
    upper: float,
) -> tuple[FloatArray, float, int]:
    evaluations = 0

    def bounded_function(value: float) -> FloatArray:
        nonlocal evaluations
        evaluations += 1
        if evaluations > _MAX_EVALS_PER_INTEGRAL:
            raise RuntimeError("CRM posterior quadrature exceeded its 4,200-evaluation bound")
        return function(value)

    value, error, info = quad_vec(
        bounded_function,
        lower,
        upper,
        epsabs=1e-12,
        epsrel=2e-10,
        norm="max",
        cache_size=1_048_576,
        limit=_QUAD_LIMIT,
        workers=1,
        full_output=True,
    )
    values = np.asarray(value, dtype=float)
    if not info.success or not np.isfinite(values).all() or not np.isfinite(error):
        raise ArithmeticError(f"CRM posterior quadrature failed: {info.message}")
    return values, float(error), evaluations


def _posterior_moments(
    doses: NDArray[np.int64],
    outcomes: NDArray[np.int8],
    skeleton: FloatArray,
    beta_sd: float,
    convention: Literal["full", "native_truncated_numerator"],
) -> tuple[float, float, int]:
    successes = np.bincount(doses - 1, weights=outcomes, minlength=skeleton.size)
    failures = np.bincount(doses - 1, weights=1 - outcomes, minlength=skeleton.size)
    used = (successes + failures) > 0
    log_a = np.log(-np.log(skeleton[used]))
    successes = successes[used]
    failures = failures[used]

    left, right = -max(8.0 * beta_sd, 8.0), max(8.0 * beta_sd, 8.0)
    for _ in range(80):
        if _posterior_score(left, log_a, successes, failures, beta_sd) > 0.0 and _posterior_score(
            right, log_a, successes, failures, beta_sd
        ) < 0.0:
            break
        left *= 2.0
        right *= 2.0
        if max(abs(left), abs(right)) > 1e6:
            raise ArithmeticError("CRM posterior mode could not be bracketed")
    else:
        raise ArithmeticError("CRM posterior mode search exceeded its iteration bound")
    mode = float(
        brentq(
            lambda b: _posterior_score(b, log_a, successes, failures, beta_sd),
            left,
            right,
            xtol=1e-12,
            rtol=1e-14,
            maxiter=200,
        )
    )
    curvature = _posterior_curvature(mode, log_a, successes, failures, beta_sd)
    if not np.isfinite(curvature) or curvature >= 0.0:
        raise ArithmeticError("CRM posterior mode has invalid curvature")
    scale = 1.0 / sqrt(-curvature)
    peak = _log_posterior(mode, log_a, successes, failures, beta_sd)
    if not np.isfinite(peak) or not np.isfinite(scale):
        raise ArithmeticError("CRM posterior mode is not representable")

    def integrand(z: float) -> FloatArray:
        beta = mode + scale * z
        log_weight = _log_posterior(beta, log_a, successes, failures, beta_sd) - peak
        if np.isnan(log_weight) or log_weight == float("inf"):
            raise ArithmeticError("CRM posterior quadrature encountered invalid log weight")
        weight = float(np.exp(min(0.0, log_weight))) if np.isfinite(log_weight) else 0.0
        return np.asarray([weight, z * weight, z * z * weight])

    total, error, evaluations = _integral(integrand, -np.inf, np.inf)
    norm = float(total[0])
    if norm <= 0.0:
        raise ArithmeticError("CRM posterior normalizing integral is zero")
    if convention == "full":
        moment = total
        rel_error = error / norm
    else:
        lo_z = (-10.0 - mode) / scale
        hi_z = (10.0 - mode) / scale
        if lo_z >= hi_z:
            raise ArithmeticError("native CRM moment interval is not representable")
        # Split the exact finite source window around the posterior mode so
        # narrow priors are not missed by a single very wide transformed
        # interval. The seven symmetric split radii give at most 15 bounded
        # integrals; the work preflight accounts for that exact maximum.
        cuts = {lo_z, hi_z}
        for multiple in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            cuts.add(max(lo_z, min(hi_z, -multiple)))
            cuts.add(max(lo_z, min(hi_z, multiple)))
        ordered = sorted(cuts)
        moment = np.zeros(3, dtype=float)
        numerator_error = 0.0
        numerator_evals = 0
        for lower, upper in zip(ordered[:-1], ordered[1:], strict=True):
            if lower == upper:
                continue
            part, part_error, part_evals = _integral(integrand, lower, upper)
            moment += part
            numerator_error += part_error
            numerator_evals += part_evals
        evaluations += numerator_evals
        rel_error = max(error / norm, numerator_error / norm)
    if rel_error > 2e-7:
        raise ArithmeticError("CRM posterior quadrature did not meet its error tolerance")
    if convention == "full":
        mean_z = float(moment[1] / norm)
        variance_z = float(moment[2] / norm - mean_z * mean_z)
        tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(float(moment[2] / norm)))
        if variance_z < -tolerance:
            raise ArithmeticError("CRM posterior variance is materially negative")
        mean = mode + scale * mean_z
        variance = scale * scale * max(0.0, variance_z)
    else:
        mass = float(moment[0] / norm)
        first = float(moment[1] / norm)
        second = float(moment[2] / norm)
        mean = mode * mass + scale * first
        # These are the source's truncated numerators over the full normalizer,
        # so this diagnostic is not necessarily a probability variance.
        variance = (
            mode * mode * mass * (1.0 - mass)
            + 2.0 * mode * scale * first * (1.0 - mass)
            + scale * scale * (second - first * first)
        )
        tolerance = 128.0 * np.finfo(float).eps * max(1.0, abs(mean) ** 2)
        if variance < -tolerance:
            raise ArithmeticError("native CRM truncated-numerator variance is negative")
        variance = max(0.0, variance)
    if not np.isfinite([mean, variance]).all():
        raise ArithmeticError("CRM posterior moments are not finite")
    return float(mean), float(variance), evaluations


def _plugin_curve(skeleton: FloatArray, beta_mean: float) -> FloatArray:
    log_t = np.log(-np.log(skeleton)) + beta_mean
    probabilities = np.empty_like(skeleton)
    for i, value in enumerate(log_t):
        if value > _LOG_MAX:
            probabilities[i] = 0.0
        elif value < _LOG_MIN_SUBNORMAL:
            probabilities[i] = 1.0
        else:
            probabilities[i] = exp(-exp(float(value)))
    return probabilities


def _information_for_patient(skeleton_probability: float, event: int) -> float:
    """Negative source log-likelihood second derivative at beta=0."""
    t = -log(skeleton_probability)
    if event:
        return t
    if t < 1e-4:
        # d*t*(t - (1-exp(-t))) / (1-exp(-t))**2, with ratios
        # expanded to avoid cancellation for skeletons adjacent to one.
        if t == 0.0:
            return 0.0
        q_over_t = 1.0 - t / 2.0 + t * t / 6.0 - t**3 / 24.0 + t**4 / 120.0
        numerator_over_t2 = 0.5 - t / 6.0 + t * t / 24.0 - t**3 / 120.0
        return exp(-t) * t * numerator_over_t2 / (q_over_t * q_over_t)
    q = -np.expm1(-t)
    numerator = t + np.expm1(-t)
    result = exp(-t) * t * numerator / (q * q)
    if not np.isfinite(result) or result < 0.0:
        raise ArithmeticError("CRM likelihood information is not representable")
    return float(result)


@dataclass(frozen=True)
class CRMPriorESSSimulation:
    """Adaptive CRM paths and two explicitly distinct prior-information matches."""

    true_toxicity: FloatArray
    skeleton: FloatArray
    target: float
    beta_sd: float
    starting_dose: int
    dose_indices: NDArray[np.int64]
    outcomes: NDArray[np.int8]
    outcome_uniforms: FloatArray
    beta_mean_before_patient: FloatArray
    final_beta_mean: FloatArray
    final_beta_variance: FloatArray
    selected_dose: NDArray[np.int64]
    replication_information: FloatArray
    prior_information: float
    mean_subset_information: FloatArray
    information_gap: FloatArray
    continuous_ess: float | None
    native_grid_ess: float
    matching: Literal["continuous", "native_grid"]
    ess_estimate: float | None
    crossing_status: Literal["reached", "not_reached"]
    quadrature_evaluations: int
    work_units: int
    posterior_moments: Literal["full", "native_truncated_numerator"]
    replications: int
    max_patients: int


def simulate_crm_prior_ess(
    true_toxicity: ArrayLike,
    skeleton: ArrayLike,
    target: float,
    *,
    max_patients: int,
    replications: int = 100,
    beta_sd: float = sqrt(1.34),
    starting_dose: int = 1,
    matching: Literal["continuous", "native_grid"] = "continuous",
    posterior_moments: Literal["full", "native_truncated_numerator"] = "full",
    rng: int | np.random.Generator | None = None,
    outcome_uniforms: ArrayLike | None = None,
    max_work: int = 50_000_000,
) -> CRMPriorESSSimulation:
    """Simulate adaptive empirical-CRM trials and calculate the prior ESS path.

    Each trial enrolls one patient at a time. The posterior mean beta under
    ``Normal(0, beta_sd**2)`` is transformed through the prior skeleton, then
    the dose closest to ``target`` is chosen (lower dose wins exact ties). An
    upward move is capped at the current dose after a DLT and at current+1
    after a non-DLT. Dose indices in inputs and outputs are one-based.

    ``posterior_moments='full'`` integrates all posterior moments over the
    real line. ``'native_truncated_numerator'`` reproduces dfcrm's convention
    of dividing moments integrated on [-10,10] by a normalizing integral over
    the full real line. The source's 50-point approximation is separately
    reported as ``native_grid_ess``; it is applied to the Rao--Blackwellized
    mean path and is not the source's finite-subset random stream.

    If ``outcome_uniforms`` is supplied, it must have shape
    ``(replications, max_patients)`` with values in [0,1), allowing exact
    outcome-tape replay. Otherwise ``rng`` supplies the Bernoulli draws.
    ``max_work`` limits a conservative upper bound on dose-cell likelihood
    work (quadrature nodes multiplied by the maximum number of skeleton doses)
    before any random numbers are consumed. Each individual quadrature call
    also has an enforced 4,200-node ceiling.
    """
    raw_truth = np.asarray(true_toxicity)
    raw_skeleton = np.asarray(skeleton)
    if raw_truth.ndim != 1 or raw_skeleton.ndim != 1 or raw_truth.size != raw_skeleton.size:
        raise ValueError("true_toxicity and skeleton must be matching one-dimensional vectors")
    if not 1 <= raw_truth.size <= _MAX_DOSES or np.iscomplexobj(raw_truth) or np.iscomplexobj(raw_skeleton):
        raise ValueError("CRM requires 1..20 real-valued dose probabilities")
    truth = np.asarray(raw_truth, dtype=float)
    prior = np.asarray(raw_skeleton, dtype=float)
    if not np.isfinite(truth).all() or np.any((truth < 0.0) | (truth > 1.0)):
        raise ValueError("true_toxicity must be finite probabilities in [0,1]")
    if not np.isfinite(prior).all() or np.any((prior <= 0.0) | (prior >= 1.0)):
        raise ValueError("skeleton values must lie strictly between zero and one")
    if np.any(np.diff(prior) <= 0.0):
        raise ValueError("skeleton must be strictly increasing")
    target_value = scalar(target, "target")
    if not 0.0 < target_value < 1.0:
        raise ValueError("target must lie strictly between zero and one")
    patients = _integer(max_patients, "max_patients", 1, _MAX_PATIENTS)
    reps = _integer(replications, "replications", 1, _MAX_REPLICATIONS)
    sd = _positive_logit_parameter(beta_sd, "beta_sd")
    prior_variance = sd * sd
    if not np.isfinite(prior_variance) or prior_variance <= 0.0:
        raise ValueError("beta_sd squared must be representable")
    prior_information = 1.0 / prior_variance
    if not np.isfinite(prior_information):
        raise ValueError("inverse beta prior variance must be representable")
    start = _integer(starting_dose, "starting_dose", 1, int(prior.size))
    if matching not in ("continuous", "native_grid"):
        raise ValueError("matching must be 'continuous' or 'native_grid'")
    if posterior_moments not in ("full", "native_truncated_numerator"):
        raise ValueError("unsupported posterior_moments convention")
    if reps * patients > _MAX_TAPE_CELLS:
        raise ValueError("replication-by-patient result arrays exceed 200,000 cells")
    posterior_fits = reps * patients
    integrals_per_fit = 16 if posterior_moments == "native_truncated_numerator" else 1
    # Every quadrature node evaluates the likelihood over at most K dose cells.
    max_work_bound = (
        posterior_fits
        * int(prior.size)
        * integrals_per_fit
        * _MAX_EVALS_PER_INTEGRAL
    )
    work_limit = _integer(max_work, "max_work", 1, _MAX_WORK)
    if max_work_bound > work_limit:
        raise ValueError("conservative posterior quadrature work bound exceeds max_work")

    if outcome_uniforms is not None:
        raw_uniforms = np.asarray(outcome_uniforms)
        if raw_uniforms.shape != (reps, patients) or np.iscomplexobj(raw_uniforms):
            raise ValueError("outcome_uniforms must have shape (replications,max_patients)")
        uniforms = np.asarray(raw_uniforms, dtype=float)
        if not np.isfinite(uniforms).all() or np.any((uniforms < 0.0) | (uniforms >= 1.0)):
            raise ValueError("outcome_uniforms must be finite values in [0,1)")
        generator = None
    else:
        if isinstance(rng, np.random.Generator):
            generator = rng
        elif rng is None or (isinstance(rng, (int, np.integer)) and not isinstance(rng, bool)):
            generator = np.random.default_rng(rng)
        else:
            raise ValueError("rng must be an integer seed, Generator, or None")
        uniforms = np.empty((reps, patients), dtype=float)

    dose_paths = np.empty((reps, patients), dtype=np.int64)
    outcomes = np.empty((reps, patients), dtype=np.int8)
    beta_before = np.empty((reps, patients), dtype=float)
    final_beta = np.empty(reps, dtype=float)
    final_variance = np.empty(reps, dtype=float)
    selected = np.empty(reps, dtype=np.int64)
    replicate_information = np.empty(reps, dtype=float)
    total_evaluations = 0

    for rep in range(reps):
        doses = np.empty(patients, dtype=np.int64)
        y = np.empty(patients, dtype=np.int8)
        for patient in range(patients):
            if patient == 0:
                chosen = start
                beta_before[rep, patient] = 0.0
            else:
                mean_beta, _, evals = _posterior_moments(
                    doses[:patient], y[:patient], prior, sd, posterior_moments
                )
                beta_before[rep, patient] = mean_beta
                total_evaluations += evals
                plugin = _plugin_curve(prior, mean_beta)
                recommendation = int(np.argmin(np.abs(plugin - target_value))) + 1
                cap = int(doses[patient - 1]) if y[patient - 1] else int(doses[patient - 1]) + 1
                chosen = min(recommendation, cap)
            doses[patient] = chosen
            if generator is not None:
                uniforms[rep, patient] = float(generator.random())
            y[patient] = int(uniforms[rep, patient] < truth[chosen - 1])

        final_mean, final_var, evals = _posterior_moments(doses, y, prior, sd, posterior_moments)
        total_evaluations += evals
        final_beta[rep] = final_mean
        final_variance[rep] = final_var
        selected[rep] = int(np.argmin(np.abs(_plugin_curve(prior, final_mean) - target_value))) + 1
        replicate_information[rep] = sum(
            _information_for_patient(float(prior[dose - 1]), int(event))
            for dose, event in zip(doses, y, strict=True)
        )
        dose_paths[rep], outcomes[rep] = doses, y

    mean_full_information = float(np.mean(replicate_information))
    subset_sizes = np.arange(patients + 1, dtype=float)
    subset_information = mean_full_information * (subset_sizes / patients)
    gap = prior_information - subset_information
    crossed = np.flatnonzero(gap <= 0.0)
    if crossed.size:
        hi = int(crossed[0])
        if gap[hi] == 0.0 or hi == 0:
            continuous = float(hi)
        else:
            lo = hi - 1
            continuous = float(lo + gap[lo] / (gap[lo] - gap[hi]))
        crossing_status: Literal["reached", "not_reached"] = "reached"
    else:
        continuous = None
        crossing_status = "not_reached"

    native_grid_x = np.linspace(0.0, float(patients), 50)
    native_grid_gap = prior_information - mean_full_information * (native_grid_x / patients)
    native_grid_ess = float(native_grid_x[int(np.argmin(np.abs(native_grid_gap)))])
    if matching == "continuous":
        estimate = continuous
    else:
        estimate = native_grid_ess

    return CRMPriorESSSimulation(
        _freeze(truth),
        _freeze(prior),
        target_value,
        sd,
        start,
        _freeze(dose_paths, dtype=np.int64),
        _freeze(outcomes, dtype=np.int8),
        _freeze(uniforms),
        _freeze(beta_before),
        _freeze(final_beta),
        _freeze(final_variance),
        _freeze(selected, dtype=np.int64),
        _freeze(replicate_information),
        prior_information,
        _freeze(subset_information),
        _freeze(gap),
        continuous,
        native_grid_ess,
        matching,
        estimate,
        crossing_status,
        total_evaluations,
        max_work_bound,
        posterior_moments,
        reps,
        patients,
    )
