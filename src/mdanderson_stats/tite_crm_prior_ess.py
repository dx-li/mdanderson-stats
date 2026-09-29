"""Scalar-start TITE-CRM trial simulation and prior-information ESS paths.

The dfcrm empirical model is ``p(d, beta) = d**exp(beta)`` with a Gaussian
prior on beta. BayesESS's arrival-time curvature weight and an explicit
elapsed-follow-up alternative are kept as separate criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp, roots_hermitenorm

from ._validation import FloatArray, scalar
from .crm_prior_ess import (
    _freeze,
    _integer,
    _no_event_loglik_curvature,
    _plugin_curve,
)

_MAX_DOSES = 20
_MAX_PATIENTS = 200
_MAX_REPLICATIONS = 1_000
_MAX_TAPE_CELLS = 200_000
_MAX_WORK = 1_000_000_000
_ORDERS = (32, 64, 128, 256, 512, 1024)
_MAX_NODES = sum(_ORDERS)
_RTOL = 2e-7


def _log_likelihood(
    beta: FloatArray,
    doses: NDArray[np.int64],
    outcomes: NDArray[np.int8],
    weights: FloatArray,
    skeleton: FloatArray,
) -> FloatArray:
    result = np.zeros(beta.shape, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        exp_beta = np.exp(beta)
        for dose, event, weight in zip(doses, outcomes, weights, strict=True):
            log_dose = log(float(skeleton[int(dose) - 1]))
            if event:
                result += exp_beta * log_dose
            elif weight:
                log_probability = exp_beta * log_dose
                failure_log_probability = np.log(-np.expm1(log_probability))
                if weight == 1.0:
                    result += failure_log_probability
                else:
                    result += np.logaddexp(
                        np.log1p(-weight), np.log(weight) + failure_log_probability
                    )
    result[np.isnan(result)] = -np.inf
    return result


def _posterior_moments(
    doses: NDArray[np.int64],
    outcomes: NDArray[np.int8],
    weights: FloatArray,
    skeleton: FloatArray,
    beta_sd: float,
    convention: Literal["full", "native_truncated_numerator"] = "full",
) -> tuple[float, float, float, int]:
    """Positive-weight prior-normal quadrature; unresolved refinement is an error."""
    previous: tuple[float, float, float] | None = None
    last_error = float("inf")
    evaluations = 0
    log_denominator: float | None = None
    for order in _ORDERS:
        nodes, weights_gh = roots_hermitenorm(order)
        positive_weights = weights_gh > 0.0
        nodes = np.asarray(nodes)[positive_weights]
        weights_gh = weights_gh[positive_weights]
        log_weights = np.log(weights_gh) - 0.5 * np.log(2.0 * np.pi)
        log_weights += _log_likelihood(nodes * beta_sd, doses, outcomes, weights, skeleton)
        finite = np.isfinite(log_weights)
        if not finite.any():
            raise ArithmeticError("TITE-CRM posterior has zero representable quadrature mass")
        log_norm = float(logsumexp(log_weights[finite]))
        normalized = np.exp(log_weights[finite] - log_norm)
        z = nodes[finite]
        mean_z = float(normalized @ z)
        second_z = float(normalized @ (z * z))
        variance_z = second_z - mean_z * mean_z
        tolerance = 128 * np.finfo(float).eps * max(1.0, abs(second_z))
        if variance_z < -tolerance:
            raise ArithmeticError("TITE-CRM posterior variance is materially negative")
        log_denominator = log_norm
        mean = beta_sd * mean_z
        variance = beta_sd * beta_sd * max(0.0, variance_z)
        evaluations += order
        if previous is not None:
            last_error = max(
                abs(mean - previous[0]) / max(1.0, abs(mean)),
                abs(variance - previous[1]) / max(1.0, abs(variance)),
                abs(log_norm - previous[2]),
            )
            if last_error <= _RTOL:
                break
        previous = mean, variance, log_norm
    else:
        raise ArithmeticError(
            f"TITE-CRM posterior quadrature did not stabilize; relative change={last_error:g}"
        )
    assert log_denominator is not None
    if convention == "native_truncated_numerator":
        # dfcrm uses the full-real denominator but restricts both moment
        # numerators to beta in [-10,10]. Split the interval so narrow prior
        # scales remain resolved rather than integrating one wide interval.
        cuts = {-10.0, 10.0}
        for multiple in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
            cuts.add(max(-10.0, min(10.0, -multiple * beta_sd)))
            cuts.add(max(-10.0, min(10.0, multiple * beta_sd)))
        ordered = sorted(cuts)
        moments = np.zeros(3)
        absolute_error = 0.0
        used = 0
        for left, right in zip(ordered[:-1], ordered[1:], strict=True):
            if right <= left:
                continue
            # Gauss-Legendre refinement is deterministic and bounded. The
            # normalizing scale is chosen from the interval's node values.
            previous_part: NDArray | None = None
            part_error = float("inf")
            for order in (32, 64, 128, 256):
                nodes, gl_weights = leggauss(order)
                beta = (left + right) / 2 + (right - left) / 2 * nodes
                z = beta / beta_sd
                log_values = -0.5 * z * z - np.log(beta_sd * np.sqrt(2 * np.pi))
                log_values += _log_likelihood(beta, doses, outcomes, weights, skeleton)
                peak = float(np.max(log_values))
                if not np.isfinite(peak):
                    part = np.zeros(3)
                elif peak - log_denominator > np.log(np.finfo(float).max):
                    raise ArithmeticError("native TITE numerator is not representable")
                else:
                    factor = np.exp(log_values - peak)
                    scale = ((right - left) / 2) * np.exp(peak - log_denominator)
                    part = (
                        np.asarray(
                            [
                                np.dot(gl_weights, factor),
                                np.dot(gl_weights, factor * beta),
                                np.dot(gl_weights, factor * beta * beta),
                            ]
                        )
                        * scale
                    )
                used += order
                if previous_part is not None:
                    part_error = float(
                        np.max(np.abs(part - previous_part)) / max(1.0, float(np.max(np.abs(part))))
                    )
                    if part_error <= _RTOL:
                        break
                previous_part = part
            else:
                raise ArithmeticError("native TITE moment numerator did not stabilize")
            moments += part
            absolute_error += part_error
        mean = float(moments[1])
        # This follows dfcrm's truncated numerator convention; it is a source
        # diagnostic, not necessarily the variance of a normalized posterior.
        variance = float(moments[2] - mean * mean)
        last_error = max(last_error, absolute_error)
        evaluations += used
    if not np.isfinite([mean, variance, last_error]).all():
        raise ArithmeticError("TITE-CRM posterior moments are not representable")
    return mean, variance, last_error, evaluations


def _source_second_derivative(dose_probability: float, weight: float, outcome: int) -> float:
    """The second derivative used by BayesESS ``getDiff`` at beta=0."""
    t = -log(dose_probability)
    if outcome:
        return -t
    if weight == 1.0:
        return _no_event_loglik_curvature(log(t))
    if weight == 0.0:
        return 0.0
    q = (1.0 - weight) + weight * (-np.expm1(-t))
    if q <= 0.0 or not np.isfinite(q):
        raise ArithmeticError("TITE-CRM ESS curvature denominator is not representable")
    r = weight * dose_probability * t / q
    value = r * (1.0 - t - r)
    if not np.isfinite(value):
        raise ArithmeticError("TITE-CRM ESS curvature is not representable")
    return float(value)


@dataclass(frozen=True)
class TITECRMPriorESSSimulation:
    """Read-only trial paths and curvature ESS diagnostics.

    ``observed_outcomes`` is genuinely as-of for ``criterion="followup"``;
    for ``criterion="native_arrival"`` it intentionally stores eventual
    outcomes, matching the BayesESS ESS-only behavior.
    """

    true_toxicity: FloatArray
    skeleton: FloatArray
    target: float
    beta_sd: float
    obswin: float
    rate: float
    accrual: Literal["fixed", "poisson"]
    criterion: Literal["native_arrival", "followup"]
    assessment_delay: float | None
    starting_dose: int
    posterior_moments: Literal["full", "native_truncated_numerator"]
    dose_indices: NDArray[np.int64]
    latent_outcomes: NDArray[np.int8]
    observed_outcomes: NDArray[np.int8]
    arrivals: FloatArray
    event_delays: FloatArray
    event_study_times: FloatArray
    assessment_weights: FloatArray
    arrival_uniforms: FloatArray | None
    outcome_uniforms: FloatArray
    event_time_uniforms: FloatArray
    beta_mean_before_patient: FloatArray
    final_beta_mean: FloatArray
    final_beta_variance: FloatArray
    selected_dose: NDArray[np.int64]
    replication_second_derivative: FloatArray
    prior_precision: float
    mean_subset_second_derivative: FloatArray
    information_gap: FloatArray
    continuous_ess: float | None
    native_grid_ess: float
    matching: Literal["continuous", "native_grid"]
    ess_estimate: float | None
    crossing_status: Literal["reached", "not_reached"]
    posterior_relative_error: FloatArray
    final_posterior_relative_error: FloatArray
    quadrature_evaluations: int
    work_units: int
    replications: int
    max_patients: int
    seed: int | None


def simulate_tite_crm_prior_ess(
    true_toxicity: ArrayLike,
    skeleton: ArrayLike,
    target: float,
    *,
    max_patients: int,
    obswin: float,
    rate: float,
    accrual: Literal["fixed", "poisson"],
    criterion: Literal["native_arrival", "followup"],
    assessment_delay: float | None = None,
    posterior_moments: Literal["full", "native_truncated_numerator"] = "full",
    starting_dose: int = 1,
    replications: int = 100,
    beta_sd: float = sqrt(1.34),
    matching: Literal["continuous", "native_grid"] = "continuous",
    rng: int | np.random.Generator | None = None,
    arrival_uniforms: ArrayLike | None = None,
    outcome_uniforms: ArrayLike | None = None,
    event_time_uniforms: ArrayLike | None = None,
    max_work: int = 50_000_000,
) -> TITECRMPriorESSSimulation:
    """Run scalar-start TITE-CRM and calculate expected-subset prior ESS.

    ``native_arrival`` reproduces BayesESS's ESS-only defect: eventual outcomes
    are paired with ``min(arrival/obswin,1)``. ``followup`` requires an explicit
    ``assessment_delay`` and uses events observed at ``last_arrival+delay``;
    pending latent DLTs remain unobserved. Dose updates always use elapsed
    follow-up and the dfcrm linear weighting likelihood. Fixed accrual spacing
    is ``obswin/rate``; Poisson interarrival times are exponential with that
    mean. Scalar-start begins at dose one and caps each upward move at
    ``current+1`` irrespective of the previous DLT. Full-real posterior
    moments (or the explicit native truncated-numerator convention) and
    deterministic positive-weight Gauss-Hermite refinement are
    used; unresolved quadrature raises. The returned convergence field is the
    successive-order discrepancy, not a rigorous quadrature bound.

    Supply either a complete replay tape set or ``rng``. Tapes have shape
    ``(replications,max_patients)``; Poisson ``arrival_uniforms`` are required
    and map from (0,1) by ``-log(U)``. Fixed accrual has no arrival tape.
    Outcome/event-time uniforms lie in [0,1). Without tapes, ``rng`` generates
    and the result retains every random tape for exact Python replay.
    Native finite-subset sampling is Rao-Blackwellized: expected subset
    second derivative at size m is m/M times the complete path value.
    """
    raw_truth, raw_skeleton = np.asarray(true_toxicity), np.asarray(skeleton)
    if raw_truth.ndim != 1 or raw_skeleton.ndim != 1 or raw_truth.shape != raw_skeleton.shape:
        raise ValueError("true_toxicity and skeleton must be matching one-dimensional vectors")
    if (
        not 1 <= raw_truth.size <= _MAX_DOSES
        or np.iscomplexobj(raw_truth)
        or np.iscomplexobj(raw_skeleton)
    ):
        raise ValueError("CRM requires 1..20 real-valued dose probabilities")
    truth, prior = np.asarray(raw_truth, dtype=float), np.asarray(raw_skeleton, dtype=float)
    if not np.isfinite(truth).all() or np.any((truth < 0) | (truth > 1)):
        raise ValueError("true_toxicity must contain finite probabilities in [0,1]")
    if (
        not np.isfinite(prior).all()
        or np.any((prior <= 0) | (prior >= 1))
        or np.any(np.diff(prior) <= 0)
    ):
        raise ValueError("skeleton must be finite, strictly increasing, and inside (0,1)")
    target_value = scalar(target, "target")
    if not 0 < target_value < 1:
        raise ValueError("target must lie strictly between zero and one")
    patients = _integer(max_patients, "max_patients", 1, _MAX_PATIENTS)
    reps = _integer(replications, "replications", 1, _MAX_REPLICATIONS)
    if reps * patients > _MAX_TAPE_CELLS:
        raise ValueError("replication-by-patient arrays exceed 200,000 cells")
    if accrual not in ("fixed", "poisson") or criterion not in ("native_arrival", "followup"):
        raise ValueError("accrual and criterion must use documented values")
    if criterion == "followup":
        if assessment_delay is None:
            raise ValueError("assessment_delay is required for criterion='followup'")
        delay = scalar(assessment_delay, "assessment_delay")
        if delay < 0:
            raise ValueError("assessment_delay must be nonnegative")
    else:
        if assessment_delay is not None:
            raise ValueError("assessment_delay is only valid for criterion='followup'")
        delay = None
    window, rate_value = scalar(obswin, "obswin"), scalar(rate, "rate")
    if (
        window <= 0
        or rate_value <= 0
        or not np.isfinite(window / rate_value)
        or window / rate_value <= 0
    ):
        raise ValueError("obswin and rate must define a positive representable interval")
    sd = scalar(beta_sd, "beta_sd")
    if sd <= 0 or sd > 100 or not np.isfinite(sd * sd) or sd * sd <= 0:
        raise ValueError("beta_sd must lie in (0,100] with representable variance")
    prior_precision = 1.0 / (sd * sd)
    if not np.isfinite(prior_precision):
        raise ValueError("inverse beta prior variance must be representable")
    if posterior_moments not in ("full", "native_truncated_numerator"):
        raise ValueError("unsupported posterior_moments convention")
    start_dose = _integer(starting_dose, "starting_dose", 1, int(prior.size))
    if matching not in ("continuous", "native_grid"):
        raise ValueError("matching must be 'continuous' or 'native_grid'")

    shape = (reps, patients)
    checked: dict[str, NDArray | None] = {}
    for name, tape in (
        ("arrival_uniforms", arrival_uniforms),
        ("outcome_uniforms", outcome_uniforms),
        ("event_time_uniforms", event_time_uniforms),
    ):
        if tape is None:
            checked[name] = None
            continue
        raw = np.asarray(tape)
        if raw.shape != shape or np.iscomplexobj(raw):
            raise ValueError(f"{name} must be a real array with shape {shape}")
        checked[name] = np.asarray(raw, dtype=float)
    any_tape = any(value is not None for value in checked.values())
    if any_tape and rng is not None:
        raise ValueError("supply either replay tapes or rng, not both")
    if any_tape and (checked["outcome_uniforms"] is None or checked["event_time_uniforms"] is None):
        raise ValueError("replay requires both outcome_uniforms and event_time_uniforms")
    if accrual == "poisson" and any_tape and checked["arrival_uniforms"] is None:
        raise ValueError("Poisson replay requires arrival_uniforms")
    if accrual == "fixed" and checked["arrival_uniforms"] is not None:
        raise ValueError("arrival_uniforms must be omitted for fixed accrual")
    if checked["arrival_uniforms"] is not None:
        a = checked["arrival_uniforms"]
        assert a is not None
        if not np.isfinite(a).all() or np.any((a <= 0) | (a >= 1)):
            raise ValueError("Poisson arrival_uniforms must lie in (0,1)")
    for key in ("outcome_uniforms", "event_time_uniforms"):
        if checked[key] is not None:
            value = checked[key]
            assert value is not None
            if not np.isfinite(value).all() or np.any((value < 0) | (value >= 1)):
                raise ValueError(f"{key} must lie in [0,1)")

    max_work_value = _integer(max_work, "max_work", 1, _MAX_WORK)
    # Each trial has at most M posterior fits, each can refine through every
    # listed quadrature order and evaluate at most M observations per node.
    native_numerator_nodes = (
        15 * sum((32, 64, 128, 256)) if posterior_moments == "native_truncated_numerator" else 0
    )
    work_bound = reps * patients * (_MAX_NODES + native_numerator_nodes) * patients
    if work_bound > max_work_value:
        raise ValueError("conservative TITE-CRM quadrature work exceeds max_work")
    if not isinstance(rng, np.random.Generator) and not (
        rng is None or (isinstance(rng, (int, np.integer)) and not isinstance(rng, bool))
    ):
        raise ValueError("rng must be an integer seed, Generator, or None")
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    seed_value = (
        int(rng) if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool) else None
    )
    arrival_tape = checked["arrival_uniforms"]
    outcome_tape = checked["outcome_uniforms"]
    event_tape = checked["event_time_uniforms"]
    if arrival_tape is None and accrual == "poisson":
        arrival_tape = generator.random(shape)
        if np.any(arrival_tape == 0):
            raise ArithmeticError("generated Poisson arrival uniform rounded to zero")
    if outcome_tape is None:
        outcome_tape = generator.random(shape)
    if event_tape is None:
        event_tape = generator.random(shape)
    assert outcome_tape is not None and event_tape is not None
    if accrual == "poisson":
        assert arrival_tape is not None

    arrivals = np.empty(shape)
    doses = np.empty(shape, dtype=np.int64)
    latent = np.empty(shape, dtype=np.int8)
    observed = np.empty(shape, dtype=np.int8)
    event_delays = np.full(shape, np.inf)
    ess_weights = np.empty(shape)
    beta_before = np.zeros(shape)
    posterior_errors = np.zeros(shape)
    final_errors = np.empty(reps)
    final_beta, final_variance = np.empty(reps), np.empty(reps)
    selected, replicate_curvature = np.empty(reps, dtype=np.int64), np.empty(reps)
    evaluations = 0
    actual_work_units = 0

    for rep in range(reps):
        if accrual == "fixed":
            arrivals[rep] = (window / rate_value) * np.arange(1, patients + 1)
            if (
                not np.isfinite(arrivals[rep]).all()
                or arrivals[rep, 0] <= 0
                or np.any(np.diff(arrivals[rep]) <= 0)
            ):
                raise ArithmeticError(
                    f"fixed arrival times are unrepresentable in replication {rep}"
                )
        else:
            assert arrival_tape is not None
            increments = -np.log(arrival_tape[rep]) * (window / rate_value)
            arrivals[rep] = np.cumsum(increments)
            if not np.isfinite(arrivals[rep]).all() or np.any(np.diff(arrivals[rep]) <= 0):
                raise ArithmeticError(
                    f"Poisson arrival times are unrepresentable in replication {rep}"
                )
        for i in range(patients):
            if i == 0:
                dose = start_dose
            else:
                elapsed = np.minimum(np.maximum(arrivals[rep, i] - arrivals[rep, :i], 0.0), window)
                asof = event_delays[rep, :i] <= elapsed
                w = elapsed / window
                w[asof] = 1.0
                mean, _, error, used = _posterior_moments(
                    doses[rep, :i], asof.astype(np.int8), w, prior, sd, posterior_moments
                )
                beta_before[rep, i], posterior_errors[rep, i] = mean, error
                evaluations += used
                actual_work_units += used * i
                rec = int(np.argmin(np.abs(_plugin_curve(prior, mean) - target_value))) + 1
                dose = min(rec, int(doses[rep, i - 1]) + 1)
            doses[rep, i] = dose
            event = int(outcome_tape[rep, i] < truth[dose - 1])
            latent[rep, i] = event
            if event:
                event_delays[rep, i] = event_tape[rep, i] * window
        mean, variance, error, used = _posterior_moments(
            doses[rep], latent[rep], np.ones(patients), prior, sd, posterior_moments
        )
        final_beta[rep], final_variance[rep] = mean, variance
        final_errors[rep] = error
        evaluations += used
        actual_work_units += used * patients
        selected[rep] = int(np.argmin(np.abs(_plugin_curve(prior, mean) - target_value))) + 1
        if criterion == "native_arrival":
            ess_y = latent[rep]
            ew = np.minimum(arrivals[rep] / window, 1.0)
        else:
            assert delay is not None
            gap = arrivals[rep, -1] - arrivals[rep]
            elapsed = np.full(patients, window)
            short = gap < window
            remaining = window - gap[short]
            elapsed[short] = np.where(delay >= remaining, window, gap[short] + delay)
            ess_y = (event_delays[rep] <= elapsed).astype(np.int8)
            ew = elapsed / window
            ew[ess_y == 1] = 1.0
        observed[rep], ess_weights[rep] = ess_y, ew
        replicate_curvature[rep] = sum(
            _source_second_derivative(float(prior[d - 1]), float(w), int(y))
            for d, w, y in zip(doses[rep], ew, ess_y, strict=True)
        )

    mean_curvature = float(np.mean(replicate_curvature))
    m = np.arange(patients + 1, dtype=float)
    subset = mean_curvature * (m / patients)
    gap = prior_precision + subset
    exact = np.flatnonzero(gap == 0.0)
    crossing = (
        np.flatnonzero(((gap[:-1] < 0.0) & (gap[1:] > 0.0)) | ((gap[:-1] > 0.0) & (gap[1:] < 0.0)))
        if not exact.size
        else np.empty(0, dtype=int)
    )
    if exact.size:
        continuous: float | None = float(exact[0])
        status: Literal["reached", "not_reached"] = "reached"
    elif crossing.size:
        lo = int(crossing[0])
        continuous = float(lo + gap[lo] / (gap[lo] - gap[lo + 1]))
        status = "reached"
    else:
        continuous, status = None, "not_reached"
    grid = np.linspace(0.0, float(patients), 50)
    native_grid = float(
        grid[int(np.argmin(np.abs(prior_precision + mean_curvature * grid / patients)))]
    )
    estimate = continuous if matching == "continuous" else native_grid

    event_study_times = np.full(shape, np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        event_study_times[latent == 1] = arrivals[latent == 1] + event_delays[latent == 1]
    if not np.isfinite(event_study_times[latent == 1]).all():
        raise ArithmeticError("absolute event study time is not representable")
    return TITECRMPriorESSSimulation(
        _freeze(truth),
        _freeze(prior),
        target_value,
        sd,
        window,
        rate_value,
        accrual,
        criterion,
        delay,
        start_dose,
        posterior_moments,
        _freeze(doses, dtype=np.int64),
        _freeze(latent, dtype=np.int8),
        _freeze(observed, dtype=np.int8),
        _freeze(arrivals),
        _freeze(event_delays),
        _freeze(event_study_times),
        _freeze(ess_weights),
        _freeze(arrival_tape) if arrival_tape is not None else None,
        _freeze(outcome_tape),
        _freeze(event_tape),
        _freeze(beta_before),
        _freeze(final_beta),
        _freeze(final_variance),
        _freeze(selected, dtype=np.int64),
        _freeze(replicate_curvature),
        prior_precision,
        _freeze(subset),
        _freeze(gap),
        continuous,
        native_grid,
        matching,
        estimate,
        status,
        _freeze(posterior_errors),
        _freeze(final_errors),
        evaluations,
        actual_work_units,
        reps,
        patients,
        seed_value,
    )
