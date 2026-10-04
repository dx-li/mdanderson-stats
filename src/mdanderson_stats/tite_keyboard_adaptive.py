"""Adaptive TITE-Keyboard timing weights under the paper's joint model.

This is an explicit Python sampler for the paper's adaptive timing scheme. It
uses the exact observed-data likelihood for the shared event-time shapes when
estimating timing weights, then supplies posterior-mean weights to the existing
approximate effective-binomial Keyboard decision rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, lgamma, log, log2, sqrt

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaincc, betaln, logsumexp

from ._validation import FloatArray, count, finite, scalar
from .boin import _owned
from .hierarchical_binomial import ChainSummary, summarize_chains
from .keyboard import KeyboardDesign
from .tite_keyboard import TITEKeyboardDecision, tite_keyboard_decision

_MAX_DOSES = 100
_MAX_PATIENTS = 200
_MAX_CELLS = 2_000_000
_MAX_WORK = 50_000_000
_MAX_CHAINS = 8
_MAX_DRAWS = 100_000
_MAX_WARMUP = 100_000


@dataclass(frozen=True)
class TITEKeyboardAdaptiveWeights:
    """Posterior-mean pending weights and inspectable sampler diagnostics.

    ``pending_weights`` is dose-aligned, preserving each dose's input patient
    order. The diagnostics are empirical convergence checks, not guarantees.
    """

    pending_weights: tuple[FloatArray, ...]
    log_shape_summary: ChainSummary
    weight_mean: FloatArray
    weight_mcse: FloatArray
    weight_split_rhat: FloatArray
    acceptance_rate: FloatArray
    draws: int
    warmup: int
    evaluations: int
    work_units: int
    prior_only: bool
    diagnostics_passed: bool


@dataclass(frozen=True)
class TITEKeyboardAdaptiveDecision:
    """An approximate Keyboard decision paired with its adaptive-weight fit."""

    decision: TITEKeyboardDecision
    weights: TITEKeyboardAdaptiveWeights


def tite_keyboard_adaptive_decision(
    design: KeyboardDesign,
    patients: ArrayLike,
    toxicities: ArrayLike,
    observed_dlt_times: Sequence[ArrayLike],
    pending_followup: Sequence[ArrayLike],
    current_dose: int,
    window: float,
    *,
    lambda_prior: tuple[float, float],
    gamma_prior: tuple[float, float],
    chains: int = 4,
    draws: int = 4_000,
    warmup: int = 1_000,
    rng: np.random.Generator,
    max_work: int = _MAX_WORK,
    max_split_rhat: float = 1.1,
    max_weight_mcse: float = 0.025,
    pending_fraction_limit: float | None = 0.5,
    eliminated: ArrayLike | None = None,
) -> TITEKeyboardAdaptiveDecision:
    """Estimate adaptive weights and apply the existing TITE-Keyboard controller.

    Invalid controller state is checked before the sampler consumes ``rng``.
    If the returned sampling diagnostics do not meet the explicit thresholds,
    this decision wrapper raises; callers can inspect ``tite_keyboard_adaptive_weights``
    directly to review a non-passing fit.
    """
    if not isinstance(design, KeyboardDesign):
        raise TypeError("design must be a KeyboardDesign")
    n = _bounded_dose_counts(patients, "patients")
    y = _bounded_dose_counts(toxicities, "toxicities")
    if n.shape != y.shape or not 2 <= n.size <= _MAX_DOSES or np.any(y > n):
        raise ValueError("require matching patient/toxicity counts for 2..100 doses")
    if np.any((n < 1) | (n > _MAX_PATIENTS)) or n.sum() > _MAX_PATIENTS:
        raise ValueError(f"require 1..{_MAX_PATIENTS} patients and at most {_MAX_PATIENTS} total")
    if eliminated is not None:
        if isinstance(eliminated, np.ndarray):
            if eliminated.shape != n.shape or eliminated.dtype != np.bool_:
                raise ValueError("eliminated must be a matching boolean vector")
        elif (
            not isinstance(eliminated, (list, tuple))
            or len(eliminated) != n.size
            or any(not isinstance(value, (bool, np.bool_)) for value in eliminated)
        ):
            raise ValueError("eliminated must be a matching boolean vector")
    pending_rows = _dose_vectors(pending_followup, "pending_followup")
    event_rows = _dose_vectors(observed_dlt_times, "observed_dlt_times")
    if len(pending_rows) != len(n) or len(event_rows) != len(n):
        raise ValueError("event-time and pending-follow-up inputs must match the dose count")
    duration = _bounded_scalar(window, "window")
    if duration <= 0:
        raise ValueError("window must be positive")
    completed = np.empty(len(n), dtype=float)
    checked_events: list[FloatArray] = []
    checked_pending: list[FloatArray] = []
    for dose, (event_row, pending_row) in enumerate(zip(event_rows, pending_rows, strict=True)):
        event_values = _bounded_patient_vector(event_row, f"observed_dlt_times[{dose}]")
        pending_values = _bounded_patient_vector(pending_row, f"pending_followup[{dose}]")
        if np.any((event_values <= 0) | (event_values >= duration)):
            raise ValueError("observed DLT times must be strictly inside (0, window)")
        if np.any((pending_values < 0) | (pending_values >= duration)):
            raise ValueError("pending follow-up must be in [0, window)")
        if event_values.size != int(y[dose]):
            raise ValueError("observed_dlt_times count must equal observed toxicities per dose")
        checked_events.append(event_values)
        checked_pending.append(pending_values)
        completed[dose] = n[dose] - y[dose] - pending_values.size
        if completed[dose] < 0:
            raise ValueError("pending patients cannot exceed observed non-DLT patients")
    if (
        sum(
            int(y[index]) + int(completed[index]) + checked_pending[index].size
            for index in range(len(n))
        )
        > _MAX_PATIENTS
    ):
        raise ValueError(f"total patient count must not exceed {_MAX_PATIENTS}")
    # Reuse the controller's complete state/rule validation before the sampler
    # consumes rng. The actual sampled weights are injected below.
    tite_keyboard_decision(
        design,
        patients,
        toxicities,
        checked_pending,
        current_dose,
        duration,
        pending_fraction_limit=pending_fraction_limit,
        eliminated=eliminated,
    )
    fit = tite_keyboard_adaptive_weights(
        checked_events,
        completed,
        checked_pending,
        duration,
        lambda_prior=lambda_prior,
        gamma_prior=gamma_prior,
        chains=chains,
        draws=draws,
        warmup=warmup,
        rng=rng,
        max_work=max_work,
        max_split_rhat=max_split_rhat,
        max_weight_mcse=max_weight_mcse,
    )
    if not fit.diagnostics_passed:
        raise ArithmeticError(
            "adaptive timing sampler did not meet the configured R-hat/weight-MCSE checks"
        )
    decision = tite_keyboard_decision(
        design,
        patients,
        toxicities,
        checked_pending,
        current_dose,
        duration,
        pending_weights=fit.pending_weights,
        pending_fraction_limit=pending_fraction_limit,
        eliminated=eliminated,
    )
    return TITEKeyboardAdaptiveDecision(decision, fit)


def tite_keyboard_adaptive_weights(
    observed_dlt_times: Sequence[ArrayLike],
    completed_nondlt: ArrayLike,
    pending_followup: Sequence[ArrayLike],
    window: float,
    *,
    lambda_prior: tuple[float, float],
    gamma_prior: tuple[float, float],
    chains: int = 4,
    draws: int = 4_000,
    warmup: int = 1_000,
    rng: np.random.Generator,
    max_work: int = _MAX_WORK,
    max_split_rhat: float = 1.1,
    max_weight_mcse: float = 0.025,
) -> TITEKeyboardAdaptiveWeights:
    """Estimate adaptive conditional DLT-time CDF weights.

    At each dose, observed DLT event times contribute their scaled-Beta
    densities; completed non-DLT assessments contribute ``1-p``; and each
    pending DLT-free assessment contributes ``1-p*F(u/window)``. The nuisance
    dose toxicity probability has the paper's independent Beta(1,1) prior and
    is integrated out exactly using a nonnegative beta-mixture recurrence.
    The shared timing shapes have explicit independent Gamma(shape, rate)
    priors supplied by the caller; these are a Python prior specification, not
    a recovered native default.

    A componentwise random-walk Metropolis sampler runs on ``(log(lambda),
    log(gamma))``. Posterior mean CDF weights are returned with classic split
    R-hat and batch-means MCSE checks. A decision wrapper should refuse to act
    when ``diagnostics_passed`` is false. This function does not claim that
    fixed-run MCMC diagnostics prove convergence.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy Generator")

    event_rows = _dose_vectors(observed_dlt_times, "observed_dlt_times")
    pending_rows = _dose_vectors(pending_followup, "pending_followup")
    if len(event_rows) != len(pending_rows):
        raise ValueError("event-time and pending-follow-up inputs must have equal dose counts")
    doses = len(event_rows)
    if not 1 <= doses <= _MAX_DOSES:
        raise ValueError(f"dose count must be in 1..{_MAX_DOSES}")
    completed = _bounded_counts(completed_nondlt, "completed_nondlt", doses)
    duration = _bounded_scalar(window, "window")
    if duration <= 0:
        raise ValueError("window must be positive")

    events: list[FloatArray] = []
    pending: list[FloatArray] = []
    total_patients = 0
    for dose, (event_values, pending_values) in enumerate(
        zip(event_rows, pending_rows, strict=True)
    ):
        event = _bounded_patient_vector(event_values, f"observed_dlt_times[{dose}]")
        followup = _bounded_patient_vector(pending_values, f"pending_followup[{dose}]")
        if np.any((event <= 0) | (event >= duration)):
            raise ValueError("observed DLT times must be strictly inside (0, window)")
        if np.any((followup < 0) | (followup >= duration)):
            raise ValueError("pending follow-up must be in [0, window)")
        total_patients += event.size + int(completed[dose]) + followup.size
        if event.size + int(completed[dose]) + followup.size > _MAX_PATIENTS:
            raise ValueError(f"dose {dose + 1} exceeds {_MAX_PATIENTS} patients")
        with np.errstate(under="ignore", over="ignore", invalid="ignore"):
            normalized_event = event / duration
            normalized_followup = followup / duration
        if (
            np.any(~np.isfinite(normalized_event))
            or np.any(normalized_event >= 1)
            or np.any((event > 0) & (normalized_event == 0))
        ):
            raise ValueError("normalized observed DLT times are not representable")
        if (
            np.any(~np.isfinite(normalized_followup))
            or np.any(normalized_followup >= 1)
            or np.any((followup > 0) & (normalized_followup == 0))
        ):
            raise ValueError("normalized pending follow-up is not representable")
        events.append(normalized_event)
        pending.append(normalized_followup)
    if total_patients > _MAX_PATIENTS:
        raise ValueError(f"total patient count must not exceed {_MAX_PATIENTS}")

    lambda_shape, lambda_rate = _gamma_prior(lambda_prior, "lambda_prior")
    gamma_shape, gamma_rate = _gamma_prior(gamma_prior, "gamma_prior")
    chain_count = _bounded_integer(chains, "chains", 2, _MAX_CHAINS)
    draw_count = _bounded_integer(draws, "draws", 8, _MAX_DRAWS)
    warmup_count = _bounded_integer(warmup, "warmup", 0, _MAX_WARMUP)
    work_limit = _bounded_integer(max_work, "max_work", 1, _MAX_WORK)
    rhat_limit = _bounded_scalar(max_split_rhat, "max_split_rhat")
    weight_error_limit = _bounded_scalar(max_weight_mcse, "max_weight_mcse")
    if rhat_limit <= 1 or weight_error_limit <= 0:
        raise ValueError("diagnostic limits must be finite and positive (R-hat > 1)")

    pending_count = sum(values.size for values in pending)
    state_cells = chain_count * draw_count * (16 + 5 * pending_count)
    if state_cells > _MAX_CELLS:
        raise ValueError(f"posterior draws exceed the {_MAX_CELLS}-cell storage bound")
    eval_work = (
        sum((values.size + 1) ** 2 for values in pending)
        + sum(values.size for values in events)
        + doses
    )
    iterations = warmup_count + draw_count
    diagnostic_work = (
        chain_count
        * draw_count
        * (pending_count * (2 + ceil(log2(draw_count + 1))) + 2 * (2 + ceil(log2(draw_count + 1))))
    )
    setup_work = sum(values.size + 1 for values in pending) + sum(values.size for values in events)
    worst_work = (
        chain_count * (1 + 2 * iterations) * eval_work
        + chain_count * draw_count * pending_count
        + diagnostic_work
        + setup_work
    )
    if worst_work > work_limit:
        raise ValueError("max_work is below the conservative sampler work bound")

    event_stats = (
        sum(values.size for values in events),
        sum(float(np.log(values).sum()) for values in events if values.size),
        sum(float(np.log1p(-values).sum()) for values in events if values.size),
    )
    dose_integrals = [
        _dose_beta_integrals(len(events[index]), int(completed[index]), values.size)
        for index, values in enumerate(pending)
    ]
    log_prior_modes = np.array(
        [log(lambda_shape) - log(lambda_rate), log(gamma_shape) - log(gamma_rate)]
    )
    log_float_limits = np.log([np.finfo(float).tiny, np.finfo(float).max])
    if np.any(~np.isfinite(log_prior_modes)) or np.any(
        (log_prior_modes < log_float_limits[0] + 2) | (log_prior_modes > log_float_limits[1] - 2)
    ):
        raise ValueError("Gamma prior modes are outside the representable shape range")
    prior_only = not any(values.size for values in events) and not any(
        np.any(values > 0) for values in pending
    )
    if prior_only:
        shape_draws = np.empty((chain_count, draw_count, 2), dtype=float)
        for chain in range(chain_count):
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                shape_draws[chain, :, 0] = np.exp(
                    _log_gamma_draws(lambda_shape, lambda_rate, draw_count, rng)
                )
                shape_draws[chain, :, 1] = np.exp(
                    _log_gamma_draws(gamma_shape, gamma_rate, draw_count, rng)
                )
        if np.any(~np.isfinite(shape_draws)) or np.any(shape_draws <= 0):
            raise ArithmeticError("Gamma prior generated an unrepresentable timing shape")
        log_shapes = np.log(shape_draws)
        acceptance = np.full(chain_count, np.nan)
        evaluations = 0
        work_units = setup_work
    else:
        shape_draws, acceptance, evaluations, work_units = _sample_shapes(
            pending,
            event_stats,
            dose_integrals,
            lambda_prior=(lambda_shape, lambda_rate),
            gamma_prior=(gamma_shape, gamma_rate),
            chains=chain_count,
            draws=draw_count,
            warmup=warmup_count,
            rng=rng,
            eval_work=eval_work,
            setup_work=setup_work,
        )
        log_shapes = np.log(shape_draws)

    weights_draws = np.empty((chain_count, draw_count, pending_count), dtype=float)
    for chain in range(chain_count):
        for draw in range(draw_count):
            lam, gam = shape_draws[chain, draw]
            if pending_count:
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    weights_draws[chain, draw] = np.concatenate(
                        [betainc(lam, gam, values) for values in pending]
                    )
    shape_summary = summarize_chains(log_shapes)
    weight_means = weights_draws.mean(axis=(0, 1))
    if pending_count:
        weight_summary = summarize_chains(weights_draws)
        weight_mcse = weight_summary.batch_mean_mcse
        weight_rhat = weight_summary.split_rhat
        variable = np.var(weights_draws.reshape((-1, pending_count)), axis=0) > 0
        bad_rhat = variable & (~np.isfinite(weight_rhat) | (weight_rhat > rhat_limit))
        bad_mcse = ~np.isfinite(weight_mcse) | (weight_mcse > weight_error_limit)
    else:
        weight_mcse = np.empty(0, dtype=float)
        weight_rhat = np.empty(0, dtype=float)
        bad_rhat = np.empty(0, dtype=bool)
        bad_mcse = np.empty(0, dtype=bool)
    parameter_rhat = shape_summary.split_rhat
    diagnostics_passed = bool(
        np.all(np.isfinite(parameter_rhat))
        and np.all(parameter_rhat <= rhat_limit)
        and not np.any(bad_rhat)
        and not np.any(bad_mcse)
    )
    aligned_weights: list[FloatArray] = []
    offset = 0
    for values in pending:
        aligned_weights.append(_owned(weight_means[offset : offset + values.size]))
        offset += values.size

    return TITEKeyboardAdaptiveWeights(
        tuple(aligned_weights),
        shape_summary,
        _owned(weight_means),
        _owned(weight_mcse),
        _owned(weight_rhat),
        _owned(acceptance),
        draw_count,
        warmup_count,
        evaluations,
        work_units + chain_count * draw_count * pending_count + diagnostic_work,
        prior_only,
        diagnostics_passed,
    )


def _sample_shapes(
    pending: list[FloatArray],
    event_stats: tuple[int, float, float],
    dose_integrals: list[FloatArray],
    *,
    lambda_prior: tuple[float, float],
    gamma_prior: tuple[float, float],
    chains: int,
    draws: int,
    warmup: int,
    rng: np.random.Generator,
    eval_work: int,
    setup_work: int,
) -> tuple[FloatArray, FloatArray, int, int]:
    total_iterations = warmup + draws
    samples = np.empty((chains, draws, 2), dtype=float)
    acceptance = np.zeros(chains, dtype=float)
    starts = np.array(
        [
            [-0.7, -0.7],
            [-0.7, 0.7],
            [0.7, -0.7],
            [0.7, 0.7],
            [-1.4, 0.0],
            [1.4, 0.0],
            [0.0, -1.4],
            [0.0, 1.4],
        ]
    )
    samples_used = 0
    work_used = 0
    widths = np.ones((chains, 2), dtype=float) * 0.5
    for chain in range(chains):
        prior_modes = np.array(
            [
                log(lambda_prior[0]) - log(lambda_prior[1]),
                log(gamma_prior[0]) - log(gamma_prior[1]),
            ]
        )
        state = prior_modes + starts[chain]
        current = _log_target(
            state, event_stats, pending, dose_integrals, lambda_prior, gamma_prior
        )
        samples_used += 1
        work_used += eval_work
        if not np.isfinite(current):
            raise ArithmeticError("initial adaptive timing posterior state is not finite")
        accepted = 0
        for iteration in range(total_iterations):
            for coordinate in range(2):
                candidate_state = state.copy()
                candidate_state[coordinate] += float(rng.normal(0.0, widths[chain, coordinate]))
                candidate = _log_target(
                    candidate_state,
                    event_stats,
                    pending,
                    dose_integrals,
                    lambda_prior,
                    gamma_prior,
                )
                work_used += eval_work
                samples_used += 1
                log_uniform = np.log1p(-float(rng.random()))
                moved = log_uniform < candidate - current
                if moved:
                    state, current = candidate_state, candidate
                    if iteration >= warmup:
                        accepted += 1
                if iteration < warmup:
                    rate = 1.0 / sqrt(iteration + 10.0)
                    widths[chain, coordinate] = float(
                        np.clip(
                            np.exp(
                                np.log(widths[chain, coordinate]) + rate * (float(moved) - 0.44)
                            ),
                            0.02,
                            4.0,
                        )
                    )
            if iteration >= warmup:
                samples[chain, iteration - warmup] = np.exp(state)
        acceptance[chain] = accepted / (2 * draws)
    return samples, acceptance, samples_used, work_used + setup_work


def _log_target(
    log_shapes: FloatArray,
    event_stats: tuple[int, float, float],
    pending: list[FloatArray],
    dose_integrals: list[FloatArray],
    lambda_prior: tuple[float, float],
    gamma_prior: tuple[float, float],
) -> float:
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        shapes = np.exp(log_shapes)
        if np.any(~np.isfinite(shapes)) or np.any(shapes <= 0):
            return -np.inf
        lam, gam = (float(value) for value in shapes)
        lp = _log_gamma_density_log_coordinate(lam, lambda_prior)
        lp += _log_gamma_density_log_coordinate(gam, gamma_prior)
        if not np.isfinite(lp):
            return -np.inf
        event_count, sum_log_time, sum_log_complement = event_stats
        lp += (lam - 1) * sum_log_time + (gam - 1) * sum_log_complement
        lp -= event_count * betaln(lam, gam)
        for followup, log_integrals in zip(pending, dose_integrals, strict=True):
            lp += _log_marginal_dose(lam, gam, followup, log_integrals)
        return float(lp) if np.isfinite(lp) else -np.inf


def _log_marginal_dose(
    lam: float, gam: float, followup: FloatArray, log_integrals: FloatArray
) -> float:
    log_coefficient = np.array([0.0])
    survival = betaincc(lam, gam, followup)
    if np.any(~np.isfinite(survival)) or np.any((survival < 0) | (survival > 1)):
        return -np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        for value in survival:
            log_survival = float(np.log(value))
            updated = np.full(log_coefficient.size + 1, -np.inf)
            updated[:-1] = np.logaddexp(updated[:-1], log_coefficient)
            updated[1:] = np.logaddexp(updated[1:], log_coefficient + log_survival)
            log_coefficient = updated
    result = float(logsumexp(log_coefficient + log_integrals))
    return result


def _dose_beta_integrals(events: int, completed: int, pending: int) -> FloatArray:
    k = np.arange(pending + 1, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        values = betaln(events + k + 1, completed + pending - k + 1)
    if np.any(~np.isfinite(values)):
        raise ArithmeticError("dose Beta-mixture integral is not representable")
    return values


def _log_gamma_density_log_coordinate(value: float, prior: tuple[float, float]) -> float:
    shape, rate = prior
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return float(shape * log(value) - rate * value + shape * log(rate) - lgamma(shape))


def _dose_vectors(value: Sequence[ArrayLike], name: str) -> list[ArrayLike]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain one bounded vector per dose")
    if len(value) > _MAX_DOSES:
        raise ValueError(f"{name} has more than {_MAX_DOSES} doses")
    return list(value)


def _bounded_patient_vector(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size > _MAX_PATIENTS or value.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) > _MAX_PATIENTS or any(
            isinstance(item, (list, tuple, np.ndarray)) or np.ndim(item) != 0 for item in value
        ):
            raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    else:
        raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    values = finite(value, name)
    if values.ndim != 1 or values.size > _MAX_PATIENTS:
        raise ValueError(f"{name} must be a bounded one-dimensional real vector")
    return np.array(values, dtype=float, copy=True)


def _bounded_counts(value: ArrayLike, name: str, doses: int) -> FloatArray:
    values = _bounded_patient_vector(value, name)
    if values.shape != (doses,) or np.any(values != np.floor(values)):
        raise ValueError(f"{name} must be a nonnegative integer vector with {doses} values")
    result = count(values, name)
    if np.any(result > _MAX_PATIENTS):
        raise ValueError(f"{name} exceeds the per-dose patient bound")
    return result


def _bounded_dose_counts(value: ArrayLike, name: str) -> FloatArray:
    values = _bounded_patient_vector(value, name)
    if values.size > _MAX_DOSES or np.any(values != np.floor(values)):
        raise ValueError(f"{name} must be a bounded dose-count vector")
    return count(values, name)


def _gamma_prior(value: tuple[float, float], name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be (shape, rate)")
    for item in value:
        if isinstance(item, np.ndarray):
            if item.ndim != 0 or item.dtype.kind not in "biuf":
                raise ValueError(f"{name} shape and rate must be real scalars")
        elif not np.isscalar(item):
            raise ValueError(f"{name} shape and rate must be real scalars")
        if np.iscomplexobj(item):
            raise ValueError(f"{name} shape and rate must be real scalars")
    prior = finite(value, name)
    if prior.shape != (2,) or np.any(prior <= 0) or not 0.01 <= prior[0] <= 1e6:
        raise ValueError(f"{name} requires shape in [.01, 1e6] and a positive finite rate")
    return float(prior[0]), float(prior[1])


def _log_gamma_draws(shape: float, rate: float, size: int, rng: np.random.Generator) -> FloatArray:
    """Draw log Gamma(shape, rate) values without multiplying by inverse rate."""
    if shape < 1:
        gamma_values = rng.gamma(shape + 1, size=size)
        uniforms = rng.random(size)
        if np.any(gamma_values <= 0) or np.any(uniforms <= 0):
            raise ArithmeticError("random generator returned an unrepresentable Gamma draw")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.log(gamma_values) + np.log(uniforms) / shape - log(rate)
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("Gamma prior draw is outside the representable log range")
        return result
    gamma_values = rng.gamma(shape, size=size)
    if np.any(gamma_values <= 0):
        raise ArithmeticError("random generator returned an unrepresentable Gamma draw")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.log(gamma_values) - log(rate)
    if np.any(~np.isfinite(result)):
        raise ArithmeticError("Gamma prior draw is outside the representable log range")
    return result


def _bounded_integer(value: int, name: str, lower: int, upper: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    number = _bounded_scalar(value, name)
    if number != int(number) or not lower <= number <= upper:
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(number)


def _bounded_scalar(value: object, name: str) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or value.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a real scalar")
    elif not np.isscalar(value):
        raise ValueError(f"{name} must be a real scalar")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real scalar")
    return scalar(value, name)  # type: ignore[arg-type]
