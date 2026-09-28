"""Moment and effective-sample-size calibration for EffTox Gaussian priors.

The calibration objective follows Thall et al. (2014), Eq. 3. Marginal
probability moments are evaluated by deterministic adaptive quadrature. For
efficacy, the Gaussian quadratic coefficient is integrated as part of the
linear predictor variance; toxicity can optionally condition its slope to be
positive, matching :class:`EffToxPrior`'s monotone default. The paper's
rounded calibration table does not establish whether that truncation was used
internally, so the choice is explicit and the result records it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad_vec
from scipy.optimize import OptimizeResult, minimize
from scipy.special import expit, log_ndtr, ndtri_exp

from ._validation import FloatArray, scalar
from .efftox_model import EffToxPrior, _owned, efftox_standardize

EffToxOutcome = Literal["efficacy", "toxicity"]

_MAX_QUADRATURE_ORDER = 256
_MAX_ITERATIONS = 5_000
_MAX_EVALUATIONS = 5_000
_DEFAULT_EVALUATIONS = 2_000
_MAX_QUADRATURE_WORK = 1_000_000_000
_MEAN_BOUNDS = (-1.0e4, 1.0e4)
_SD_BOUNDS = (1.0e-4, 100.0)


@dataclass(frozen=True)
class EffToxPriorMoments:
    """Induced Bernoulli-probability moments and moment-matched beta ESS."""

    outcome: EffToxOutcome
    mean: FloatArray
    variance: FloatArray
    effective_sample_size: FloatArray
    standardized_doses: FloatArray
    monotone_toxicity: bool
    quadrature_order: int
    quadrature_error: float
    quadrature_converged: bool


@dataclass(frozen=True)
class EffToxCalibration:
    """Calibrated prior and achieved efficacy/toxicity marginal summaries.

    The optimizer uses documented Python bounds on the four calibrated
    hyperparameters. ``*_boundary_hits`` identifies any active limits. The
    adaptive integration error and convergence are reported separately from
    optimizer convergence.
    """

    prior: EffToxPrior
    standardized_doses: FloatArray
    efficacy_elicited_mean: FloatArray
    toxicity_elicited_mean: FloatArray
    efficacy: EffToxPriorMoments
    toxicity: EffToxPriorMoments
    efficacy_target_ess: float
    toxicity_target_ess: float
    efficacy_objective: float
    toxicity_objective: float
    efficacy_optimizer_success: bool
    toxicity_optimizer_success: bool
    efficacy_optimizer_message: str
    toxicity_optimizer_message: str
    efficacy_iterations: int
    toxicity_iterations: int
    efficacy_evaluations: int
    toxicity_evaluations: int
    quadrature_order: int
    quadrature_error: float
    quadrature_converged: bool
    efficacy_boundary_hits: tuple[str, ...]
    toxicity_boundary_hits: tuple[str, ...]
    mean_bounds: tuple[float, float]
    sd_bounds: tuple[float, float]
    zero_dose_shift: bool


def _strict_scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must be a finite real scalar")
    return scalar(float(raw), name)


def _order(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("quadrature_order must be an integer")
    result = int(value)
    if not 8 <= result <= _MAX_QUADRATURE_ORDER:
        raise ValueError(f"quadrature_order must be in [8, {_MAX_QUADRATURE_ORDER}]")
    return result


def _normal_logit_moments(
    eta_mean: FloatArray, eta_sd: FloatArray, *, limit: int
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Integrate logistic-normal moments with adaptive standard-normal quadrature."""
    means = np.empty(eta_mean.size, dtype=np.float64)
    variances = np.empty_like(means)
    bernoulli_variances = np.empty_like(means)
    maximum_error = 0.0
    density_scale = 1.0 / np.sqrt(2.0 * np.pi)
    for index, (location, scale) in enumerate(zip(eta_mean, eta_sd, strict=True)):
        if scale == 0.0:
            means[index] = expit(location)
            variances[index] = 0.0
            bernoulli_variances[index] = means[index] * expit(-location)
            continue

        def tails(z: float) -> FloatArray:
            return (
                density_scale
                * np.exp(-0.5 * z * z)
                * np.array([expit(location + scale * z), expit(-location - scale * z)])
            )

        tail_integral, tail_error, info = quad_vec(
            tails, -np.inf, np.inf, epsabs=2e-13, epsrel=2e-12, limit=limit, full_output=True
        )
        if not info.success or not np.isfinite(tail_integral).all():
            raise ArithmeticError("adaptive logistic-normal mean integration did not converge")
        if abs(float(np.sum(tail_integral)) - 1.0) > 1e-8:
            raise ArithmeticError("logistic-normal probability mass did not normalize")
        maximum_error = max(maximum_error, float(tail_error))
        success_mean, failure_mean = (float(v) for v in tail_integral)
        mean = success_mean if success_mean <= 0.5 else 1.0 - failure_mean
        rare_mean = failure_mean if mean > 0.5 else success_mean

        def centered_square(z: float) -> float:
            probability = (
                expit(-location - scale * z) if mean > 0.5 else expit(location + scale * z)
            )
            return density_scale * np.exp(-0.5 * z * z) * (probability - rare_mean) ** 2

        variance, variance_error, variance_info = quad_vec(
            centered_square,
            -np.inf,
            np.inf,
            epsabs=1e-28,
            epsrel=2e-10,
            limit=limit,
            full_output=True,
        )
        if (
            not variance_info.success
            or not np.isfinite(variance)
            or variance < 0.0
            or not np.isfinite(variance_error)
        ):
            raise ArithmeticError("adaptive logistic-normal variance integration did not converge")
        means[index] = mean
        variances[index] = float(variance)
        bernoulli_variances[index] = mean * failure_mean
        maximum_error = max(maximum_error, float(variance_error))
    return means, variances, bernoulli_variances, maximum_error


def _truncated_toxicity_moments(
    x: FloatArray,
    intercept_mean: float,
    intercept_sd: float,
    slope_mean: float,
    slope_sd: float,
    order: int,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Integrate the logistic-normal predictor conditional on a positive slope."""
    if slope_sd == 0.0:
        return _normal_logit_moments(
            intercept_mean + slope_mean * x,
            np.full_like(x, intercept_sd),
            limit=max(200, order * 4),
        )
    threshold = -slope_mean / slope_sd
    if threshold < -12.0:
        # Conditioning changes the normal slope by less than machine precision.
        return _normal_logit_moments(
            intercept_mean + slope_mean * x,
            np.sqrt(intercept_sd**2 + (slope_sd * x) ** 2),
            limit=max(200, order * 4),
        )
    if threshold > 12.0:
        raise ArithmeticError("positive-slope prior is too rare for stable conditioning")

    means = np.empty(x.size, dtype=np.float64)
    variances = np.empty_like(means)
    bernoulli_variances = np.empty_like(means)
    limit = max(200, order * 4)
    maximum_error = 0.0
    for index, dose_code in enumerate(x):
        if dose_code == 0.0:
            point = _normal_logit_moments(
                np.array([intercept_mean]), np.array([intercept_sd]), limit=limit
            )
            means[index] = point[0][0]
            variances[index] = point[1][0]
            bernoulli_variances[index] = point[2][0]
            maximum_error = max(maximum_error, point[3])
            continue
        if intercept_sd == 0.0:
            # The conditional predictor is just a transformed positive normal.
            log_survival = float(log_ndtr(-threshold))

            def slope_tail(u: float) -> FloatArray:
                z = -ndtri_exp(log_survival + np.log1p(-u))
                eta = intercept_mean + dose_code * (slope_mean + slope_sd * z)
                return np.array([expit(eta), expit(-eta)])

            tails, tail_error, info = quad_vec(
                slope_tail, 0.0, 1.0, epsabs=2e-13, epsrel=2e-12, limit=limit, full_output=True
            )
            if not info.success or not np.isfinite(tails).all():
                raise ArithmeticError("positive-slope probability integration did not converge")
            if abs(float(np.sum(tails)) - 1.0) > 1e-8:
                raise ArithmeticError("positive-slope probability mass did not normalize")
            mean = float(tails[0] if tails[0] <= 0.5 else 1.0 - tails[1])
            rare_mean = float(tails[1] if mean > 0.5 else tails[0])

            def slope_variance(u: float) -> float:
                first, second = slope_tail(u)
                difference = (second if mean > 0.5 else first) - rare_mean
                return difference * difference

            variance, variance_error, variance_info = quad_vec(
                slope_variance, 0.0, 1.0, epsabs=1e-28, epsrel=2e-10, limit=limit, full_output=True
            )
            if not variance_info.success or not np.isfinite(variance) or variance < 0.0:
                raise ArithmeticError("positive-slope variance integration did not converge")
            means[index] = mean
            variances[index] = float(variance)
            bernoulli_variances[index] = mean * float(tails[1])
            maximum_error = max(maximum_error, float(tail_error), float(variance_error))
            continue

        eta_sd = float(np.hypot(intercept_sd, dose_code * slope_sd))
        eta_mean = intercept_mean + dose_code * slope_mean
        intercept_ratio = eta_sd / (slope_sd * intercept_sd)
        slope_ratio = dose_code * slope_sd / intercept_sd
        log_conditioning = float(log_ndtr(slope_mean / slope_sd))

        def weighted_tails(z: float) -> FloatArray:
            conditional_slope = intercept_ratio * slope_mean + slope_ratio * z
            log_weight = (
                -0.5 * z * z
                - 0.5 * np.log(2.0 * np.pi)
                + float(log_ndtr(conditional_slope))
                - log_conditioning
            )
            if log_weight > 700.0:
                raise ArithmeticError("conditional predictor density is not representable")
            eta = eta_mean + eta_sd * z
            return np.exp(log_weight) * np.array([expit(eta), expit(-eta)])

        tails, tail_error, info = quad_vec(
            weighted_tails,
            -np.inf,
            np.inf,
            epsabs=2e-13,
            epsrel=2e-12,
            limit=limit,
            full_output=True,
        )
        if not info.success or not np.isfinite(tails).all():
            raise ArithmeticError("positive-slope probability integration did not converge")
        if abs(float(np.sum(tails)) - 1.0) > 1e-8:
            raise ArithmeticError("positive-slope probability mass did not normalize")
        mean = float(tails[0] if tails[0] <= 0.5 else 1.0 - tails[1])
        rare_mean = float(tails[1] if mean > 0.5 else tails[0])

        def centered_square(z: float) -> float:
            conditional_slope = intercept_ratio * slope_mean + slope_ratio * z
            log_weight = (
                -0.5 * z * z
                - 0.5 * np.log(2.0 * np.pi)
                + float(log_ndtr(conditional_slope))
                - log_conditioning
            )
            eta = eta_mean + eta_sd * z
            tail_probability = expit(-eta) if mean > 0.5 else expit(eta)
            return np.exp(log_weight) * (tail_probability - rare_mean) ** 2

        variance, variance_error, variance_info = quad_vec(
            centered_square,
            -np.inf,
            np.inf,
            epsabs=1e-28,
            epsrel=2e-10,
            limit=limit,
            full_output=True,
        )
        if not variance_info.success or not np.isfinite(variance) or variance < 0.0:
            raise ArithmeticError("positive-slope variance integration did not converge")
        means[index] = mean
        variances[index] = float(variance)
        bernoulli_variances[index] = mean * float(tails[1])
        maximum_error = max(maximum_error, float(tail_error), float(variance_error))
    return means, variances, bernoulli_variances, maximum_error


def _moments_from_codes(
    x: FloatArray,
    *,
    outcome: EffToxOutcome,
    intercept_mean: float,
    slope_mean: float,
    intercept_sd: float,
    slope_sd: float,
    curvature_mean: float,
    curvature_sd: float,
    monotone_toxicity: bool,
    order: int,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    limit = max(200, order * 4)
    if outcome == "efficacy":
        # Independent Gaussian coefficients make the complete linear
        # predictor Gaussian at each x, including uncertainty in beta_E2.
        eta_mean = intercept_mean + slope_mean * x + curvature_mean * x**2
        eta_sd = np.sqrt(intercept_sd**2 + (slope_sd * x) ** 2 + (curvature_sd * x**2) ** 2)
        means, variances, bernoulli_variance, error = _normal_logit_moments(
            eta_mean, eta_sd, limit=limit
        )
    elif outcome == "toxicity" and monotone_toxicity and slope_sd > 0:
        means, variances, bernoulli_variance, error = _truncated_toxicity_moments(
            x, intercept_mean, intercept_sd, slope_mean, slope_sd, order
        )
    else:
        eta_mean = intercept_mean + slope_mean * x
        eta_sd = np.sqrt(intercept_sd**2 + (slope_sd * x) ** 2)
        means, variances, bernoulli_variance, error = _normal_logit_moments(
            eta_mean, eta_sd, limit=limit
        )

    if outcome == "efficacy":
        fixed = (
            (intercept_sd == 0.0)
            & ((x == 0.0) | (slope_sd == 0.0))
            & ((x == 0.0) | (curvature_sd == 0.0))
        )
    else:
        fixed = (intercept_sd == 0.0) & ((x == 0.0) | (slope_sd == 0.0))
    variances[fixed] = 0.0
    if (
        not np.isfinite(means).all()
        or not np.isfinite(variances).all()
        or np.any(variances < 0.0)
        or np.any((variances == 0.0) & ~fixed)
    ):
        raise ArithmeticError("quadrature did not resolve finite prior probability variance")
    ess = (
        np.divide(
            bernoulli_variance,
            variances,
            out=np.full_like(variances, np.inf),
            where=variances > 0.0,
        )
        - 1.0
    )
    if np.any(np.isnan(ess)) or np.any(np.isneginf(ess)):
        raise ArithmeticError("moment-matched prior ESS is not representable")
    return means, variances, ess, error


def efftox_prior_moments(
    doses: ArrayLike,
    prior: EffToxPrior,
    *,
    outcome: EffToxOutcome,
    quadrature_order: int = 48,
    zero_dose_shift: bool = True,
) -> EffToxPriorMoments:
    """Compute marginal probability mean, variance, and beta-matched ESS.

    ``doses`` are physical dose values and are standardized with the package's
    existing convention. All coefficient means, SDs, and the monotone-toxicity
    flag are read from ``prior`` so moment calculations cannot silently drift
    from the prior used by ``fit_efftox``. ``quadrature_order`` sets the
    adaptive integrator's subinterval budget (four times the supplied value);
    it is not a fixed Gaussian-node count.
    """
    if outcome not in ("efficacy", "toxicity"):
        raise ValueError("outcome must be 'efficacy' or 'toxicity'")
    if not isinstance(prior, EffToxPrior):
        raise TypeError("prior must be an EffToxPrior")
    order = _order(quadrature_order)
    x = efftox_standardize(doses, zero_dose_shift=zero_dose_shift)
    means, variances, ess, error = _moments_from_codes(
        x,
        outcome=outcome,
        intercept_mean=float(prior.mean[2] if outcome == "efficacy" else prior.mean[0]),
        slope_mean=float(prior.mean[3] if outcome == "efficacy" else prior.mean[1]),
        intercept_sd=float(prior.sd[2] if outcome == "efficacy" else prior.sd[0]),
        slope_sd=float(prior.sd[3] if outcome == "efficacy" else prior.sd[1]),
        curvature_mean=float(prior.mean[4]) if outcome == "efficacy" else 0.0,
        curvature_sd=float(prior.sd[4]) if outcome == "efficacy" else 0.0,
        monotone_toxicity=bool(prior.monotone_toxicity),
        order=order,
    )
    return EffToxPriorMoments(
        outcome,
        _owned(means),
        _owned(variances),
        _owned(ess),
        x,
        bool(prior.monotone_toxicity),
        order,
        error,
        True,
    )


def _calibrate_outcome(
    x: FloatArray,
    elicited: FloatArray,
    *,
    outcome: EffToxOutcome,
    target_ess: float,
    curvature_mean: float,
    curvature_sd: float,
    monotone_toxicity: bool,
    order: int,
    max_iterations: int,
    max_evaluations: int,
) -> tuple[OptimizeResult, EffToxPriorMoments, float]:
    endpoint_logits = np.log(elicited / (1.0 - elicited))
    span = float(x[-1] - x[0])
    initial_slope = float((endpoint_logits[-1] - endpoint_logits[0]) / span)
    initial_intercept = float(endpoint_logits[0] - initial_slope * x[0])
    p_endpoint = elicited[[0, -1]]
    x_endpoint = x[[0, -1]]
    target_variance = p_endpoint * (1.0 - p_endpoint) / (target_ess + 1.0)
    # Paper initialization: endpoint logits determine the line, while the
    # common starting SD balances endpoint beta-moment variances.
    starting_sd_squared = np.sqrt(np.prod(target_variance)) / float(
        np.mean((p_endpoint * (1.0 - p_endpoint)) ** 2 * (1.0 + x_endpoint**2))
    )
    if not np.isfinite(starting_sd_squared) or starting_sd_squared <= 0.0:
        raise ArithmeticError("paper's initial prior SD is not representable")
    start_sd = float(np.clip(np.sqrt(starting_sd_squared), *_SD_BOUNDS))
    start = np.array([initial_intercept, initial_slope, np.log(start_sd), np.log(start_sd)])

    def moments(parameters: FloatArray) -> EffToxPriorMoments:
        means, variances, ess, error = _moments_from_codes(
            x,
            outcome=outcome,
            intercept_mean=float(parameters[0]),
            slope_mean=float(parameters[1]),
            intercept_sd=float(np.exp(parameters[2])),
            slope_sd=float(np.exp(parameters[3])),
            curvature_mean=curvature_mean,
            curvature_sd=curvature_sd if outcome == "efficacy" else 0.0,
            monotone_toxicity=monotone_toxicity,
            order=order,
        )
        return EffToxPriorMoments(
            outcome,
            _owned(means),
            _owned(variances),
            _owned(ess),
            x,
            monotone_toxicity,
            order,
            error,
            True,
        )

    def objective(parameters: FloatArray) -> float:
        try:
            current = moments(parameters)
        except ArithmeticError:
            # Saturated trial vertices can have no numerically resolvable
            # probability variance; keep the optimizer inside resolvable
            # regions without hiding failures at the reported optimum.
            return 1e100
        mean_penalty = float(np.sum((elicited - current.mean) ** 2))
        ess_penalty = 0.1 * float((np.mean(current.effective_sample_size) - target_ess) ** 2)
        sd_penalty = 0.02 * float((np.exp(parameters[2]) - np.exp(parameters[3])) ** 2)
        value = mean_penalty + ess_penalty + sd_penalty
        return value if np.isfinite(value) else 1e100

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        bounds=(
            _MEAN_BOUNDS,
            _MEAN_BOUNDS,
            (np.log(_SD_BOUNDS[0]), np.log(_SD_BOUNDS[1])),
            (np.log(_SD_BOUNDS[0]), np.log(_SD_BOUNDS[1])),
        ),
        options={
            "maxiter": max_iterations,
            "maxfev": max_evaluations,
            "xatol": 1e-7,
            "fatol": 1e-9,
        },
    )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    final_moments = moments(final_parameters)
    final_objective = objective(final_parameters)
    return result, final_moments, final_objective


def _boundary_hits(parameters: ArrayLike) -> tuple[str, ...]:
    values = np.asarray(parameters, dtype=np.float64)
    bounds = (
        _MEAN_BOUNDS,
        _MEAN_BOUNDS,
        (np.log(_SD_BOUNDS[0]), np.log(_SD_BOUNDS[1])),
        (np.log(_SD_BOUNDS[0]), np.log(_SD_BOUNDS[1])),
    )
    names = ("intercept_mean", "slope_mean", "intercept_sd", "slope_sd")
    hits: list[str] = []
    for name, value, (lower, upper) in zip(names, values, bounds, strict=True):
        if np.isclose(value, lower, rtol=1e-8, atol=1e-10):
            hits.append(f"{name}:lower")
        if np.isclose(value, upper, rtol=1e-8, atol=1e-10):
            hits.append(f"{name}:upper")
    return tuple(hits)


def calibrate_efftox_prior(
    doses: ArrayLike,
    efficacy_means: ArrayLike,
    toxicity_means: ArrayLike,
    *,
    target_ess: float = 0.9,
    efficacy_target_ess: float | None = None,
    toxicity_target_ess: float | None = None,
    curvature_mean: float = 0.0,
    curvature_sd: float = 0.2,
    association_mean: float = 0.0,
    association_sd: float = 1.0,
    monotone_toxicity: bool = True,
    quadrature_order: int = 40,
    max_iterations: int = 1_500,
    max_evaluations: int = _DEFAULT_EVALUATIONS,
    zero_dose_shift: bool = True,
) -> EffToxCalibration:
    """Calibrate independent efficacy and toxicity regression priors.

    Each outcome minimizes the paper's three-term objective independently.
    The E quadratic coefficient and association have explicit normal priors;
    the default toxic slope is truncated positive to agree with the package's
    monotone EffTox model. Set ``monotone_toxicity=False`` to calibrate an
    untruncated Gaussian slope instead. This is a Python deterministic-
    quadrature implementation; it does not claim exact Windows-program
    optimizer or integration parity. Hypermeans are bounded to +/-1e4 and
    hyper-SDs to [1e-4,100], matching the supported EffTox prior domain except
    for the positive SD floor needed by log-scale optimization.
    """
    x = efftox_standardize(doses, zero_dose_shift=zero_dose_shift)
    elicited_e = np.asarray(efficacy_means)
    elicited_t = np.asarray(toxicity_means)
    if np.iscomplexobj(elicited_e) or np.iscomplexobj(elicited_t):
        raise ValueError("elicited means must be real")
    elicited_e = np.asarray(elicited_e, dtype=np.float64)
    elicited_t = np.asarray(elicited_t, dtype=np.float64)
    if elicited_e.shape != x.shape or elicited_t.shape != x.shape:
        raise ValueError("elicited efficacy and toxicity means must match the dose vector")
    if not np.isfinite(elicited_e).all() or not np.isfinite(elicited_t).all():
        raise ValueError("elicited means must be finite")
    if np.any((elicited_e <= 0.0) | (elicited_e >= 1.0)) or np.any(
        (elicited_t <= 0.0) | (elicited_t >= 1.0)
    ):
        raise ValueError("elicited probabilities must be strictly between zero and one")
    if not isinstance(monotone_toxicity, (bool, np.bool_)):
        raise ValueError("monotone_toxicity must be boolean")
    target = _strict_scalar(target_ess, "target_ess")
    curv_mean = _strict_scalar(curvature_mean, "curvature_mean")
    curv_sd = _strict_scalar(curvature_sd, "curvature_sd")
    assoc_mean = _strict_scalar(association_mean, "association_mean")
    assoc_sd = _strict_scalar(association_sd, "association_sd")
    target_e = (
        target
        if efficacy_target_ess is None
        else _strict_scalar(efficacy_target_ess, "efficacy_target_ess")
    )
    target_t = (
        target
        if toxicity_target_ess is None
        else _strict_scalar(toxicity_target_ess, "toxicity_target_ess")
    )
    if (
        not 0.01 <= target <= 100
        or not 0.01 <= target_e <= 100
        or not 0.01 <= target_t <= 100
        or abs(curv_mean) > 10
        or not 0 <= curv_sd <= 10
    ):
        raise ValueError(
            "ESS targets must be in [0.01,100]; curvature prior must be finite and bounded"
        )
    if abs(assoc_mean) > 10 or not 0 < assoc_sd <= 10:
        raise ValueError("association prior mean must be within +/-10 and SD in (0,10]")
    order = _order(quadrature_order)
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or not 100 <= max_iterations <= _MAX_ITERATIONS
    ):
        raise ValueError(f"max_iterations must be in [100, {_MAX_ITERATIONS}]")
    if (
        isinstance(max_evaluations, (bool, np.bool_))
        or not isinstance(max_evaluations, (int, np.integer))
        or not 100 <= max_evaluations <= _MAX_EVALUATIONS
    ):
        raise ValueError(f"max_evaluations must be in [100, {_MAX_EVALUATIONS}]")
    # quad_vec's limit is a subinterval cap, not an evaluation count. This
    # work-unit estimate conservatively bounds integrals, doses, optimizer
    # evaluations, and quadrature subdivisions before fitting starts.
    base_limit = max(200, 4 * order)
    quadrature_work = (
        4 * int(max_evaluations) * x.size * base_limit * 42 + 4 * x.size * base_limit * 42
    )
    if quadrature_work > _MAX_QUADRATURE_WORK:
        raise ValueError("prior calibration exceeds the deterministic quadrature work budget")
    result_e, moments_e, objective_e = _calibrate_outcome(
        x,
        elicited_e,
        outcome="efficacy",
        target_ess=target_e,
        curvature_mean=curv_mean,
        curvature_sd=curv_sd,
        monotone_toxicity=bool(monotone_toxicity),
        order=order,
        max_iterations=int(max_iterations),
        max_evaluations=int(max_evaluations),
    )
    result_t, moments_t, objective_t = _calibrate_outcome(
        x,
        elicited_t,
        outcome="toxicity",
        target_ess=target_t,
        curvature_mean=0.0,
        curvature_sd=0.0,
        monotone_toxicity=bool(monotone_toxicity),
        order=order,
        max_iterations=int(max_iterations),
        max_evaluations=int(max_evaluations),
    )
    prior_mean = np.array(
        [result_t.x[0], result_t.x[1], result_e.x[0], result_e.x[1], curv_mean, assoc_mean]
    )
    prior_sd = np.array(
        [
            np.clip(np.exp(result_t.x[2]), *_SD_BOUNDS),
            np.clip(np.exp(result_t.x[3]), *_SD_BOUNDS),
            np.clip(np.exp(result_e.x[2]), *_SD_BOUNDS),
            np.clip(np.exp(result_e.x[3]), *_SD_BOUNDS),
            curv_sd,
            assoc_sd,
        ]
    )
    prior = EffToxPrior(prior_mean, prior_sd, bool(monotone_toxicity))
    return EffToxCalibration(
        prior,
        x,
        _owned(elicited_e),
        _owned(elicited_t),
        moments_e,
        moments_t,
        target_e,
        target_t,
        objective_e,
        objective_t,
        bool(result_e.success),
        bool(result_t.success),
        str(result_e.message),
        str(result_t.message),
        int(result_e.nit),
        int(result_t.nit),
        int(result_e.nfev),
        int(result_t.nfev),
        order,
        max(moments_e.quadrature_error, moments_t.quadrature_error),
        moments_e.quadrature_converged and moments_t.quadrature_converged,
        _boundary_hits(result_e.x),
        _boundary_hits(result_t.x),
        _MEAN_BOUNDS,
        _SD_BOUNDS,
        bool(zero_dose_shift),
    )
