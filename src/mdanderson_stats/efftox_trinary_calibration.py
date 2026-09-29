"""Python calibration for independent continuation-ratio EffTox priors.

The historical EffTox calibration description does not specify a trinary
calibration procedure. This implementation makes the elicitation policy
explicit: toxicity is calibrated first; conditional efficacy is then fit to
marginal efficacy means and marginal beta-moment ESS, integrating over both
independent prior blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import FloatArray
from .efftox_calibration import (
    _DEFAULT_EVALUATIONS,
    _MAX_EVALUATIONS,
    _MAX_ITERATIONS,
    _MAX_QUADRATURE_WORK,
    _MEAN_BOUNDS,
    _SD_BOUNDS,
    _boundary_hits,
    _calibrate_outcome,
    _order,
    _owned,
    _strict_scalar,
    _truncated_toxicity_moments,
)
from .efftox_model import efftox_standardize
from .efftox_trinary_model import EffToxTrinaryPrior


@dataclass(frozen=True)
class EffToxTrinaryPriorMoments:
    """Marginal prior probability moments and moment-matched beta ESS."""

    efficacy_mean: FloatArray
    efficacy_variance: FloatArray
    efficacy_effective_sample_size: FloatArray
    toxicity_mean: FloatArray
    toxicity_variance: FloatArray
    toxicity_effective_sample_size: FloatArray
    conditional_efficacy_mean: FloatArray
    conditional_efficacy_variance: FloatArray
    standardized_doses: FloatArray
    quadrature_order: int
    quadrature_error: float


@dataclass(frozen=True)
class EffToxTrinaryCalibration:
    """Trinary prior fit, elicited targets and optimizer diagnostics."""

    prior: EffToxTrinaryPrior
    standardized_doses: FloatArray
    efficacy_elicited_mean: FloatArray
    toxicity_elicited_mean: FloatArray
    efficacy_mean_residual: FloatArray
    toxicity_mean_residual: FloatArray
    moments: EffToxTrinaryPriorMoments
    toxicity_target_ess: float
    efficacy_target_ess: float
    toxicity_objective: float
    efficacy_objective: float
    toxicity_optimizer_success: bool
    efficacy_optimizer_success: bool
    toxicity_optimizer_message: str
    efficacy_optimizer_message: str
    toxicity_iterations: int
    efficacy_iterations: int
    toxicity_evaluations: int
    efficacy_evaluations: int
    toxicity_boundary_hits: tuple[str, ...]
    efficacy_boundary_hits: tuple[str, ...]
    quadrature_order: int
    quadrature_error: float
    zero_dose_shift: bool


def _logistic_moments(
    x: FloatArray, mean: float, sd: float, slope_mean: float, slope_sd: float, order: int
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    means, variances, bernoulli_variance, error = _truncated_toxicity_moments(
        x, mean, sd, slope_mean, slope_sd, order
    )
    return means, variances, bernoulli_variance, error


def _joint_moments(
    x: FloatArray,
    toxicity: tuple[FloatArray, FloatArray, FloatArray],
    q_parameters: ArrayLike,
    order: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float]:
    t_mean, t_variance, t_bernoulli_variance = toxicity
    values = np.asarray(q_parameters, dtype=np.float64)
    q_mean, q_variance, q_bernoulli_variance, error = _logistic_moments(
        x,
        float(values[0]),
        float(np.exp(values[2])),
        float(values[1]),
        float(np.exp(values[3])),
        order,
    )
    t_not_mean = _stable_complement(t_mean, t_bernoulli_variance)
    combined = _combine_moments(
        t_mean, t_not_mean, t_variance, q_mean, q_variance, q_bernoulli_variance
    )
    return (*combined, q_mean, q_variance, error)


def _stable_complement(mean: FloatArray, bernoulli_variance: FloatArray) -> FloatArray:
    complement = 1.0 - mean
    np.divide(bernoulli_variance, mean, out=complement, where=mean > 0.5)
    return complement


def _combine_moments(
    t_mean: FloatArray,
    t_not_mean: FloatArray,
    t_variance: FloatArray,
    q_mean: FloatArray,
    q_variance: FloatArray,
    q_bernoulli_variance: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    q_not_mean = _stable_complement(q_mean, q_bernoulli_variance)
    efficacy_mean = t_not_mean * q_mean
    efficacy_variance = (
        t_not_mean**2 * q_variance + q_mean**2 * t_variance + t_variance * q_variance
    )
    efficacy_variance[(t_variance == 0.0) & (q_variance == 0.0)] = 0.0
    if np.any(efficacy_variance < -1e-12) or not np.isfinite(efficacy_variance).all():
        raise ArithmeticError("marginal efficacy variance is invalid")
    efficacy_variance = np.maximum(efficacy_variance, 0.0)
    bernoulli_variance = efficacy_mean * (t_mean + t_not_mean * q_not_mean)
    ess = (
        np.divide(
            bernoulli_variance,
            efficacy_variance,
            out=np.full_like(efficacy_variance, np.inf),
            where=efficacy_variance > 0,
        )
        - 1.0
    )
    if np.any(np.isnan(ess)) or np.any(np.isneginf(ess)):
        raise ArithmeticError("marginal efficacy ESS is invalid")
    return efficacy_mean, efficacy_variance, ess


def efftox_trinary_prior_moments(
    doses: ArrayLike,
    prior: EffToxTrinaryPrior,
    *,
    quadrature_order: int = 40,
    zero_dose_shift: bool = True,
) -> EffToxTrinaryPriorMoments:
    """Compute marginal efficacy/toxicity probability means, variances and ESS."""
    if not isinstance(prior, EffToxTrinaryPrior):
        raise TypeError("prior must be an EffToxTrinaryPrior")
    x = efftox_standardize(doses, zero_dose_shift=zero_dose_shift)
    order = _order(quadrature_order)
    t_mean, t_variance, t_bernoulli_variance, t_error = _truncated_toxicity_moments(
        x,
        float(prior.mean[0]),
        float(prior.sd[0]),
        float(prior.mean[1]),
        float(prior.sd[1]),
        order,
    )
    if prior.sd[3] == 0.0:
        from .efftox_calibration import _normal_logit_moments

        eta_mean = float(prior.mean[2]) + float(prior.mean[3]) * x
        eta_sd = np.sqrt(float(prior.sd[2]) ** 2 + (float(prior.sd[3]) * x) ** 2)
        q_mean, q_variance, q_bernoulli_variance, q_error = _normal_logit_moments(
            eta_mean, eta_sd, limit=max(200, order * 4)
        )
    else:
        q_mean, q_variance, q_bernoulli_variance, q_error = _logistic_moments(
            x,
            float(prior.mean[2]),
            float(prior.sd[2]),
            float(prior.mean[3]),
            float(prior.sd[3]),
            order,
        )
    t_not_mean = _stable_complement(t_mean, t_bernoulli_variance)
    e_mean, e_variance, e_ess = _combine_moments(
        t_mean, t_not_mean, t_variance, q_mean, q_variance, q_bernoulli_variance
    )
    t_ess = (
        np.divide(
            t_bernoulli_variance,
            t_variance,
            out=np.full_like(t_variance, np.inf),
            where=t_variance > 0,
        )
        - 1.0
    )
    return EffToxTrinaryPriorMoments(
        _owned(e_mean),
        _owned(e_variance),
        _owned(e_ess),
        _owned(t_mean),
        _owned(t_variance),
        _owned(t_ess),
        _owned(q_mean),
        _owned(q_variance),
        x,
        order,
        max(t_error, q_error),
    )


def calibrate_efftox_trinary_prior(
    doses: ArrayLike,
    efficacy_means: ArrayLike,
    toxicity_means: ArrayLike,
    *,
    target_ess: float = 0.9,
    efficacy_target_ess: float | None = None,
    toxicity_target_ess: float | None = None,
    quadrature_order: int = 40,
    max_iterations: int = 1_500,
    max_evaluations: int = _DEFAULT_EVALUATIONS,
    zero_dose_shift: bool = True,
) -> EffToxTrinaryCalibration:
    """Fit positive-slope Gaussian prior blocks to trinary probability targets.

    Elicited efficacy is the marginal probability ``E[(1-T)Q]``; toxicity is
    ``E[T]``. The optimizer first fits T, then fits Q while matching marginal
    efficacy means and marginal beta-moment ESS. This sequential policy is a
    documented Python choice, not a claimed historical Windows default.
    """
    x = efftox_standardize(doses, zero_dose_shift=zero_dose_shift)
    raw_e, raw_t = np.asarray(efficacy_means), np.asarray(toxicity_means)
    if np.iscomplexobj(raw_e) or np.iscomplexobj(raw_t):
        raise ValueError("elicited means must be real")
    e, t = np.asarray(raw_e, dtype=np.float64), np.asarray(raw_t, dtype=np.float64)
    if (
        e.shape != x.shape
        or t.shape != x.shape
        or not np.isfinite(e).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("elicited means must be finite and match the dose vector")
    if np.any((e <= 0) | (e >= 1) | (t <= 0) | (t >= 1)):
        raise ValueError("elicited probability means must lie strictly between zero and one")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("toxicity means must be nondecreasing for the positive-slope prior")
    target = _strict_scalar(target_ess, "target_ess")
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
    if any(not 0.01 <= value <= 100 for value in (target, target_e, target_t)):
        raise ValueError("ESS targets must be in [0.01,100]")
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
    work = 8 * int(max_evaluations) * x.size * max(200, 4 * order) * 42
    if work > _MAX_QUADRATURE_WORK:
        raise ValueError("trinary prior calibration exceeds deterministic quadrature work budget")

    t_fit, t_moments, t_objective = _calibrate_outcome(
        x,
        t,
        outcome="toxicity",
        target_ess=target_t,
        curvature_mean=0.0,
        curvature_sd=0.0,
        monotone_toxicity=True,
        order=order,
        max_iterations=int(max_iterations),
        max_evaluations=int(max_evaluations),
    )
    t_pair = _truncated_toxicity_moments(
        x,
        float(t_fit.x[0]),
        float(np.exp(t_fit.x[2])),
        float(t_fit.x[1]),
        float(np.exp(t_fit.x[3])),
        order,
    )[:3]
    t_not_mean = _stable_complement(t_pair[0], t_pair[2])
    if np.any(e >= t_not_mean):
        raise ValueError(
            "elicited marginal efficacy must be below one minus fitted mean toxicity at every dose"
        )
    conditional_targets = e / t_not_mean
    if np.any((conditional_targets <= 0) | (conditional_targets >= 1)):
        raise ValueError("elicited means imply conditional efficacy outside (0,1)")
    endpoint = np.log(conditional_targets[[0, -1]] / (1.0 - conditional_targets[[0, -1]]))
    slope = float((endpoint[1] - endpoint[0]) / (x[-1] - x[0]))
    intercept = float(endpoint[0] - slope * x[0])
    start_sd = 0.7
    start = np.array([intercept, np.clip(slope, -6.0, 6.0), np.log(start_sd), np.log(start_sd)])

    def objective(parameters: FloatArray) -> float:
        try:
            means, _, ess, _, _, _ = _joint_moments(x, t_pair, parameters, order)
        except (ArithmeticError, ValueError, OverflowError):
            return 1e100
        value = float(
            np.sum((e - means) ** 2)
            + 0.1 * (np.mean(ess) - target_e) ** 2
            + 0.02 * (np.exp(parameters[2]) - np.exp(parameters[3])) ** 2
        )
        return value if np.isfinite(value) else 1e100

    q_fit = minimize(
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
            "maxiter": int(max_iterations),
            "maxfev": int(max_evaluations),
            "xatol": 1e-7,
            "fatol": 1e-9,
        },
    )
    prior = EffToxTrinaryPrior(
        np.asarray([t_fit.x[0], t_fit.x[1], q_fit.x[0], q_fit.x[1]], dtype=np.float64),
        np.asarray(
            [np.exp(t_fit.x[2]), np.exp(t_fit.x[3]), np.exp(q_fit.x[2]), np.exp(q_fit.x[3])],
            dtype=np.float64,
        ),
    )
    moments = efftox_trinary_prior_moments(
        doses, prior, quadrature_order=order, zero_dose_shift=zero_dose_shift
    )
    return EffToxTrinaryCalibration(
        prior,
        x,
        _owned(e),
        _owned(t),
        _owned(moments.efficacy_mean - e),
        _owned(moments.toxicity_mean - t),
        moments,
        target_t,
        target_e,
        t_objective,
        float(objective(q_fit.x)),
        bool(t_fit.success),
        bool(q_fit.success),
        str(t_fit.message),
        str(q_fit.message),
        int(t_fit.nit),
        int(q_fit.nit),
        int(t_fit.nfev),
        int(q_fit.nfev),
        _boundary_hits(t_fit.x),
        _boundary_hits(q_fit.x),
        order,
        moments.quadrature_error,
        bool(zero_dose_shift),
    )
