"""Physician-elicited single-agent priors for the ToxFinder model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import exp, isfinite, log, log1p

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import least_squares
from scipy.special import gammainc, gammaincc, gammaincinv, gammaln, roots_legendre

from ._validation import FloatArray, finite, scalar
from .toxfinder_model import ToxFinderPrior

_QUADRATURE_ORDER = 128
_MIN_SHAPE = 1e-6
_MAX_SHAPE = 1e4
_MAX_STARTS = 6
_PROBABILITY_TOLERANCE = 2e-5
_ADAPTIVE_ERROR_TOLERANCE = 5e-7


def _readonly(values: ArrayLike) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ToxFinderAgentElicitation:
    """Fitted independent Gamma priors for one agent and equation diagnostics."""

    alpha_shape: float
    alpha_scale: float
    beta_shape: float
    beta_scale: float
    alpha_mean: float
    alpha_variance: float
    beta_mean: float
    beta_variance: float
    negligible_probability: float
    over_target_probability: float
    probability_residuals: FloatArray
    probability_quadrature_errors: FloatArray
    moment_residual: float
    converged: bool
    optimizer_nfev: int
    quadrature_order: int


@dataclass(frozen=True)
class ToxFinderElicitationResult:
    """Two single-agent elicitation fits assembled into a model prior."""

    prior: ToxFinderPrior
    agent1: ToxFinderAgentElicitation
    agent2: ToxFinderAgentElicitation


def _dose_ratios(doses: ArrayLike) -> FloatArray:
    raw = _fixed_vector(doses, 4, "elicited_doses")
    if np.iscomplexobj(raw):
        raise ValueError(
            "elicited doses must contain (negligible, AD, high-toxicity, above-target)"
        )
    values = finite(raw, "elicited_doses")
    if (
        np.any(values <= 0)
        or values[0] >= values[1]
        or values[2] <= values[1]
        or values[3] <= values[1]
    ):
        raise ValueError("doses must satisfy 0 < negligible < AD, with both later doses above AD")
    ratios = values / values[1]
    if np.any(~np.isfinite(ratios)) or ratios[0] >= 1 or ratios[2] <= 1 or ratios[3] <= 1:
        raise ValueError("elicited dose ratios must be finite and positive")
    return ratios


def _real_scalar(value: float, name: str) -> float:
    if isinstance(value, np.ndarray):
        raw = value
    elif np.isscalar(value):
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a real scalar")
    if raw.shape != () or np.iscomplexobj(raw) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar")
    return scalar(raw.item(), name)


def _fixed_vector(value: ArrayLike, size: int, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if len(value) != size:
            raise ValueError(f"{name} must contain exactly {size} values")
        for item in value:
            if isinstance(item, (list, tuple, dict)):
                raise ValueError(f"{name} must be a flat vector")
            if isinstance(item, np.ndarray) and item.ndim != 0:
                raise ValueError(f"{name} must be a flat vector")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a flat vector of length {size}")
    if raw.shape != (size,):
        raise ValueError(f"{name} must be a flat vector of length {size}")
    return raw


def _interaction_moments(value: ArrayLike, name: str, *, variance: bool) -> FloatArray:
    raw = _fixed_vector(value, 2, name)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real alpha3 and beta3 values")
    values = finite(raw, name)
    if np.any(values < 0 if variance else values <= 0):
        condition = "nonnegative" if variance else "positive"
        raise ValueError(f"{name} values must be {condition}")
    return values


def _probabilities(
    log_shapes: FloatArray,
    *,
    alpha_mean: float,
    ratios: FloatArray,
    low_probability: float,
    high_probability: float,
    target: float,
    confidence: float,
    uniform_nodes: FloatArray,
    uniform_weights: FloatArray,
) -> tuple[float, float, float, float, float, float]:
    alpha_shape, beta_shape = np.exp(log_shapes)
    alpha_scale = alpha_mean / alpha_shape
    high_mean = high_probability / (1.0 - high_probability)
    target_mean = alpha_mean
    mgf_ratio = high_mean / target_mean
    log_mgf_ratio = log(mgf_ratio)
    log_ratio3 = log(float(ratios[2]))
    beta_scale = -np.expm1(-log_mgf_ratio / beta_shape) / log_ratio3
    quantiles = gammaincinv(beta_shape, uniform_nodes)
    beta_values = beta_scale * quantiles
    if np.any(~np.isfinite(beta_values)) or np.any(beta_values < 0):
        raise ArithmeticError("Gamma quadrature produced unrepresentable beta values")
    logit_low = log(low_probability) - log1p(-low_probability)
    logit_target = log(target) - log1p(-target)
    # The quantile transform integrates under the Gamma law using normalized
    # uniform-space Gauss-Legendre weights.
    p_low = float(
        np.dot(
            uniform_weights,
            _conditional_values(
                float(alpha_shape),
                float(alpha_scale),
                beta_values,
                float(ratios[0]),
                logit_low,
                upper=False,
            ),
        )
    )
    p_above = float(
        np.dot(
            uniform_weights,
            _conditional_values(
                float(alpha_shape),
                float(alpha_scale),
                beta_values,
                float(ratios[3]),
                logit_target,
                upper=True,
            ),
        )
    )
    return (
        p_low,
        p_above,
        float(alpha_shape),
        float(alpha_scale),
        float(beta_shape),
        float(beta_scale),
    )


def _conditional_values(
    shape: float,
    scale: float,
    beta_values: FloatArray,
    ratio: float,
    log_odds: float,
    *,
    upper: bool,
) -> FloatArray:
    log_argument = log_odds - beta_values * log(ratio) - log(scale)
    with np.errstate(over="ignore", under="ignore"):
        argument = np.exp(log_argument)
    if upper:
        values = np.asarray(gammaincc(shape, argument), dtype=np.float64)
        values[np.isneginf(log_argument)] = 1.0
    else:
        values = np.asarray(gammainc(shape, argument), dtype=np.float64)
        values[np.isposinf(log_argument)] = 1.0
    return values


def _adaptive_probability(
    *,
    alpha_shape: float,
    alpha_scale: float,
    beta_shape: float,
    beta_scale: float,
    dose_ratio: float,
    probability: float,
    upper: bool,
) -> tuple[float, float]:
    """Integrate a probability constraint directly over the beta Gamma density."""
    log_odds = log(probability) - log1p(-probability)
    log_ratio = log(dose_ratio)

    def integrand(beta: float) -> float:
        if beta <= 0:
            return 0.0
        log_density = (
            (beta_shape - 1.0) * log(beta)
            - beta / beta_scale
            - gammaln(beta_shape)
            - beta_shape * log(beta_scale)
        )
        if log_density > 700:
            raise ArithmeticError("Gamma density exceeds the adaptive integration range")
        density = float(np.exp(log_density))
        log_argument = log_odds - beta * log_ratio - log(alpha_scale)
        if upper and log_argument < -745:
            conditional = 1.0
        elif not upper and log_argument > 709:
            conditional = 1.0
        elif log_argument < -745:
            conditional = 0.0
        else:
            argument = float(np.exp(min(log_argument, 709.0)))
            conditional = (
                float(gammaincc(alpha_shape, argument))
                if upper
                else float(gammainc(alpha_shape, argument))
            )
        return density * conditional

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            value, error = quad(
                integrand,
                0.0,
                np.inf,
                epsabs=1e-10,
                epsrel=1e-10,
                limit=256,
            )
    except (IntegrationWarning, OverflowError) as exc:
        raise ArithmeticError("adaptive Gamma-probability integration did not converge") from exc
    if not np.isfinite(value) or not np.isfinite(error):
        raise ArithmeticError("adaptive Gamma-probability integration was non-finite")
    return float(value), float(error)


def _equation_residuals(
    log_shapes: FloatArray,
    *,
    alpha_mean: float,
    ratios: FloatArray,
    low_probability: float,
    high_probability: float,
    target: float,
    confidence: float,
    nodes: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    p_low, p_above, *_ = _probabilities(
        log_shapes,
        alpha_mean=alpha_mean,
        ratios=ratios,
        low_probability=low_probability,
        high_probability=high_probability,
        target=target,
        confidence=confidence,
        uniform_nodes=nodes,
        uniform_weights=weights,
    )
    return np.array([p_low - confidence, p_above - confidence], dtype=np.float64)


def elicit_toxfinder_agent_prior(
    doses: ArrayLike,
    target: float,
    *,
    negligible_probability: float = 0.05,
    high_toxicity_probability: float = 0.60,
    confidence: float = 0.99,
    quadrature_order: int = _QUADRATURE_ORDER,
) -> ToxFinderAgentElicitation:
    """Fit the paper's four Gamma-prior equations for one agent.

    ``doses`` is ``(d1, AD, d3, d4)``: negligible-toxicity dose, target-dose
    reference, high-toxicity dose, and dose almost certainly above target.
    The second equation fixes ``E(alpha)=target/(1-target)``; the third
    analytically determines the Gamma scale of beta for each beta shape. The
    remaining two probability equations are solved by bounded least squares.
    The optimizer bounds, starts, and quadrature order are Python numerical
    conventions, not a claim about the native executable's solver.
    """
    ratios = _dose_ratios(doses)
    goal = _real_scalar(target, "target")
    low = _real_scalar(negligible_probability, "negligible_probability")
    high = _real_scalar(high_toxicity_probability, "high_toxicity_probability")
    certainty = _real_scalar(confidence, "confidence")
    if not 0 < low < goal < high < 1:
        raise ValueError("probabilities must satisfy 0 < negligible < target < high-toxicity < 1")
    if not 0.5 < certainty < 1:
        raise ValueError("confidence must be in (0.5, 1)")
    if (
        isinstance(quadrature_order, (bool, np.bool_))
        or not isinstance(quadrature_order, (int, np.integer))
        or not 32 <= quadrature_order <= 512
    ):
        raise ValueError("quadrature_order must be an integer from 32 through 512")
    alpha_mean = goal / (1.0 - goal)
    high_mean = high / (1.0 - high)
    if high_mean <= alpha_mean:
        raise ValueError("the high-toxicity odds must exceed the target odds for dose d3 > AD")
    nodes, weights = roots_legendre(int(quadrature_order))
    nodes = (nodes + 1.0) / 2.0
    weights = weights / 2.0
    lower, upper = log(_MIN_SHAPE), log(_MAX_SHAPE)
    starts = ((0.5, 2.0), (1.5, 8.0), (3.0, 20.0), (10.0, 50.0), (0.15, 30.0), (30.0, 3.0))
    solutions = []
    optimizer_nfev = 0
    for alpha_start, beta_start in starts[:_MAX_STARTS]:

        def residual_function(log_shapes: FloatArray) -> FloatArray:
            return _equation_residuals(
                log_shapes,
                alpha_mean=alpha_mean,
                ratios=ratios,
                low_probability=low,
                high_probability=high,
                target=goal,
                confidence=certainty,
                nodes=nodes,
                weights=weights,
            )

        result = least_squares(
            residual_function,
            np.log([alpha_start, beta_start]),
            bounds=([lower, lower], [upper, upper]),
            max_nfev=160,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        optimizer_nfev += int(result.nfev)
        score = float(np.max(np.abs(result.fun)))
        if result.success and isfinite(score) and score <= _PROBABILITY_TOLERANCE:
            solutions.append((score, result))
    if not solutions:
        raise ArithmeticError("elicitation equations did not converge within probability tolerance")
    best = min(solutions, key=lambda pair: pair[0])
    for _, other in solutions:
        if not np.allclose(other.x, best[1].x, rtol=1e-4, atol=1e-5):
            raise ArithmeticError(
                "elicitation equations have distinct solutions within search bounds"
            )
    _, solution = best
    if np.any(np.abs(solution.x - np.array([lower, lower])) < 1e-7) or np.any(
        np.abs(solution.x - np.array([upper, upper])) < 1e-7
    ):
        raise ArithmeticError("elicitation fit reached a Gamma-shape search boundary")
    _, _, alpha_shape, alpha_scale, beta_shape, beta_scale = _probabilities(
        np.asarray(solution.x, dtype=np.float64),
        alpha_mean=alpha_mean,
        ratios=ratios,
        low_probability=low,
        high_probability=high,
        target=goal,
        confidence=certainty,
        uniform_nodes=nodes,
        uniform_weights=weights,
    )
    alpha_variance = alpha_shape * alpha_scale * alpha_scale
    beta_mean = beta_shape * beta_scale
    beta_variance = beta_shape * beta_scale * beta_scale
    mgf_product = beta_scale * log(float(ratios[2]))
    if not 0 < mgf_product < 1:
        raise ArithmeticError("the third elicitation moment-generating function is unrepresentable")
    mgf_base_log = log1p(-mgf_product)
    log_expected_odds = log(alpha_mean) - beta_shape * mgf_base_log
    if not isfinite(log_expected_odds) or log_expected_odds > 700:
        raise ArithmeticError("the third elicitation moment is not representable")
    moment_residual = exp(log_expected_odds) - high_mean
    if abs(moment_residual) > 1e-10 * max(1.0, high_mean):
        raise ArithmeticError("the third elicitation moment does not meet its equation")
    adaptive_low, error_low = _adaptive_probability(
        alpha_shape=alpha_shape,
        alpha_scale=alpha_scale,
        beta_shape=beta_shape,
        beta_scale=beta_scale,
        dose_ratio=float(ratios[0]),
        probability=low,
        upper=False,
    )
    adaptive_above, error_above = _adaptive_probability(
        alpha_shape=alpha_shape,
        alpha_scale=alpha_scale,
        beta_shape=beta_shape,
        beta_scale=beta_scale,
        dose_ratio=float(ratios[3]),
        probability=goal,
        upper=True,
    )
    probability_residuals = np.array(
        [adaptive_low - certainty, adaptive_above - certainty], dtype=np.float64
    )
    probability_errors = np.array([error_low, error_above], dtype=np.float64)
    if np.max(np.abs(probability_residuals)) > _PROBABILITY_TOLERANCE:
        raise ArithmeticError(
            "adaptive integration shows that the fitted probability equations miss tolerance"
        )
    if np.max(probability_errors) > _ADAPTIVE_ERROR_TOLERANCE:
        raise ArithmeticError("adaptive integration error exceeds its tolerance")
    moments = np.array([alpha_scale, alpha_variance, beta_mean, beta_variance], dtype=np.float64)
    if np.any(~np.isfinite(moments)) or np.any(moments <= 0):
        raise ArithmeticError("elicited Gamma moments are not finite and positive")
    return ToxFinderAgentElicitation(
        alpha_shape=alpha_shape,
        alpha_scale=alpha_scale,
        beta_shape=beta_shape,
        beta_scale=beta_scale,
        alpha_mean=alpha_mean,
        alpha_variance=alpha_variance,
        beta_mean=beta_mean,
        beta_variance=beta_variance,
        negligible_probability=adaptive_low,
        over_target_probability=adaptive_above,
        probability_residuals=_readonly(probability_residuals),
        probability_quadrature_errors=_readonly(probability_errors),
        moment_residual=moment_residual,
        converged=True,
        optimizer_nfev=optimizer_nfev,
        quadrature_order=int(quadrature_order),
    )


def elicit_toxfinder_prior(
    agent1_doses: ArrayLike,
    agent2_doses: ArrayLike,
    target: float,
    *,
    interaction_mean: ArrayLike,
    interaction_variance: ArrayLike,
    negligible_probability: float = 0.05,
    high_toxicity_probability: float = 0.60,
    confidence: float = 0.99,
    quadrature_order: int = _QUADRATURE_ORDER,
) -> ToxFinderElicitationResult:
    """Elicit both single-agent priors and combine them with explicit interaction moments."""
    # Validate every fixed input before beginning either optimizer.
    _dose_ratios(agent1_doses)
    _dose_ratios(agent2_doses)
    interaction_mean_values = _interaction_moments(
        interaction_mean, "interaction_mean", variance=False
    )
    interaction_variance_values = _interaction_moments(
        interaction_variance, "interaction_variance", variance=True
    )
    free = interaction_variance_values > 0
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        interaction_shape = np.exp(
            2 * np.log(interaction_mean_values)
            - np.log(np.where(free, interaction_variance_values, 1.0))
        )
        interaction_scale = np.exp(
            np.log(np.where(free, interaction_variance_values, 1.0))
            - np.log(interaction_mean_values)
        )
    if np.any(
        free
        & (
            (interaction_shape < 1e-8)
            | ~np.isfinite(interaction_shape)
            | (interaction_scale == 0)
            | ~np.isfinite(interaction_scale)
        )
    ):
        raise ValueError("interaction Gamma moments must have representable shape and scale")
    first = elicit_toxfinder_agent_prior(
        agent1_doses,
        target,
        negligible_probability=negligible_probability,
        high_toxicity_probability=high_toxicity_probability,
        confidence=confidence,
        quadrature_order=quadrature_order,
    )
    second = elicit_toxfinder_agent_prior(
        agent2_doses,
        target,
        negligible_probability=negligible_probability,
        high_toxicity_probability=high_toxicity_probability,
        confidence=confidence,
        quadrature_order=quadrature_order,
    )
    means = [
        first.alpha_mean,
        first.beta_mean,
        second.alpha_mean,
        second.beta_mean,
        *interaction_mean_values,
    ]
    variances = [
        first.alpha_variance,
        first.beta_variance,
        second.alpha_variance,
        second.beta_variance,
        *interaction_variance_values,
    ]
    prior = ToxFinderPrior(means, variances)
    return ToxFinderElicitationResult(prior=prior, agent1=first, agent2=second)
