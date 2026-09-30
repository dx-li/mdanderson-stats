"""Centered-dose 2010 GAO probabilities with a Gaussian-copula joint law."""

import math
from dataclasses import dataclass
from decimal import Decimal, DecimalException, localcontext

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import U2OETProbabilities, _real
from .u2oet_gao import (
    _MAX_CELLS,
    _gaussian_joint,
    _log_ordinal_from_log_hazard,
    _positive_scalar,
)


def _dose_grid_2010(value: ArrayLike, name: str) -> FloatArray:
    doses = _real(value, name)
    if (
        doses.ndim != 1
        or not 2 <= doses.size <= 5
        or np.any(doses < 0)
        or np.any(np.diff(doses) <= 0)
    ):
        raise ValueError(f"{name} must be 2–5 strictly increasing nonnegative raw doses")
    return doses


def _decimal_log_bracket(first: float, second: float, gamma: float) -> float | None:
    """Evaluate a cancellation-prone interaction bracket above float precision."""
    high, low = max(first, second), min(first, second)
    try:
        with localcontext() as context:
            context.prec = 80
            high_d, low_d = Decimal.from_float(high), Decimal.from_float(low)
            # Factor exp(high) before evaluating. Add the unit and interaction
            # terms first so exact cancellation leaves the small positive
            # exp(low-high) term visible (e.g. eta=(1000,0), gamma=-1).
            scaled = Decimal(1) + Decimal.from_float(gamma) * low_d.exp() + (low_d - high_d).exp()
            if scaled <= 0:
                return None
            return float(high_d + scaled.ln())
    except DecimalException as exc:
        raise ArithmeticError(
            "negative gamma interaction is outside the supported Decimal exponent range"
        ) from exc


@dataclass(frozen=True)
class U2OETGAO2010Marginal:
    """One 2010 GAO endpoint, with threshold-by-agent coefficients.

    ``intercepts`` and ``slopes`` have shape ``(levels - 1, 2)``. Slopes act
    on each supplied dose grid after centering at its mean. ``lambda_`` is
    positive; ``gamma`` is shared across thresholds and may be negative only
    where it gives valid continuation probabilities on the requested grid.
    """

    intercepts: FloatArray
    slopes: FloatArray
    lambda_: float
    gamma: float

    def __init__(
        self,
        intercepts: ArrayLike,
        slopes: ArrayLike,
        lambda_: float,
        gamma: float,
    ) -> None:
        alpha = _real(intercepts, "intercepts")
        beta = _real(slopes, "slopes")
        lam = _positive_scalar(lambda_, "lambda_")
        interaction = _real(gamma, "gamma")
        if alpha.ndim != 2 or not 1 <= alpha.shape[0] <= 3 or alpha.shape[1] != 2:
            raise ValueError("intercepts must have shape (levels - 1, 2), with 2–4 levels")
        if beta.shape != alpha.shape:
            raise ValueError("slopes must have the same threshold-by-agent shape as intercepts")
        if interaction.ndim != 0:
            raise ValueError("gamma must be a finite scalar")
        object.__setattr__(self, "intercepts", _freeze(alpha))
        object.__setattr__(self, "slopes", _freeze(beta))
        object.__setattr__(self, "lambda_", lam)
        object.__setattr__(self, "gamma", float(interaction))


def _centered_grid(doses: FloatArray) -> FloatArray:
    """Center using offsets from the first dose to avoid an overflowing mean."""
    offsets = doses - doses[0]
    scale = float(np.max(offsets))
    if scale == 0.0:
        raise ValueError("dose grid must contain at least two distinct values")
    centered = scale * (offsets / scale - np.mean(offsets / scale))
    if not np.all(np.isfinite(centered)):
        raise ArithmeticError("centered dose values exceed floating-point range")
    return centered


def _log_hazard(
    doses1: FloatArray,
    doses2: FloatArray,
    parameters: U2OETGAO2010Marginal,
) -> FloatArray:
    alpha = np.asarray(parameters.intercepts)
    beta = np.asarray(parameters.slopes)
    d1, d2 = _centered_grid(doses1), _centered_grid(doses2)
    with np.errstate(over="ignore", invalid="ignore"):
        eta1 = alpha[:, 0, None] + beta[:, 0, None] * d1[None, :]
        eta2 = alpha[:, 1, None] + beta[:, 1, None] * d2[None, :]
    if not np.all(np.isfinite(eta1)) or not np.all(np.isfinite(eta2)):
        raise ArithmeticError("2010 GAO centered linear predictor exceeds floating-point range")

    log_positive = np.logaddexp(eta1[:, :, None], eta2[:, None, :])
    if parameters.gamma > 0.0:
        with np.errstate(over="ignore", invalid="ignore"):
            eta_sum = eta1[:, :, None] + eta2[:, None, :]
        if not np.all(np.isfinite(eta_sum)):
            raise ArithmeticError("2010 GAO interaction predictor exceeds floating-point range")
        with np.errstate(over="ignore", invalid="ignore"):
            log_interaction = np.log(parameters.gamma) + eta_sum
        if not np.all(np.isfinite(log_interaction)):
            raise ArithmeticError("2010 GAO interaction term exceeds floating-point range")
        log_sum = np.logaddexp(log_positive, log_interaction)
    elif parameters.gamma == 0.0:
        log_sum = log_positive
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            eta_sum = eta1[:, :, None] + eta2[:, None, :]
        if not np.all(np.isfinite(eta_sum)):
            raise ArithmeticError("2010 GAO interaction predictor exceeds floating-point range")
        log_abs_gamma = np.log(-parameters.gamma)
        log_sum = np.empty_like(log_positive)
        for index in np.ndindex(log_positive.shape):
            first = float(eta1[index[0], index[1]])
            second = float(eta2[index[0], index[2]])
            combined = float(eta_sum[index])
            if max(abs(first), abs(second), abs(combined), abs(log_abs_gamma)) <= 300.0:
                # Preserve the represented gamma in the signed sum. This is
                # important when exp(eta1)=exp(eta2)=1 near gamma=-2.
                bracket = math.fsum(
                    (math.exp(first), math.exp(second), parameters.gamma * math.exp(combined))
                )
                magnitude = (
                    math.exp(first) + math.exp(second) + abs(parameters.gamma * math.exp(combined))
                )
                if bracket <= 0.0 or bracket <= 1e-8 * magnitude:
                    precise = _decimal_log_bracket(first, second, parameters.gamma)
                    if precise is None:
                        raise ValueError(
                            "negative gamma makes the 2010 GAO continuation probability invalid "
                            "on at least one dose pair and threshold"
                        )
                    log_sum[index] = precise
                else:
                    log_sum[index] = math.log(bracket)
                continue

            log_negative = log_abs_gamma + combined
            if not np.isfinite(log_negative):
                raise ArithmeticError("2010 GAO interaction term exceeds floating-point range")
            positive_log = float(log_positive[index])
            delta = log_negative - positive_log
            gap = -delta
            cancellation_bound = max(
                1e-8,
                64.0 * np.finfo(float).eps * max(1.0, abs(log_negative), abs(positive_log)),
            )
            if abs(gap) <= cancellation_bound:
                precise = _decimal_log_bracket(first, second, parameters.gamma)
                if precise is None:
                    raise ValueError(
                        "negative gamma makes the 2010 GAO continuation probability invalid "
                        "on at least one dose pair and threshold"
                    )
                log_sum[index] = precise
                continue
            if gap < -cancellation_bound:
                raise ValueError(
                    "negative gamma makes the 2010 GAO continuation probability invalid "
                    "on at least one dose pair and threshold"
                )
            # log(exp(eta1) + exp(eta2) - |gamma| exp(eta1+eta2));
            # expm1/log avoid loss when the scaled terms are well separated.
            log_sum[index] = float(log_positive[index]) + math.log(-math.expm1(delta))

    log_lambda_z = np.log(parameters.lambda_) + log_sum
    if not np.all(np.isfinite(log_lambda_z)):
        raise ArithmeticError("2010 GAO link predictor exceeds floating-point range")
    log_one_plus = np.logaddexp(0.0, log_lambda_z)
    log_log_one_plus = np.empty_like(log_one_plus)
    small = log_lambda_z < -36.0
    log_log_one_plus[small] = log_lambda_z[small]
    log_log_one_plus[~small] = np.log(log_one_plus[~small])
    with np.errstate(over="ignore", invalid="ignore"):
        log_hazard = log_log_one_plus - np.log(parameters.lambda_)
    if not np.all(np.isfinite(log_hazard)):
        raise ArithmeticError("2010 GAO continuation hazard exceeds floating-point range")
    # Threshold first internally; the package result uses dose1,dose2,category.
    return np.moveaxis(log_hazard, 0, -1)


def u2oet_gao2010_probabilities(
    doses1: ArrayLike,
    doses2: ArrayLike,
    *,
    efficacy: U2OETGAO2010Marginal,
    toxicity: U2OETGAO2010Marginal,
    association: float = 0.0,
) -> U2OETProbabilities:
    """Evaluate the 2010 centered-dose GAO model on a dose-pair grid.

    Each endpoint uses its own ``lambda_`` and ``gamma``; ``gamma`` is shared
    over that endpoint's ordinal thresholds. The ordinal continuation model is
    coupled with a Gaussian copula. Negative interactions are accepted when
    the continuation bracket remains positive at every grid pair and
    threshold; invalid parameter/grid combinations raise ``ValueError``.
    """
    d1, d2 = _dose_grid_2010(doses1, "doses1"), _dose_grid_2010(doses2, "doses2")
    if not isinstance(efficacy, U2OETGAO2010Marginal) or not isinstance(
        toxicity, U2OETGAO2010Marginal
    ):
        raise ValueError("efficacy and toxicity must be U2OETGAO2010Marginal values")
    rho_array = _real(association, "association")
    if rho_array.ndim != 0 or abs(float(rho_array)) > 1.0:
        raise ValueError("association must be a scalar in [-1,1]")
    rho = float(rho_array)
    efficacy_categories = efficacy.intercepts.shape[0] + 1
    toxicity_categories = toxicity.intercepts.shape[0] + 1
    cells = int(d1.size) * int(d2.size) * efficacy_categories * toxicity_categories
    if cells > _MAX_CELLS:
        raise ValueError("2010 GAO joint probability grid exceeds the bounded cell budget")

    log_e = _log_ordinal_from_log_hazard(_log_hazard(d1, d2, efficacy))
    log_t = _log_ordinal_from_log_hazard(_log_hazard(d1, d2, toxicity))
    log_joint = _gaussian_joint(log_e, log_t, rho)
    if np.any(np.isnan(log_joint)) or np.any(log_joint > 1e-12):
        raise ArithmeticError("2010 GAO Gaussian-copula log probabilities are invalid")
    joint = np.exp(log_joint)
    if (
        np.max(np.abs(np.sum(joint, axis=(-2, -1)) - 1.0)) > 2e-11
        or np.max(np.abs(np.sum(joint, axis=-1) - np.exp(log_e))) > 2e-11
        or np.max(np.abs(np.sum(joint, axis=-2) - np.exp(log_t))) > 2e-11
    ):
        raise ArithmeticError("2010 GAO copula failed probability or marginal checks")
    return U2OETProbabilities(_freeze(log_e), _freeze(log_t), _freeze(log_joint))
