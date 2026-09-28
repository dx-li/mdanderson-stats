"""Explicit-coefficient GAO marginals with a Gaussian-copula joint law."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri_exp

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import U2OETProbabilities, _real
from .u2oet_scenario import _rectangle

_MAX_CELLS = 20_000_000
_LOG_HALF = -np.log(2.0)
_LOG_MAX = np.log(np.finfo(float).max)


def _dose_grid(value: ArrayLike, name: str) -> FloatArray:
    doses = _real(value, name)
    if (
        doses.ndim != 1
        or not 2 <= doses.size <= 5
        or np.any(doses <= 0)
        or np.any(np.diff(doses) <= 0)
    ):
        raise ValueError(f"{name} must be 2–5 strictly increasing positive raw doses")
    return doses


def _positive_scalar(value: object, name: str) -> float:
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite positive scalar")
    result = float(scalar)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


@dataclass(frozen=True)
class U2OETGAOMarginal:
    """One ordinal GAO marginal with threshold-by-agent coefficients.

    ``intercepts`` and ``slopes`` have shape ``(levels - 1, 2)``. Slopes are
    unrestricted real values and use raw dose units. ``lambda_`` is the
    positive outcome-specific generalized Aranda–Ordaz parameter.
    """

    intercepts: FloatArray
    slopes: FloatArray
    lambda_: float

    def __init__(
        self,
        intercepts: ArrayLike,
        slopes: ArrayLike,
        lambda_: float,
    ) -> None:
        alpha = _real(intercepts, "intercepts")
        beta = _real(slopes, "slopes")
        lam = _positive_scalar(lambda_, "lambda_")
        if alpha.ndim != 2 or not 1 <= alpha.shape[0] <= 3 or alpha.shape[1] != 2:
            raise ValueError("intercepts must have shape (levels - 1, 2), with 2–4 levels")
        if beta.shape != alpha.shape:
            raise ValueError("slopes must have the same threshold-by-agent shape as intercepts")
        object.__setattr__(self, "intercepts", _freeze(alpha))
        object.__setattr__(self, "slopes", _freeze(beta))
        object.__setattr__(self, "lambda_", lam)


def _log_ordinal_from_log_hazard(log_hazard: FloatArray) -> FloatArray:
    """GCR category log probabilities from log conditional hazards."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        hazard = np.exp(log_hazard)
    hazard = np.where(log_hazard > _LOG_MAX, np.inf, hazard)
    with np.errstate(divide="ignore", under="ignore", invalid="ignore"):
        log_gamma = np.log(-np.expm1(-hazard))
    tiny = log_hazard < -36.0
    log_gamma = np.where(tiny, log_hazard, log_gamma)
    if np.any(np.isnan(log_gamma)):
        raise ArithmeticError("GAO continuation probabilities are not representable")
    log_categories = np.empty((*log_hazard.shape[:-1], log_hazard.shape[-1] + 1))
    log_categories[..., 0] = -hazard[..., 0]
    if log_hazard.shape[-1] > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            prefix = np.cumsum(log_gamma, axis=-1)
            log_categories[..., 1:-1] = prefix[..., :-1] - hazard[..., 1:]
            log_categories[..., -1] = prefix[..., -1]
    else:
        log_categories[..., -1] = log_gamma[..., 0]
    if np.any(np.isnan(log_categories)):
        raise ArithmeticError("GAO ordinal log probabilities are not representable")
    return log_categories


def _gao_log_marginal(
    doses1: FloatArray,
    doses2: FloatArray,
    parameters: U2OETGAOMarginal,
    kappa: float,
) -> FloatArray:
    alpha = np.asarray(parameters.intercepts)
    beta = np.asarray(parameters.slopes)
    with np.errstate(over="ignore", invalid="ignore"):
        eta1 = alpha[:, 0, None, None] + beta[:, 0, None, None] * doses1[None, :, None]
        eta2 = alpha[:, 1, None, None] + beta[:, 1, None, None] * doses2[None, None, :]
        interaction = np.log(kappa) + eta1 + eta2
    if not np.all(np.isfinite(eta1)) or not np.all(np.isfinite(eta2)):
        raise ArithmeticError("GAO raw-dose linear predictor exceeds floating-point range")
    if not np.all(np.isfinite(interaction)):
        raise ArithmeticError("GAO interaction predictor exceeds floating-point range")
    log_sum = np.logaddexp(np.logaddexp(eta1, eta2), interaction)
    log_link_argument = np.log(parameters.lambda_) + log_sum
    if not np.all(np.isfinite(log_link_argument)):
        raise ArithmeticError("GAO link predictor exceeds floating-point range")
    log_softplus = np.empty_like(log_link_argument)
    small = log_link_argument < -36.0
    log_softplus[small] = log_link_argument[small]
    log_softplus[~small] = np.log(np.logaddexp(0.0, log_link_argument[~small]))
    log_hazard = log_softplus - np.log(parameters.lambda_)
    # Category arrays use dose-1, dose-2, category axes.
    return _log_ordinal_from_log_hazard(np.moveaxis(log_hazard, 0, -1))


def _normal_boundaries(log_probability: FloatArray) -> FloatArray:
    """Latent-normal interval boundaries using stable log-tail quantiles."""
    log_cdf = np.logaddexp.accumulate(log_probability)[:-1]
    log_survival = np.logaddexp.accumulate(log_probability[::-1])[::-1][1:]
    interior = np.empty(log_cdf.shape, dtype=float)
    lower_tail = log_cdf <= _LOG_HALF
    interior[lower_tail] = ndtri_exp(log_cdf[lower_tail])
    interior[~lower_tail] = -ndtri_exp(log_survival[~lower_tail])
    if not np.all(np.isfinite(interior)) or np.any(np.diff(interior) < 0):
        raise ArithmeticError("GAO Gaussian-copula thresholds are not representable")
    return np.concatenate(([-np.inf], interior, [np.inf]))


def _gaussian_joint(
    log_efficacy: FloatArray,
    log_toxicity: FloatArray,
    association: float,
) -> FloatArray:
    shape = (*log_efficacy.shape[:2], log_efficacy.shape[-1], log_toxicity.shape[-1])
    if association == 0.0:
        return log_efficacy[..., :, None] + log_toxicity[..., None, :]

    joint = np.empty(shape, dtype=float)
    for dose_index in np.ndindex(log_efficacy.shape[:2]):
        log_e = log_efficacy[dose_index]
        log_t = log_toxicity[dose_index]
        e = np.exp(log_e)
        t = np.exp(log_t)
        if abs(association) == 1.0:
            e_edges = np.concatenate(([0.0], np.cumsum(e)))
            t_edges = np.concatenate(([0.0], np.cumsum(t)))
            e_edges[-1] = 1.0
            t_edges[-1] = 1.0
            if association > 0:
                t_low, t_high = t_edges[:-1], t_edges[1:]
            else:
                t_low, t_high = 1.0 - t_edges[1:], 1.0 - t_edges[:-1]
            cell = np.maximum(
                0.0,
                np.minimum(e_edges[1:, None], t_high[None, :])
                - np.maximum(e_edges[:-1, None], t_low[None, :]),
            )
            with np.errstate(divide="ignore"):
                joint[dose_index] = np.log(cell)
            continue

        e_boundaries = _normal_boundaries(log_e)
        t_boundaries = _normal_boundaries(log_t)
        for e_category, t_category in np.ndindex((e.size, t.size)):
            if not np.isfinite(log_e[e_category]) or not np.isfinite(log_t[t_category]):
                joint[dose_index][e_category, t_category] = -np.inf
                continue
            probability, _ = _rectangle(
                float(e_boundaries[e_category]),
                float(e_boundaries[e_category + 1]),
                float(t_boundaries[t_category]),
                float(t_boundaries[t_category + 1]),
                association,
            )
            if probability <= 0.0:
                raise ArithmeticError(
                    "positive GAO marginal categories produced an unresolved Gaussian rectangle"
                )
            joint[dose_index][e_category, t_category] = np.log(probability)
    return joint


def u2oet_gao_probabilities(
    doses1: ArrayLike,
    doses2: ArrayLike,
    *,
    efficacy: U2OETGAOMarginal,
    toxicity: U2OETGAOMarginal,
    kappa: float,
    association: float = 0.0,
) -> U2OETProbabilities:
    """Evaluate explicit GAO marginals and their Gaussian-copula joint law.

    GAO uses raw dose values and a single positive ``kappa`` shared by both
    outcomes. The outcome-specific ``lambda_`` values are stored on each
    marginal. Gaussian rectangles reuse the package's bounded deterministic
    quadrature. Extremely small positive rectangles that underflow are rejected;
    this is not a rare-tail log-rectangle integrator.
    """
    d1, d2 = _dose_grid(doses1, "doses1"), _dose_grid(doses2, "doses2")
    if not isinstance(efficacy, U2OETGAOMarginal) or not isinstance(toxicity, U2OETGAOMarginal):
        raise ValueError("efficacy and toxicity must be U2OETGAOMarginal values")
    kappa_value = _positive_scalar(kappa, "kappa")
    rho_array = _real(association, "association")
    if rho_array.ndim != 0 or abs(float(rho_array)) > 1.0:
        raise ValueError("association must be a scalar in [-1,1]")
    rho = float(rho_array)
    category_cells = efficacy.intercepts.shape[0] + 1
    toxicity_cells = toxicity.intercepts.shape[0] + 1
    cells = int(d1.size) * int(d2.size) * category_cells * toxicity_cells
    if cells > _MAX_CELLS:
        raise ValueError("GAO joint probability grid exceeds the bounded cell budget")

    log_e = _gao_log_marginal(d1, d2, efficacy, kappa_value)
    log_t = _gao_log_marginal(d1, d2, toxicity, kappa_value)
    log_joint = _gaussian_joint(log_e, log_t, rho)
    if np.any(np.isnan(log_joint)) or np.any(log_joint > 1e-12):
        raise ArithmeticError("GAO Gaussian-copula log probabilities are invalid")
    joint = np.exp(log_joint)
    if (
        np.max(np.abs(np.sum(joint, axis=(-2, -1)) - 1.0)) > 2e-11
        or np.max(np.abs(np.sum(joint, axis=-1) - np.exp(log_e))) > 2e-11
        or np.max(np.abs(np.sum(joint, axis=-2) - np.exp(log_t))) > 2e-11
    ):
        raise ArithmeticError("GAO Gaussian copula failed probability or marginal checks")
    return U2OETProbabilities(_freeze(log_e), _freeze(log_t), _freeze(log_joint))
