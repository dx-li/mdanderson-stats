"""Ordinal UAROET probability model with a Gaussian-copula association.

This implements the published continuation-logit marginals and Gaussian
copula probability model. Parameter priors, posterior sampling and allocation
are kept in separate modules.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import log_expit

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet_scenario import _boundaries, _rectangle

_MAX_DOSES = 5
_MAX_LEVELS = 4


def _real_matrix(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.size > 10_000:
        raise ValueError(f"{name} exceeds the bounded UAROET input size")
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    answer = int(raw)
    if not minimum <= answer <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return answer


def _scalar(value: object, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def uaroet_parameter_names(
    dose_count: int,
    efficacy_levels: int,
    toxicity_levels: int,
    *,
    monotone_efficacy: bool = True,
    monotone_toxicity: bool = True,
) -> tuple[str, ...]:
    """Name all independent normal coordinates, excluding latent correlation.

    Monotone outcomes use one baseline logit per threshold followed by
    nonnegative dose increments. Unstructured outcomes use one logit per
    dose and threshold. These coordinates are Python-facing explicit priors;
    native prior elicitation is not implied.
    """
    doses = _integer(dose_count, "dose_count", 1, _MAX_DOSES)
    levels = (
        _integer(efficacy_levels, "efficacy_levels", 2, _MAX_LEVELS),
        _integer(toxicity_levels, "toxicity_levels", 2, _MAX_LEVELS),
    )
    if not isinstance(monotone_efficacy, (bool, np.bool_)) or not isinstance(
        monotone_toxicity, (bool, np.bool_)
    ):
        raise ValueError("monotonicity flags must be boolean")
    result: list[str] = []
    for endpoint, level_count, monotone in zip(
        ("efficacy", "toxicity"), levels, (monotone_efficacy, monotone_toxicity), strict=True
    ):
        thresholds = level_count - 1
        if monotone:
            result.extend(f"{endpoint}.baseline.{cut + 1}" for cut in range(thresholds))
            result.extend(
                f"{endpoint}.increment.{dose}.{cut + 1}"
                for dose in range(1, doses)
                for cut in range(thresholds)
            )
        else:
            result.extend(
                f"{endpoint}.logit.{dose}.{cut + 1}"
                for dose in range(doses)
                for cut in range(thresholds)
            )
    return tuple(result)


def uaroet_logits(
    parameters: ArrayLike,
    *,
    dose_count: int,
    efficacy_levels: int,
    toxicity_levels: int,
    monotone_efficacy: bool = True,
    monotone_toxicity: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Build dose-by-threshold continuation logits from named flat coordinates."""
    names = uaroet_parameter_names(
        dose_count,
        efficacy_levels,
        toxicity_levels,
        monotone_efficacy=monotone_efficacy,
        monotone_toxicity=monotone_toxicity,
    )
    raw = np.asarray(parameters)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or raw.size != len(names):
        raise ValueError("parameters must be a real vector matching uaroet_parameter_names")
    values = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("parameters must be finite")
    matrices: list[FloatArray] = []
    offset = 0
    for levels, monotone in (
        (efficacy_levels, monotone_efficacy),
        (toxicity_levels, monotone_toxicity),
    ):
        thresholds = levels - 1
        if monotone:
            baseline = values[offset : offset + thresholds]
            offset += thresholds
            increments = values[offset : offset + (dose_count - 1) * thresholds].reshape(
                dose_count - 1, thresholds
            )
            offset += (dose_count - 1) * thresholds
            if np.any(increments < 0):
                raise ValueError("monotone dose increments must be nonnegative")
            logits = np.empty((dose_count, thresholds), dtype=float)
            logits[0] = baseline
            if dose_count > 1:
                logits[1:] = baseline + np.cumsum(increments, axis=0)
        else:
            logits = (
                values[offset : offset + dose_count * thresholds]
                .reshape(dose_count, thresholds)
                .copy()
            )
            offset += dose_count * thresholds
        if not np.all(np.isfinite(logits)):
            raise ArithmeticError("assembled continuation logits exceed floating-point range")
        matrices.append(logits)
    return _freeze(matrices[0]), _freeze(matrices[1])


def _marginal_log_probabilities(logits: FloatArray) -> FloatArray:
    """Stable continuation-logit probabilities, with category last."""
    log_continue = log_expit(logits)
    log_stop = log_expit(-logits)
    result = np.empty((*logits.shape[:-1], logits.shape[-1] + 1), dtype=float)
    result[..., 0] = log_stop[..., 0]
    if logits.shape[-1] > 1:
        result[..., 1:-1] = np.cumsum(log_continue[..., :-1], axis=-1) + log_stop[..., 1:]
    result[..., -1] = np.sum(log_continue, axis=-1)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("ordinal log probabilities are not finite")
    return result


@dataclass(frozen=True)
class UAROETProbabilities:
    """Marginal and joint outcome probabilities; axes are dose, efficacy, toxicity."""

    efficacy: FloatArray
    toxicity: FloatArray
    joint: FloatArray
    log_efficacy: FloatArray
    log_toxicity: FloatArray
    log_joint: FloatArray
    quadrature_error: FloatArray

    def expected_utility(self, utility: ArrayLike) -> FloatArray:
        """Compute one expected utility per dose from an efficacy-by-toxicity table."""
        u = _real_matrix(utility, "utility")
        if u.shape != self.joint.shape[1:]:
            raise ValueError("utility must have efficacy-by-toxicity shape")
        value = np.sum(self.joint * u, axis=(-2, -1))
        if not np.all(np.isfinite(value)):
            raise ArithmeticError("expected utility is not finite")
        return _freeze(value)


def _joint_from_marginals(
    efficacy: FloatArray,
    toxicity: FloatArray,
    association: float,
    *,
    budget: list[int] | None = None,
    max_evaluations: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    doses, e_levels = efficacy.shape
    t_levels = toxicity.shape[1]
    joint = np.empty((doses, e_levels, t_levels), dtype=float)
    errors = np.zeros_like(joint)
    for dose in range(doses):
        e, t = efficacy[dose], toxicity[dose]
        if association == 0:
            joint[dose] = e[:, None] * t[None, :]
            continue
        if abs(association) == 1:
            ec, tc = np.r_[0.0, np.cumsum(e)], np.r_[0.0, np.cumsum(t)]
            if association > 0:
                lower, upper = tc[:-1], tc[1:]
            else:
                lower, upper = 1 - tc[1:], 1 - tc[:-1]
            joint[dose] = np.maximum(
                0.0,
                np.minimum(ec[1:, None], upper[None, :])
                - np.maximum(ec[:-1, None], lower[None, :]),
            )
            continue
        eb, tb = _boundaries(e), _boundaries(t)
        for i, j in np.ndindex((e_levels, t_levels)):
            if e[i] == 0 or t[j] == 0:
                joint[dose, i, j] = 0.0
            else:
                if budget is not None:
                    if max_evaluations is None or budget[0] >= max_evaluations:
                        raise RuntimeError(
                            "UAROET likelihood/rectangle work exceeded max_evaluations"
                        )
                    budget[0] += 1
                    if len(budget) > 1:
                        budget[1] += 1
                joint[dose, i, j], errors[dose, i, j] = _rectangle(
                    float(eb[i]), float(eb[i + 1]), float(tb[j]), float(tb[j + 1]), association
                )
    if not np.all(np.isfinite(joint)):
        raise ArithmeticError("Gaussian-copula joint probabilities are nonfinite")
    if np.any(joint < 0) or np.any(np.abs(joint.sum(axis=(1, 2)) - 1) > 1e-11):
        raise ArithmeticError("Gaussian-copula joint probabilities do not sum to one")
    if np.any(np.abs(joint.sum(axis=2) - efficacy) > 1e-11) or np.any(
        np.abs(joint.sum(axis=1) - toxicity) > 1e-11
    ):
        raise ArithmeticError("Gaussian copula failed to preserve its marginals")
    return joint, errors


def uaroet_probabilities(
    efficacy_logits: ArrayLike,
    toxicity_logits: ArrayLike,
    *,
    association: float = 0.0,
) -> UAROETProbabilities:
    """Evaluate UAROET logistic marginals and their Gaussian-copula joint law.

    Inputs are dose-by-threshold continuation logits. The latent-normal
    association can be anywhere in [-1,1], including direct limiting copulas.
    Output outcome axes are efficacy then toxicity.
    """
    e_theta = _real_matrix(efficacy_logits, "efficacy_logits")
    t_theta = _real_matrix(toxicity_logits, "toxicity_logits")
    if (
        e_theta.ndim != 2
        or t_theta.ndim != 2
        or e_theta.shape[0] != t_theta.shape[0]
        or not 1 <= e_theta.shape[0] <= _MAX_DOSES
        or not 1 <= e_theta.shape[1] < _MAX_LEVELS
        or not 1 <= t_theta.shape[1] < _MAX_LEVELS
    ):
        raise ValueError("logits must be matching dose-by-threshold matrices with 1..3 thresholds")
    rho_array = np.asarray(association)
    if rho_array.ndim != 0 or rho_array.dtype.kind not in "iuf":
        raise ValueError("association must be a finite real scalar")
    rho = float(rho_array)
    if not isfinite(rho) or abs(rho) > 1:
        raise ValueError("association must lie in [-1,1]")
    log_e = _marginal_log_probabilities(e_theta)
    log_t = _marginal_log_probabilities(t_theta)
    with np.errstate(under="ignore"):
        e = np.exp(log_e)
        t = np.exp(log_t)
    if np.any(e.sum(axis=1) <= 0) or np.any(t.sum(axis=1) <= 0):
        raise ArithmeticError("ordinal marginal probabilities are unrepresentable")
    # Renormalize only roundoff in exponentiated log-category probabilities.
    e /= e.sum(axis=1, keepdims=True)
    t /= t.sum(axis=1, keepdims=True)
    joint, errors = _joint_from_marginals(e, t, rho)
    with np.errstate(divide="ignore"):
        log_joint = log_e[:, :, None] + log_t[:, None, :] if rho == 0 else np.log(joint)
    return UAROETProbabilities(
        _freeze(e),
        _freeze(t),
        _freeze(joint),
        _freeze(log_e),
        _freeze(log_t),
        _freeze(log_joint),
        _freeze(errors),
    )
