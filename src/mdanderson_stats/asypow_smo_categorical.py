"""Original ASYPOW SMO for independent multinomial and ordinal groups."""

from math import fsum

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite
from .asypow_categorical_constraints import _fixed_categorical_null
from .asypow_smo import SMOPower, _poisson_log_kl
from .boin import _owned


def _masses(parameters: FloatArray, ordinal: bool) -> FloatArray:
    if ordinal:
        mass = np.diff(
            np.column_stack((np.zeros(len(parameters)), parameters, np.ones(len(parameters)))),
            axis=1,
        )
        if np.any(mass <= 0):
            raise ValueError("cumulative probabilities must increase strictly within (0,1)")
        return mass
    if np.any((parameters <= 0) | (parameters >= 1)):
        raise ValueError("category probabilities must be strictly between zero and one")
    remainder = np.array([fsum([1.0, *(-row)]) for row in parameters])
    if np.any(remainder <= 0):
        raise ValueError("category probability rows must sum to less than one")
    return np.column_stack((parameters, remainder))


def _categorical(
    parameters: ArrayLike,
    null: ArrayLike | None,
    constraints: ArrayLike | None,
    group_size: ArrayLike,
    subtract_df: bool,
    ordinal: bool,
) -> SMOPower:
    p = finite(parameters, "parameters")
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or not 1 <= p.size <= 500 or not p.shape[1]:
        raise ValueError("parameters need nonempty group rows and at most 500 entries")
    if not isinstance(subtract_df, (bool, np.bool_)):
        raise ValueError("subtract_df must be boolean")
    mass = _masses(p, ordinal)
    weight = np.broadcast_to(finite(group_size, "group_size"), (len(p),))
    if np.any(weight <= 0):
        raise ValueError("group_size must be positive")
    log_weight = np.log(weight) - logsumexp(np.log(weight))
    if constraints is not None:
        if null is not None:
            raise ValueError("supply either null parameters or constraints, not both")
        q, df = _fixed_categorical_null(p, mass, constraints, ordinal, log_weight)
    elif null is None:
        if len(p) < 2:
            raise ValueError("equality testing requires at least two groups")
        normalized = np.exp(log_weight)
        normalized /= normalized.sum()
        # Centering preserves an exactly common null and close alternatives.
        low = p.min(axis=0)
        pooled = np.clip(low + normalized @ (p - low), low, p.max(axis=0))
        q = np.broadcast_to(pooled, p.shape)
        df = (len(p) - 1) * p.shape[1]
    else:
        q = np.broadcast_to(finite(null, "null parameters"), p.shape)
        df = p.size
    null_mass = _masses(q, ordinal)
    # The linear terms sum to zero for probability distributions. Using
    # nonnegative generalized-KL summands avoids cancellation near the null.
    with np.errstate(under="ignore"):
        w = float(
            np.exp(np.log(2) + logsumexp(log_weight[:, None] + _poisson_log_kl(mass, null_mass)))
        )
    if not np.isfinite(w):
        raise ArithmeticError("categorical divergence is not representable")
    return SMOPower(w, df, _owned(q), bool(subtract_df))


def asypow_smo_multinomial(
    probabilities: ArrayLike,
    *,
    null_probabilities: ArrayLike | None = None,
    constraints: ArrayLike | None = None,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
) -> SMOPower:
    """SMO for K-1 category masses per group; the final mass is 1-sum(p).

    Omit the null to compare all group distributions, or supply a vector/row
    matrix fixing every free probability. Null parameters retain group rows.
    """
    return _categorical(
        probabilities, null_probabilities, constraints, group_size, subtract_df, False
    )


def asypow_smo_ordinal(
    cumulative: ArrayLike,
    *,
    null_cumulative: ArrayLike | None = None,
    constraints: ArrayLike | None = None,
    group_size: ArrayLike = 1,
    subtract_df: bool = True,
) -> SMOPower:
    """SMO for K-1 strictly increasing cumulative probabilities per group.

    Omit the null to compare all group distributions, or supply a vector/row
    matrix fixing every cumulative probability. The terminal 1 is implicit.
    """
    return _categorical(cumulative, null_cumulative, constraints, group_size, subtract_df, True)
