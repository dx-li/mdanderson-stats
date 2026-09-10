"""DRDIST's seven-operation distribution calculator using shared numerical kernels."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray, finite
from .cdflib_chisq import cdf_chisq
from .cdflib_f import cdf_f
from .cdflib_normal import cdf_normal
from .cdflib_t import cdf_t
from .numerics import normal_tails

_OPERATIONS = ("inverse_f", "f", "inverse_normal", "normal", "inverse_t", "t", "chi_square")


def drdist(
    operation: int | str,
    value: ArrayLike,
    *,
    df: ArrayLike | None = None,
    numerator_df: ArrayLike | None = None,
    denominator_df: ArrayLike | None = None,
    mean: ArrayLike | None = None,
    sd: ArrayLike | None = None,
) -> FloatArray:
    """Evaluate a native menu operation (1..7) or its corresponding name.

    F and chi-square probabilities are upper tails; normal and t are lower
    tails. Inverse F accepts an upper-tail probability, inverse normal/t a
    lower-tail probability. Numerical parameters broadcast; results are immutable.
    """
    if isinstance(operation, (int, np.integer)) and not isinstance(operation, (bool, np.bool_)):
        if not 1 <= operation <= 7:
            raise ValueError("DRDIST menu operation must be in 1..7")
        name = _OPERATIONS[int(operation) - 1]
    elif isinstance(operation, str) and operation in _OPERATIONS:
        name = operation
    else:
        raise ValueError("unknown DRDIST operation")
    x = finite(value, "value")
    normal = name in ("normal", "inverse_normal")
    f_distribution = name in ("f", "inverse_f")
    if (
        not normal
        and (mean is not None or sd is not None)
        or not f_distribution
        and (numerator_df is not None or denominator_df is not None)
        or (normal or f_distribution)
        and df is not None
    ):
        raise ValueError("omit parameters that do not belong to the selected distribution")
    inverse = name.startswith("inverse_")
    if inverse and np.any((x < 0) | (x > 1)):
        raise ValueError("inverse probabilities must be in [0,1]")
    interior = (x > 0) & (x < 1)
    safe = np.where(interior, x, 0.5) if inverse else x
    if f_distribution:
        if inverse:
            answer = cdf_f(2, ccum=safe, dfn=numerator_df, dfd=denominator_df).f
            answer = np.where(x == 0, np.inf, np.where(x == 1, 0, answer))
        else:
            answer = cdf_f(f=x, dfn=numerator_df, dfd=denominator_df).ccum
    elif normal:
        mu = finite(0 if mean is None else mean, "mean")
        sigma = finite(1 if sd is None else sd, "sd")
        if np.any(sigma <= 0):
            raise ValueError("sd must be positive")
        if inverse:
            z = cdf_normal(2, cum=safe).x
            with np.errstate(over="ignore", invalid="ignore"):
                answer = mu + sigma * z
                answer = np.where(np.isfinite(answer), answer, sigma * (mu / sigma + z))
            if np.any(~np.isfinite(answer)):
                raise ArithmeticError(
                    "normal quantile transformation is not representable; rescale units"
                )
            answer = np.where(x == 0, -np.inf, np.where(x == 1, np.inf, answer))
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                delta = x - mu
                z = np.where(np.isfinite(delta), delta / sigma, x / sigma - mu / sigma)
            answer = normal_tails(z)[0]
    elif name in ("t", "inverse_t"):
        if inverse:
            answer = cdf_t(2, cum=safe, df=df).t
            answer = np.where(x == 0, -np.inf, np.where(x == 1, np.inf, answer))
        else:
            answer = cdf_t(t=x, df=df).cum
    else:
        answer = cdf_chisq(x=x, df=df).ccum
    return _freeze(answer)
