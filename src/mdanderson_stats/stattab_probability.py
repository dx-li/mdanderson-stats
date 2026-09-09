"""STATTAB discrete terms with consistent truncation and direct probability factors."""

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze, _pair
from ._validation import FloatArray, finite
from .cdflib_beta_factors import _positive_parts, _scaled
from .cdflib_gamma_factor import _large_log_factor
from .cdflib_gamma_support import _positive_log_gamma


def _nonnegative(value: ArrayLike, name: str) -> FloatArray:
    result = finite(value, name)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _result(value: FloatArray) -> FloatArray:
    if np.any(~np.isfinite(value) | (value < 0) | (value > 1)):
        raise ArithmeticError("discrete probability could not be evaluated within [0,1]")
    return _freeze(value)


def _power(p: FloatArray, q: FloatArray, n: FloatArray) -> FloatArray:
    # Use the small supplied complement even when the base rounds to one.
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        exponent = np.where(p <= q, np.log(p), np.log1p(-q))
        return np.where(n == 0, 1.0, np.exp(n * exponent))


def _beta_term(
    a: FloatArray, b: FloatArray, p: FloatArray, q: FloatArray, divisor: FloatArray
) -> FloatArray:
    result = np.zeros(a.shape)
    active = (p > 0) & (q > 0)
    # Apply the probability normalization before the final exponentiation,
    # avoiding rounding a subnormal beta factor before dividing it.
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        prefactor, exponent, denominator = _positive_parts(
            a[active], b[active], p[active], q[active], np.zeros(np.count_nonzero(active))
        )
        result[active] = _scaled(prefactor, exponent - np.log(divisor[active]), denominator)
    return result


def stattab_binomial_term(
    successes: ArrayLike,
    trials: ArrayLike,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
) -> FloatArray:
    """P(S=floor(successes)) for Binomial(floor(trials), p).

    Finite nonnegative inputs broadcast; successes must not exceed trials before
    truncation. Supply p, its complement q, or a consistent pair. Zero trials
    give unit mass at zero. Results are immutable float64 arrays.
    """
    s, n, pp, qq = np.broadcast_arrays(
        _nonnegative(successes, "successes"),
        _nonnegative(trials, "trials"),
        *_pair(p, q, "p,q"),
    )
    if np.any(s > n):
        raise ValueError("successes must not exceed trials before truncation")
    s, n = np.floor(s), np.floor(n)
    result = np.empty(s.shape)
    zero, all_success = s == 0, (s == n) & (s > 0)
    result[zero] = _power(qq[zero], pp[zero], n[zero])
    result[all_success] = _power(pp[all_success], qq[all_success], n[all_success])
    interior = ~(zero | all_success)
    a, b = s[interior], n[interior] - s[interior]
    with np.errstate(under="ignore"):
        small, large = np.minimum(a, b), np.maximum(a, b)
        divisor = small / (1 + small / large)
    result[interior] = _beta_term(a, b, pp[interior], qq[interior], divisor)
    return _result(result)


def stattab_negative_binomial_term(
    failures: ArrayLike,
    successes: ArrayLike,
    p: ArrayLike | None = None,
    q: ArrayLike | None = None,
) -> FloatArray:
    """P(F=floor(failures)) before floor(successes) successes, chance p.

    Counts are finite/nonnegative and truncate independently. Zero required
    successes give unit mass at zero failures, including p=0. With positive
    required successes and p=0, every finite failure count has zero mass.
    """
    f, s, pp, qq = np.broadcast_arrays(
        _nonnegative(failures, "failures"),
        _nonnegative(successes, "successes"),
        *_pair(p, q, "p,q"),
    )
    f, s = np.floor(f), np.floor(s)
    result = np.zeros(f.shape)
    zero = f == 0
    result[zero] = _power(pp[zero], qq[zero], s[zero])
    interior = (f > 0) & (s > 0)
    result[interior] = _beta_term(s[interior], f[interior], pp[interior], qq[interior], f[interior])
    return _result(result)


def stattab_poisson_term(events: ArrayLike, mean: ArrayLike) -> FloatArray:
    """P(N=floor(events)) for a Poisson mean, including the zero-mean boundary.

    Both inputs are finite/nonnegative and broadcast. Direct log factors retain
    near-center accuracy for huge counts and avoid subtracting adjacent CDFs.
    """
    n, rate = np.broadcast_arrays(_nonnegative(events, "events"), _nonnegative(mean, "mean"))
    n = np.floor(n)
    result = np.zeros(n.shape)
    with np.errstate(over="ignore", under="ignore"):
        zero = n == 0
        result[zero] = np.exp(-rate[zero])
        active = (n > 0) & (rate > 0)
        small = active & (n < 8)
        nn, rr = n[small], rate[small]
        result[small] = np.exp(nn * np.log(rr) - rr - _positive_log_gamma(nn + 1))
        large = active & (n >= 8)
        nn, rr = n[large], rate[large]
        result[large] = np.exp(_large_log_factor(nn, rr) - np.log(nn))
    return _result(result)
