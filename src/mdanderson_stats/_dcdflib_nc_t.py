"""Signed noncentral-t tails, with logarithmic conditional-normal integration."""

from math import exp, log, pi

import numpy as np
from scipy.integrate import quad
from scipy.special import exp1, gammainc, gammaincc
from scipy.stats import nct

from ._validation import FloatArray
from .cdflib_nc_t import _negative_tail
from .dcdflib_gamma import _log_gamma_ratio
from .dcdflib_normal import cumnor
from .dcdflib_t import _small_tail


def _gamma_logs(df: float, log_ratio: float) -> tuple[float, float]:
    """Log chi-square tails at df*ratio**2 without forming either product."""
    logz = log(df) - log(2) + 2 * log_ratio
    if logz < -36:
        # The relative correction in the lower-gamma series is O(exp(logz)).
        correction = float(_log_gamma_ratio(np.asarray(df / 2)))
        lp = df * ((logz - correction) / 2)
        return lp, log(-np.expm1(lp)) if lp < 0 else -np.inf
    if logz > log(np.finfo(float).max):
        return 0.0, -np.inf
    # Preserve the exact mean coordinate when the ratio is near one;
    # exp(log(df/2)) loses many probability ulps for a concentrated gamma.
    z = (df / 2) * exp(2 * log_ratio) if abs(log_ratio) < 1 and df >= 1e-300 else exp(logz)
    if df < 1e-12:
        # Divide before multiplying so df/2 need not be representable.
        coefficient = float(exp1(z)) / 2
        lq = log(df) + log(coefficient) if coefficient > 0 else -np.inf
        return float(np.log1p(-exp(lq))), lq
    p, q = float(gammainc(df / 2, z)), float(gammaincc(df / 2, z))
    if not np.isfinite(p + q) or min(p, q) < 0 or max(p, q) > 1:
        raise ArithmeticError("legacy noncentral t conditional gamma kernel failed")
    return log(p) if p > 0 else -np.inf, log(q) if q > 0 else -np.inf


def _conditional_tail(t: float, df: float, nc: float, *, upper: bool) -> float:
    """Integrate over a positive normal numerator; nc>=0 and t!=0."""
    logt = log(abs(t))
    if t < 0:
        # u=z*(nc+1) keeps the normal density resolved in the far left tail.
        scale = nc + 1
        if nc > 40:
            return 0.0  # The entire probability is bounded by Phi(-nc).

        def integrand_log(u: float) -> float:
            if u <= 0:
                return -np.inf
            z = u / scale
            lp, _ = _gamma_logs(df, log(z) - logt)
            return -nc * z - z * z / 2 + lp

        points = [0.01, 0.1, 0.5, 1, 2, 4, 8, 16, 32]
        low, high = 0.0, np.inf
        outside = -nc * nc / 2 - log(2 * pi) / 2 - log(scale)
    else:
        # z=nc+u; truncation beyond 40 standard deviations is below float range.
        def integrand_log(u: float) -> float:
            z = nc + u
            if z <= 0:
                return -np.inf
            relative = ((nc - t) + u) / t
            log_ratio = float(np.log1p(relative)) if abs(relative) < 0.5 else log(z) - logt
            lp, lq = _gamma_logs(df, log_ratio)
            return -u * u / 2 + (lp if upper else lq)

        low, high = -min(nc, 40.0), 40.0
        points = [v for v in (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20) if low < v < high]
        points.append(max(low + 0.01, min(t - nc, 39.0)))
        outside = -log(2 * pi) / 2
    peak = max(integrand_log(v) for v in points)
    if peak == -np.inf:
        return 0.0

    def integrand(value: float) -> float:
        return exp(integrand_log(value) - peak)

    result = quad(integrand, low, high, epsabs=0, epsrel=2e-10, limit=250, full_output=1)
    value, error = result[:2]
    if len(result) != 3 or not np.isfinite(value) or error > 1e-8 * abs(value):
        raise ArithmeticError("legacy noncentral t conditional integration failed")
    return exp(outside + peak + log(value)) if value > 0 else 0.0


def _tails(t: FloatArray, df: FloatArray, nc: FloatArray) -> tuple[FloatArray, FloatArray]:
    # Reflection permits one implementation of the difficult negative-t tail.
    reflected = nc < 0
    tt, nn = np.where(reflected, -t, t), np.abs(nc)
    p, q = np.empty(t.shape), np.empty(t.shape)
    central = nn == 0
    small = _small_tail(tt[central], df[central])
    p[central] = np.where(tt[central] < 0, small, 1 - small)
    q[central] = np.where(tt[central] < 0, 1 - small, small)
    zero = ~central & (tt == 0)
    p[zero], q[zero] = cumnor(-nn[zero])
    active = ~(central | zero)
    large_df = active & (df >= 1e20)
    if np.any(large_df):
        # Delta-method denominator fluctuation must remain when nc is huge.
        a, d, n = tt[large_df], df[large_df], nn[large_df]
        scale = np.hypot(1, (a / np.sqrt(d)) / np.sqrt(2))
        with np.errstate(over="ignore", invalid="ignore"):
            delta = (a - n) / scale
        delta = np.where(np.isfinite(delta), delta, a / scale - n / scale)
        delta -= (a / d) / (4 * scale)
        delta = np.clip(delta, -np.finfo(float).max, np.finfo(float).max)
        lp, uq = cumnor(delta)
        # First Edgeworth correction: the standardized third cumulant of
        # t*sqrt(V/df)-Z is t**3/(4*df**2*scale**3). It matters when t is
        # huge, even at large df; omitting it loses relative tail accuracy.
        correction = np.zeros(delta.shape)
        finite_tail = np.abs(delta) < 40
        z = delta[finite_tail]
        ratio = (a[finite_tail] / np.sqrt(d[finite_tail])) / scale[finite_tail]
        skewness = (ratio**3 / 4) / np.sqrt(d[finite_tail])
        correction[finite_tail] = (
            skewness * (z * z - 1) * np.exp(-z * z / 2) / (6 * np.sqrt(2 * np.pi))
        )
        p[large_df], q[large_df] = lp + correction, uq - correction
    active &= ~large_df
    tiny_df = active & (df < 1e-50) & (nn <= 8)
    # Uniform over finite t: df*|log(df/t**2)| is < 3e-47 here;
    # Phi(-nc)>=Phi(-8)>6e-16, so this error is negligible in either tail.
    p[tiny_df], q[tiny_df] = cumnor(-nn[tiny_df])
    active &= ~tiny_df
    wide = active & ((df < 1e-10) | (np.abs(tt) > 1e100) | (nn > 1e4))
    regular = active & ~wide
    if np.any(regular):
        p[regular] = nct.sf(-tt[regular], df[regular], -nn[regular])
        q[regular] = nct.sf(tt[regular], df[regular], nn[regular])
        wide |= regular & (~np.isfinite(p) | ~np.isfinite(q))
    negative = regular & ~wide & (tt < 0) & (p < 1e-6)
    for i in np.flatnonzero(negative):
        p.flat[i] = _negative_tail(float(tt.flat[i]), float(df.flat[i]), float(nn.flat[i]))
        q.flat[i] = 1 - p.flat[i]
    for i in np.flatnonzero(wide):
        x, d, n = float(tt.flat[i]), float(df.flat[i]), float(nn.flat[i])
        if x < 0:
            p.flat[i] = _conditional_tail(x, d, n, upper=False)
            q.flat[i] = 1 - p.flat[i]
        else:
            normal, _ = cumnor(-n)
            p.flat[i] = float(normal) + _conditional_tail(x, d, n, upper=False)
            q.flat[i] = _conditional_tail(x, d, n, upper=True)
    if np.any(np.abs(p + q - 1) > 1e-8):
        raise ArithmeticError("legacy noncentral t tail pair is inconsistent")
    tolerance = 8 * np.finfo(float).eps
    p = np.where((p > 1) & (p <= 1 + tolerance), 1.0, p)
    q = np.where((q > 1) & (q <= 1 + tolerance), 1.0, q)
    if np.any(~np.isfinite(p) | ~np.isfinite(q) | (p < 0) | (q < 0) | (p > 1) | (q > 1)):
        raise ArithmeticError("legacy noncentral t tail evaluation failed")
    p, q = np.where(p <= q, p, 1 - q), np.where(p <= q, 1 - p, q)
    return np.where(reflected, q, p), np.where(reflected, p, q)
