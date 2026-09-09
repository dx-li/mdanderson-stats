"""Vectorized small-pool PCR Poisson-mean likelihood fits."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _freeze(a: ArrayLike) -> FloatArray:
    x = np.asarray(a, dtype=np.float64)
    return np.frombuffer(x.tobytes(), dtype=np.float64).reshape(x.shape)


def _mask(a: ArrayLike) -> BoolArray:
    x = np.asarray(a, dtype=np.bool_)
    return np.frombuffer(x.tobytes(), dtype=np.bool_).reshape(x.shape)


@dataclass(frozen=True)
class SPPCRMeanFit:
    """Immutable fits, with final axes DNA level and allele in count arrays.

    Other arrays have shape ``batch + (alleles,)``. Variance is inverse observed
    information for interior fits and NaN at the zero boundary (interior=False).
    Log likelihood omits binomial coefficients and uses the disclosed fit counts.
    mu_lower/mu_upper bracket the fitted score root; both are zero at the boundary.
    """

    dna: FloatArray
    original_seen: FloatArray
    wells: FloatArray
    seen: FloatArray
    unseen: FloatArray
    mu: FloatArray
    variance: FloatArray
    interior: BoolArray
    adjusted: BoolArray
    log_likelihood: FloatArray
    mu_lower: FloatArray
    mu_upper: FloatArray


def sppcr_fit_means(
    dna: ArrayLike,
    seen: ArrayLike,
    wells: ArrayLike,
    *,
    saturation: str = "raise",
    max_iterations: int = 2048,
) -> SPPCRMeanFit:
    """Fit independent allele Poisson means to repeated detection counts.

    dna is a positive one-dimensional vector; seen has shape (..., levels,
    alleles). wells is scalar or broadcastable to (..., levels), with positive
    integer counts. seen counts must lie between zero and the matching wells.
    All counts are integers below 2**53. Leading batch axes may be empty.

    saturation='raise' rejects an allele detected in every well at every level.
    saturation='half' moves half a detection to nondetection at the first minimum
    DNA level for such alleles only. Never-seen alleles retain their zero MLE.
    No hidden pseudo-observations or source upper search bound are imposed.
    """
    d = finite(dna, "dna")
    original = count(seen, "seen")
    n = count(wells, "wells")
    if d.ndim != 1 or d.size == 0 or np.any(d <= 0):
        raise ValueError("dna must be a nonempty positive one-dimensional vector")
    if original.ndim < 2 or original.shape[-2] != d.size or original.shape[-1] == 0:
        raise ValueError("seen must have shape (..., DNA levels, nonempty alleles)")
    if saturation not in ("raise", "half"):
        raise ValueError("saturation must be 'raise' or 'half'")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    n = np.broadcast_to(n, original.shape[:-1])
    if np.any(n <= 0) or np.any(original > n[..., None]):
        raise ValueError("wells must be positive and seen must not exceed wells")
    s = original.copy()
    u = n[..., None] - s
    adjusted = np.all(u == 0, axis=-2)
    if np.any(adjusted):
        if saturation == "raise":
            raise ValueError(
                "fully detected allele has no finite mean MLE; choose saturation='half'"
            )
        i = int(np.argmin(d))
        s[..., i, :] -= 0.5 * adjusted
        u[..., i, :] += 0.5 * adjusted
        if np.any((original[..., i, :] - s[..., i, :])[adjusted] != 0.5):
            raise ArithmeticError("half-count adjustment is not representable for these counts")
    interior = np.any(s > 0, axis=-2)
    # Fit tau=mu*max(dna) to avoid dependence on the arbitrary DNA unit.
    scale = float(d.max())
    w = d / scale
    if np.any(w == 0):
        raise ArithmeticError("DNA-level ratio underflows float64")
    weight = w[:, None]
    numerator = np.sum(s, axis=-2)
    denominator = np.sum(u * weight, axis=-2)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        upper = numerator / denominator
    if np.any(~np.isfinite(upper)) or np.any((upper == 0) & interior):
        raise ArithmeticError("finite root bracket is not representable")
    lower = np.zeros_like(upper)
    active = interior.copy()
    # tau*score = sum(seen*x/expm1(x)) - tau*sum(unseen*w).
    # x/expm1(x) <= 1 proves upper=sum(seen)/sum(unseen*w) brackets the root.
    for _ in range(max_iterations):
        if not np.any(active):
            break
        middle = lower + (upper - lower) / 2
        x = middle[..., None, :] * weight
        p = -np.expm1(-x)
        h = np.ones_like(x)
        np.divide(x * np.exp(-x), p, out=h, where=p > 0)
        sign = np.sum(s * h, axis=-2) - middle * denominator
        lower = np.where(active & (sign > 0), middle, lower)
        upper = np.where(active & (sign <= 0), middle, upper)
        active &= (upper - lower) > 8 * np.finfo(float).eps * upper
    if np.any(active):
        raise ArithmeticError("SPPCR root search exceeded max_iterations")
    tau = lower + (upper - lower) / 2
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        mu = tau / scale
        mu_lower, mu_upper = lower / scale, upper / scale
        x = tau[..., None, :] * weight
        logp = np.log(-np.expm1(-x))
        terms = np.zeros_like(s)
        np.multiply(s, logp, out=terms, where=s > 0)
        log_likelihood = np.sum(terms - u * x, axis=-2)
        log_information_terms = np.full_like(s, -np.inf)
        raw = np.log(s) + 2 * np.log(d[:, None]) - x - 2 * logp
        np.copyto(log_information_terms, raw, where=s > 0)
        log_information = np.logaddexp.reduce(log_information_terms, axis=-2)
        variance = np.where(interior, np.exp(-log_information), np.nan)
    if (
        np.any(~np.isfinite(mu))
        or np.any(interior & (mu == 0))
        or np.any(~np.isfinite(mu_lower))
        or np.any(~np.isfinite(mu_upper))
        or np.any(~np.isfinite(log_likelihood))
        or np.any(~np.isfinite(variance[interior]))
    ):
        raise ArithmeticError("fit or observed variance exceeds representable float64 range")
    return SPPCRMeanFit(
        _freeze(d),
        _freeze(original),
        _freeze(n),
        _freeze(s),
        _freeze(u),
        _freeze(mu),
        _freeze(variance),
        _mask(interior),
        _mask(adjusted),
        _freeze(log_likelihood),
        _freeze(mu_lower),
        _freeze(mu_upper),
    )
