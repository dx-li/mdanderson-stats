"""BNORM conjugate updating for normal observations with known or unknown variance."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtri
from scipy.stats import invgamma, norm, t

from ._validation import FloatArray, count, finite


def _owned(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=float, copy=True)
    result.flags.writeable = False
    return result


def _positive(value: ArrayLike, name: str) -> FloatArray:
    result = finite(value, name)
    if np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


def _exp(log_value: FloatArray) -> FloatArray:
    with np.errstate(over="ignore", under="ignore"):
        value = np.exp(log_value)
    if np.any(~np.isfinite(value) | (value == 0)):
        raise ArithmeticError("positive posterior parameter is not representable in float64")
    return value


def _center(a: FloatArray, b: FloatArray, wa: FloatArray, wb: FloatArray) -> FloatArray:
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1)
    scaled = (wa * (a / scale) + wb * (b / scale)) * scale
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = np.where(wa >= wb, a + (b - a) * wb, b + (a - b) * wa)
    return np.where(np.isfinite(shifted), shifted, scaled)


def _mass(probability: ArrayLike) -> FloatArray:
    p = finite(probability, "probability")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("interval probability must lie in (0, 1)")
    return p


def _interval(center: ArrayLike, half_width: ArrayLike) -> FloatArray:
    center, half_width = np.broadcast_arrays(center, half_width)
    with np.errstate(over="ignore"):
        result = np.stack((center - half_width, center + half_width), axis=-1)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("interval endpoints are not representable in float64")
    return _owned(result)


@dataclass(frozen=True, init=False)
class NormalSample:
    """Batched sufficient statistics; sample_sd uses the unbiased n-1 denominator.

    Omit sample_sd for a known-variance analysis from mean and size alone.
    Unknown-variance analyses require it for sizes above one. Observations occupy
    the final axis in from_data; other axes represent independent data sets.
    """

    mean: FloatArray
    size: FloatArray
    sum_squares: FloatArray

    def __init__(self, mean: ArrayLike, size: ArrayLike, *, sample_sd: ArrayLike | None = None):
        m, n = np.broadcast_arrays(finite(mean, "mean"), count(size, "size"))
        if np.any(n < 1):
            raise ValueError("sample size must be at least one")
        if sample_sd is None:
            ss = np.where(n == 1, 0.0, np.nan)
        else:
            m, n, sd = np.broadcast_arrays(m, n, finite(sample_sd, "sample_sd"))
            if np.any(sd < 0) or np.any((n == 1) & (sd != 0)):
                raise ValueError("sample_sd must be nonnegative and zero for size one")
            with np.errstate(over="ignore"):
                ss = (sd * np.sqrt(n - 1)) ** 2
            if np.any(~np.isfinite(ss) | ((sd > 0) & (n > 1) & (ss == 0))):
                raise ArithmeticError("sample sum of squares is not representable")
        for name, value in (("mean", m), ("size", n), ("sum_squares", ss)):
            object.__setattr__(self, name, _owned(value))

    @classmethod
    def from_data(cls, observations: ArrayLike) -> "NormalSample":
        x = finite(observations, "observations")
        if x.ndim == 0 or x.shape[-1] < 1:
            raise ValueError("observations require a nonempty final sample axis")
        # Center before summing to preserve small variation around large offsets.
        # Fall back to scaled means only where the initial subtraction overflows.
        base = x[..., 0]
        with np.errstate(over="ignore", invalid="ignore"):
            differences = x - base[..., None]
            shifted_mean = base + np.sum(differences / x.shape[-1], axis=-1)
        scale = np.max(np.abs(x), axis=-1)
        scale = np.where(scale == 0, 1, scale)
        fallback = (x / scale[..., None]).mean(axis=-1) * scale
        mean = np.where(np.all(np.isfinite(differences), axis=-1), shifted_mean, fallback)
        with np.errstate(over="ignore"):
            residual = x - mean[..., None]
        if np.any(~np.isfinite(residual)):
            raise ArithmeticError("sample spread is not representable")
        spread = np.max(np.abs(residual), axis=-1)
        spread = np.where(spread == 0, 1, spread)
        sd = (
            np.sqrt(np.sum((residual / spread[..., None]) ** 2, axis=-1) / max(x.shape[-1] - 1, 1))
            * spread
        )
        return cls(mean, x.shape[-1], sample_sd=sd)

    @property
    def sample_variance(self) -> FloatArray:
        return np.divide(
            self.sum_squares,
            self.size - 1,
            out=np.full_like(self.size, np.nan),
            where=self.size > 1,
        )

    def confidence_interval(
        self, probability: ArrayLike = 0.95, *, observation_variance: ArrayLike | None = None
    ) -> FloatArray:
        """Data-only frequentist interval, separate from posterior uncertainty."""
        p = _mass(probability)
        if observation_variance is not None:
            variance = _positive(observation_variance, "observation_variance")
            width = -ndtri((1 - p) / 2) * np.sqrt(variance) / np.sqrt(self.size)
        else:
            if np.any(self.size < 2) or np.any(~np.isfinite(self.sum_squares)):
                raise ValueError(
                    "unknown-variance confidence intervals require size >= 2 and sample_sd"
                )
            width = -t.ppf((1 - p) / 2, self.size - 1) * np.sqrt(self.sample_variance / self.size)
        return _interval(self.mean, width)


@dataclass(frozen=True, init=False)
class NormalMeanPosterior:
    """Normal distribution of an unknown mean with a known observation variance."""

    location: FloatArray
    variance: FloatArray

    def __init__(self, location: ArrayLike = 0, variance: ArrayLike = 10):
        m, v = np.broadcast_arrays(finite(location, "location"), _positive(variance, "variance"))
        object.__setattr__(self, "location", _owned(m))
        object.__setattr__(self, "variance", _owned(v))

    def update(
        self, sample: NormalSample, *, observation_variance: ArrayLike
    ) -> "NormalMeanPosterior":
        v = _positive(observation_variance, "observation_variance")
        lp, ld = -np.log(self.variance), np.log(sample.size) - np.log(v)
        total = np.logaddexp(lp, ld)
        # Normalize scaled precisions directly so equal inputs have exact 1/2
        # weights even when the absolute log precisions are very large.
        high = np.maximum(lp, ld)
        wp, wd = np.exp(lp - high), np.exp(ld - high)
        denominator = wp + wd
        mean = _center(self.location, sample.mean, wp / denominator, wd / denominator)
        return NormalMeanPosterior(mean, _exp(-total))

    def pdf(self, value: ArrayLike) -> FloatArray:
        return norm.pdf(value, loc=self.location, scale=np.sqrt(self.variance))

    def cdf(self, value: ArrayLike) -> FloatArray:
        return norm.cdf(value, loc=self.location, scale=np.sqrt(self.variance))

    def sf(self, value: ArrayLike) -> FloatArray:
        return norm.sf(value, loc=self.location, scale=np.sqrt(self.variance))

    def credible_interval(self, probability: ArrayLike = 0.95) -> FloatArray:
        """Symmetric highest-density interval (also equal-tailed)."""
        return _interval(
            self.location, -ndtri((1 - _mass(probability)) / 2) * np.sqrt(self.variance)
        )


@dataclass(frozen=True, init=False)
class NormalInverseGamma:
    """mu|variance ~ Normal(location, variance/mean_precision); variance ~ IG(shape, scale).

    IG density is scale**shape / Gamma(shape) * v**(-shape-1) * exp(-scale/v).
    All priors are proper. Parameters broadcast and are stored in owned arrays.
    """

    location: FloatArray
    mean_precision: FloatArray
    shape: FloatArray
    scale: FloatArray

    def __init__(
        self,
        location: ArrayLike = 0,
        mean_precision: ArrayLike = 4,
        shape: ArrayLike = 2,
        scale: ArrayLike = 3,
    ):
        values = np.broadcast_arrays(
            finite(location, "location"),
            _positive(mean_precision, "mean_precision"),
            _positive(shape, "shape"),
            _positive(scale, "scale"),
        )
        for name, value in zip(
            ("location", "mean_precision", "shape", "scale"), values, strict=True
        ):
            object.__setattr__(self, name, _owned(value))

    def update(self, sample: NormalSample) -> "NormalInverseGamma":
        if np.any(~np.isfinite(sample.sum_squares)):
            raise ValueError("unknown-variance updating requires sample_sd or raw observations")
        k, n = self.mean_precision, sample.size
        with np.errstate(over="ignore"):
            total, shape = k + n, self.shape + n / 2
        if np.any(~np.isfinite(total)) or np.any(~np.isfinite(shape)):
            raise ArithmeticError("posterior shape or mean precision overflowed")
        wp, wd = k / total, n / total
        location = _center(self.location, sample.mean, wp, wd)
        unit = np.maximum(np.maximum(np.abs(self.location), np.abs(sample.mean)), 1)
        delta = self.location / unit - sample.mean / unit
        with np.errstate(over="ignore", divide="ignore"):
            direct = np.abs(self.location - sample.mean)
            log_delta = np.where(
                np.isfinite(direct), np.log(direct), np.log(np.abs(delta)) + np.log(unit)
            )
        with np.errstate(divide="ignore"):
            log_between = np.log(k) + np.log(wd) + 2 * log_delta - np.log(2)
            log_within = np.log(sample.sum_squares) - np.log(2)
        log_scale = np.logaddexp(np.log(self.scale), np.logaddexp(log_within, log_between))
        return NormalInverseGamma(location, total, shape, _exp(log_scale))

    @property
    def mean_scale(self) -> FloatArray:
        """Scale of marginal Student t, not its standard deviation."""
        return _exp((np.log(self.scale) - np.log(self.shape) - np.log(self.mean_precision)) / 2)

    @property
    def mean_expectation(self) -> FloatArray:
        return np.where(self.shape > 0.5, self.location, np.nan)

    @property
    def variance_expectation(self) -> FloatArray:
        return np.divide(
            self.scale, self.shape - 1, out=np.full_like(self.scale, np.inf), where=self.shape > 1
        )

    def mean_pdf(self, value: ArrayLike) -> FloatArray:
        return t.pdf(value, 2 * self.shape, loc=self.location, scale=self.mean_scale)

    def mean_cdf(self, value: ArrayLike) -> FloatArray:
        return t.cdf(value, 2 * self.shape, loc=self.location, scale=self.mean_scale)

    def mean_sf(self, value: ArrayLike) -> FloatArray:
        return t.sf(value, 2 * self.shape, loc=self.location, scale=self.mean_scale)

    def mean_interval(self, probability: ArrayLike = 0.95) -> FloatArray:
        return _interval(
            self.location, -t.ppf((1 - _mass(probability)) / 2, 2 * self.shape) * self.mean_scale
        )

    def variance_pdf(self, value: ArrayLike) -> FloatArray:
        return invgamma.pdf(value, self.shape, scale=self.scale)

    def variance_cdf(self, value: ArrayLike) -> FloatArray:
        return invgamma.cdf(value, self.shape, scale=self.scale)

    def variance_sf(self, value: ArrayLike) -> FloatArray:
        return invgamma.sf(value, self.shape, scale=self.scale)

    def variance_interval(
        self, probability: ArrayLike = 0.95, *, method: str = "highest-density"
    ) -> FloatArray:
        """Inverse-gamma credible interval, solved at unit scale for stability."""
        if method not in ("highest-density", "equal-tail"):
            raise ValueError("method must be highest-density or equal-tail")
        a, scale, p = np.broadcast_arrays(self.shape, self.scale, _mass(probability))
        budget = 1 - p
        lo, hi = np.zeros_like(p), budget.copy()
        left = budget / 2
        if method == "highest-density":
            for _ in range(100):
                x, y = invgamma.ppf(left, a), invgamma.isf(budget - left, a)
                residual = -(a + 1) * (np.log(x) - np.log(y)) - 1 / x + 1 / y
                lo, hi = np.where(residual < 0, left, lo), np.where(residual >= 0, left, hi)
                middle = lo + (hi - lo) / 2
                middle = np.where(residual == 0, left, middle)
                if np.all(middle == left):
                    break
                left = middle
        x, y = invgamma.ppf(left, a), invgamma.isf(budget - left, a)
        mass = invgamma.cdf(y, a) - invgamma.cdf(x, a)
        residual = -(a + 1) * (np.log(x) - np.log(y)) - 1 / x + 1 / y
        if np.any(~np.isfinite(mass)) or np.any(np.abs(mass - p) > 2e-9):
            raise ArithmeticError("inverse-gamma interval mass could not be resolved")
        if method == "highest-density" and (
            np.any(~np.isfinite(residual)) or np.any(np.abs(residual) > 2e-7)
        ):
            raise ArithmeticError("inverse-gamma highest-density endpoints did not converge")
        with np.errstate(over="ignore", under="ignore"):
            result = np.stack((scale * x, scale * y), axis=-1)
        if np.any(~np.isfinite(result) | (result <= 0)):
            raise ArithmeticError("inverse-gamma interval endpoints are not representable")
        return _owned(result)
