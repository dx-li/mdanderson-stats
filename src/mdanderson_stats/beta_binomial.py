"""Conjugate beta-binomial updating, credible sets and sequential trial simulation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc, betaincc, betainccinv, betaincinv, betaln, xlog1py, xlogy

from ._validation import FloatArray, count, finite, scalar


def _owned(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _probability(value: ArrayLike) -> FloatArray:
    p = finite(value, "probability")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probability must lie in [0, 1]")
    return p


@dataclass(frozen=True)
class BetaCredibleSet:
    """One or two closed intervals; unused second interval has NaN endpoints.

    The final axes of intervals are (component, endpoint), both of length two.
    Endpoint complements are calculated directly to retain precision near one.
    A U-shaped highest-density set has two components. Uniform densities use
    a central interval because their highest-density set is not unique.
    """

    intervals: FloatArray
    complements: FloatArray
    probability: FloatArray
    method: str


@dataclass(frozen=True, init=False)
class BetaBinomialPosterior:
    """A broadcast collection of proper Beta(alpha, beta) distributions.

    The same object represents a prior or posterior. Shapes must be finite and
    positive, with a representable finite sum. Updates take success and failure
    counts separately, avoiding ambiguity about whether a count is a total.
    """

    alpha: FloatArray
    beta: FloatArray

    def __init__(self, alpha: ArrayLike = 1, beta: ArrayLike = 1):
        a, b = np.broadcast_arrays(finite(alpha, "alpha"), finite(beta, "beta"))
        with np.errstate(over="ignore"):
            total = a + b
        if np.any((a <= 0) | (b <= 0)) or not np.all(np.isfinite(total)):
            raise ValueError("beta shapes must be positive with a finite sum")
        object.__setattr__(self, "alpha", _owned(a))
        object.__setattr__(self, "beta", _owned(b))

    @property
    def mean(self) -> FloatArray:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> FloatArray:
        total = self.alpha + self.beta
        return (self.alpha / total) * (self.beta / total) / (total + 1)

    def update(self, successes: ArrayLike, failures: ArrayLike) -> "BetaBinomialPosterior":
        s, f = count(successes, "successes"), count(failures, "failures")
        return BetaBinomialPosterior(self.alpha + s, self.beta + f)

    def logpdf(self, probability: ArrayLike) -> FloatArray:
        p = _probability(probability)
        return xlogy(self.alpha - 1, p) + xlog1py(self.beta - 1, -p) - betaln(self.alpha, self.beta)

    def pdf(self, probability: ArrayLike) -> FloatArray:
        """Density; infinite endpoint values are legitimate for shapes below one."""
        with np.errstate(over="ignore"):
            return np.exp(self.logpdf(probability))

    def cdf(self, probability: ArrayLike) -> FloatArray:
        return betainc(self.alpha, self.beta, _probability(probability))

    def sf(self, probability: ArrayLike) -> FloatArray:
        """Upper tail computed directly rather than by subtracting the CDF."""
        return betaincc(self.alpha, self.beta, _probability(probability))

    def quantile(self, probability: ArrayLike) -> FloatArray:
        return betaincinv(self.alpha, self.beta, _probability(probability))

    def credible_set(
        self, probability: ArrayLike = 0.95, *, method: str = "highest-density"
    ) -> BetaCredibleSet:
        """Highest-density (possibly disconnected) or equal-tail credible set.

        A vectorized bisection allocates probability between tails to equalize
        endpoint log densities. Direct complementary quantiles protect endpoints
        close to one. Unrepresentable quantiles or failed mass/density checks
        raise ArithmeticError rather than returning a misleading interval.
        """
        if method not in ("highest-density", "equal-tail"):
            raise ValueError("method must be highest-density or equal-tail")
        a, b, mass = np.broadcast_arrays(self.alpha, self.beta, _probability(probability))
        if np.any((mass <= 0) | (mass >= 1)):
            raise ValueError("credible probability must lie strictly between zero and one")
        a, b, mass = a.ravel(), b.ravel(), mass.ravel()
        u_shaped = (a < 1) & (b < 1) & (method == "highest-density")
        interior = (a > 1) & (b > 1) & (method == "highest-density")
        active = u_shaped | interior
        # For a connected set, t is excluded left mass. For a U-shaped set,
        # t is included left mass. The right mass is budget - t in either case.
        budget = np.where(u_shaped, mass, 1 - mass)
        left, right = np.zeros_like(mass), budget.copy()
        t = budget / 2
        decreasing = (a <= 1) & (b >= 1) & ((a < 1) | (b > 1))
        increasing = (a >= 1) & (b <= 1) & ((a > 1) | (b < 1))
        if method == "highest-density":
            t[decreasing] = 0
            t[increasing] = budget[increasing]
        for _ in range(100):
            if not np.any(active):
                break
            aa, bb, tt = a[active], b[active], t[active]
            rr = budget[active] - tt
            x = betaincinv(aa, bb, tt)
            xc = betainccinv(bb, aa, tt)
            y = betainccinv(aa, bb, rr)
            yc = betaincinv(bb, aa, rr)
            with np.errstate(divide="ignore", invalid="ignore"):
                difference = (aa - 1) * (np.log(x) - np.log(y)) + (bb - 1) * (
                    np.log(xc) - np.log(yc)
                )
            # The density difference increases with t for a unimodal density
            # and decreases for a U-shaped density.
            move_left = np.where(u_shaped[active], difference > 0, difference < 0)
            left[active] = np.where(move_left, tt, left[active])
            right[active] = np.where(move_left, right[active], tt)
            middle = left + (right - left) / 2
            exact = np.zeros_like(active)
            exact[active] = difference == 0
            converged = (middle == t) | exact
            t[active & ~exact] = middle[active & ~exact]
            active &= ~converged
        x = betaincinv(a, b, t)
        xc = betainccinv(b, a, t)
        y = betainccinv(a, b, budget - t)
        yc = betaincinv(b, a, budget - t)
        intervals = np.full((mass.size, 2, 2), np.nan)
        complements = np.full_like(intervals, np.nan)
        intervals[:, 0] = np.stack((x, y), axis=-1)
        complements[:, 0] = np.stack((xc, yc), axis=-1)
        intervals[u_shaped, 0] = np.stack((np.zeros_like(x[u_shaped]), x[u_shaped]), -1)
        intervals[u_shaped, 1] = np.stack((y[u_shaped], np.ones_like(y[u_shaped])), -1)
        complements[u_shaped, 0] = np.stack((np.ones_like(x[u_shaped]), xc[u_shaped]), -1)
        complements[u_shaped, 1] = np.stack((yc[u_shaped], np.zeros_like(y[u_shaped])), -1)
        # Compute each tail using whichever endpoint representation is smaller.
        lower_mass = np.where(x <= 0.5, betainc(a, b, x), betaincc(b, a, xc))
        upper_mass = np.where(yc <= 0.5, betainc(b, a, yc), betaincc(a, b, y))
        achieved = np.where(u_shaped, lower_mass + upper_mass, 1 - lower_mass - upper_mass)
        if not np.all(np.isfinite(achieved)) or np.any(np.abs(achieved - mass) > 2e-9):
            raise ArithmeticError("beta credible-set mass cannot be resolved in float64")
        balanced = u_shaped | interior
        with np.errstate(divide="ignore", invalid="ignore"):
            residual = (a - 1) * (np.log(x) - np.log(y)) + (b - 1) * (np.log(xc) - np.log(yc))
        if np.any(~np.isfinite(residual[balanced])) or np.any(np.abs(residual[balanced]) > 2e-7):
            raise ArithmeticError("beta highest-density endpoints did not converge")
        shape = np.broadcast_shapes(self.alpha.shape, np.shape(probability))
        return BetaCredibleSet(
            _owned(intervals.reshape((*shape, 2, 2))),
            _owned(complements.reshape((*shape, 2, 2))),
            _owned(mass.reshape(shape)),
            method,
        )


@dataclass(frozen=True)
class BetaBinomialSequence:
    """Cohort outcomes and posterior path; posterior's last axis includes the prior."""

    successes: FloatArray
    failures: FloatArray
    cumulative_successes: FloatArray
    cumulative_failures: FloatArray
    posterior: BetaBinomialPosterior

    @property
    def observed_rate(self) -> FloatArray:
        n = self.cumulative_successes + self.cumulative_failures
        return np.divide(self.cumulative_successes, n, out=np.full_like(n, np.nan), where=n > 0)


def beta_binomial_sequence(
    successes: ArrayLike,
    failures: ArrayLike,
    *,
    alpha: ArrayLike = 1,
    beta: ArrayLike = 1,
) -> BetaBinomialSequence:
    """Update after each cohort on the last axis; leading axes broadcast priors."""
    s, f = np.broadcast_arrays(count(successes, "successes"), count(failures, "failures"))
    if s.ndim == 0:
        raise ValueError("cohort outcomes must have a final cohort axis")
    prior = BetaBinomialPosterior(alpha, beta)
    shape = np.broadcast_shapes(s.shape[:-1], prior.alpha.shape)
    s, f = np.broadcast_to(s, (*shape, s.shape[-1])), np.broadcast_to(f, (*shape, f.shape[-1]))
    cs, cf = np.cumsum(s, axis=-1), np.cumsum(f, axis=-1)
    if np.any(cs + cf >= 2**53):
        raise ValueError("cumulative trial size must be smaller than 2**53")
    zero = np.zeros((*shape, 1))
    aa = np.broadcast_to(prior.alpha, shape)[..., None] + np.concatenate((zero, cs), axis=-1)
    bb = np.broadcast_to(prior.beta, shape)[..., None] + np.concatenate((zero, cf), axis=-1)
    return BetaBinomialSequence(
        _owned(s), _owned(f), _owned(cs), _owned(cf), BetaBinomialPosterior(aa, bb)
    )


def simulate_beta_binomial(
    probability: float,
    *,
    cohort_size: int = 3,
    cohorts: int = 10,
    trials: int = 1,
    alpha: ArrayLike = 1,
    beta: ArrayLike = 1,
    rng: np.random.Generator,
) -> BetaBinomialSequence:
    """Independent binomial cohorts with fixed true probability and explicit RNG.

    Priors can be scalar or broadcast to (trials,). Continue a trial with the last
    posterior shapes and the same generator. NumPy seeds do not reproduce R draws.
    """
    p = scalar(probability, "probability")
    _probability(p)
    for name, value in (("cohort_size", cohort_size), ("cohorts", cohorts), ("trials", trials)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer")
        if value < 1 or value >= 2**53:
            raise ValueError(f"{name} must be a positive integer smaller than 2**53")
    if cohort_size * cohorts >= 2**53:
        raise ValueError("cumulative trial size must be smaller than 2**53")
    prior = BetaBinomialPosterior(alpha, beta)
    np.broadcast_to(prior.alpha, (trials,))
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    s = rng.binomial(cohort_size, p, size=(trials, cohorts))
    return beta_binomial_sequence(s, cohort_size - s, alpha=prior.alpha, beta=prior.beta)
