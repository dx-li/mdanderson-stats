"""Bayesian two-agent toxicity model used by ToxFinder."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp, polygamma

from ._validation import FloatArray, count, finite
from .hierarchical_binomial import ChainSummary, summarize_chains

_LOG_MAX = float(np.log(np.finfo(float).max))
_MAX_PAIRS = 100
_MAX_SUBJECTS = 10_000
_MAX_EVALS = 2_000_000


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ToxFinderPrior:
    """Independent gamma priors in (alpha1,beta1,alpha2,beta2,alpha3,beta3) order.

    Variance zero is an explicit extension that fixes the corresponding parameter
    at its mean. Positive variances use the usual gamma shape/scale conversion.
    """

    mean: ArrayLike
    variance: ArrayLike

    def __post_init__(self) -> None:
        raw_mean, raw_variance = np.asarray(self.mean), np.asarray(self.variance)
        if raw_mean.shape != (6,) or raw_variance.shape != (6,):
            raise ValueError("prior mean and variance must each have six values")
        mean, variance = finite(raw_mean, "prior mean"), finite(raw_variance, "prior variance")
        if (
            mean.shape != (6,)
            or variance.shape != (6,)
            or np.any(mean <= 0)
            or np.any(variance < 0)
        ):
            raise ValueError(
                "prior mean and variance must each have six values; means > 0, variances >= 0"
            )
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            shape = np.exp(2 * np.log(mean) - np.log(np.where(variance > 0, variance, 1.0)))
            scale = np.exp(np.log(np.where(variance > 0, variance, 1.0)) - np.log(mean))
        free = variance > 0
        if np.any(
            free & ((shape < 1e-8) | ~np.isfinite(shape) | (scale == 0) | ~np.isfinite(scale))
        ):
            raise ValueError(
                "gamma prior shape and scale must be representable; shape must be >= 1e-8"
            )
        object.__setattr__(self, "mean", _readonly(mean))
        object.__setattr__(self, "variance", _readonly(variance))

    @property
    def shape(self) -> FloatArray:
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            values = np.exp(
                2 * np.log(self.mean) - np.log(np.where(self.variance > 0, self.variance, 1.0))
            )
        return _readonly(np.where(self.variance > 0, values, 1.0))

    @property
    def scale(self) -> FloatArray:
        with np.errstate(over="ignore", under="ignore"):
            values = np.exp(
                np.log(np.where(self.variance > 0, self.variance, 1.0)) - np.log(self.mean)
            )
        return _readonly(np.where(self.variance > 0, values, 0.0))


def toxfinder_standardize(doses: ArrayLike, reference_doses: ArrayLike) -> FloatArray:
    """Divide each physical agent dose by its single-agent reference dose."""
    raw_dose, raw_reference = np.asarray(doses), np.asarray(reference_doses)
    if raw_dose.size > _MAX_PAIRS * 2 or raw_dose.shape[-1:] != (2,) or raw_reference.shape != (2,):
        raise ValueError("doses must end in two values, with at most 100 dose pairs")
    dose = finite(raw_dose, "doses")
    reference = finite(raw_reference, "reference_doses")
    if (
        dose.shape[-1:] != (2,)
        or reference.shape != (2,)
        or np.any(dose < 0)
        or np.any(reference <= 0)
    ):
        raise ValueError("doses must end in two nonnegative values; references must be positive")
    if dose.size > _MAX_PAIRS * 2:
        raise ValueError("at most 100 dose pairs are supported")
    return _readonly(dose / reference)


def _dose_array(doses: ArrayLike) -> FloatArray:
    raw = np.asarray(doses)
    if raw.ndim == 1 and raw.shape == (2,):
        raw = raw.reshape((1, 2))
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] > _MAX_PAIRS:
        raise ValueError("standardized doses must have shape (n, 2), with at most 100 rows")
    result = finite(raw, "doses")
    if np.any(result < 0):
        raise ValueError("standardized doses must be nonnegative")
    return result


def _parameter_array(parameters: ArrayLike, log_parameters: bool) -> FloatArray:
    raw = np.asarray(parameters)
    if raw.ndim < 1 or raw.shape[-1] != 6 or raw.size > 200_000:
        raise ValueError("parameters must end in six values and contain at most 200,000 values")
    result = np.asarray(raw, dtype=np.float64)
    if log_parameters:
        if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
            raise ValueError("log parameters must be finite except alpha log values may be -inf")
        if np.any(np.isneginf(result[..., [1, 3, 5]])):
            raise ValueError("log beta parameters must be finite")
        return result
    if (
        not np.all(np.isfinite(result))
        or np.any(result[..., [0, 2, 4]] < 0)
        or np.any(result[..., [1, 3, 5]] <= 0)
    ):
        raise ValueError("alpha parameters must be nonnegative and beta parameters positive")
    with np.errstate(divide="ignore"):
        return np.log(result)


def toxfinder_log_probabilities(
    doses: ArrayLike, parameters: ArrayLike, *, log_parameters: bool = False
) -> FloatArray:
    """Return Cartesian log probabilities (parameter batch, dose row, cell)."""
    dose = _dose_array(doses)
    z0 = _parameter_array(parameters, log_parameters)
    batch = z0.shape[:-1]
    if int(np.prod(batch or (1,))) * dose.shape[0] > 200_000:
        raise ValueError("requested probability evaluation exceeds 200,000 parameter-dose pairs")
    z = np.broadcast_to(z0, (*batch, 6))[..., None, :]
    x1 = dose[:, 0].reshape((1,) * len(batch) + (-1,))
    x2 = dose[:, 1].reshape((1,) * len(batch) + (-1,))
    x1 = np.broadcast_to(x1, (*batch, dose.shape[0]))
    x2 = np.broadcast_to(x2, (*batch, dose.shape[0]))
    terms = np.full((*batch, dose.shape[0], 3), -np.inf)
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        for term, x, alpha, beta in ((0, x1, 0, 1), (1, x2, 2, 3)):
            positive = (x > 0) & np.isfinite(z[..., alpha])
            log_x = np.zeros_like(x)
            np.log(x, out=log_x, where=x > 0)
            magnitude = np.full_like(x, -np.inf)
            np.log(np.abs(log_x), out=magnitude, where=log_x != 0)
            powered_log = np.sign(log_x) * np.exp(z[..., beta] + magnitude)
            terms[..., term] = np.where(positive, z[..., alpha] + powered_log, -np.inf)

        positive = (x1 > 0) & (x2 > 0) & np.isfinite(z[..., 4])
        log_x1, log_x2 = np.zeros_like(x1), np.zeros_like(x2)
        np.log(x1, out=log_x1, where=x1 > 0)
        np.log(x2, out=log_x2, where=x2 > 0)
        magnitude1, magnitude2 = np.full_like(x1, -np.inf), np.full_like(x2, -np.inf)
        np.log(np.abs(log_x1), out=magnitude1, where=log_x1 != 0)
        np.log(np.abs(log_x2), out=magnitude2, where=log_x2 != 0)
        # Compute beta1*beta3*log(x1) and beta2*beta3*log(x2)
        # directly, so an extreme beta1 need not overflow before tiny beta3.
        component1 = np.sign(log_x1) * np.exp(z[..., 1] + z[..., 5] + magnitude1)
        component2 = np.sign(log_x2) * np.exp(z[..., 3] + z[..., 5] + magnitude2)
        interaction = component1 + component2
        terms[..., 2] = np.where(positive, z[..., 4] + interaction, -np.inf)
    if np.any(np.isnan(terms)):
        raise ArithmeticError(
            "model terms are not representable at the supplied doses and parameters"
        )
    log_q = logsumexp(terms, axis=-1)
    toxic = -np.logaddexp(0.0, -log_q)
    no_toxic = -np.logaddexp(0.0, log_q)
    origin = (x1 == 0) & (x2 == 0)
    toxic = np.where(origin, -np.inf, toxic)
    no_toxic = np.where(origin, 0.0, no_toxic)
    return _readonly(np.stack((no_toxic, toxic), axis=-1))


def toxfinder_probabilities(
    doses: ArrayLike, parameters: ArrayLike, *, log_parameters: bool = False
) -> FloatArray:
    """Return probabilities ordered as (no toxicity, toxicity)."""
    return _readonly(
        np.exp(toxfinder_log_probabilities(doses, parameters, log_parameters=log_parameters))
    )


def toxfinder_log_likelihood(
    doses: ArrayLike,
    toxicities: ArrayLike,
    subjects: ArrayLike,
    parameters: ArrayLike,
    *,
    log_parameters: bool = False,
) -> FloatArray:
    """Grouped Bernoulli log likelihood, with the parameter batch as leading axes."""
    dose = _dose_array(doses)
    raw_y, raw_n = np.asarray(toxicities), np.asarray(subjects)
    if raw_y.size > _MAX_PAIRS or raw_n.size > _MAX_PAIRS:
        raise ValueError("count vectors may contain at most 100 rows")
    y, n = count(raw_y, "toxicities"), count(raw_n, "subjects")
    if y.ndim != 1 or n.shape != y.shape or dose.shape != (y.size, 2) or y.size > _MAX_PAIRS:
        raise ValueError(
            "require matching vectors and a (dose-pair, 2) dose array with at most 100 rows"
        )
    if n.sum() > _MAX_SUBJECTS or np.any(y > n):
        raise ValueError("require toxicities <= subjects and at most 10,000 subjects")
    if np.any((dose == 0).all(axis=1) & (y > 0)):
        raise ValueError("toxicity is impossible at the zero-dose origin")
    logs = toxfinder_log_probabilities(dose, parameters, log_parameters=log_parameters)
    toxic_part = np.zeros_like(logs[..., 1])
    safe_part = np.zeros_like(logs[..., 0])
    np.multiply(y, logs[..., 1], out=toxic_part, where=y > 0)
    np.multiply(n - y, logs[..., 0], out=safe_part, where=n > y)
    return _readonly(np.sum(toxic_part + safe_part, axis=-1))


def _log_gamma_draw(shape: float, scale: float, rng: np.random.Generator) -> float:
    if scale == 0:
        raise ValueError("fixed prior coordinates do not use gamma sampling")
    if shape < 1:
        g = rng.gamma(shape + 1.0)
        u = rng.random()
        if u == 0 or g == 0:
            raise ArithmeticError("random generator returned an unrepresentable gamma draw")
        return float(np.log(g) + np.log(u) / shape + np.log(scale))
    g = rng.gamma(shape)
    if g == 0:
        raise ArithmeticError("random generator returned an unrepresentable gamma draw")
    return float(np.log(g) + np.log(scale))


def _log_posterior(
    z: FloatArray,
    doses: FloatArray,
    y: FloatArray,
    n: FloatArray,
    shape: FloatArray,
    scale: FloatArray,
    fixed: NDArray[np.bool_],
    fixed_z: FloatArray,
) -> float:
    free = ~fixed
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.exp(z[free] - np.log(scale[free]))
        prior = np.sum(shape[free] * z[free] - scaled)
    if not np.isfinite(prior):
        return -np.inf
    ll = float(toxfinder_log_likelihood(doses, y, n, z, log_parameters=True))
    return float(prior + ll) if np.isfinite(ll) else -np.inf


def _slice_coordinate(
    z: FloatArray, index: int, logp: float, width: float, rng: np.random.Generator, objective
) -> tuple[FloatArray, float, int]:
    evaluations = 0
    height = logp + np.log1p(-rng.random())
    left = z[index] - width * rng.random()
    right = left + width
    for _ in range(128):
        trial = z.copy()
        trial[index] = left
        evaluations += 1
        if objective(trial) <= height:
            break
        left -= width
    else:
        raise ArithmeticError("slice sampler left bracket exceeded 128 steps")
    for _ in range(128):
        trial = z.copy()
        trial[index] = right
        evaluations += 1
        if objective(trial) <= height:
            break
        right += width
    else:
        raise ArithmeticError("slice sampler right bracket exceeded 128 steps")
    for _ in range(512):
        proposal = z.copy()
        proposal[index] = rng.uniform(left, right)
        value = objective(proposal)
        evaluations += 1
        if value >= height:
            proposal.flags.writeable = True
            return proposal, value, evaluations
        if proposal[index] < z[index]:
            left = proposal[index]
        else:
            right = proposal[index]
    raise ArithmeticError("slice sampler failed to find a point in 512 proposals")


@dataclass(frozen=True)
class ToxFinderFit:
    standardized_doses: FloatArray
    toxicities: FloatArray
    subjects: FloatArray
    log_parameters: FloatArray
    log_likelihood: FloatArray
    probabilities: FloatArray
    parameter_summary: ChainSummary
    evaluations: int
    warmup: int

    @property
    def parameters(self) -> FloatArray:
        """Exponentiated draws; extreme log draws can underflow or overflow."""
        with np.errstate(under="ignore", over="ignore"):
            return _readonly(np.exp(self.log_parameters))


def fit_toxfinder(
    doses: ArrayLike,
    toxicities: ArrayLike,
    subjects: ArrayLike,
    *,
    prior: ToxFinderPrior,
    rng: np.random.Generator,
    draws: int = 500,
    warmup: int = 250,
    chains: int = 2,
) -> ToxFinderFit:
    """Fit the two-agent model by componentwise slice sampling in log parameters."""
    dose = _dose_array(doses)
    raw_y, raw_n = np.asarray(toxicities), np.asarray(subjects)
    if raw_y.size > _MAX_PAIRS or raw_n.size > _MAX_PAIRS:
        raise ValueError("count vectors may contain at most 100 rows")
    y, n = count(raw_y, "toxicities"), count(raw_n, "subjects")
    if (
        y.ndim != 1
        or n.shape != y.shape
        or dose.shape != (y.size, 2)
        or not 1 <= y.size <= _MAX_PAIRS
    ):
        raise ValueError("require matching vectors and 1..100 dose pairs")
    if n.sum() > _MAX_SUBJECTS or np.any(y > n) or np.any((dose == 0).all(axis=1) & (y > 0)):
        raise ValueError("invalid counts, more than 10,000 subjects, or toxicity at the origin")
    for value, name, lower in ((draws, "draws", 8), (warmup, "warmup", 0), (chains, "chains", 2)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < lower
        ):
            raise ValueError(f"{name} must be an integer >= {lower}")
    if chains > 4 or draws > 10_000 or (warmup + draws) * chains * 6 * y.size > _MAX_EVALS:
        raise ValueError("requested fit exceeds resource limits")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    shape, scale = prior.shape, prior.scale
    fixed = prior.variance == 0
    fixed_z = np.log(prior.mean)
    all_z = np.empty((chains, draws, 6))
    all_ll = np.empty((chains, draws))
    evaluations = 0
    for chain in range(chains):
        z = fixed_z.copy()
        for j in np.flatnonzero(~fixed):
            z[j] = _log_gamma_draw(float(shape[j]), float(scale[j]), rng)

        objective_evaluations = 0

        def objective(value: FloatArray) -> float:
            nonlocal objective_evaluations
            objective_evaluations += 1
            if objective_evaluations > _MAX_EVALS:
                raise ValueError("actual sampler evaluations exceed the resource limit")
            return _log_posterior(value, dose, y, n, shape, scale, fixed, fixed_z)

        lp = objective(z)
        evaluations += 1
        if not np.isfinite(lp):
            raise ArithmeticError("initial log posterior is not representable")
        widths = np.ones(6)
        widths[~fixed] = np.sqrt(np.maximum(polygamma(1, shape[~fixed]), 1e-4))
        for iteration in range(warmup + draws):
            if y.sum() == 0 and n.sum() == 0:
                for j in np.flatnonzero(~fixed):
                    z[j] = _log_gamma_draw(float(shape[j]), float(scale[j]), rng)
                lp = objective(z)
                evaluations += 1
            else:
                for j in np.flatnonzero(~fixed):
                    z, lp, used = _slice_coordinate(z, int(j), lp, float(widths[j]), rng, objective)
                    evaluations += used
            if iteration >= warmup:
                k = iteration - warmup
                all_z[chain, k] = z
                all_ll[chain, k] = float(
                    toxfinder_log_likelihood(dose, y, n, z, log_parameters=True)
                )
    probabilities = toxfinder_probabilities(dose, all_z, log_parameters=True)
    return ToxFinderFit(
        _readonly(dose),
        _readonly(y),
        _readonly(n),
        _readonly(all_z),
        _readonly(all_ll),
        probabilities,
        summarize_chains(all_z),
        evaluations,
        warmup,
    )
