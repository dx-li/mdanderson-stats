"""Independent prior draws and beta moment-matching information for U2OET."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import _joint, _marginal, _real, u2oet_standardize
from .u2oet_decision import _integer
from .u2oet_fit import _parameters, u2oet_parameter_names


def _positive_normal(mu: float, sd: float, size: int, rng: np.random.Generator) -> FloatArray:
    """Normal rejection above its mean; exponential envelope in the far tail."""
    result = np.empty(size)
    pending = np.arange(size)
    if mu < 0:
        a = -mu / sd
        if not np.isfinite(a):
            raise ValueError("truncated-normal standardized bound exceeds floating-point range")
        rate = a / 2 + np.hypot(a, 2) / 2
    for _ in range(10000):
        if mu >= 0:
            values = rng.normal(mu, sd, pending.size)
            accept = values > 0
        else:
            excess = rng.exponential(size=pending.size) / rate
            accept = np.log1p(-rng.random(pending.size)) <= -0.5 * (excess - 1 / rate) ** 2
            # mu + sd*(a+excess) would cancel in the extreme truncated tail.
            values = sd * excess
        result[pending[accept]] = values[accept]
        pending = pending[~accept]
        if not pending.size:
            if not np.all(np.isfinite(result)) or np.any(result <= 0):
                raise ArithmeticError("positive normal draws exceed floating-point range")
            return result
    raise ArithmeticError("positive normal rejection sampler did not finish")


@dataclass(frozen=True)
class U2OETPriorDraws:
    names: tuple[str, ...]
    parameters: FloatArray
    joint: FloatArray


def sample_u2oet_prior(
    doses1: ArrayLike,
    doses2: ArrayLike,
    *,
    efficacy_levels: int,
    toxicity_levels: int,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    model: str = "pds",
    centering: str = "log",
    draws: int = 10000,
    rng: np.random.Generator,
) -> U2OETPriorDraws:
    """IID prior draws in the same named coordinates as fit_u2oet; no MCMC."""
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    u2oet_standardize(d1)
    u2oet_standardize(d2)
    names = u2oet_parameter_names(efficacy_levels, toxicity_levels, model=model)
    efficacy_levels = _integer(efficacy_levels, "efficacy_levels", 2, 4)
    toxicity_levels = _integer(toxicity_levels, "toxicity_levels", 2, 4)
    if centering not in ("log", "linear") or (model == "cmi" and centering != "log"):
        raise ValueError("invalid centering; CMI requires log")
    mu, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if mu.shape != (len(names) - 1,) or sd.shape != mu.shape or np.any(sd <= 0):
        raise ValueError(
            "prior arrays must match named coordinates excluding association; SDs positive"
        )
    draws = _integer(draws, "draws", 2, 1_000_000)
    shape = (d1.size, d2.size, efficacy_levels, toxicity_levels)
    if draws * int(np.prod(shape)) > 20_000_000:
        raise ValueError("prior joint draws exceed 20 million cells")
    values = np.empty((draws, len(names)))
    for i, name in enumerate(names[:-1]):
        values[:, i] = (
            _positive_normal(float(mu[i]), float(sd[i]), draws, rng)
            if ".slope." in name
            else rng.normal(mu[i], sd[i], draws)
        )
    values[:, -1] = rng.uniform(-1, 1, draws)
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("normal prior draws exceed floating-point range")
    split = next(i for i, name in enumerate(names) if name.startswith("toxicity."))
    joint = np.empty((draws, *shape))
    for i, row in enumerate(values):
        e = _parameters(row[:split], efficacy_levels, model)
        t = _parameters(row[split:-1], toxicity_levels, model)
        assert e is not None and t is not None
        joint[i] = np.exp(
            _joint(_marginal(d1, d2, e, centering), _marginal(d1, d2, t, centering), float(row[-1]))
        )
    return U2OETPriorDraws(names, _freeze(values), _freeze(joint))


@dataclass(frozen=True)
class U2OETPriorESS:
    """Cell ESS arrays have axes agent1, agent2, ordinal category.

    This is prior information, not MCMC effective sample size. NaN means the
    marginal is identically zero/one; infinity means constant in the interior.
    Zero denotes the limiting endpoint-mixture case, not a proper beta prior.
    """

    efficacy: FloatArray
    toxicity: FloatArray
    mean: float
    maximum: float
    draws: int


def _beta_ess(probabilities: FloatArray) -> FloatArray:
    probabilities = np.clip(probabilities, 0, 1)
    mean = probabilities.mean(axis=0)
    variance = np.mean((probabilities - mean) ** 2, axis=0)
    variance = np.where(np.all(probabilities == probabilities[0], axis=0), 0, variance)
    bound = mean * (1 - mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        answer = bound / variance - 1
    interior = (mean > 0) & (mean < 1)
    answer = np.where(interior, answer, np.nan)
    # A [0,1] population has variance <= mean*(1-mean). Only roundoff may
    # produce a slightly negative finite estimate with population moments.
    if np.any(answer < -1e-12):
        raise ArithmeticError("inconsistent probability moments for beta matching")
    return np.maximum(answer, 0)


def u2oet_prior_ess(joint_draws: ArrayLike) -> U2OETPriorESS:
    """Guide section 1.5: moment-match each ordinal marginal, then average.

    Uses empirical population variance (divisor draws), preserving the moment
    inequality. No undefined cell is dropped from the reported mean/maximum.
    """
    p = _real(joint_draws, "joint_draws")
    if (
        p.ndim != 5
        or p.shape[0] < 2
        or p.size > 20_000_000
        or any(not 2 <= n <= 5 for n in p.shape[1:3])
        or any(not 2 <= n <= 4 for n in p.shape[3:])
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("require >=2 prior joint draws on valid dose and outcome grids")
    if not np.all(np.abs(p.sum(axis=(-2, -1)) - 1) <= 1e-12):
        raise ValueError("each draw/dose must have unit joint probability")
    e = _beta_ess(p.sum(axis=-1))
    t = _beta_ess(p.sum(axis=-2))
    all_cells = np.r_[e.ravel(), t.ravel()]
    return U2OETPriorESS(
        _freeze(e), _freeze(t), float(all_cells.mean()), float(all_cells.max()), p.shape[0]
    )
