"""U2OET posterior sampling with explicit priors and complete ordinal outcomes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .u2oet import U2OETMarginal, _joint, _marginal, _real, u2oet_standardize
from .u2oet_decision import _integer


def u2oet_parameter_names(
    efficacy_levels: int, toxicity_levels: int, *, model: str = "pds"
) -> tuple[str, ...]:
    """Python ordering, not the unverified native prior-file ordering.

    All named coordinates have independent normal priors except association,
    which is uniform on [-1,1]. Slopes are normals truncated below at zero.
    Power/link coordinates are logarithms, so their physical priors are lognormal.
    """
    if model not in ("pds", "cmi", "pds+cmi"):
        raise ValueError("model must be pds, cmi or pds+cmi")
    names: list[str] = []
    for outcome, levels in (("efficacy", efficacy_levels), ("toxicity", toxicity_levels)):
        levels = _integer(levels, "levels", 2, 4)
        names.extend(f"{outcome}.intercept.{y}" for y in range(1, levels))
        names.extend(f"{outcome}.slope.{y}.{a}" for y in range(1, levels) for a in (1, 2))
        if model != "cmi":
            names.extend(f"{outcome}.log_power.{a}" for a in (1, 2))
        names.append(f"{outcome}.log_link")
        if model != "pds":
            names.append(f"{outcome}.interaction")
    return (*names, "association")


def _parameters(values: FloatArray, levels: int, model: str) -> U2OETMarginal | None:
    length = levels - 1
    beta = values[length : 3 * length].reshape(length, 2)
    if np.any(beta <= 0):
        return None
    offset = 3 * length
    log_power = values[offset : offset + 2] if model != "cmi" else np.zeros(2)
    offset += 2 if model != "cmi" else 0
    logarithms = np.r_[log_power, values[offset]]
    with np.errstate(over="ignore", under="ignore"):
        positive = np.exp(logarithms)
    if not np.all(np.isfinite(positive)) or np.any(positive == 0):
        raise ArithmeticError(
            "power/link prior draws exceed floating-point range; revise scaling/prior"
        )
    interaction = 0.0 if model == "pds" else float(values[offset + 1])
    return U2OETMarginal(values[:length], beta, positive[:2], float(positive[2]), interaction)


@dataclass(frozen=True)
class U2OETFit:
    """Retained coordinates and probabilities with explicit chain/draw axes.

    Use summarize_chains on parameters and clinically relevant probability or
    utility summaries. Sampling completion does not certify convergence.
    """

    names: tuple[str, ...]
    parameters: FloatArray
    joint: FloatArray
    log_likelihood: FloatArray
    association_acceptance: FloatArray
    likelihood_evaluations: int
    warmup: int
    model: str


def fit_u2oet(
    doses1: ArrayLike,
    doses2: ArrayLike,
    counts: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    model: str = "pds",
    centering: str = "log",
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    rng: np.random.Generator,
) -> U2OETFit:
    """Blocked elliptical slice plus independent-uniform association updates.

    Prior arrays follow u2oet_parameter_names, excluding association. Means/SDs
    for slopes describe the underlying normal before truncation. Initial rows
    follow the full ordering (one row per chain); no Jacobian is needed because
    retained power/link coordinates themselves are normal logarithms.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    u2oet_standardize(d1)
    u2oet_standardize(d2)
    n = _real(counts, "counts")
    if (
        n.ndim != 4
        or n.shape[:2] != (d1.size, d2.size)
        or any(not 2 <= size <= 4 for size in n.shape[2:])
        or np.any(n < 0)
        or np.any(n != np.floor(n))
        or n.sum() >= 2**53
    ):
        raise ValueError("counts must be integer agent1-by-agent2-by-efficacy-by-toxicity cells")
    names = u2oet_parameter_names(n.shape[2], n.shape[3], model=model)
    if centering not in ("log", "linear") or (model == "cmi" and centering != "log"):
        raise ValueError("invalid centering; CMI requires log")
    mu, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    dimension = len(names) - 1
    if mu.shape != (dimension,) or sd.shape != mu.shape or np.any(sd <= 0):
        raise ValueError(
            "prior arrays must match named coordinates excluding association; SDs positive"
        )
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    if draws * chains * n.size > 20_000_000:
        raise ValueError("retained joint draws exceed 20 million cells")
    split = next(i for i, name in enumerate(names) if name.startswith("toxicity."))
    blocks = (slice(0, split), slice(split, dimension))
    levels = n.shape[2:]
    if initial is None:
        start = np.tile(np.r_[mu, 0.0], (chains, 1))
        # Valid deterministic centers; supplied dispersed starts are preferable
        # for a substantive convergence assessment.
        for i, name in enumerate(names[:-1]):
            if ".slope." in name:
                start[:, i] = max(mu[i], sd[i])
    else:
        start = _real(initial, "initial").copy()
        if start.shape != (chains, dimension + 1):
            raise ValueError("initial must have one full coordinate row per chain")
    if np.any(np.abs(start[:, -1]) > 1):
        raise ValueError("initial association must lie in [-1,1]")
    parameters = np.empty((chains, draws, dimension + 1))
    joint = np.empty((chains, draws, *n.shape))
    likelihood = np.empty((chains, draws))
    accepted = np.zeros(chains)
    evaluations = 0
    observed = n > 0

    def evaluate(e: FloatArray, t: FloatArray, rho: float) -> tuple[float, FloatArray]:
        nonlocal evaluations
        evaluations += 1
        log_joint = _joint(e, t, rho)
        ll = float(np.sum(n[observed] * log_joint[observed]))
        if np.isnan(ll) or ll == np.inf:
            raise ArithmeticError("invalid posterior likelihood")
        return ll, log_joint

    for chain in range(chains):
        state = start[chain].copy()
        marginals = []
        for block, category_count in zip(blocks, levels, strict=True):
            marginal = _parameters(state[block], category_count, model)
            if marginal is None:
                raise ValueError("initial slopes must be positive")
            marginals.append(_marginal(d1, d2, marginal, centering))
        ll, log_joint = evaluate(marginals[0], marginals[1], float(state[-1]))
        if not np.isfinite(ll):
            raise ValueError("initial state must have finite likelihood")
        for iteration in range(warmup + draws):
            for which, block in enumerate(blocks):
                centered = state[block] - mu[block]
                direction = rng.normal(size=centered.size) * sd[block]
                height = ll + np.log1p(-rng.random())
                angle = rng.uniform(0, 2 * np.pi)
                lower, upper = angle - 2 * np.pi, angle
                for _ in range(1000):
                    proposal = mu[block] + centered * np.cos(angle) + direction * np.sin(angle)
                    marginal = _parameters(proposal, levels[which], model)
                    if marginal is not None:
                        candidate = _marginal(d1, d2, marginal, centering)
                        e, t = (
                            (candidate, marginals[1]) if which == 0 else (marginals[0], candidate)
                        )
                        trial_ll, trial_joint = evaluate(e, t, float(state[-1]))
                        if trial_ll >= height:
                            state[block] = proposal
                            marginals[which] = candidate
                            ll, log_joint = trial_ll, trial_joint
                            break
                    if angle < 0:
                        lower = angle
                    else:
                        upper = angle
                    angle = rng.uniform(lower, upper)
                else:
                    raise ArithmeticError(
                        f"elliptical slice failed at chain {chain}, iteration {iteration}"
                    )
            rho = rng.uniform(-1, 1)
            trial_ll, trial_joint = evaluate(marginals[0], marginals[1], rho)
            if np.log1p(-rng.random()) < trial_ll - ll:
                state[-1] = rho
                ll, log_joint = trial_ll, trial_joint
                if iteration >= warmup:
                    accepted[chain] += 1
            if iteration >= warmup:
                index = iteration - warmup
                parameters[chain, index] = state
                joint[chain, index] = np.exp(log_joint)
                likelihood[chain, index] = ll
    return U2OETFit(
        names,
        _freeze(parameters),
        _freeze(joint),
        _freeze(likelihood),
        _freeze(accepted / draws),
        evaluations,
        warmup,
        model,
    )
