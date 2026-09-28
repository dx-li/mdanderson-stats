"""Explicit-prior posterior fitting for the GAO U2OET model.

Prior coordinates are a Python convention, not inferred native prior-file
ordering: threshold intercepts/slopes and association Fisher-z are Gaussian;
endpoint lambdas and shared kappa are sampled as logarithms. Priors are
independent in these retained coordinates, and zero SD fixes a coordinate.
The sampler uses elliptical slice updates; diagnostics are estimates, not
convergence certificates. Gaussian rectangle failures are propagated.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import ChainSummary, summarize_chains
from .u2oet import _real
from .u2oet_decision import _integer
from .u2oet_gao import U2OETGAOMarginal, _dose_grid, u2oet_gao_probabilities

_MAX_RETAINED_CELLS = 20_000_000
_MAX_LIKELIHOOD_EVALUATIONS = 1_000_000
_MAX_WORK_UNITS = 100_000_000
_MAX_SLICE_STEPS = 1000


def u2oet_gao_parameter_names(efficacy_levels: int, toxicity_levels: int) -> tuple[str, ...]:
    """Return the retained coordinate names and ordering used by this fitter."""
    e = _integer(efficacy_levels, "efficacy_levels", 2, 4)
    t = _integer(toxicity_levels, "toxicity_levels", 2, 4)
    names: list[str] = []
    for outcome, levels in (("efficacy", e), ("toxicity", t)):
        for threshold in range(1, levels):
            names.extend(
                (
                    f"{outcome}.intercept.{threshold}.agent1",
                    f"{outcome}.intercept.{threshold}.agent2",
                    f"{outcome}.slope.{threshold}.agent1",
                    f"{outcome}.slope.{threshold}.agent2",
                )
            )
        names.append(f"{outcome}.log_lambda")
    return (*names, "shared.log_kappa", "association.fisher_z")


@dataclass(frozen=True)
class U2OETGAOFit:
    """Posterior draws and diagnostics; arrays retain chain and draw axes."""

    names: tuple[str, ...]
    parameters: FloatArray
    joint: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    counts: FloatArray
    toxicity_only: FloatArray
    prior_mean: FloatArray
    prior_sd: FloatArray
    likelihood_evaluations: int
    likelihood_work_units: int
    warmup: int
    efficacy_levels: int
    toxicity_levels: int


def _input_shape(value: ArrayLike, name: str) -> tuple[int, ...]:
    """Read dimensions before numeric conversion or allocation where possible."""
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc


def _marginal(coordinates: FloatArray, levels: int) -> U2OETGAOMarginal:
    thresholds = levels - 1
    block = coordinates[: 4 * thresholds].reshape(thresholds, 4)
    with np.errstate(over="ignore", under="ignore"):
        lam = float(np.exp(coordinates[4 * thresholds]))
    if not np.isfinite(lam) or lam <= 0:
        raise ArithmeticError("GAO log_lambda draw is outside the representable positive range")
    return U2OETGAOMarginal(block[:, :2], block[:, 2:], lam)


def _likelihood(
    d1: FloatArray,
    d2: FloatArray,
    counts: FloatArray,
    toxicity_only: FloatArray,
    coordinates: FloatArray,
) -> tuple[float, FloatArray]:
    e_levels, t_levels = counts.shape[-2:]
    n_e, n_t = 4 * (e_levels - 1) + 1, 4 * (t_levels - 1) + 1
    with np.errstate(over="ignore", under="ignore"):
        kappa = float(np.exp(coordinates[n_e + n_t]))
    if not np.isfinite(kappa) or kappa <= 0:
        raise ArithmeticError("GAO log_kappa draw is outside the representable positive range")
    rho = float(np.tanh(coordinates[n_e + n_t + 1]))
    if abs(rho) == 1.0:
        raise ArithmeticError(
            "association Fisher-z exceeds the representable interior correlation range"
        )
    probabilities = u2oet_gao_probabilities(
        d1,
        d2,
        efficacy=_marginal(coordinates[:n_e], e_levels),
        toxicity=_marginal(coordinates[n_e : n_e + n_t], t_levels),
        kappa=kappa,
        association=rho,
    )
    log_joint = probabilities.log_joint
    observed = counts > 0
    ll = float(np.sum(counts[observed] * log_joint[observed]))
    observed_t = toxicity_only > 0
    if np.any(observed_t):
        ll += float(np.sum(toxicity_only[observed_t] * probabilities.log_toxicity[observed_t]))
    if np.isnan(ll) or ll == np.inf:
        raise ArithmeticError("GAO observed-data log likelihood is invalid")
    return ll, probabilities.joint


def fit_u2oet_gao(
    doses1: ArrayLike,
    doses2: ArrayLike,
    counts: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    toxicity_only: ArrayLike | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    rng: np.random.Generator,
    max_likelihood_evaluations: int = 1_000_000,
    max_work: int = 100_000_000,
) -> U2OETGAOFit:
    """Fit GAO under caller-supplied independent normal priors.

    ``counts`` axes are dose1, dose2, efficacy category, toxicity category;
    ``toxicity_only`` optionally has dose1, dose2, toxicity category axes.
    A supplied ``initial`` has one coordinate vector or one vector per chain.
    Otherwise chains start at the prior center; dispersed starts are preferable
    for assessing multimodality. Work units count one full dose/category grid
    per likelihood evaluation; the rectangle integrator has a separate fixed
    subdivision limit. The hard-capped budgets are checked before each call.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    d1, d2 = _dose_grid(doses1, "doses1"), _dose_grid(doses2, "doses2")
    count_shape = _input_shape(counts, "counts")
    if (
        len(count_shape) != 4
        or count_shape[:2] != (d1.size, d2.size)
        or any(not 2 <= size <= 4 for size in count_shape[2:])
    ):
        raise ValueError("counts must have dose-by-dose-by-category shape with 2–4 categories")
    n = _real(counts, "counts")
    if np.any(n < 0) or np.any(n != np.floor(n)) or n.sum() >= 2**53:
        raise ValueError("counts must be integer dose-by-dose-by-efficacy-by-toxicity cells")
    if toxicity_only is None:
        nt = np.zeros((*n.shape[:2], n.shape[-1]))
    else:
        expected_toxicity_shape = (*n.shape[:2], n.shape[-1])
        if _input_shape(toxicity_only, "toxicity_only") != expected_toxicity_shape:
            raise ValueError("toxicity_only must have dose-by-dose-by-toxicity shape")
        nt = _real(toxicity_only, "toxicity_only")
    if (
        nt.shape != (*n.shape[:2], n.shape[-1])
        or np.any(nt < 0)
        or np.any(nt != np.floor(nt))
        or n.sum() + nt.sum() >= 2**53
    ):
        raise ValueError("toxicity_only must be integer dose-by-dose-by-toxicity counts")
    e_levels, t_levels = n.shape[-2:]
    names = u2oet_gao_parameter_names(e_levels, t_levels)
    dimension = len(names)
    if _input_shape(prior_mean, "prior_mean") != (dimension,):
        raise ValueError("prior_mean must have one value per named GAO coordinate")
    if _input_shape(prior_sd, "prior_sd") != (dimension,):
        raise ValueError("prior_sd must have one value per named GAO coordinate")
    mu, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if mu.shape != (dimension,) or sd.shape != mu.shape or np.any(sd < 0):
        raise ValueError("prior_mean and prior_sd must match named coordinates; SDs may be zero")
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    max_likelihood_evaluations = _integer(
        max_likelihood_evaluations, "max_likelihood_evaluations", 1, _MAX_LIKELIHOOD_EVALUATIONS
    )
    max_work = _integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    joint_shape = (*n.shape[:2], e_levels, t_levels)
    joint_cells = int(np.prod(joint_shape))
    # Include raw/frozen result arrays and the ordered/width buffers used by
    # summarize_chains; the 20M-cell cap bounds peak live arrays.
    retained = chains * draws * (14 * dimension + 2 * joint_cells + 2)
    if retained > _MAX_RETAINED_CELLS:
        raise ValueError("retained GAO posterior arrays exceed the 20-million-cell budget")
    free = np.flatnonzero(sd > 0)
    minimum_evaluations = chains * (1 + (warmup + draws if free.size else 0))
    if minimum_evaluations > max_likelihood_evaluations:
        raise ValueError("minimum chain evaluations exceed max_likelihood_evaluations")
    if minimum_evaluations * joint_cells > max_work:
        raise ValueError("minimum GAO likelihood work exceeds max_work")

    if initial is None:
        starts = np.broadcast_to(mu, (chains, dimension)).copy()
    else:
        initial_shape = _input_shape(initial, "initial")
        if initial_shape not in ((dimension,), (chains, dimension)):
            raise ValueError("initial must have one coordinate vector or one vector per chain")
        starts = _real(initial, "initial").copy()
        if starts.shape == (dimension,):
            starts = np.broadcast_to(starts, (chains, dimension)).copy()
        if starts.shape != (chains, dimension):
            raise ValueError("initial must have one coordinate vector or one vector per chain")
    if np.any(starts[:, sd == 0] != mu[sd == 0]):
        raise ValueError(
            "initial coordinates with zero prior SD must equal their fixed prior means"
        )

    parameters = np.empty((chains, draws, dimension), dtype=float)
    joint = np.empty((chains, draws, *joint_shape), dtype=float)
    log_likelihood = np.empty((chains, draws), dtype=float)
    evaluations = 0
    work = 0

    def evaluate(state: FloatArray) -> tuple[float, FloatArray]:
        nonlocal evaluations, work
        if evaluations >= max_likelihood_evaluations:
            raise ArithmeticError("GAO likelihood-evaluation budget exhausted")
        if work + joint_cells > max_work:
            raise ArithmeticError("GAO likelihood-work budget exhausted")
        ll, p = _likelihood(d1, d2, n, nt, state)
        evaluations += 1
        work += joint_cells
        return ll, p

    for chain in range(chains):
        state = starts[chain].copy()
        ll, probabilities = evaluate(state)
        if not np.isfinite(ll):
            raise ValueError(f"initial state has zero GAO likelihood in chain {chain}")
        for iteration in range(warmup + draws):
            if free.size:
                center = mu[free]
                centered = state[free] - center
                direction = rng.normal(size=free.size) * sd[free]
                height = ll + np.log1p(-rng.random())
                angle = rng.uniform(0.0, 2.0 * np.pi)
                lower, upper = angle - 2.0 * np.pi, angle
                for _ in range(_MAX_SLICE_STEPS):
                    proposal = state.copy()
                    proposal[free] = center + centered * np.cos(angle) + direction * np.sin(angle)
                    if not np.all(np.isfinite(proposal)):
                        raise ArithmeticError(
                            "GAO elliptical proposal exceeds floating-point range"
                        )
                    trial_ll, trial_probabilities = evaluate(proposal)
                    if trial_ll >= height:
                        state, ll, probabilities = proposal, trial_ll, trial_probabilities
                        break
                    if angle < 0:
                        lower = angle
                    else:
                        upper = angle
                    angle = rng.uniform(lower, upper)
                else:
                    raise ArithmeticError(f"GAO elliptical-slice update failed in chain {chain}")
            if iteration >= warmup:
                out = iteration - warmup
                parameters[chain, out] = state
                joint[chain, out] = probabilities
                log_likelihood[chain, out] = ll

    frozen_parameters = _freeze(parameters)
    return U2OETGAOFit(
        names=names,
        parameters=frozen_parameters,
        joint=_freeze(joint),
        log_likelihood=_freeze(log_likelihood),
        parameter_summary=summarize_chains(frozen_parameters),
        counts=_freeze(n),
        toxicity_only=_freeze(nt),
        prior_mean=_freeze(mu),
        prior_sd=_freeze(sd),
        likelihood_evaluations=evaluations,
        likelihood_work_units=work,
        warmup=warmup,
        efficacy_levels=e_levels,
        toxicity_levels=t_levels,
    )
