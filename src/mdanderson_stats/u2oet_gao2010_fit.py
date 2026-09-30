"""Bounded posterior fitting for the original centered-dose 2010 GAO model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import ChainSummary, summarize_chains
from .u2oet import _real
from .u2oet_decision import _integer
from .u2oet_gao2010 import (
    U2OETGAO2010Marginal,
    _dose_grid_2010,
    _InvalidGAO2010Domain,
    u2oet_gao2010_probabilities,
)

_MAX_RETAINED_CELLS = 20_000_000
_MAX_LIKELIHOOD_EVALUATIONS = 1_000_000
_MAX_WORK_UNITS = 100_000_000
_MAX_SLICE_STEPS = 1000


def u2oet_gao2010_parameter_names(efficacy_levels: int, toxicity_levels: int) -> tuple[str, ...]:
    """Return Gaussian coordinate names followed by raw uniform association.

    Prior mean/SD arrays correspond to every returned name except the final
    ``association`` coordinate. For each endpoint and threshold, coefficient
    order is intercept agent 1, intercept agent 2, slope agent 1, slope agent
    2, followed by log-link-shape and interaction coordinates.
    """
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
        names.extend((f"{outcome}.log_lambda", f"{outcome}.gamma"))
    return (*names, "association")


@dataclass(frozen=True)
class U2OETGAO2010Fit:
    """Read-only retained parameter/probability draws and sampler diagnostics."""

    names: tuple[str, ...]
    parameters: FloatArray
    joint: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    doses1: FloatArray
    doses2: FloatArray
    counts: FloatArray
    toxicity_only: FloatArray
    prior_mean: FloatArray
    prior_sd: FloatArray
    association_acceptance: FloatArray | None
    likelihood_evaluations: int
    likelihood_work_units: int
    warmup: int
    efficacy_levels: int
    toxicity_levels: int
    fixed_association: float | None
    prior_support: str


def _input_shape(value: ArrayLike, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    try:
        return tuple(int(size) for size in shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc


def _endpoint_marginal(coordinates: FloatArray, levels: int) -> U2OETGAO2010Marginal:
    thresholds = levels - 1
    coefficient_count = 4 * thresholds
    coefficients = coordinates[:coefficient_count].reshape(thresholds, 4)
    with np.errstate(over="ignore", under="ignore"):
        lambda_ = float(np.exp(coordinates[coefficient_count]))
    if not np.isfinite(lambda_) or lambda_ <= 0.0:
        raise ArithmeticError("2010 GAO log_lambda is outside the representable range")
    return U2OETGAO2010Marginal(
        coefficients[:, :2],
        coefficients[:, 2:],
        lambda_,
        float(coordinates[coefficient_count + 1]),
    )


def _likelihood(
    doses1: FloatArray,
    doses2: FloatArray,
    counts: FloatArray,
    toxicity_only: FloatArray,
    coordinates: FloatArray,
    efficacy_levels: int,
    toxicity_levels: int,
) -> tuple[float, FloatArray]:
    e_dimension = 4 * (efficacy_levels - 1) + 2
    t_dimension = 4 * (toxicity_levels - 1) + 2
    efficacy = _endpoint_marginal(coordinates[:e_dimension], efficacy_levels)
    toxicity = _endpoint_marginal(
        coordinates[e_dimension : e_dimension + t_dimension], toxicity_levels
    )
    association = float(coordinates[-1])
    probability = u2oet_gao2010_probabilities(
        doses1,
        doses2,
        efficacy=efficacy,
        toxicity=toxicity,
        association=association,
    )
    value = probability.loglikelihood(counts, toxicity_only=toxicity_only)
    if np.isnan(value) or value == np.inf:
        raise ArithmeticError("2010 GAO observed-data log likelihood is invalid")
    return value, probability.joint


def fit_u2oet_gao2010(
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
    fixed_association: float | None = None,
    rng: np.random.Generator,
    max_likelihood_evaluations: int = _MAX_LIKELIHOOD_EVALUATIONS,
    max_work: int = _MAX_WORK_UNITS,
) -> U2OETGAO2010Fit:
    """Fit the 2010 GAO model under explicit independent Gaussian coordinates.

    The Gaussian coordinates are threshold coefficients, ``log(lambda)`` and
    endpoint-specific ``gamma``. Their independently supplied Normal priors
    are restricted jointly to values whose interaction bracket is positive at
    every dose-grid pair and threshold. There is one joint validity indicator,
    with no gamma-conditional prior renormalization. Association has a separate
    Uniform(-1,1) prior; ``fixed_association`` replaces that prior by a fixed
    value for conditional/reference analyses.

    The named-coordinate order is returned by
    :func:`u2oet_gao2010_parameter_names`; ``prior_mean`` and ``prior_sd`` omit
    its final association coordinate. Zero SD fixes a Gaussian coordinate.
    A supplied ``initial`` has one full raw-coordinate vector or one vector per
    chain. Otherwise chains start at the Gaussian prior center and association
    zero (or the fixed value); every initial state is validated before the RNG
    is used. Dispersed valid starts are preferable for convergence assessment.

    Each endpoint's Gaussian coordinates use elliptical slice updates, with
    invalid grid states assigned zero target density. Free association uses an
    independent Uniform(-1,1) Metropolis proposal. This is a Python sampler,
    not a reproduction of the paper's Gibbs/two-level algorithm. Diagnostics
    are estimates and do not establish convergence. Work counts one full
    dose/category likelihood grid per evaluation; the Gaussian rectangle
    integrator has its own fixed quadrature budget.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    d1, d2 = _dose_grid_2010(doses1, "doses1"), _dose_grid_2010(doses2, "doses2")
    count_shape = _input_shape(counts, "counts")
    if (
        len(count_shape) != 4
        or count_shape[:2] != (d1.size, d2.size)
        or any(not 2 <= size <= 4 for size in count_shape[2:])
    ):
        raise ValueError("counts must be dose-by-dose-by-efficacy-by-toxicity cells, 2–4 each")
    n = _real(counts, "counts")
    if np.any(n < 0) or np.any(n != np.floor(n)) or n.sum() >= 2**53:
        raise ValueError("counts must be nonnegative integer joint outcome cells")
    if toxicity_only is None:
        nt = np.zeros((*n.shape[:2], n.shape[-1]))
    else:
        expected_partial_shape = (*n.shape[:2], n.shape[-1])
        if _input_shape(toxicity_only, "toxicity_only") != expected_partial_shape:
            raise ValueError("toxicity_only must have dose-by-dose-by-toxicity shape")
        nt = _real(toxicity_only, "toxicity_only")
    if (
        nt.shape != (*n.shape[:2], n.shape[-1])
        or np.any(nt < 0)
        or np.any(nt != np.floor(nt))
        or n.sum() + nt.sum() >= 2**53
    ):
        raise ValueError("toxicity_only must be nonnegative integer toxicity counts")

    efficacy_levels, toxicity_levels = n.shape[-2:]
    full_names = u2oet_gao2010_parameter_names(efficacy_levels, toxicity_levels)
    gaussian_names = full_names[:-1]
    dimension = len(gaussian_names)
    if _input_shape(prior_mean, "prior_mean") != (dimension,):
        raise ValueError("prior_mean must have one value per named Gaussian coordinate")
    if _input_shape(prior_sd, "prior_sd") != (dimension,):
        raise ValueError("prior_sd must have one value per named Gaussian coordinate")
    mu, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if mu.shape != (dimension,) or sd.shape != mu.shape or np.any(sd < 0):
        raise ValueError("prior_mean and prior_sd must match named Gaussian coordinates; SD>=0")
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    max_likelihood_evaluations = _integer(
        max_likelihood_evaluations,
        "max_likelihood_evaluations",
        1,
        _MAX_LIKELIHOOD_EVALUATIONS,
    )
    max_work = _integer(max_work, "max_work", 1, _MAX_WORK_UNITS)

    fixed_rho: float | None
    if fixed_association is None:
        fixed_rho = None
    else:
        fixed_value = _real(fixed_association, "fixed_association")
        if fixed_value.ndim != 0 or abs(float(fixed_value)) > 1.0:
            raise ValueError("fixed_association must be a scalar in [-1,1]")
        fixed_rho = float(fixed_value)

    e_dimension = 4 * (efficacy_levels - 1) + 2
    endpoint_blocks = (slice(0, e_dimension), slice(e_dimension, dimension))
    active_blocks = sum(bool(np.any(sd[block] > 0)) for block in endpoint_blocks)
    active_association = int(fixed_rho is None)
    per_iteration_minimum = active_blocks + active_association
    minimum_evaluations = chains * (1 + (warmup + draws) * per_iteration_minimum)
    if minimum_evaluations > max_likelihood_evaluations:
        raise ValueError("minimum evaluations exceed max_likelihood_evaluations")

    joint_shape = (*n.shape[:2], efficacy_levels, toxicity_levels)
    joint_cells = int(np.prod(joint_shape))
    minimum_work = minimum_evaluations * joint_cells
    if minimum_work > max_work:
        raise ValueError("minimum likelihood work exceeds max_work")
    retained = chains * draws * (14 * (dimension + 1) + 2 * joint_cells + 2)
    retained += n.size + nt.size
    if retained > _MAX_RETAINED_CELLS:
        raise ValueError("retained 2010 GAO posterior arrays exceed the 20-million-cell budget")

    if initial is None:
        starts = np.empty((chains, dimension + 1), dtype=float)
        starts[:, :dimension] = mu
        starts[:, -1] = 0.0 if fixed_rho is None else fixed_rho
    else:
        initial_shape = _input_shape(initial, "initial")
        if initial_shape not in ((dimension + 1,), (chains, dimension + 1)):
            raise ValueError("initial must have one full coordinate vector or one per chain")
        starts = _real(initial, "initial").copy()
        if starts.shape == (dimension + 1,):
            starts = np.broadcast_to(starts, (chains, dimension + 1)).copy()
        if starts.shape != (chains, dimension + 1):
            raise ValueError("initial must have one full coordinate vector or one per chain")
    if np.any(np.abs(starts[:, -1]) > 1.0):
        raise ValueError("initial association must lie in [-1,1]")
    if fixed_rho is not None and np.any(starts[:, -1] != fixed_rho):
        raise ValueError("initial association must equal fixed_association")
    for row in starts:
        if np.any(row[:-1][sd == 0] != mu[sd == 0]):
            raise ValueError("initial fixed Gaussian coordinates must equal prior_mean")

    parameters = np.empty((chains, draws, dimension + 1), dtype=float)
    joint = np.empty((chains, draws, *joint_shape), dtype=float)
    log_likelihood = np.empty((chains, draws), dtype=float)
    acceptance_count = np.zeros(chains, dtype=float)
    evaluations = 0
    work = 0

    def evaluate(state: FloatArray) -> tuple[float, FloatArray | None]:
        nonlocal evaluations, work
        if evaluations >= max_likelihood_evaluations:
            raise ArithmeticError("2010 GAO likelihood-evaluation budget exhausted")
        if work + joint_cells > max_work:
            raise ArithmeticError("2010 GAO likelihood-work budget exhausted")
        try:
            value, probabilities = _likelihood(
                d1, d2, n, nt, state, efficacy_levels, toxicity_levels
            )
        except _InvalidGAO2010Domain:
            evaluations += 1
            work += joint_cells
            return -np.inf, None
        evaluations += 1
        work += joint_cells
        return value, probabilities

    # Check every start and its observed-data likelihood before the first RNG
    # draw, so invalid caller state cannot partially advance the Generator.
    log_values = np.empty(chains, dtype=float)
    start_probabilities: list[FloatArray] = []
    for chain, row in enumerate(starts):
        value, probabilities = evaluate(row)
        if not np.isfinite(value) or probabilities is None:
            raise ValueError(f"initial state is invalid or has zero likelihood in chain {chain}")
        log_values[chain] = value
        start_probabilities.append(probabilities)

    for chain in range(chains):
        state = starts[chain].copy()
        log_value = float(log_values[chain])
        probabilities = start_probabilities[chain]
        for iteration in range(warmup + draws):
            for block in endpoint_blocks:
                free = np.flatnonzero(sd[block] > 0) + block.start
                if free.size == 0:
                    continue
                center = mu[free]
                centered = state[free] - center
                direction = rng.normal(size=free.size) * sd[free]
                height = log_value + np.log1p(-rng.random())
                angle = rng.uniform(0.0, 2.0 * np.pi)
                lower, upper = angle - 2.0 * np.pi, angle
                for _ in range(_MAX_SLICE_STEPS):
                    proposal = state.copy()
                    proposal[free] = center + centered * np.cos(angle) + direction * np.sin(angle)
                    if not np.all(np.isfinite(proposal)):
                        raise ArithmeticError("2010 GAO elliptical proposal is not finite")
                    trial_value, trial_probabilities = evaluate(proposal)
                    if trial_value >= height and trial_probabilities is not None:
                        state, log_value = proposal, trial_value
                        probabilities = trial_probabilities
                        break
                    if angle < 0.0:
                        lower = angle
                    else:
                        upper = angle
                    angle = rng.uniform(lower, upper)
                else:
                    raise ArithmeticError(
                        f"2010 GAO elliptical-slice update failed in chain {chain}"
                    )

            if fixed_rho is None:
                proposal = state.copy()
                proposal[-1] = rng.uniform(-1.0, 1.0)
                trial_value, trial_probabilities = evaluate(proposal)
                log_uniform = np.log1p(-rng.random())
                if log_uniform < trial_value - log_value and trial_probabilities is not None:
                    state, log_value = proposal, trial_value
                    probabilities = trial_probabilities
                    if iteration >= warmup:
                        acceptance_count[chain] += 1.0

            if iteration >= warmup:
                out = iteration - warmup
                parameters[chain, out] = state
                joint[chain, out] = probabilities
                log_likelihood[chain, out] = log_value

    frozen_parameters = _freeze(parameters)
    return U2OETGAO2010Fit(
        names=full_names,
        parameters=frozen_parameters,
        joint=_freeze(joint),
        log_likelihood=_freeze(log_likelihood),
        parameter_summary=summarize_chains(frozen_parameters),
        doses1=_freeze(d1),
        doses2=_freeze(d2),
        counts=_freeze(n),
        toxicity_only=_freeze(nt),
        prior_mean=_freeze(mu),
        prior_sd=_freeze(sd),
        association_acceptance=(
            None if fixed_rho is not None else _freeze(acceptance_count / draws)
        ),
        likelihood_evaluations=evaluations,
        likelihood_work_units=work,
        warmup=warmup,
        efficacy_levels=efficacy_levels,
        toxicity_levels=toxicity_levels,
        fixed_association=fixed_rho,
        prior_support=(
            "independent Gaussian coordinates jointly restricted to the supplied-grid-valid "
            "domain; association is fixed"
            if fixed_rho is not None
            else "independent Gaussian coordinates jointly restricted to the supplied-grid-valid "
            "domain; association has Uniform(-1,1) prior"
        ),
    )
