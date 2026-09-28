"""Coefficient-domain wavelet functional mixed-model likelihood and sampler.

The inputs are already in an orthogonal coefficient basis. This module does
not implement a transform or empirical-Bayes initialization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit, log_ndtr
from scipy.stats import truncnorm

from ._validation import FloatArray, finite
from .hierarchical_binomial import ChainSummary, summarize_chains

_MAX_DESIGN_CELLS = 2_000_000
_MAX_RETAINED_CELLS = 2_000_000
_MAX_WORK = 250_000_000
_MAX_ROWS = 500
_MAX_FIXED_EFFECTS = 100
_MAX_RANDOM_EFFECTS = 500
_MAX_COEFFICIENTS = 512


def _owned(value: ArrayLike) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


def _owned_bool(value: ArrayLike) -> NDArray[np.bool_]:
    array = np.array(value, dtype=np.bool_, copy=True)
    array.flags.writeable = False
    return array


def _owned_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.array(value, dtype=np.int64, copy=True)
    array.flags.writeable = False
    return array


def _finite_real(value: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    return finite(raw, name)


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return result


def _positive_array(value: ArrayLike | None, name: str) -> FloatArray | None:
    if value is None:
        return None
    array = _finite_real(value, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return _owned(array)


@dataclass(frozen=True)
class WFMMPrior:
    """Explicit spike-and-slab and inverse-gamma prior parameters.

    ``inclusion_probability`` and ``slab_variance`` are scalar or
    coefficient-specific arrays. A zero slab variance is permitted only for
    a zero inclusion probability, representing the canonical all-spike limit.
    Variance priors use inverse-gamma
    shape/scale density proportional to ``v**(-a-1)*exp(-b/v)``. Random
    variance prior arrays may be omitted only when the model has no random
    effects; variance priors are required only when estimating those
    variances.
    """

    inclusion_probability: ArrayLike
    slab_variance: ArrayLike
    random_shape: ArrayLike | None = None
    random_scale: ArrayLike | None = None
    residual_shape: ArrayLike | None = None
    residual_scale: ArrayLike | None = None

    def __post_init__(self) -> None:
        raw_pi = np.asarray(self.inclusion_probability)
        raw_tau = np.asarray(self.slab_variance)
        max_prior_cells = _MAX_FIXED_EFFECTS * _MAX_COEFFICIENTS
        if (
            raw_pi.ndim > 2
            or raw_tau.ndim > 2
            or raw_pi.size > max_prior_cells
            or raw_tau.size > max_prior_cells
        ):
            raise ValueError("coefficient prior arrays exceed supported shape bounds")
        if raw_pi.ndim > 0 and raw_tau.ndim > 0 and raw_pi.shape != raw_tau.shape:
            raise ValueError("inclusion_probability and slab_variance array shapes must match")
        pi = _finite_real(self.inclusion_probability, "inclusion_probability")
        tau = _finite_real(self.slab_variance, "slab_variance")
        if np.any((pi < 0.0) | (pi > 1.0)):
            raise ValueError("inclusion_probability must lie in [0,1]")
        if np.any(tau < 0.0) or np.any((tau == 0.0) & (pi != 0.0)):
            raise ValueError(
                "zero slab_variance is allowed only when inclusion_probability is zero"
            )
        object.__setattr__(self, "inclusion_probability", _owned(pi))
        object.__setattr__(self, "slab_variance", _owned(tau))
        for name in ("random_shape", "random_scale", "residual_shape", "residual_scale"):
            object.__setattr__(self, name, _positive_array(getattr(self, name), name))


def _component_array(
    value: ArrayLike | None,
    shape: tuple[int, int],
    name: str,
    *,
    positive: bool = False,
    default_empty: bool = False,
) -> FloatArray:
    if value is None:
        if default_empty and shape[0] == 0:
            return np.empty(shape, dtype=np.float64)
        raise ValueError(f"{name} must be supplied explicitly")
    array = _finite_real(value, name)
    if array.ndim == 0:
        result = np.full(shape, float(array), dtype=np.float64)
    elif array.shape == shape:
        result = np.array(array, copy=True)
    elif shape[0] == 1 and array.shape == (shape[1],):
        result = np.array(array[None, :], copy=True)
    else:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} is not representable")
    return result


def _coefficient_prior(
    value: ArrayLike,
    shape: tuple[int, int],
    name: str,
    *,
    probability: bool = False,
    allow_zero: bool = False,
) -> FloatArray:
    array = _finite_real(value, name)
    if array.ndim == 0:
        result = np.full(shape, float(array), dtype=np.float64)
    elif array.shape == shape:
        result = np.array(array, copy=True)
    else:
        raise ValueError(f"{name} must be scalar or have shape {shape}")
    if probability:
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError(f"{name} must lie in [0,1]")
    elif np.any(result < 0.0) or (not allow_zero and np.any(result == 0.0)):
        qualifier = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _strata(value: ArrayLike | None, size: int, name: str) -> NDArray[np.int64]:
    if value is None:
        return np.zeros(size, dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (size,):
        raise ValueError(f"{name} must have length {size}")
    numeric = _finite_real(raw, name)
    if (
        np.any(numeric < 0.0)
        or np.any(numeric > max(0, size - 1))
        or np.any(numeric != np.floor(numeric))
    ):
        raise ValueError(f"{name} must contain nonnegative integer group labels")
    labels = numeric.astype(np.int64)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError(f"{name} group labels must be contiguous from zero")
    return labels


def _logdet_and_solve(
    covariance: FloatArray, response: FloatArray
) -> tuple[FloatArray, FloatArray, float]:
    try:
        factor = cho_factor(covariance, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError("marginal coefficient covariance is not positive definite") from exc
    solved_response = cho_solve(factor, response, check_finite=False)
    logdet = 2.0 * float(np.log(np.diag(factor[0])).sum())
    if not np.all(np.isfinite(solved_response)) or not np.isfinite(logdet):
        raise ArithmeticError("marginal Gaussian solve is nonfinite")
    return factor, solved_response, logdet


def _marginal_covariance(
    random_design: FloatArray,
    random_strata: NDArray[np.int64],
    random_variances: FloatArray,
    residual_strata: NDArray[np.int64],
    residual_variances: FloatArray,
    coefficient: int,
) -> FloatArray:
    rows = residual_strata.size
    covariance = np.zeros((rows, rows), dtype=np.float64)
    for group in range(random_variances.shape[0]):
        columns = random_strata == group
        block = random_design[:, columns]
        covariance += random_variances[group, coefficient] * (block @ block.T)
    diagonal = np.arange(rows)
    covariance[diagonal, diagonal] += residual_variances[residual_strata, coefficient]
    if not np.all(np.isfinite(covariance)):
        raise ArithmeticError("marginal covariance is outside floating-point range")
    return covariance


def _marginal_log_likelihood(
    residual: FloatArray,
    random_design: FloatArray,
    random_strata: NDArray[np.int64],
    random_variances: FloatArray,
    residual_strata: NDArray[np.int64],
    residual_variances: FloatArray,
    coefficient: int,
) -> float:
    covariance = _marginal_covariance(
        random_design,
        random_strata,
        random_variances,
        residual_strata,
        residual_variances,
        coefficient,
    )
    _, solved, logdet = _logdet_and_solve(covariance, residual)
    quadratic = float(residual @ solved)
    value = -0.5 * (residual.size * np.log(2.0 * np.pi) + logdet + quadratic)
    if not np.isfinite(value):
        raise ArithmeticError("marginal Gaussian log likelihood is nonfinite")
    return value


def _log_inverse_gamma_ratio(new: float, old: float, shape: float, scale: float) -> float:
    """Stable log IG(new)/IG(old); normalizing constants cancel."""
    extended = np.longdouble
    log_ratio = np.log(extended(new)) - np.log(extended(old))
    reciprocal_difference = 1 / extended(new) - 1 / extended(old)
    result = -(extended(shape) + 1) * log_ratio - extended(scale) * reciprocal_difference
    if np.isnan(result):
        raise ArithmeticError("inverse-gamma log density ratio is indeterminate")
    if result > np.finfo(np.float64).max:
        return np.inf
    if result < -np.finfo(np.float64).max:
        return -np.inf
    return float(result)


def _propose_positive(current: float, sd: float, rng: np.random.Generator) -> float:
    lower = -current / sd
    proposal = float(truncnorm.rvs(lower, np.inf, loc=current, scale=sd, random_state=rng))
    if not np.isfinite(proposal) or proposal <= 0.0:
        raise ArithmeticError("positive variance proposal is not representable")
    return proposal


def _draw_fixed_coefficient(
    bhat: float,
    variance: float,
    inclusion_probability: float,
    slab_variance: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    if inclusion_probability == 0.0:
        return 0.0, False
    if np.isinf(variance):
        included = inclusion_probability == 1.0 or rng.random() < inclusion_probability
        if not included:
            return 0.0, False
        draw = np.sqrt(slab_variance) * rng.normal()
        if not np.isfinite(draw):
            raise ArithmeticError("fixed-effect prior draw is nonfinite")
        return float(draw), True
    if inclusion_probability == 1.0:
        included = True
    else:
        log_tau_over_v = np.log(slab_variance) - np.log(variance)
        slab_fraction = float(expit(log_tau_over_v))
        z = bhat / np.sqrt(variance)
        log_odds = (
            np.log(inclusion_probability)
            - np.log1p(-inclusion_probability)
            - 0.5 * np.logaddexp(0.0, log_tau_over_v)
            + 0.5 * z * z * slab_fraction
        )
        included = bool(rng.random() < expit(log_odds))
    if not included:
        return 0.0, False
    fraction = float(expit(np.log(slab_variance) - np.log(variance)))
    posterior_mean = bhat * fraction
    posterior_variance = variance * fraction
    draw = posterior_mean + np.sqrt(posterior_variance) * rng.normal()
    if not np.isfinite(draw):
        raise ArithmeticError("fixed-effect posterior draw is nonfinite")
    return float(draw), True


def _sample_random_effects(
    residual: FloatArray,
    random_design: FloatArray,
    random_strata: NDArray[np.int64],
    random_variances: FloatArray,
    residual_strata: NDArray[np.int64],
    residual_variances: FloatArray,
    coefficient: int,
    rng: np.random.Generator,
) -> FloatArray:
    levels = random_design.shape[1]
    if levels == 0:
        return np.empty(0, dtype=np.float64)
    q = random_variances[random_strata, coefficient]
    root_q = np.sqrt(q)
    scaled_design = random_design * root_q
    precision = (
        np.eye(levels)
        + (scaled_design.T / residual_variances[residual_strata, coefficient]) @ scaled_design
    )
    rhs = scaled_design.T @ (residual / residual_variances[residual_strata, coefficient])
    try:
        factor = np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as exc:
        raise ArithmeticError(
            "conditional random-effect precision is not positive definite"
        ) from exc
    mean_standardized = solve_triangular(
        factor.T,
        solve_triangular(factor, rhs, lower=True, check_finite=False),
        lower=False,
        check_finite=False,
    )
    noise = solve_triangular(factor.T, rng.normal(size=levels), lower=False, check_finite=False)
    draw = root_q * (mean_standardized + noise)
    if not np.all(np.isfinite(draw)):
        raise ArithmeticError("random-effect posterior draw is nonfinite")
    return draw


@dataclass(frozen=True)
class WFMMCoefficientFit:
    """Posterior draws for coefficient-domain fixed/random effects and variances."""

    coefficients: FloatArray
    inclusion_indicators: NDArray[np.bool_]
    random_variances: FloatArray
    residual_variances: FloatArray
    random_effects: FloatArray | None
    log_likelihood: FloatArray
    summary: ChainSummary
    random_strata: NDArray[np.int64]
    residual_strata: NDArray[np.int64]
    coefficient_partition: NDArray[np.int64]
    coefficient_scale: NDArray[np.int64]
    warmup: int
    estimate_variances: bool
    variance_acceptance: FloatArray
    likelihood_evaluations: int


def fit_wfmm_coefficients(
    coefficients: ArrayLike,
    fixed_design: ArrayLike,
    random_design: ArrayLike | None = None,
    *,
    prior: WFMMPrior,
    random_variance: ArrayLike | None = None,
    residual_variance: ArrayLike,
    random_strata: ArrayLike | None = None,
    residual_strata: ArrayLike | None = None,
    coefficient_partition: ArrayLike | None = None,
    coefficient_scale: ArrayLike | None = None,
    estimate_variances: bool = True,
    proposal_sd: tuple[ArrayLike | None, ArrayLike] | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 2,
    sample_random_effects: bool = False,
    rng: np.random.Generator,
) -> WFMMCoefficientFit:
    """Fit an orthogonal-coefficient WFMM by marginal Bayesian MCMC.

    ``coefficients`` is N×K, ``fixed_design`` is N×P and ``random_design`` is
    N×M (or omitted). Random-effect columns are grouped by ``random_strata``;
    rows share residual variance components according to ``residual_strata``.
    Random and residual variances are supplied as starting values. They are
    sampled only with ``estimate_variances=True``; then positive-truncated
    normal proposals and inverse-gamma priors are used. ``proposal_sd`` is a
    pair ``(random, residual)`` in that order. The model uses independent
    random-effect levels and independent residual rows, with coefficient-
    specific variance components. This fits the wavelet-domain model, not a
    generic mixed model; input transforms and empirical-Bayes defaults are
    intentionally outside this API.
    """
    if not isinstance(prior, WFMMPrior):
        raise ValueError("prior must be a WFMMPrior")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    if not isinstance(estimate_variances, (bool, np.bool_)):
        raise ValueError("estimate_variances must be boolean")
    if not isinstance(sample_random_effects, (bool, np.bool_)):
        raise ValueError("sample_random_effects must be boolean")

    raw_d, raw_x = np.asarray(coefficients), np.asarray(fixed_design)
    if (
        raw_d.ndim != 2
        or not 1 <= raw_d.shape[0] <= _MAX_ROWS
        or not 1 <= raw_d.shape[1] <= _MAX_COEFFICIENTS
    ):
        raise ValueError("coefficients must be an N-by-K matrix within supported dimensions")
    rows, coefficient_count = raw_d.shape
    if raw_x.ndim != 2 or raw_x.shape[0] != rows or not 1 <= raw_x.shape[1] <= _MAX_FIXED_EFFECTS:
        raise ValueError("fixed_design must be an N-by-P matrix with 1..100 columns")
    fixed_count = raw_x.shape[1]
    if random_design is None:
        z = np.empty((rows, 0), dtype=np.float64)
    else:
        raw_z = np.asarray(random_design)
        if raw_z.ndim != 2 or raw_z.shape[0] != rows or raw_z.shape[1] > _MAX_RANDOM_EFFECTS:
            raise ValueError("random_design must be N-by-M with at most 500 columns")
        z = _finite_real(raw_z, "random_design")
    if rows * (fixed_count + z.shape[1] + coefficient_count) > _MAX_DESIGN_CELLS:
        raise ValueError("WFMM design and coefficient data exceed 2000000 cells")
    d, x = _finite_real(raw_d, "coefficients"), _finite_real(raw_x, "fixed_design")
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(d)) and np.all(np.isfinite(x))):
        raise ValueError("WFMM arrays must be finite")

    random_groups = (
        _strata(random_strata, z.shape[1], "random_strata")
        if z.shape[1]
        else np.empty(0, dtype=np.int64)
    )
    if z.shape[1] == 0 and random_strata is not None:
        raise ValueError("random_strata requires a nonempty random_design")
    residual_groups = _strata(residual_strata, rows, "residual_strata")
    random_group_count = int(random_groups.max()) + 1 if random_groups.size else 0
    residual_group_count = int(residual_groups.max()) + 1
    pi = _coefficient_prior(
        prior.inclusion_probability,
        (fixed_count, coefficient_count),
        "inclusion_probability",
        probability=True,
    )
    tau = _coefficient_prior(
        prior.slab_variance,
        (fixed_count, coefficient_count),
        "slab_variance",
        allow_zero=True,
    )
    if np.any((tau == 0.0) & (pi != 0.0)):
        raise ValueError("zero slab_variance is allowed only when inclusion_probability is zero")
    q = _component_array(
        random_variance,
        (random_group_count, coefficient_count),
        "random_variance",
        positive=True,
        default_empty=random_group_count == 0,
    )
    s = _component_array(
        residual_variance,
        (residual_group_count, coefficient_count),
        "residual_variance",
        positive=True,
    )
    random_shape_input = prior.random_shape
    random_scale_input = prior.random_scale
    residual_shape_input = prior.residual_shape
    residual_scale_input = prior.residual_scale
    if not estimate_variances:
        if random_group_count and random_shape_input is None:
            random_shape_input = np.ones((random_group_count, coefficient_count))
        if random_group_count and random_scale_input is None:
            random_scale_input = np.ones((random_group_count, coefficient_count))
        if residual_shape_input is None:
            residual_shape_input = np.ones((residual_group_count, coefficient_count))
        if residual_scale_input is None:
            residual_scale_input = np.ones((residual_group_count, coefficient_count))
    random_shape = _component_array(
        random_shape_input,
        (random_group_count, coefficient_count),
        "prior.random_shape",
        positive=True,
        default_empty=random_group_count == 0,
    )
    random_scale = _component_array(
        random_scale_input,
        (random_group_count, coefficient_count),
        "prior.random_scale",
        positive=True,
        default_empty=random_group_count == 0,
    )
    residual_shape = _component_array(
        residual_shape_input,
        (residual_group_count, coefficient_count),
        "prior.residual_shape",
        positive=True,
    )
    residual_scale = _component_array(
        residual_scale_input,
        (residual_group_count, coefficient_count),
        "prior.residual_scale",
        positive=True,
    )
    if estimate_variances:
        if proposal_sd is None or len(proposal_sd) != 2:
            raise ValueError("proposal_sd must be (random, residual) when estimating variances")
        q_proposal = _component_array(
            proposal_sd[0],
            (random_group_count, coefficient_count),
            "random proposal_sd",
            positive=True,
            default_empty=random_group_count == 0,
        )
        s_proposal = _component_array(
            proposal_sd[1],
            (residual_group_count, coefficient_count),
            "residual proposal_sd",
            positive=True,
        )
    else:
        if proposal_sd is not None:
            raise ValueError("proposal_sd is only used when estimate_variances=True")
        q_proposal = np.empty_like(q)
        s_proposal = np.empty_like(s)

    def coefficient_labels(value: ArrayLike | None, name: str) -> NDArray[np.int64]:
        if value is None:
            return np.zeros(coefficient_count, dtype=np.int64)
        raw = np.asarray(value)
        if raw.shape != (coefficient_count,):
            raise ValueError(f"{name} must have one nonnegative integer per coefficient")
        array = _finite_real(raw, name)
        if (
            np.any(array < 0)
            or np.any(array > np.iinfo(np.int64).max)
            or np.any(array != np.floor(array))
        ):
            raise ValueError(f"{name} must contain nonnegative integer labels")
        return array.astype(np.int64)

    partitions = coefficient_labels(coefficient_partition, "coefficient_partition")
    scales = coefficient_labels(coefficient_scale, "coefficient_scale")
    draw_count = _integer(draws, "draws", 8, 5000)
    warm = _integer(warmup, "warmup", 0, 5000)
    chain_count = _integer(chains, "chains", 2, 4)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(s)):
        raise ValueError("starting variances must be positive and finite")
    if (q.size and np.any(q <= 0.0)) or np.any(s <= 0.0):
        raise ValueError("starting variances must be positive and finite")
    kept_fixed = chain_count * draw_count * fixed_count * coefficient_count
    kept_variance = chain_count * draw_count * (q.size + s.size)
    kept_random_effects = (
        chain_count * draw_count * z.shape[1] * coefficient_count if sample_random_effects else 0
    )
    if kept_fixed * 2 + kept_variance + kept_random_effects > _MAX_RETAINED_CELLS:
        raise ValueError("retained WFMM posterior arrays exceed 2000000 cells")
    iterations = warm + draw_count
    likelihood_factorizations = (
        random_group_count + residual_group_count + 3 if estimate_variances else 2
    )
    per_coefficient_work = (
        likelihood_factorizations * rows**3
        + (z.shape[1] + random_group_count + residual_group_count + 2) * rows**2
        + rows * fixed_count
    )
    work = chain_count * iterations * coefficient_count * per_coefficient_work
    if work > _MAX_WORK:
        raise ValueError("WFMM sampling work exceeds the supported budget")
    random_sampling_work = (
        chain_count * draw_count * coefficient_count * (z.shape[1] ** 3 + rows * z.shape[1] ** 2)
    )
    if sample_random_effects and random_sampling_work > _MAX_WORK:
        raise ValueError("random-effect posterior sampling exceeds the supported budget")

    if sample_random_effects and z.shape[1] == 0:
        raise ValueError("sample_random_effects requires a nonempty random_design")
    coefficients_draws = np.empty(
        (chain_count, draw_count, fixed_count, coefficient_count), dtype=np.float64
    )
    inclusion_draws = np.empty(
        (chain_count, draw_count, fixed_count, coefficient_count), dtype=np.bool_
    )
    random_draws = np.empty(
        (chain_count, draw_count, random_group_count, coefficient_count), dtype=np.float64
    )
    residual_draws = np.empty(
        (chain_count, draw_count, residual_group_count, coefficient_count), dtype=np.float64
    )
    likelihood_draws = np.empty((chain_count, draw_count), dtype=np.float64)
    effect_draws = (
        np.empty((chain_count, draw_count, z.shape[1], coefficient_count), dtype=np.float64)
        if sample_random_effects
        else None
    )
    acceptance = np.zeros((random_group_count + residual_group_count, coefficient_count))
    proposals = np.zeros_like(acceptance)
    likelihood_evaluations = 0

    for chain in range(chain_count):
        gamma = np.zeros((fixed_count, coefficient_count), dtype=np.bool_)
        beta = np.zeros((fixed_count, coefficient_count), dtype=np.float64)
        for i in range(fixed_count):
            for k in range(coefficient_count):
                probability = pi[i, k]
                if probability == 1.0:
                    gamma[i, k] = True
                elif probability > 0.0:
                    gamma[i, k] = rng.random() < probability
                if gamma[i, k]:
                    beta[i, k] = np.sqrt(tau[i, k]) * rng.normal()

        q_chain, s_chain = q.copy(), s.copy()
        for iteration in range(iterations):
            log_likelihood = 0.0
            for k in range(coefficient_count):
                sigma = _marginal_covariance(z, random_groups, q_chain, residual_groups, s_chain, k)
                try:
                    factor = np.linalg.cholesky(sigma)
                except np.linalg.LinAlgError as exc:
                    raise ArithmeticError(
                        "marginal coefficient covariance is not positive definite"
                    ) from exc
                whitened_x = solve_triangular(factor, x, lower=True, check_finite=False)
                residual = d[:, k] - x @ beta[:, k]
                whitened_residual = solve_triangular(
                    factor, residual, lower=True, check_finite=False
                )
                for i in range(fixed_count):
                    xi = x[:, i]
                    wi = whitened_x[:, i]
                    information = float(wi @ wi)
                    old = beta[i, k]
                    without = residual + xi * old
                    whitened_without = whitened_residual + wi * old
                    if information <= 0.0 or not np.isfinite(information):
                        if information == 0.0:
                            bhat, conditional_variance = 0.0, np.inf
                        else:
                            raise ArithmeticError("fixed-effect conditional information is invalid")
                    else:
                        conditional_variance = 1.0 / information
                        bhat = conditional_variance * float(wi @ whitened_without)
                    if information > 0.0 and not np.isfinite(conditional_variance):
                        raise ArithmeticError(
                            "fixed-effect conditional variance is not representable"
                        )
                    new, included = _draw_fixed_coefficient(
                        bhat,
                        conditional_variance,
                        float(pi[i, k]),
                        float(tau[i, k]),
                        rng,
                    )
                    gamma[i, k] = included
                    beta[i, k] = new
                    residual = without - xi * new
                    whitened_residual = whitened_without - wi * new

                if estimate_variances:
                    residual = d[:, k] - x @ beta[:, k]
                    current_ll = _marginal_log_likelihood(
                        residual, z, random_groups, q_chain, residual_groups, s_chain, k
                    )
                    likelihood_evaluations += 1
                    for group in range(random_group_count):
                        proposals[group, k] += 1
                        old = float(q_chain[group, k])
                        new = _propose_positive(old, float(q_proposal[group, k]), rng)
                        candidate_q = q_chain.copy()
                        candidate_q[group, k] = new
                        candidate_ll = _marginal_log_likelihood(
                            residual,
                            z,
                            random_groups,
                            candidate_q,
                            residual_groups,
                            s_chain,
                            k,
                        )
                        likelihood_evaluations += 1
                        log_alpha = (
                            candidate_ll
                            - current_ll
                            + _log_inverse_gamma_ratio(
                                new, old, random_shape[group, k], random_scale[group, k]
                            )
                            + log_ndtr(old / q_proposal[group, k])
                            - log_ndtr(new / q_proposal[group, k])
                        )
                        if np.isnan(log_alpha):
                            raise ArithmeticError("random variance MH acceptance ratio is NaN")
                        if np.log(rng.random()) < min(0.0, float(log_alpha)):
                            q_chain[group, k] = new
                            current_ll = candidate_ll
                            acceptance[group, k] += 1
                    for group in range(residual_group_count):
                        proposals[random_group_count + group, k] += 1
                        old = float(s_chain[group, k])
                        new = _propose_positive(old, float(s_proposal[group, k]), rng)
                        candidate_s = s_chain.copy()
                        candidate_s[group, k] = new
                        candidate_ll = _marginal_log_likelihood(
                            residual,
                            z,
                            random_groups,
                            q_chain,
                            residual_groups,
                            candidate_s,
                            k,
                        )
                        likelihood_evaluations += 1
                        log_alpha = (
                            candidate_ll
                            - current_ll
                            + _log_inverse_gamma_ratio(
                                new, old, residual_shape[group, k], residual_scale[group, k]
                            )
                            + log_ndtr(old / s_proposal[group, k])
                            - log_ndtr(new / s_proposal[group, k])
                        )
                        if np.isnan(log_alpha):
                            raise ArithmeticError("residual variance MH acceptance ratio is NaN")
                        if np.log(rng.random()) < min(0.0, float(log_alpha)):
                            s_chain[group, k] = new
                            current_ll = candidate_ll
                            acceptance[random_group_count + group, k] += 1

                residual = d[:, k] - x @ beta[:, k]
                log_likelihood += _marginal_log_likelihood(
                    residual, z, random_groups, q_chain, residual_groups, s_chain, k
                )
                likelihood_evaluations += 1
                if effect_draws is not None and iteration >= warm:
                    effect_draws[chain, iteration - warm, :, k] = _sample_random_effects(
                        residual,
                        z,
                        random_groups,
                        q_chain,
                        residual_groups,
                        s_chain,
                        k,
                        rng,
                    )

            if iteration >= warm:
                index = iteration - warm
                coefficients_draws[chain, index] = beta
                inclusion_draws[chain, index] = gamma
                random_draws[chain, index] = q_chain
                residual_draws[chain, index] = s_chain
                likelihood_draws[chain, index] = log_likelihood

    acceptance_rate = np.divide(
        acceptance, proposals, out=np.zeros_like(acceptance), where=proposals > 0
    )
    return WFMMCoefficientFit(
        _owned(coefficients_draws),
        _owned_bool(inclusion_draws),
        _owned(random_draws),
        _owned(residual_draws),
        None if effect_draws is None else _owned(effect_draws),
        _owned(likelihood_draws),
        summarize_chains(coefficients_draws),
        _owned_int(random_groups),
        _owned_int(residual_groups),
        _owned_int(partitions),
        _owned_int(scales),
        warm,
        bool(estimate_variances),
        _owned(acceptance_rate),
        likelihood_evaluations,
    )
