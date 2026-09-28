"""Unknown-shape Weibull posterior fitting and Johnson GOF diagnostics.

The model uses an explicit Gaussian prior on transformed Weibull parameters;
BCSTTE does not publish its executable prior/fitter details in the cached guide.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, finite, scalar
from .bayesian_chi_square import BayesianChiSquare, bayesian_chi_square_cdf
from .boin import _owned
from .hierarchical_binomial import ChainSummary, summarize_chains

_MAX_RETAINED_CELLS = 20_000_000
_MAX_LIKELIHOOD_EVALUATIONS = 1_000_000
_MAX_WORK_UNITS = 100_000_000
_MAX_SLICE_STEPS = 1000
_LOG_FLOAT_MIN = float(np.log(np.nextafter(0.0, 1.0)))
_LOG_FLOAT_MAX = float(np.log(np.finfo(float).max))


@dataclass(frozen=True)
class WeibullUnknownShapeGOF:
    """Joint posterior draws and Johnson diagnostic for complete Weibull data.

    ``parameters[..., 0]`` is log shape and ``parameters[..., 1]`` is log scale
    centered by ``log_scale_offset``. Add the offset to recover the absolute
    log-scale coordinate. Posterior samples keep the shape/scale pairing used
    for every observed-time CDF evaluation.
    """

    parameter_names: tuple[str, str]
    parameters: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    times: FloatArray
    log_scale_offset: float
    prior_mean: FloatArray
    prior_covariance: FloatArray
    likelihood_evaluations: int
    likelihood_work_units: int
    warmup: int
    diagnostic: BayesianChiSquare


def _input_shape(value: ArrayLike, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(size) for size in shape)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a rectangular array") from exc

    def nested_shape(item: object, remaining: int) -> tuple[int, ...]:
        if np.isscalar(item):
            return ()
        if remaining <= 0 or not isinstance(item, Sequence):
            raise ValueError(f"{name} must be a small rectangular array")
        if len(item) > 16:
            raise ValueError(f"{name} must be a small rectangular array")
        child_shapes = [nested_shape(child, remaining - 1) for child in item]
        if not child_shapes:
            return (0,)
        if any(child != child_shapes[0] for child in child_shapes[1:]):
            raise ValueError(f"{name} must be rectangular")
        return (len(item), *child_shapes[0])

    try:
        shape = nested_shape(value, 3)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a small rectangular array") from exc
    return shape


def _bounded_integer(value: ArrayLike, name: str, minimum: int, maximum: int) -> int:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = scalar(cast(float, value), name)
    if result != int(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(result)


def _relative_log_times(x: FloatArray) -> tuple[FloatArray, float]:
    reference = float(x[0])
    offset = float(np.log(reference))
    relative = np.empty_like(x)
    with np.errstate(over="ignore", under="ignore"):
        near = (x >= reference * 0.5) & (x <= reference * 1.5)
    relative[near] = np.log1p((x[near] - reference) / reference)
    relative[~near] = np.log(x[~near]) - offset
    return relative, offset


def _log_likelihood(coordinates: FloatArray, relative_log_times: FloatArray) -> float:
    log_shape, centered_log_scale = map(float, coordinates)
    if not np.isfinite(log_shape) or not np.isfinite(centered_log_scale):
        raise ArithmeticError("Weibull log-parameter proposal is nonfinite")
    if not _LOG_FLOAT_MIN <= log_shape <= _LOG_FLOAT_MAX:
        raise ArithmeticError("Weibull shape proposal is outside representable range")
    shape = float(np.exp(log_shape))
    log_ratio = relative_log_times - centered_log_scale
    if np.any(~np.isfinite(log_ratio)):
        return -np.inf
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_hazard = shape * log_ratio
        if np.any(np.isnan(log_hazard)):
            raise ArithmeticError("Weibull hazard evaluation became indeterminate")
        if np.any(log_hazard > _LOG_FLOAT_MAX):
            return -np.inf
        hazard = np.exp(log_hazard)
        # Parameter-dependent kernel; the omitted -sum(log(times)) is
        # constant in the parameters and restored for public likelihoods.
        result = float(
            relative_log_times.size * log_shape + shape * float(np.sum(log_ratio)) - np.sum(hazard)
        )
    if np.isnan(result) or result == np.inf:
        raise ArithmeticError("Weibull log likelihood is invalid")
    return result


def weibull_unknown_shape_bayesian_gof(
    times: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    bins: int | None = None,
    critical_probability: float = 0.95,
    rng: np.random.Generator,
    max_likelihood_evaluations: int = 1_000_000,
    max_work: int = 100_000_000,
) -> WeibullUnknownShapeGOF:
    """Fit an unknown-shape Weibull model using serial elliptical slice sampling.

    The complete-data density is the guide's Weibull density with shape ``beta``
    and scale ``eta``. The caller supplies a proper bivariate Normal prior for
    ``(log(beta), log(eta))``. Prior arrays use absolute coordinates; retained
    log-scale samples are centered for time-unit stability. No prior defaults,
    censoring, rounding transform, or BCSTTE executable parity are claimed.

    Overflowing-hazard states have zero representable likelihood and are
    rejected by the slice bracket. Nonrepresentable shape coordinates, invalid
    initial states, numerical failures and exhausted hard work budgets raise.
    The returned chain/draw arrays retain paired parameters and log likelihood;
    one paired posterior draw supplies all observed-time CDF values.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    time_shape = getattr(times, "shape", None)
    if time_shape is None:
        sequence = cast(Sequence[object], times)
        try:
            n = len(sequence)
        except TypeError as exc:
            raise ValueError("times must be a one-dimensional vector") from exc
        if not 2 <= n <= _MAX_RETAINED_CELLS:
            raise ValueError("require at least two complete positive event times")
        for item in sequence:
            if not np.isscalar(item):
                raise ValueError("times must be a one-dimensional vector")
            if np.iscomplexobj(item):
                raise ValueError("times must be real")
    else:
        if len(time_shape) != 1:
            raise ValueError("times must be a one-dimensional vector")
        n = int(time_shape[0])
        if not 2 <= n <= _MAX_RETAINED_CELLS:
            raise ValueError("require at least two complete positive event times")
        if np.iscomplexobj(times):
            raise ValueError("times must be real")

    if _input_shape(prior_mean, "prior_mean") != (2,):
        raise ValueError("prior_mean must contain log shape and log scale means")
    if _input_shape(prior_covariance, "prior_covariance") != (2, 2):
        raise ValueError("prior_covariance must be a 2x2 matrix")
    if np.iscomplexobj(prior_mean) or np.iscomplexobj(prior_covariance):
        raise ValueError("prior_mean and prior_covariance must be real")
    mean = finite(prior_mean, "prior_mean")
    covariance = finite(prior_covariance, "prior_covariance")
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("prior_mean and prior_covariance have invalid shapes")
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("prior_covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_covariance must be positive definite") from exc
    if not _LOG_FLOAT_MIN <= mean[0] <= _LOG_FLOAT_MAX:
        raise ValueError("prior mean log shape must represent a positive finite shape")

    size = _bounded_integer(draws, "draws", 8, 100_000)
    burn = _bounded_integer(warmup, "warmup", 0, 100_000)
    chain_count = _bounded_integer(chains, "chains", 2, 16)
    bin_count = (
        max(2, int(np.floor(n**0.4 + 0.5)))
        if bins is None
        else _bounded_integer(bins, "bins", 2, 1000)
    )
    if np.iscomplexobj(critical_probability):
        raise ValueError("critical_probability must be real")
    level = scalar(critical_probability, "critical_probability")
    if not 0 < level < 1:
        raise ValueError("bins must be 2..1000 and critical_probability must lie in (0,1)")
    evaluation_limit = _bounded_integer(
        max_likelihood_evaluations,
        "max_likelihood_evaluations",
        1,
        _MAX_LIKELIHOOD_EVALUATIONS,
    )
    work_limit = _bounded_integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    if not 1 <= evaluation_limit <= _MAX_LIKELIHOOD_EVALUATIONS:
        raise ValueError("max_likelihood_evaluations exceeds the supported limit")
    total_draws = chain_count * size
    cdf_cells = total_draws * n
    summary_cells = total_draws * bin_count
    chunk_cells = min(total_draws, 256) * n
    retained_cells = cdf_cells + 3 * summary_cells + 30 * total_draws + 8 * n + 4 * chunk_cells
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("Weibull posterior and diagnostic arrays exceed 20 million cells")
    minimum_evaluations = chain_count * (1 + burn + size)
    if minimum_evaluations > evaluation_limit:
        raise ValueError("minimum chain evaluations exceed max_likelihood_evaluations")
    if minimum_evaluations * n > work_limit:
        raise ValueError("minimum likelihood work exceeds max_work")

    if initial is not None:
        initial_shape = _input_shape(initial, "initial")
        if initial_shape not in ((2,), (chain_count, 2)):
            raise ValueError("initial must be one log-parameter vector or one per chain")
        if np.iscomplexobj(initial):
            raise ValueError("initial must be real")
        start_abs = finite(initial, "initial")
        if start_abs.shape == (2,):
            starts = np.broadcast_to(start_abs, (chain_count, 2)).copy()
        else:
            starts = start_abs.copy()
        if starts.shape != (chain_count, 2):
            raise ValueError("initial must be one log-parameter vector or one per chain")
    else:
        starts = np.broadcast_to(mean, (chain_count, 2)).copy()

    x = finite(times, "times")
    if x.ndim != 1 or np.any(x <= 0):
        raise ValueError("require at least two complete positive event times")
    relative, offset = _relative_log_times(x)
    log_likelihood_constant = -(n * offset + float(np.sum(relative)))
    centered_mean = mean.copy()
    centered_mean[1] -= offset
    centered_starts = starts.copy()
    centered_starts[:, 1] -= offset
    if np.any(~np.isfinite(centered_mean)) or np.any(~np.isfinite(centered_starts)):
        raise ArithmeticError("centered Weibull log-scale coordinates are not representable")

    evaluations, work = 0, 0

    def evaluate(state: FloatArray) -> float:
        nonlocal evaluations, work
        if evaluations >= evaluation_limit:
            raise ArithmeticError("Weibull likelihood-evaluation budget exhausted")
        if work + n > work_limit:
            raise ArithmeticError("Weibull likelihood-work budget exhausted")
        result = _log_likelihood(state, relative)
        evaluations += 1
        work += n
        return result

    parameters = np.empty((chain_count, size, 2), dtype=float)
    log_likelihood = np.empty((chain_count, size), dtype=float)
    for chain in range(chain_count):
        state = centered_starts[chain].copy()
        ll = evaluate(state)
        if not np.isfinite(ll):
            raise ValueError(f"initial Weibull state has zero likelihood in chain {chain}")
        for iteration in range(burn + size):
            direction = cholesky @ rng.standard_normal(2)
            height = ll + np.log1p(-rng.random())
            angle = float(rng.uniform(0, 2 * np.pi))
            lower, upper = angle - 2 * np.pi, angle
            for _ in range(_MAX_SLICE_STEPS):
                proposal = (
                    centered_mean
                    + (state - centered_mean) * np.cos(angle)
                    + direction * np.sin(angle)
                )
                trial_ll = evaluate(proposal)
                if trial_ll >= height:
                    state, ll = proposal, trial_ll
                    break
                if angle < 0:
                    lower = angle
                else:
                    upper = angle
                angle = float(rng.uniform(lower, upper))
            else:
                raise ArithmeticError(f"Weibull elliptical-slice update failed in chain {chain}")
            if iteration >= burn:
                draw = iteration - burn
                parameters[chain, draw] = state
                log_likelihood[chain, draw] = ll + log_likelihood_constant

    all_cdf = np.empty((total_draws, n), dtype=float)
    flat = parameters.reshape((total_draws, 2))
    for start in range(0, total_draws, 256):
        stop = min(start + 256, total_draws)
        shape_logs = flat[start:stop, 0, None]
        centered_log_scale = flat[start:stop, 1, None]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            shape = np.exp(shape_logs)
            log_hazard = shape * (relative[None, :] - centered_log_scale)
            hazard = np.exp(log_hazard)
            all_cdf[start:stop] = -np.expm1(-hazard)
        if np.any(np.isnan(all_cdf[start:stop])):
            raise ArithmeticError("Weibull posterior CDF became indeterminate")
    diagnostic = bayesian_chi_square_cdf(all_cdf, bins=bin_count, critical_probability=level)
    frozen_parameters = _owned(parameters)
    return WeibullUnknownShapeGOF(
        ("log_shape", "centered_log_scale"),
        frozen_parameters,
        _owned(log_likelihood),
        summarize_chains(frozen_parameters),
        _owned(x),
        offset,
        _owned(mean),
        _owned(covariance),
        evaluations,
        work,
        burn,
        diagnostic,
    )
