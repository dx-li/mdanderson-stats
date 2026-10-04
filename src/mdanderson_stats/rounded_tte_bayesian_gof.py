"""Bayesian fitting for rounded TTE observations under explicit transformed priors."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ._rounded_tte_likelihood import rounded_tte_log_probabilities
from ._validation import FloatArray, finite, scalar
from .bayesian_chi_square import BayesianChiSquare, bayesian_chi_square_discrete_cdf
from .boin import _owned
from .hierarchical_binomial import ChainSummary, summarize_chains
from .tte_family_bayesian_gof import _sample_elliptical_slice_chains
from .weibull_unknown_shape_gof import (
    _LOG_FLOAT_MAX,
    _LOG_FLOAT_MIN,
    _MAX_LIKELIHOOD_EVALUATIONS,
    _MAX_WORK_UNITS,
    _bounded_integer,
    _input_shape,
    _relative_log_times,
)

_FAMILY_DIMENSIONS = {
    "exponential": 1,
    "weibull": 2,
    "gamma": 2,
    "inverse_gamma": 2,
    "log_logistic": 2,
    "lognormal": 2,
    "log_odds_rate": 3,
}
_PARAMETER_NAMES = {
    "exponential": ("centered_log_scale",),
    "weibull": ("log_shape", "centered_log_scale"),
    "gamma": ("log_shape", "centered_log_scale"),
    "inverse_gamma": ("log_shape", "centered_log_scale"),
    "log_logistic": ("log_shape", "centered_log_scale"),
    "lognormal": ("centered_log_location", "log_sigma"),
    "log_odds_rate": ("log_shape", "centered_log_scale", "log_c"),
}
_MAX_OBSERVATIONS = 1_000_000
_MAX_RETAINED_PARAMETER_CELLS = 2_000_000
_MAX_DRAW_OBSERVATION_CELLS = 1_000_000


@dataclass(frozen=True)
class RoundedTTEBayesianGOF:
    """Paired posterior draws and optional randomized rounded-data diagnostic.

    Scale/location coordinates are centered for time-unit stability. Add
    ``time_offset`` to a centered log scale or log-location to recover its
    absolute coordinate; exponential uses the reciprocal-rate parameterization
    and follows the same scale convention.
    """

    family: str
    parameter_names: tuple[str, ...]
    parameters: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    interval_lower: FloatArray
    interval_upper: FloatArray
    time_offset: float
    prior_mean: FloatArray
    prior_covariance: FloatArray
    initial_parameters: FloatArray
    draws: int
    chains: int
    likelihood_evaluations: int
    likelihood_work_units: int
    warmup: int
    bins: int
    critical_probability: float
    compute_diagnostic: bool
    diagnostic: BayesianChiSquare | None


def _bounded_vector_shape(value: object, name: str) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a one-dimensional real vector")
        length = value.size
    elif isinstance(value, (list, tuple)):
        length = len(value)
        if not 1 <= length <= _MAX_OBSERVATIONS:
            raise ValueError(f"{name} must contain 1..{_MAX_OBSERVATIONS} values")
        for item in value:
            if isinstance(item, np.ndarray):
                if item.ndim != 0 or item.dtype.kind not in "fiu":
                    raise ValueError(f"{name} must contain real scalar values")
                item = item.item()
            if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"{name} must contain real scalar values")
    else:
        raise TypeError(f"{name} must be a NumPy array or a bounded list/tuple")
    if not 1 <= length <= _MAX_OBSERVATIONS:
        raise ValueError(f"{name} must contain 1..{_MAX_OBSERVATIONS} values")
    return length


def _freeze_vector(value: object, name: str, expected: int) -> FloatArray:
    try:
        result = finite(cast(ArrayLike, value), name)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if result.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},)")
    return result.copy()


def _absolute_to_centered(value: FloatArray, family: str, offset: float) -> FloatArray:
    centered = value.copy()
    if family == "lognormal":
        centered[0] -= offset
    elif family == "exponential":
        centered[0] -= offset
    else:
        centered[1] -= offset
    if np.any(~np.isfinite(centered)):
        raise ArithmeticError("centered prior or initial coordinates are not representable")
    return centered


def rounded_tte_bayesian_gof(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    family: str,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    rng: np.random.Generator,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    bins: int | None = None,
    critical_probability: float = 0.95,
    max_likelihood_evaluations: int = 1_000_000,
    max_work: int = 100_000_000,
    compute_diagnostic: bool = True,
) -> RoundedTTEBayesianGOF:
    """Fit interval-rounded positive times with a transformed-Gaussian prior.

    ``lower`` and ``upper`` are the observed interval endpoints, with finite
    bounds satisfying ``0 <= lower < upper``. For observations rounded to the
    nearest integer ``t``, callers may use ``(max(0, t - .5), t + .5)``.
    Family names are ``exponential``, ``weibull``, ``gamma``,
    ``inverse_gamma``, ``log_logistic``, ``lognormal``, and ``log_odds_rate``.
    The caller supplies a proper Gaussian prior in absolute coordinates:
    exponential log scale; lognormal ``(mu=E[log(T)], log(sigma))``; and for
    other families log shape/log scale, with log ``c`` for log-odds-rate. The
    result centers only log scale or ``mu``; add ``time_offset`` back to
    recover the absolute coordinate. This is a Python prior contract; it does
    not reproduce family-specific conjugate priors of other fitters or assert
    BCSTTE executable defaults.

    Posterior parameters are sampled jointly by elliptical slice sampling.
    When requested, the randomized Johnson diagnostic uses the paired
    posterior CDF interval for every observed rounded value. CDF endpoint
    collapse in floating-point arithmetic raises an error; pass
    ``compute_diagnostic=False`` to fit without that diagnostic. This method
    does not handle right censoring. The likelihood evaluation and
    draw-by-observation cells are each bounded at one million; retained
    parameter cells are bounded at two million.
    """
    if not isinstance(family, str) or family not in _FAMILY_DIMENSIONS:
        raise ValueError(f"unsupported TTE family {family!r}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    lower_size = _bounded_vector_shape(lower, "lower")
    upper_size = _bounded_vector_shape(upper, "upper")
    if lower_size != upper_size or lower_size < 2:
        raise ValueError("lower and upper must have the same length of at least two")
    n = lower_size
    dimension = _FAMILY_DIMENSIONS[family]
    if _input_shape(prior_mean, "prior_mean") != (dimension,):
        raise ValueError(f"prior_mean must have shape ({dimension},)")
    if _input_shape(prior_covariance, "prior_covariance") != (dimension, dimension):
        raise ValueError(f"prior_covariance must have shape ({dimension},{dimension})")
    if np.iscomplexobj(prior_mean) or np.iscomplexobj(prior_covariance):
        raise ValueError("prior arrays must be real")
    mean = finite(prior_mean, "prior_mean").copy()
    covariance = finite(prior_covariance, "prior_covariance").copy()
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("prior_covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_covariance must be positive definite") from exc
    if family in ("weibull", "gamma", "inverse_gamma", "log_logistic", "log_odds_rate"):
        if not _LOG_FLOAT_MIN <= mean[0] <= _LOG_FLOAT_MAX:
            raise ValueError("prior mean log shape must represent a positive finite shape")

    draw_count = _bounded_integer(draws, "draws", 8, 100_000)
    warm = _bounded_integer(warmup, "warmup", 0, 100_000)
    chain_count = _bounded_integer(chains, "chains", 2, 16)
    bin_count = (
        max(2, int(np.floor(n**0.4 + 0.5)))
        if bins is None
        else _bounded_integer(bins, "bins", 2, 1000)
    )
    level = scalar(critical_probability, "critical_probability")
    if not 0 < level < 1:
        raise ValueError("critical_probability must lie in (0,1)")
    evaluation_limit = _bounded_integer(
        max_likelihood_evaluations,
        "max_likelihood_evaluations",
        1,
        _MAX_LIKELIHOOD_EVALUATIONS,
    )
    work_limit = _bounded_integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    if not isinstance(compute_diagnostic, (bool, np.bool_)):
        raise ValueError("compute_diagnostic must be Boolean")
    total_draws = chain_count * draw_count
    draw_observation_cells = total_draws * n
    retained_parameter_cells = total_draws * dimension
    if draw_observation_cells > _MAX_DRAW_OBSERVATION_CELLS:
        raise ValueError("posterior draw-by-observation work exceeds one million cells")
    if retained_parameter_cells > _MAX_RETAINED_PARAMETER_CELLS:
        raise ValueError("retained posterior parameters exceed two million cells")
    if compute_diagnostic and total_draws * bin_count > 2_000_000:
        raise ValueError("diagnostic count cells exceed two million")
    minimum_evaluations = chain_count * (2 + warm + draw_count)
    diagnostic_work = total_draws * n if compute_diagnostic else 0
    if (
        minimum_evaluations > evaluation_limit
        or minimum_evaluations * n + diagnostic_work > work_limit
    ):
        raise ValueError("minimum chain work exceeds the requested likelihood budget")

    lower_values = _freeze_vector(lower, "lower", n)
    upper_values = _freeze_vector(upper, "upper", n)
    if np.any(lower_values < 0) or np.any(upper_values <= lower_values):
        raise ValueError("intervals must satisfy 0 <= lower < upper")
    positive = upper_values[upper_values > 0]
    offset = float(np.log(positive[0]))
    positive_lower = lower_values > 0
    relative_lower = np.full(n, -np.inf)
    positive_lower_values = lower_values[positive_lower]
    combined = np.concatenate((upper_values[:1], positive_lower_values, upper_values))
    relative_combined, _ = _relative_log_times(combined)
    positive_lower_count = positive_lower_values.size
    relative_lower[positive_lower] = relative_combined[1 : 1 + positive_lower_count]
    relative_upper = relative_combined[1 + positive_lower_count :]
    for row in np.flatnonzero(positive_lower):
        lower_time = float(lower_values[row])
        upper_time = float(upper_values[row])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            close_upper = upper_time <= lower_time * 1.5
        if close_upper:
            width = float(np.log1p((upper_time - lower_time) / lower_time))
            relative_upper[row] = relative_lower[row] + width
    if np.any(~np.isfinite(relative_upper)) or np.any(
        positive_lower & ~np.isfinite(relative_lower)
    ):
        raise ArithmeticError("centered interval endpoints are not representable")
    if np.any(relative_lower >= relative_upper):
        raise ValueError("interval endpoints collapse after centered log transformation")

    centered_mean = _absolute_to_centered(mean, family, offset)
    if initial is None:
        initial_absolute = np.broadcast_to(mean, (chain_count, dimension)).copy()
        starts = np.broadcast_to(centered_mean, (chain_count, dimension)).copy()
    else:
        initial_shape = _input_shape(initial, "initial")
        if initial_shape not in ((dimension,), (chain_count, dimension)):
            raise ValueError("initial must be one transformed vector or one vector per chain")
        if np.iscomplexobj(initial):
            raise ValueError("initial must be real")
        starts_abs = finite(initial, "initial")
        if starts_abs.shape == (dimension,):
            starts_abs = np.broadcast_to(starts_abs, (chain_count, dimension)).copy()
        initial_absolute = starts_abs.copy()
        starts = np.stack(
            [_absolute_to_centered(row, family, offset) for row in starts_abs], axis=0
        )
    if np.any(~np.isfinite(starts)):
        raise ArithmeticError("centered initial coordinates are not representable")

    evaluations = 0
    work = 0

    def evaluate(state: FloatArray) -> float:
        nonlocal evaluations, work
        if evaluations >= evaluation_limit or work + n > work_limit:
            raise ArithmeticError("rounded-TTE likelihood budget exhausted")
        helper_work = [0]
        log_mass, _, _ = rounded_tte_log_probabilities(
            relative_lower,
            relative_upper,
            state,
            family,
            work_counter=helper_work,
            work_limit=work_limit - work - n,
        )
        value = float(np.sum(log_mass, dtype=np.float64))
        evaluations += 1
        work += n + helper_work[0]
        if np.isnan(value) or value == np.inf:
            raise ArithmeticError("rounded-TTE log likelihood is indeterminate")
        return value

    # Validate every chain start before the sampler consumes random values.
    for chain, start in enumerate(starts):
        try:
            start_likelihood = evaluate(start)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"initial {family} state is invalid for rounded intervals in chain {chain}"
            ) from exc
        if not np.isfinite(start_likelihood):
            raise ValueError(
                f"initial {family} state has zero representable rounded likelihood in chain {chain}"
            )

    parameters, log_likelihood = _sample_elliptical_slice_chains(
        centered_mean=centered_mean,
        cholesky=cholesky,
        centered_starts=starts,
        draws=draw_count,
        warmup=warm,
        rng=rng,
        evaluate=evaluate,
        likelihood_adjustment=0.0,
        label=family,
    )
    diagnostic = None
    if compute_diagnostic:
        left_cdf = np.empty((total_draws, n), dtype=float)
        right_cdf = np.empty((total_draws, n), dtype=float)
        flat = parameters.reshape(total_draws, dimension)
        for start in range(0, total_draws, 64):
            stop = min(start + 64, total_draws)
            for draw_index in range(start, stop):
                if work + n > work_limit:
                    raise ArithmeticError("rounded-TTE diagnostic work budget exhausted")
                helper_work = [0]
                _, left_log, right_log = rounded_tte_log_probabilities(
                    relative_lower,
                    relative_upper,
                    flat[draw_index],
                    family,
                    work_counter=helper_work,
                    work_limit=work_limit - work - n,
                )
                work += n + helper_work[0]
                with np.errstate(under="ignore", over="ignore", invalid="ignore"):
                    left_cdf[draw_index] = np.exp(left_log)
                    right_cdf[draw_index] = np.exp(right_log)
        if np.any(~np.isfinite(left_cdf)) or np.any(~np.isfinite(right_cdf)):
            raise ArithmeticError("rounded-TTE diagnostic CDF is not representable")
        if np.any(left_cdf >= right_cdf):
            raise ArithmeticError(
                "rounded interval CDF mass collapses in floating-point arithmetic; "
                "use compute_diagnostic=False to retain the fit without the diagnostic"
            )
        diagnostic = bayesian_chi_square_discrete_cdf(
            left_cdf,
            right_cdf,
            rng=rng,
            bins=bin_count,
            critical_probability=level,
        )
    return RoundedTTEBayesianGOF(
        family,
        _PARAMETER_NAMES[family],
        _owned(parameters),
        _owned(log_likelihood),
        summarize_chains(parameters),
        _owned(lower_values),
        _owned(upper_values),
        offset,
        _owned(mean),
        _owned(covariance),
        _owned(initial_absolute),
        draw_count,
        chain_count,
        evaluations,
        work,
        warm,
        bin_count,
        level,
        bool(compute_diagnostic),
        diagnostic,
    )
