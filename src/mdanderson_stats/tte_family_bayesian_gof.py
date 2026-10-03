"""Bayesian chi-square workflows for four BCSTTE TTE families.

The guide supplies family densities and survival functions, but not prior or
fitting defaults. Gamma, inverse-Gamma and log-logistic workflows require a
proper correlated Gaussian prior on log(shape), log(scale); the generalized
log-odds-rate workflow requires one on log(shape), log(scale), log(c). Right
censoring uses ordinary event-density and right-tail-survival likelihoods.
Johnson's diagnostic is returned only for complete samples because the guide
does not define a censored-data CDF diagnostic.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import gammainc, gammaincc, gammaln, log_expit

from ._log_odds_rate import log_odds_rate_components
from ._validation import FloatArray, finite, scalar
from .bayesian_chi_square import BayesianChiSquare, _event_indicator, bayesian_chi_square_cdf
from .boin import _owned
from .cdflib_gamma_factor import _large_log_factor
from .cdflib_gamma_support import _local_log_gamma
from .hierarchical_binomial import ChainSummary, summarize_chains
from .weibull_unknown_shape_gof import (
    _LOG_FLOAT_MAX,
    _LOG_FLOAT_MIN,
    _MAX_LIKELIHOOD_EVALUATIONS,
    _MAX_RETAINED_CELLS,
    _MAX_SLICE_STEPS,
    _MAX_WORK_UNITS,
    _bounded_integer,
    _input_shape,
    _relative_log_times,
)

NDArrayBool = NDArray[np.bool_]


def _log1mexp(value: FloatArray) -> FloatArray:
    """Stable log(1-exp(value)) for nonpositive log probabilities."""
    result = np.empty_like(value)
    split = value < -np.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        result[split] = np.log1p(-np.exp(value[split]))
        result[~split] = np.log(-np.expm1(value[~split]))
    return result


@dataclass(frozen=True)
class TTEFamilyBayesianGOF:
    """Joint posterior draws for a direct shape/scale TTE model.

    ``parameters[..., 0]`` is log shape and ``parameters[..., 1]`` is centered
    log scale. Add ``log_scale_offset`` to the second coordinate to recover
    absolute log scale. ``diagnostic`` is ``None`` whenever any observation is
    right-censored; no censored-data Johnson transform is specified by BCSTTE.
    """

    family: str
    parameter_names: tuple[str, str]
    parameters: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    times: FloatArray
    event: NDArrayBool
    log_scale_offset: float
    prior_mean: FloatArray
    prior_covariance: FloatArray
    likelihood_evaluations: int
    likelihood_work_units: int
    warmup: int
    diagnostic: BayesianChiSquare | None


@dataclass(frozen=True)
class LogOddsRateBayesianGOF:
    """Joint posterior draws for the generalized log-odds-rate model.

    ``parameters[..., 0]`` is log shape, ``[..., 1]`` is centered log scale,
    and ``[..., 2]`` is log ``c``. Add ``log_scale_offset`` to the second
    coordinate to recover absolute log scale. Censored fits have no Johnson
    diagnostic because no censored-data transform is specified by the source.
    """

    parameter_names: tuple[str, str, str]
    parameters: FloatArray
    log_likelihood: FloatArray
    parameter_summary: ChainSummary
    times: FloatArray
    event: NDArrayBool
    log_scale_offset: float
    prior_mean: FloatArray
    prior_covariance: FloatArray
    likelihood_evaluations: int
    likelihood_work_units: int
    warmup: int
    diagnostic: BayesianChiSquare | None


def _sample_elliptical_slice_chains(
    *,
    centered_mean: FloatArray,
    cholesky: FloatArray,
    centered_starts: FloatArray,
    draws: int,
    warmup: int,
    rng: np.random.Generator,
    evaluate: Callable[[FloatArray], float],
    likelihood_adjustment: float,
    label: str,
) -> tuple[FloatArray, FloatArray]:
    """Run the shared bounded elliptical-slice sampler in any dimension."""
    chain_count = centered_starts.shape[0]
    dimension = centered_mean.size
    parameters = np.empty((chain_count, draws, dimension), dtype=float)
    log_likelihood = np.empty((chain_count, draws), dtype=float)
    for chain in range(chain_count):
        state = centered_starts[chain].copy()
        ll = evaluate(state)
        if not np.isfinite(ll):
            raise ValueError(f"initial {label} state has zero representable likelihood")
        for iteration in range(warmup + draws):
            direction = cholesky @ rng.standard_normal(dimension)
            height = ll + np.log1p(-rng.random())
            angle = float(rng.uniform(0, 2 * np.pi))
            lower, upper = angle - 2 * np.pi, angle
            for _ in range(_MAX_SLICE_STEPS):
                proposal = (
                    centered_mean
                    + (state - centered_mean) * np.cos(angle)
                    + direction * np.sin(angle)
                )
                trial = evaluate(proposal)
                if trial >= height:
                    state, ll = proposal, trial
                    break
                if angle < 0:
                    lower = angle
                else:
                    upper = angle
                angle = float(rng.uniform(lower, upper))
            else:
                raise ArithmeticError(f"{label} elliptical-slice update failed in chain {chain}")
            if iteration >= warmup:
                draw = iteration - warmup
                parameters[chain, draw] = state
                log_likelihood[chain, draw] = ll + likelihood_adjustment
    return parameters, log_likelihood


def _log_regularized_gamma_tail(
    shape: float,
    log_x: FloatArray,
    *,
    lower: bool,
    work_counter: list[int] | None = None,
    work_limit: int = 0,
) -> FloatArray:
    """Log P(a,x) or log Q(a,x), retaining tails after scipy underflows."""
    result = np.empty_like(log_x)
    log_min, log_max = _LOG_FLOAT_MIN, _LOG_FLOAT_MAX
    for index, lx0 in np.ndenumerate(log_x):
        lx = float(lx0)
        if np.isnan(lx):
            result[index] = np.nan
        elif lx == -np.inf or lx < log_min:
            log_gamma_a1 = (
                float(_local_log_gamma(np.asarray([shape]))[0])
                if shape < 0.6
                else float(gammaln(shape + 1.0))
            )
            log_lower = shape * lx - log_gamma_a1
            if lower:
                # P(a,x) = x**a/Gamma(a+1) * (1 + O(x)).
                result[index] = log_lower
            else:
                result[index] = _log1mexp(np.asarray([log_lower]))[0]
        elif lx > log_max:
            result[index] = 0.0 if lower else -np.inf
        else:
            x = float(np.exp(lx))
            direct = float(gammainc(shape, x) if lower else gammaincc(shape, x))
            if direct > 0.0:
                result[index] = np.log(direct)
            elif lower and x < shape + 1.0:
                # Series for the lower regularized incomplete gamma ratio.
                term = total = 1.0 / shape
                ap = shape
                for _ in range(512):
                    if work_counter is not None:
                        work_counter[0] += 1
                        if work_counter[0] > work_limit:
                            raise ArithmeticError("incomplete-gamma tail work budget exhausted")
                    ap += 1.0
                    term *= x / ap
                    total += term
                    if abs(term) <= abs(total) * 2.0e-16:
                        break
                else:
                    raise ArithmeticError("lower incomplete-gamma tail did not converge")
                result[index] = float(_unit_gamma_log_factor(shape, np.asarray([lx]))[0]) + np.log(
                    total
                )
            elif not lower and x >= shape + 1.0:
                # Continued fraction for the upper regularized incomplete gamma.
                tiny = np.finfo(float).tiny / np.finfo(float).eps
                b = x + 1.0 - shape
                c = 1.0 / tiny
                d = 1.0 / b
                h = d
                for i in range(1, 513):
                    if work_counter is not None:
                        work_counter[0] += 1
                        if work_counter[0] > work_limit:
                            raise ArithmeticError("incomplete-gamma tail work budget exhausted")
                    an = -float(i) * (float(i) - shape)
                    b += 2.0
                    d = an * d + b
                    if abs(d) < tiny:
                        d = tiny
                    c = b + an / c
                    if abs(c) < tiny:
                        c = tiny
                    d = 1.0 / d
                    delta = d * c
                    h *= delta
                    if abs(delta - 1.0) <= 4.0e-15:
                        break
                else:
                    raise ArithmeticError("upper incomplete-gamma tail did not converge")
                if h <= 0.0 or not np.isfinite(h):
                    raise ArithmeticError("upper incomplete-gamma continued fraction is invalid")
                result[index] = float(_unit_gamma_log_factor(shape, np.asarray([lx]))[0]) + np.log(
                    h
                )
            else:
                # The opposite tail is near one; log1p preserves its complement.
                complement = float(gammaincc(shape, x) if lower else gammainc(shape, x))
                result[index] = np.log1p(-complement)
    return result


def _unit_gamma_log_factor(shape: float, log_value: FloatArray) -> FloatArray:
    """Compute ``a*log(x)-x-lgamma(a)`` without large-shape cancellation."""
    result = np.empty_like(log_value)
    tiny = log_value < _LOG_FLOAT_MIN
    huge = log_value > _LOG_FLOAT_MAX
    ordinary = ~(tiny | huge)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result[tiny] = shape * log_value[tiny] - gammaln(shape)
        result[huge] = -np.inf
        if np.any(ordinary):
            x = np.exp(log_value[ordinary])
            if shape >= 8.0:
                result[ordinary] = _large_log_factor(np.full_like(x, shape), x)
            else:
                result[ordinary] = shape * log_value[ordinary] - x - gammaln(shape)
    if np.any(np.isnan(result)) or np.any(result == np.inf):
        raise ArithmeticError("Gamma event log density is not representable")
    return result


def _family_log_likelihood(
    coordinates: FloatArray,
    relative: FloatArray,
    event: NDArrayBool,
    family: str,
    zero_censor: NDArrayBool | None = None,
    tail_work_counter: list[int] | None = None,
    tail_work_limit: int = 0,
) -> float:
    log_shape, log_scale = map(float, coordinates)
    if not np.isfinite(log_shape) or not np.isfinite(log_scale):
        raise ArithmeticError(f"{family} log-parameter proposal is nonfinite")
    if not _LOG_FLOAT_MIN <= log_shape <= _LOG_FLOAT_MAX:
        return -np.inf
    shape = float(np.exp(log_shape))
    log_ratio = relative - log_scale
    if np.any(~np.isfinite(log_ratio)):
        return -np.inf
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        censor_eval = ~event
        if zero_censor is not None:
            censor_eval &= ~zero_censor
        if family == "gamma":
            lx = log_ratio
            event_ll = _unit_gamma_log_factor(shape, lx) - relative
            cens_ll = np.zeros_like(relative)
            cens_ll[censor_eval] = _log_regularized_gamma_tail(
                shape,
                lx[censor_eval],
                lower=False,
                work_counter=tail_work_counter,
                work_limit=tail_work_limit,
            )
        elif family == "inverse_gamma":
            lx = -log_ratio
            event_ll = _unit_gamma_log_factor(shape, lx) - relative
            cens_ll = np.zeros_like(relative)
            cens_ll[censor_eval] = _log_regularized_gamma_tail(
                shape,
                lx[censor_eval],
                lower=True,
                work_counter=tail_work_counter,
                work_limit=tail_work_limit,
            )
        elif family == "log_logistic":
            z = shape * log_ratio
            event_ll = (
                log_shape - shape * log_scale + (shape - 1.0) * relative + 2.0 * log_expit(-z)
            )
            cens_ll = log_expit(-z)
        else:
            raise ValueError(f"unsupported TTE family {family!r}")
    terms = np.where(event, event_ll, cens_ll)
    if zero_censor is not None:
        terms[zero_censor] = 0.0
    if np.any(np.isnan(terms)) or np.any(terms == np.inf):
        raise ArithmeticError(f"{family} likelihood evaluation became indeterminate")
    value = float(np.sum(terms, dtype=np.float64))
    if np.isnan(value) or value == np.inf:
        raise ArithmeticError(f"{family} likelihood sum became indeterminate")
    return value


def _family_cdf(
    log_shape: FloatArray, centered_log_scale: FloatArray, relative: FloatArray, family: str
) -> FloatArray:
    shape, log_ratio = np.broadcast_arrays(
        np.exp(log_shape), relative[None, :] - centered_log_scale
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if family == "gamma":
            cdf = gammainc(shape, np.exp(log_ratio))
        elif family == "inverse_gamma":
            cdf = gammaincc(shape, np.exp(-log_ratio))
        else:
            z = shape * log_ratio
            cdf = np.exp(log_expit(z))
        if family in ("gamma", "inverse_gamma"):
            log_argument = log_ratio if family == "gamma" else -log_ratio
            underflow = log_argument < _LOG_FLOAT_MIN
            if np.any(underflow):
                a, lx = shape[underflow], log_argument[underflow]
                log_gamma_a1 = np.empty_like(a)
                small_shape = a < 0.6
                log_gamma_a1[small_shape] = _local_log_gamma(a[small_shape])
                log_gamma_a1[~small_shape] = gammaln(a[~small_shape] + 1.0)
                log_lower = a * lx - log_gamma_a1
                if family == "gamma":
                    cdf[underflow] = np.exp(log_lower)
                else:
                    cdf[underflow] = np.exp(_log1mexp(log_lower))
    return np.asarray(cdf, dtype=float)


def _family_bayesian_gof(
    family: str,
    times: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    event: ArrayLike | None,
    draws: int,
    warmup: int,
    chains: int,
    initial: ArrayLike | None,
    bins: int | None,
    critical_probability: float,
    rng: np.random.Generator,
    max_likelihood_evaluations: int,
    max_work: int,
) -> TTEFamilyBayesianGOF:
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
            raise ValueError("times must contain at least two observations")
        if any(not np.isscalar(item) or np.iscomplexobj(item) for item in sequence):
            raise ValueError("times must be a one-dimensional real vector")
    else:
        if len(time_shape) != 1:
            raise ValueError("times must be a one-dimensional vector")
        n = int(time_shape[0])
        if not 2 <= n <= _MAX_RETAINED_CELLS or np.iscomplexobj(times):
            raise ValueError("times must be a one-dimensional real vector with n >= 2")
    events = _event_indicator(event, n)
    if _input_shape(prior_mean, "prior_mean") != (2,):
        raise ValueError("prior_mean must contain log shape and log scale means")
    if _input_shape(prior_covariance, "prior_covariance") != (2, 2):
        raise ValueError("prior_covariance must be a 2x2 matrix")
    if np.iscomplexobj(prior_mean) or np.iscomplexobj(prior_covariance):
        raise ValueError("prior arrays must be real")
    mean, covariance = (
        finite(prior_mean, "prior_mean"),
        finite(prior_covariance, "prior_covariance"),
    )
    if (
        mean.shape != (2,)
        or covariance.shape != (2, 2)
        or not np.array_equal(covariance, covariance.T)
    ):
        raise ValueError("prior arrays must have shapes (2,) and (2,2), covariance symmetric")
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
    level = scalar(critical_probability, "critical_probability")
    if not 0 < level < 1:
        raise ValueError("critical_probability must lie in (0,1)")
    evaluation_limit = _bounded_integer(
        max_likelihood_evaluations, "max_likelihood_evaluations", 1, _MAX_LIKELIHOOD_EVALUATIONS
    )
    work_limit = _bounded_integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    total_draws = chain_count * size
    diagnostic_cells = total_draws * n if np.all(events) else 0
    summaries = total_draws * bin_count if diagnostic_cells else 0
    retained_cells = (
        diagnostic_cells + 3 * summaries + 30 * total_draws + 8 * n + 4 * min(total_draws, 256) * n
    )
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("posterior and diagnostic arrays exceed 20 million cells")
    minimum = chain_count * (1 + burn + size)
    if minimum > evaluation_limit or minimum * n > work_limit:
        raise ValueError("minimum chain work exceeds the requested likelihood budget")
    if initial is None:
        starts = np.broadcast_to(mean, (chain_count, 2)).copy()
    else:
        start_shape = _input_shape(initial, "initial")
        if start_shape not in ((2,), (chain_count, 2)) or np.iscomplexobj(initial):
            raise ValueError("initial must be one real log-parameter vector or one per chain")
        start = finite(initial, "initial")
        starts = (
            np.broadcast_to(start, (chain_count, 2)).copy() if start.shape == (2,) else start.copy()
        )
    x = finite(times, "times")
    if x.ndim != 1 or np.any(x < 0) or np.any(x[events] <= 0):
        raise ValueError("exact events require positive times; times cannot be negative")
    # Zero-time right censoring is valid: log S(0)=0. Use a positive anchor
    # when available; an all-zero sample has a prior-only posterior.
    positive = x[x > 0]
    relative = np.zeros_like(x)
    if positive.size:
        rel_positive, offset = _relative_log_times(positive)
        relative[x > 0] = rel_positive
    else:
        offset = float(mean[1])
    centered_mean = mean.copy()
    centered_mean[1] -= offset
    starts[:, 1] -= offset
    if np.any(~np.isfinite(centered_mean)) or np.any(~np.isfinite(starts)):
        raise ArithmeticError("centered log-scale coordinates are not representable")
    evals = work = 0

    def evaluate(state: FloatArray) -> float:
        nonlocal evals, work
        if evals >= evaluation_limit or work + n > work_limit:
            raise ArithmeticError("likelihood work budget exhausted")
        tail_work = [0]
        val = _family_log_likelihood(
            state,
            relative,
            events,
            family,
            zero_censor=(x == 0),
            tail_work_counter=tail_work,
            tail_work_limit=work_limit - work - n,
        )
        evals += 1
        work += n + tail_work[0]
        return val

    parameters, log_likelihood = _sample_elliptical_slice_chains(
        centered_mean=centered_mean,
        cholesky=cholesky,
        centered_starts=starts,
        draws=size,
        warmup=burn,
        rng=rng,
        evaluate=evaluate,
        likelihood_adjustment=-int(np.count_nonzero(events)) * offset,
        label=family,
    )
    diagnostic = None
    if np.all(events):
        cdf = np.empty((total_draws, n))
        flat = parameters.reshape(total_draws, 2)
        for batch_start in range(0, total_draws, 256):
            stop = min(batch_start + 256, total_draws)
            cdf[batch_start:stop] = _family_cdf(
                flat[batch_start:stop, 0, None], flat[batch_start:stop, 1, None], relative, family
            )
        if np.any(~np.isfinite(cdf)) or np.any((cdf < 0) | (cdf > 1)):
            raise ArithmeticError("posterior CDF evaluation became invalid")
        diagnostic = bayesian_chi_square_cdf(cdf, bins=bin_count, critical_probability=level)
    return TTEFamilyBayesianGOF(
        family,
        ("log_shape", "centered_log_scale"),
        _owned(parameters),
        _owned(log_likelihood),
        summarize_chains(parameters),
        _owned(x),
        _owned(events),
        offset,
        _owned(mean),
        _owned(covariance),
        evals,
        work,
        burn,
        diagnostic,
    )


def _make_family_function(family: str) -> Callable[..., TTEFamilyBayesianGOF]:
    def fit(
        times: ArrayLike,
        *,
        prior_mean: ArrayLike,
        prior_covariance: ArrayLike,
        event: ArrayLike | None = None,
        draws: int = 1000,
        warmup: int = 500,
        chains: int = 4,
        initial: ArrayLike | None = None,
        bins: int | None = None,
        critical_probability: float = 0.95,
        rng: np.random.Generator,
        max_likelihood_evaluations: int = 1_000_000,
        max_work: int = 100_000_000,
    ) -> TTEFamilyBayesianGOF:
        """Fit under an explicit correlated Gaussian prior on log shape/scale."""
        return _family_bayesian_gof(
            family,
            times,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
            event=event,
            draws=draws,
            warmup=warmup,
            chains=chains,
            initial=initial,
            bins=bins,
            critical_probability=critical_probability,
            rng=rng,
            max_likelihood_evaluations=max_likelihood_evaluations,
            max_work=max_work,
        )

    fit.__name__ = f"{family}_bayesian_gof"
    fit.__qualname__ = fit.__name__
    return fit


gamma_bayesian_gof = _make_family_function("gamma")
inverse_gamma_bayesian_gof = _make_family_function("inverse_gamma")
log_logistic_bayesian_gof = _make_family_function("log_logistic")


def log_odds_rate_bayesian_gof(
    times: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_covariance: ArrayLike,
    event: ArrayLike | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial: ArrayLike | None = None,
    bins: int | None = None,
    critical_probability: float = 0.95,
    rng: np.random.Generator,
    max_likelihood_evaluations: int = 1_000_000,
    max_work: int = 100_000_000,
) -> LogOddsRateBayesianGOF:
    """Fit the log-odds-rate family under an explicit Gaussian log prior.

    The proper correlated Normal prior is supplied on absolute
    ``(log_shape, log_scale, log_c)`` coordinates. The sampler centers only
    log scale, returning that coordinate with ``log_scale_offset``. The prior
    is an explicit Python convention; no BCSTTE native prior or fitting
    defaults are claimed. ``event=True`` marks an exact event and ``False`` a
    right-censored observation. Censored fits have no Johnson diagnostic.
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
            raise ValueError("times must contain at least two observations")
        if any(not np.isscalar(item) or np.iscomplexobj(item) for item in sequence):
            raise ValueError("times must be a one-dimensional real vector")
    else:
        if len(time_shape) != 1:
            raise ValueError("times must be a one-dimensional vector")
        n = int(time_shape[0])
        if not 2 <= n <= _MAX_RETAINED_CELLS or np.iscomplexobj(times):
            raise ValueError("times must be a one-dimensional real vector with n >= 2")

    events = _event_indicator(event, n)
    if _input_shape(prior_mean, "prior_mean") != (3,):
        raise ValueError("prior_mean must contain log shape, log scale, and log c means")
    if _input_shape(prior_covariance, "prior_covariance") != (3, 3):
        raise ValueError("prior_covariance must be a 3x3 matrix")
    if np.iscomplexobj(prior_mean) or np.iscomplexobj(prior_covariance):
        raise ValueError("prior arrays must be real")
    mean = finite(prior_mean, "prior_mean")
    covariance = finite(prior_covariance, "prior_covariance")
    if mean.shape != (3,) or covariance.shape != (3, 3):
        raise ValueError("prior arrays must have shapes (3,) and (3,3)")
    if not np.array_equal(covariance, covariance.T):
        raise ValueError("prior_covariance must be symmetric")
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("prior_covariance must be positive definite") from exc
    if not _LOG_FLOAT_MIN <= mean[0] <= _LOG_FLOAT_MAX:
        raise ValueError("prior log-shape mean must represent a positive finite shape")

    size = _bounded_integer(draws, "draws", 8, 100_000)
    burn = _bounded_integer(warmup, "warmup", 0, 100_000)
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
    total_draws = chain_count * size
    cdf_cells = total_draws * n if np.all(events) else 0
    summary_cells = total_draws * bin_count if cdf_cells else 0
    retained_cells = (
        cdf_cells + 3 * summary_cells + 31 * total_draws + 8 * n + 4 * min(total_draws, 256) * n
    )
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("posterior and diagnostic arrays exceed 20 million cells")
    minimum = chain_count * (1 + burn + size)
    if minimum > evaluation_limit or minimum * n > work_limit:
        raise ValueError("minimum chain work exceeds the requested likelihood budget")

    if initial is None:
        starts = np.broadcast_to(mean, (chain_count, 3)).copy()
    else:
        initial_shape = _input_shape(initial, "initial")
        if initial_shape not in ((3,), (chain_count, 3)) or np.iscomplexobj(initial):
            raise ValueError("initial must be one real log-parameter vector or one per chain")
        supplied = finite(initial, "initial")
        starts = (
            np.broadcast_to(supplied, (chain_count, 3)).copy()
            if supplied.shape == (3,)
            else supplied.copy()
        )

    x = finite(times, "times")
    if x.ndim != 1 or np.any(x < 0) or np.any(x[events] <= 0):
        raise ValueError("exact events require positive times; times cannot be negative")
    positive = x[x > 0]
    relative = np.zeros_like(x)
    if positive.size:
        relative_positive, offset = _relative_log_times(positive)
        relative[x > 0] = relative_positive
    else:
        offset = float(mean[1])
    centered_mean = mean.copy()
    centered_mean[1] -= offset
    centered_starts = starts.copy()
    centered_starts[:, 1] -= offset
    if np.any(~np.isfinite(centered_mean)) or np.any(~np.isfinite(centered_starts)):
        raise ArithmeticError("centered log-scale coordinates are not representable")

    evaluations = work = 0

    def evaluate(state: FloatArray) -> float:
        nonlocal evaluations, work
        if evaluations >= evaluation_limit or work + n > work_limit:
            raise ArithmeticError("log-odds-rate likelihood budget exhausted")
        if np.any(~np.isfinite(state)) or not _LOG_FLOAT_MIN <= float(state[0]) <= _LOG_FLOAT_MAX:
            evaluations += 1
            work += n
            return -np.inf
        observed = x > 0
        if np.any(observed):
            log_density, log_survival, _ = log_odds_rate_components(
                relative[observed],
                float(state[0]),
                float(state[1]),
                float(state[2]),
            )
            terms = np.where(events[observed], log_density, log_survival)
        else:
            terms = np.zeros(0, dtype=float)
        if np.any(np.isnan(terms)) or np.any(terms == np.inf):
            raise ArithmeticError("log-odds-rate likelihood became indeterminate")
        result = float(np.sum(terms, dtype=np.float64))
        if np.isnan(result) or result == np.inf:
            raise ArithmeticError("log-odds-rate likelihood sum became indeterminate")
        evaluations += 1
        work += n
        return result

    parameters, log_likelihood = _sample_elliptical_slice_chains(
        centered_mean=centered_mean,
        cholesky=cholesky,
        centered_starts=centered_starts,
        draws=size,
        warmup=burn,
        rng=rng,
        evaluate=evaluate,
        likelihood_adjustment=-int(np.count_nonzero(events)) * offset,
        label="log-odds-rate",
    )
    diagnostic = None
    if np.all(events):
        cdf = np.empty((total_draws, n), dtype=float)
        flat = parameters.reshape((total_draws, 3))
        for batch_start in range(0, total_draws, 256):
            stop = min(batch_start + 256, total_draws)
            for row in range(batch_start, stop):
                _, _, cdf[row] = log_odds_rate_components(
                    relative,
                    float(flat[row, 0]),
                    float(flat[row, 1]),
                    float(flat[row, 2]),
                )
        if np.any(~np.isfinite(cdf)) or np.any((cdf < 0) | (cdf > 1)):
            raise ArithmeticError("log-odds-rate posterior CDF evaluation became invalid")
        diagnostic = bayesian_chi_square_cdf(cdf, bins=bin_count, critical_probability=level)

    return LogOddsRateBayesianGOF(
        ("log_shape", "centered_log_scale", "log_c"),
        _owned(parameters),
        _owned(log_likelihood),
        summarize_chains(parameters),
        _owned(x),
        _owned(events),
        offset,
        _owned(mean),
        _owned(covariance),
        evaluations,
        work,
        burn,
        diagnostic,
    )
