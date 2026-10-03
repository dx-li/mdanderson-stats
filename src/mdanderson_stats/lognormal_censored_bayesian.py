"""Right-censored log-normal posterior fitting by bounded data augmentation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_ndtr

from ._validation import FloatArray, finite, scalar
from .bayesian_chi_square import _event_indicator
from .boin import _owned
from .hierarchical_binomial import ChainSummary, summarize_chains
from .lognormal_bayesian_gof import _convex_location, _log_abs_difference

_MAX_RETAINED_CELLS = 20_000_000
_MAX_OBSERVATIONS = 1_000_000
_MAX_DRAWS = 100_000
_MAX_ITERATIONS = 100_000
_MAX_WORK = 100_000_000
_MAX_TRUNCATED_PROPOSALS = 100_000_000
_LOG_MAX = float(np.log(np.finfo(float).max))
_MAX_STANDARDIZED_BOUND = 1e150


@dataclass(frozen=True)
class LognormalRightCensoredBayesianFit:
    """Paired Gibbs draws and observed-data likelihood for right-censored data.

    The two draw arrays have shape ``(chains, draws)``. Add ``location_offset``
    to ``centered_location_samples`` for absolute log-location. The posterior
    is not conjugate after integrating censored observations; the retained
    draws are from the stated data-augmentation Gibbs chain, not from a
    Normal-Inverse-Gamma posterior with updated hyperparameters.
    """

    centered_location_samples: FloatArray
    location_offset: float
    log_variance_samples: FloatArray
    parameter_summary: ChainSummary
    log_likelihood_samples: FloatArray
    times: FloatArray
    event: NDArray[np.bool_]
    prior_location: float
    prior_location_precision: float
    prior_variance_shape: float
    prior_variance_scale: float
    likelihood_evaluations: int
    likelihood_work_units: int
    parameter_draw_work_units: int
    gibbs_updates: int
    gibbs_work_units: int
    total_work_units: int
    truncated_normal_proposals: int
    warmup: int


def _bounded_integer(value: ArrayLike, name: str, minimum: int, maximum: int) -> int:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = scalar(cast(float, value), name)
    if result != int(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(result)


def _owned_bool(value: ArrayLike) -> NDArray[np.bool_]:
    result = np.array(value, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _center_log_times(x: FloatArray, offset: float) -> FloatArray:
    positive = x > 0
    y = np.full_like(x, np.nan)
    if not np.any(positive):
        return y
    reference = float(x[np.flatnonzero(positive)[0]])
    reference_log = float(np.log(reference))
    relative = np.empty(np.count_nonzero(positive), dtype=float)
    observed = x[positive]
    with np.errstate(over="ignore", under="ignore"):
        near = (observed >= reference * 0.5) & (observed <= reference * 1.5)
    relative[near] = np.log1p((observed[near] - reference) / reference)
    relative[~near] = np.log(observed[~near]) - reference_log
    # offset is the same reference log, except in the no-positive-time case.
    y[positive] = relative + (reference_log - offset)
    if np.any(~np.isfinite(y[positive])):
        raise ArithmeticError("centered log times are not representable")
    return y


def _log_half_sse(values: FloatArray, mean: float) -> float:
    deviations = values - mean
    if np.any(~np.isfinite(deviations)):
        raise ArithmeticError("augmented log-time deviations are not representable")
    scale = float(np.max(np.abs(deviations)))
    if scale == 0:
        return -np.inf
    normalized = deviations / scale
    return float(2.0 * np.log(scale) + np.log(float(np.dot(normalized, normalized))) - np.log(2.0))


def _draw_scaled_noise(log_sigma: float, normal: float) -> float:
    if normal == 0:
        return 0.0
    log_abs = float(np.log(abs(normal)) + log_sigma)
    if not np.isfinite(log_abs) or log_abs > _LOG_MAX:
        raise ArithmeticError("normal location update exceeds floating-point range")
    with np.errstate(over="ignore", under="ignore"):
        value = float(np.copysign(np.exp(log_abs), normal))
    if not np.isfinite(value):
        raise ArithmeticError("normal location update exceeds floating-point range")
    return value


def _truncated_normal_draw(
    lower: float,
    location: float,
    log_sigma: float,
    rng: np.random.Generator,
    *,
    max_proposals: int,
    proposals_used: int,
    chain: int,
    iteration: int,
    row: int,
) -> tuple[float, int]:
    difference = lower - location
    if not np.isfinite(difference):
        raise ArithmeticError(
            f"standardized censoring bound is unrepresentable at chain {chain}, "
            f"iteration {iteration}, row {row}"
        )
    if difference == 0:
        alpha = 0.0
    else:
        log_alpha = np.log(abs(difference)) - log_sigma
        if log_alpha > np.log(_MAX_STANDARDIZED_BOUND):
            raise ArithmeticError(
                f"standardized censoring bound is too extreme at chain {chain}, "
                f"iteration {iteration}, row {row}"
            )
        with np.errstate(under="ignore"):
            alpha = float(np.copysign(np.exp(log_alpha), difference))
    if not np.isfinite(alpha) or abs(alpha) > _MAX_STANDARDIZED_BOUND:
        raise ArithmeticError(
            f"standardized censoring bound is unrepresentable at chain {chain}, "
            f"iteration {iteration}, row {row}"
        )

    for _ in range(max(0, max_proposals - proposals_used)):
        if proposals_used >= max_proposals:
            raise ArithmeticError(
                f"truncated-normal proposal budget exhausted at chain {chain}, "
                f"iteration {iteration}, row {row}"
            )
        proposals_used += 1
        if alpha <= 0:
            standard = float(rng.standard_normal())
            if standard > alpha:
                noise = _draw_scaled_noise(log_sigma, standard)
                sample = location + noise
                if not np.isfinite(sample) or sample <= lower:
                    raise ArithmeticError(
                        f"truncated-normal draw is unrepresentable at chain {chain}, "
                        f"iteration {iteration}, row {row}"
                    )
                return float(sample), proposals_used
        else:
            delta = 2.0 / (np.hypot(alpha, 2.0) + alpha)
            rate = alpha + delta
            excess = float(rng.exponential(1.0 / rate))
            log_acceptance = -0.5 * (excess - delta) ** 2
            if np.log1p(-rng.random()) <= log_acceptance:
                log_increment = log_sigma + np.log(excess)
                if not np.isfinite(log_increment) or log_increment > _LOG_MAX:
                    raise ArithmeticError(
                        f"truncated-normal increment is unrepresentable at chain {chain}, "
                        f"iteration {iteration}, row {row}"
                    )
                with np.errstate(over="ignore", under="ignore"):
                    increment = float(np.exp(log_increment))
                    sample = lower + increment
                if increment <= 0 or not np.isfinite(sample) or sample <= lower:
                    raise ArithmeticError(
                        f"truncated-normal increment is below time-coordinate precision at "
                        f"chain {chain}, iteration {iteration}, row {row}"
                    )
                return float(sample), proposals_used
    raise ArithmeticError(
        f"truncated-normal proposal budget exhausted at chain {chain}, "
        f"iteration {iteration}, row {row}"
    )


def _observed_log_likelihood(
    event_centered_logs: FloatArray,
    event_log_times: FloatArray,
    censor_centered_logs: FloatArray,
    centered_location: float,
    log_variance: float,
) -> float:
    log_sigma = 0.5 * log_variance
    result = 0.0
    for centered_log, log_event_time in zip(event_centered_logs, event_log_times, strict=True):
        difference = float(centered_log - centered_location)
        if not np.isfinite(difference):
            raise ArithmeticError("event log-time difference is not representable")
        if difference == 0:
            standardized = 0.0
        else:
            log_abs_z = np.log(abs(difference)) - log_sigma
            if log_abs_z > 0.5 * _LOG_MAX:
                raise ArithmeticError("event log-density quadratic exceeds floating-point range")
            standardized = float(np.copysign(np.exp(log_abs_z), difference))
        if not np.isfinite(log_event_time):
            raise ArithmeticError("event time Jacobian is not representable")
        result += -log_event_time - 0.5 * (np.log(2 * np.pi) + log_variance)
        result -= 0.5 * standardized * standardized

    for centered_log in censor_centered_logs:
        difference = float(centered_location - centered_log)
        if not np.isfinite(difference):
            raise ArithmeticError("censor log-time difference is not representable")
        if difference == 0:
            standardized = 0.0
        else:
            log_abs_z = np.log(abs(difference)) - log_sigma
            if log_abs_z > _LOG_MAX:
                if difference > 0:
                    # Phi(+z) rounds to one, so its log contribution is zero.
                    continue
                raise ArithmeticError("censor log-survival exceeds floating-point range")
            with np.errstate(under="ignore"):
                standardized = float(np.copysign(np.exp(log_abs_z), difference))
        contribution = float(log_ndtr(standardized))
        if not np.isfinite(contribution):
            raise ArithmeticError("censor log-survival is not representable")
        result += contribution
    if not np.isfinite(result):
        raise ArithmeticError("observed-data log likelihood is not representable")
    return float(result)


def lognormal_right_censored_bayesian_fit(
    times: ArrayLike,
    *,
    event: ArrayLike,
    prior_location: float,
    prior_location_precision: float,
    prior_variance_shape: float,
    prior_variance_scale: float,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
    max_work: int = 100_000_000,
    max_truncated_normal_proposals: int = 10_000_000,
) -> LognormalRightCensoredBayesianFit:
    """Fit a log-normal model to mixed exact/right-censored observations.

    Uses a caller-supplied proper Normal-Inverse-Gamma prior on log-time
    location and log-time variance and an exact latent-log-time Gibbs update. Censored fits
    have no Johnson diagnostic. An all-complete vector is rejected in favor of
    :func:`lognormal_complete_data_bayesian_gof`, which returns independent
    conjugate draws. A proper prior also supports an all-censored sample as a
    Python extension beyond BCSTTE's minimum-one-event input rule.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    for value, name in (
        (prior_location, "prior_location"),
        (prior_location_precision, "prior_location_precision"),
        (prior_variance_shape, "prior_variance_shape"),
        (prior_variance_scale, "prior_variance_scale"),
    ):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
    m0 = scalar(prior_location, "prior_location")
    k0 = scalar(prior_location_precision, "prior_location_precision")
    a0 = scalar(prior_variance_shape, "prior_variance_shape")
    b0 = scalar(prior_variance_scale, "prior_variance_scale")
    if k0 <= 0 or a0 <= 0 or b0 <= 0:
        raise ValueError("the Normal-Inverse-Gamma prior parameters must be positive")

    shape = getattr(times, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError("times must be a one-dimensional vector")
        n = int(shape[0])
        if np.iscomplexobj(times):
            raise ValueError("times must be real")
    else:
        sequence = cast(Sequence[object], times)
        try:
            n = len(sequence)
        except TypeError as exc:
            raise ValueError("times must be a one-dimensional vector") from exc
        if not 2 <= n <= _MAX_OBSERVATIONS:
            raise ValueError(f"times must contain 2..{_MAX_OBSERVATIONS} observations")
        if any(not np.isscalar(value) or np.iscomplexobj(value) for value in sequence):
            raise ValueError("times must be a one-dimensional real vector")
    if not 2 <= n <= _MAX_OBSERVATIONS:
        raise ValueError(f"times must contain 2..{_MAX_OBSERVATIONS} observations")
    events = _event_indicator(event, n)
    if np.all(events):
        raise ValueError("all-complete data use lognormal_complete_data_bayesian_gof")

    draw_count = _bounded_integer(draws, "draws", 8, _MAX_DRAWS)
    warm = _bounded_integer(warmup, "warmup", 0, _MAX_ITERATIONS)
    chain_count = _bounded_integer(chains, "chains", 2, 16)
    work_limit = _bounded_integer(max_work, "max_work", 1, _MAX_WORK)
    proposal_limit = _bounded_integer(
        max_truncated_normal_proposals,
        "max_truncated_normal_proposals",
        1,
        _MAX_TRUNCATED_PROPOSALS,
    )
    total_draws = chain_count * draw_count
    # Parameter/likelihood outputs plus summary sorting, widths, split chains,
    # and the summary helper's temporary arrays, conservatively bounded.
    retained_cells = 30 * total_draws + 20 * n + 64 * chain_count
    if retained_cells > _MAX_RETAINED_CELLS:
        raise ValueError("log-normal posterior arrays exceed 20 million cells")

    x = finite(times, "times").copy()
    if x.ndim != 1 or np.any(x < 0) or np.any(x[events] <= 0):
        raise ValueError("event times must be positive and censor times nonnegative")
    positive = x > 0
    zero_censor_only = not np.any(positive)
    if zero_censor_only:
        location_offset = m0
    else:
        location_offset = float(np.log(x[np.flatnonzero(positive)[0]]))
    centered_prior_location = m0 - location_offset
    if not np.isfinite(centered_prior_location):
        raise ArithmeticError("centered prior location is not representable")
    centered_logs = _center_log_times(x, location_offset)
    event_rows = np.flatnonzero(events)
    censor_rows = np.flatnonzero((~events) & positive)
    n_informative = int(event_rows.size + censor_rows.size)
    censor_count = int(censor_rows.size)
    event_centered_logs = centered_logs[event_rows]
    event_log_times = np.log(x[event_rows])
    censor_centered_logs = centered_logs[censor_rows]
    gibbs_updates = chain_count * (warm + draw_count) if censor_count else 0
    likelihood_work_bound = total_draws * n_informative
    parameter_draw_work = total_draws
    fixed_work = gibbs_updates * n_informative + likelihood_work_bound + parameter_draw_work
    minimum_proposals = gibbs_updates * censor_count
    effective_proposal_limit = min(proposal_limit, work_limit - fixed_work)
    minimum_work = fixed_work + minimum_proposals
    if minimum_work > work_limit:
        raise ValueError("minimum Gibbs, observed-likelihood, and proposal work exceeds max_work")
    if minimum_proposals > proposal_limit:
        raise ValueError("minimum truncated-normal proposals exceed their configured limit")

    centered_draws = np.empty((chain_count, draw_count), dtype=float)
    log_variance_draws = np.empty((chain_count, draw_count), dtype=float)
    log_likelihood_draws = np.empty((chain_count, draw_count), dtype=float)
    generator = rng
    proposals_used = 0
    likelihood_evaluations = 0
    likelihood_work = 0

    if n_informative == 0:
        # Every observed censor is at time zero and contributes S(0)=1 exactly.
        shape_post = a0
        log_scale_post = float(np.log(b0))
        for chain in range(chain_count):
            gamma_draws = generator.gamma(shape_post, size=draw_count)
            normal_draws = generator.standard_normal(draw_count)
            if np.any(~np.isfinite(gamma_draws)) or np.any(gamma_draws <= 0):
                raise ArithmeticError("prior inverse-gamma draws are not representable")
            log_variance_draws[chain] = log_scale_post - np.log(gamma_draws)
            for draw in range(draw_count):
                log_sigma = 0.5 * float(log_variance_draws[chain, draw])
                centered_draws[chain, draw] = centered_prior_location + _draw_scaled_noise(
                    log_sigma, float(normal_draws[draw]) / np.sqrt(k0)
                )
            log_likelihood_draws[chain] = 0.0
    elif censor_count == 0:
        # Positive event times plus zero-time censors reduce to the exact
        # complete-data NIG posterior; zero censors add no information.
        event_logs = centered_logs[events]
        event_count = event_logs.size
        ybar = float(event_logs[0] + np.mean(event_logs - event_logs[0]))
        posterior_precision = k0 + event_count
        posterior_location = _convex_location(
            centered_prior_location,
            ybar,
            k0 / posterior_precision,
            event_count / posterior_precision,
        )
        posterior_shape = a0 + event_count / 2.0
        log_sse_half = _log_half_sse(event_logs, ybar)
        log_between = (
            np.log(k0)
            + np.log(float(event_count))
            - np.log(posterior_precision)
            - np.log(2.0)
            + 2.0 * _log_abs_difference(centered_prior_location, ybar)
        )
        log_posterior_scale = float(
            np.logaddexp(np.log(b0), np.logaddexp(log_sse_half, log_between))
        )
        if (
            not np.isfinite(posterior_precision)
            or not np.isfinite(posterior_location)
            or not np.isfinite(posterior_shape)
            or not np.isfinite(log_posterior_scale)
        ):
            raise ArithmeticError("complete-data conditional NIG posterior is not representable")
        for chain in range(chain_count):
            gamma_draws = generator.gamma(posterior_shape, size=draw_count)
            normal_draws = generator.standard_normal(draw_count)
            if np.any(~np.isfinite(gamma_draws)) or np.any(gamma_draws <= 0):
                raise ArithmeticError(
                    "complete-data posterior variance draws are not representable"
                )
            log_variance_draws[chain] = log_posterior_scale - np.log(gamma_draws)
            for draw in range(draw_count):
                noise = _draw_scaled_noise(
                    0.5 * (float(log_variance_draws[chain, draw]) - np.log(posterior_precision)),
                    float(normal_draws[draw]),
                )
                centered_draws[chain, draw] = posterior_location + noise
                if not np.isfinite(centered_draws[chain, draw]):
                    raise ArithmeticError(
                        "complete-data posterior location draw is not representable"
                    )
                log_likelihood_draws[chain, draw] = _observed_log_likelihood(
                    event_centered_logs,
                    event_log_times,
                    censor_centered_logs,
                    float(centered_draws[chain, draw]),
                    float(log_variance_draws[chain, draw]),
                )
                likelihood_evaluations += 1
                likelihood_work += n_informative
    else:
        censor_rows = np.flatnonzero((~events) & positive)
        fixed_event_logs = event_centered_logs
        posterior_precision = k0 + n_informative
        posterior_shape = a0 + n_informative / 2.0
        if not np.isfinite(posterior_precision) or not np.isfinite(posterior_shape):
            raise ArithmeticError("posterior NIG parameters are not representable")
        for chain in range(chain_count):
            gamma0 = float(generator.gamma(a0))
            if not np.isfinite(gamma0) or gamma0 <= 0:
                raise ArithmeticError("initial prior variance draw is not representable")
            log_variance = np.log(b0) - np.log(gamma0)
            initial_noise = _draw_scaled_noise(
                0.5 * float(log_variance), float(generator.standard_normal()) / np.sqrt(k0)
            )
            centered_location = centered_prior_location + initial_noise
            if not np.isfinite(centered_location):
                raise ArithmeticError("initial prior location draw is not representable")
            latent = np.empty(n_informative, dtype=float)
            event_count = event_rows.size
            latent[:event_count] = fixed_event_logs
            for iteration in range(warm + draw_count):
                for offset, row in enumerate(censor_rows):
                    latent_index = event_count + offset
                    latent[latent_index], proposals_used = _truncated_normal_draw(
                        float(centered_logs[row]),
                        centered_location,
                        0.5 * float(log_variance),
                        generator,
                        max_proposals=effective_proposal_limit,
                        proposals_used=proposals_used,
                        chain=chain,
                        iteration=iteration,
                        row=int(row),
                    )
                ybar = float(latent[0] + np.mean(latent - latent[0]))
                posterior_location = _convex_location(
                    centered_prior_location,
                    ybar,
                    k0 / posterior_precision,
                    n_informative / posterior_precision,
                )
                log_sse_half = _log_half_sse(latent, ybar)
                log_between = (
                    np.log(k0)
                    + np.log(float(n_informative))
                    - np.log(posterior_precision)
                    - np.log(2.0)
                    + 2.0 * _log_abs_difference(centered_prior_location, ybar)
                )
                log_posterior_scale = float(
                    np.logaddexp(np.log(b0), np.logaddexp(log_sse_half, log_between))
                )
                if not np.isfinite(posterior_location) or not np.isfinite(log_posterior_scale):
                    raise ArithmeticError("conditional NIG update is not representable")
                gamma_draw = float(generator.gamma(posterior_shape))
                normal_draw = float(generator.standard_normal())
                if not np.isfinite(gamma_draw) or gamma_draw <= 0 or not np.isfinite(normal_draw):
                    raise ArithmeticError("conditional NIG draws are not representable")
                log_variance = log_posterior_scale - np.log(gamma_draw)
                noise = _draw_scaled_noise(
                    0.5 * (log_variance - np.log(posterior_precision)), normal_draw
                )
                centered_location = posterior_location + noise
                if not np.isfinite(centered_location) or not np.isfinite(log_variance):
                    raise ArithmeticError("conditional location/variance draw is not representable")
                if iteration >= warm:
                    draw = iteration - warm
                    centered_draws[chain, draw] = centered_location
                    log_variance_draws[chain, draw] = log_variance
                    likelihood = _observed_log_likelihood(
                        event_centered_logs,
                        event_log_times,
                        censor_centered_logs,
                        centered_location,
                        log_variance,
                    )
                    log_likelihood_draws[chain, draw] = likelihood
                    likelihood_evaluations += 1
                    likelihood_work += n_informative

    if n_informative == 0:
        # The integrated observed-data likelihood is identically one.
        likelihood_evaluations = 0
        likelihood_work = 0
    parameters = np.stack((centered_draws, log_variance_draws), axis=-1)
    summary = summarize_chains(parameters)
    return LognormalRightCensoredBayesianFit(
        _owned(centered_draws),
        location_offset,
        _owned(log_variance_draws),
        summary,
        _owned(log_likelihood_draws),
        _owned(x),
        _owned_bool(events),
        m0,
        k0,
        a0,
        b0,
        likelihood_evaluations,
        likelihood_work,
        parameter_draw_work,
        gibbs_updates,
        gibbs_updates * n_informative,
        gibbs_updates * n_informative + proposals_used + likelihood_work + parameter_draw_work,
        proposals_used,
        warm if censor_count else 0,
    )
