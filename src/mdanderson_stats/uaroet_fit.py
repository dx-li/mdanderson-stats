"""Explicit-prior posterior inference for UAROET ordinal outcomes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._uaroet_bounds import _likelihood_upper_bound
from ._validation import FloatArray
from .hierarchical_binomial import ChainSummary, summarize_chains
from .u2oet_prior import _positive_normal
from .uaroet import (
    _integer,
    _joint_from_marginals,
    _marginal_log_probabilities,
    _real_matrix,
    uaroet_logits,
    uaroet_parameter_names,
)

_MAX_DOSES = 5
_MAX_LEVELS = 4
_MAX_SUBJECTS = 10_000
_MAX_CHAINS = 20
_MAX_DRAWS = 100_000
_MAX_WARMUP = 100_000
_MAX_RETAINED = 2_000_000
_MAX_EVALUATIONS = 128_000_000


def _counts(value: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if (
        raw.ndim != 3
        or not 1 <= raw.shape[0] <= _MAX_DOSES
        or any(not 2 <= width <= _MAX_LEVELS for width in raw.shape[1:])
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError("counts must be a dose-by-efficacy-by-toxicity array on supported grids")
    numeric = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric >= 2**53)
        or numeric.sum() > _MAX_SUBJECTS
    ):
        raise ValueError(f"counts must be nonnegative integers totaling at most {_MAX_SUBJECTS}")
    return numeric.astype(np.int64)


def _positive_mask(names: tuple[str, ...]) -> NDArray[np.bool_]:
    return np.asarray([".increment." in name for name in names], dtype=bool)


def _draw_prior(
    mean: FloatArray,
    sd: FloatArray,
    positive: NDArray[np.bool_],
    size: int,
    rng: np.random.Generator,
) -> FloatArray:
    result = np.empty((size, mean.size), dtype=float)
    for index in range(mean.size):
        if positive[index]:
            result[:, index] = _positive_normal(float(mean[index]), float(sd[index]), size, rng)
        else:
            result[:, index] = rng.normal(float(mean[index]), float(sd[index]), size)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("normal prior draw exceeded floating-point range")
    return result


@dataclass(frozen=True)
class UAROETFit:
    """Posterior draws with chain and retained-draw axes."""

    parameter_names: tuple[str, ...]
    parameters: FloatArray
    association: FloatArray
    joint: FloatArray
    log_likelihood: FloatArray
    association_acceptance: FloatArray
    parameter_summary: ChainSummary
    association_summary: ChainSummary
    likelihood_evaluations: int
    rectangle_evaluations: int
    work_evaluations: int
    warmup: int
    monotone_efficacy: bool
    monotone_toxicity: bool
    fixed_association: float | None
    direct_prior: bool


def _summarize_uaroet_chains(samples: FloatArray) -> ChainSummary:
    if samples.shape[0] >= 2:
        return summarize_chains(samples)
    x = samples[0]
    ordered = np.sort(x, axis=0)
    included = max(1, int(np.ceil(0.8 * len(x))))
    widths = ordered[included - 1 :] - ordered[: len(x) - included + 1]
    index = np.argmin(widths, axis=0)
    low = np.take_along_axis(ordered, index[None], axis=0)[0]
    high = np.take_along_axis(ordered, (index + included - 1)[None], axis=0)[0]
    size = max(2, int(np.sqrt(len(x))))
    batches = len(x) // size
    batch_means = x[: batches * size].reshape((batches, size, *x.shape[1:])).mean(axis=1)
    mcse = batch_means.std(axis=0, ddof=1) / np.sqrt(batches)
    return ChainSummary(
        *[
            _freeze(np.asarray(v))
            for v in (
                x.mean(axis=0),
                np.median(x, axis=0),
                x.std(axis=0, ddof=1),
                np.stack((low, high), axis=-1),
                np.full(x.shape[1:], np.nan),
                mcse,
            )
        ]
    )


def fit_uaroet(
    counts: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    toxicity_only_counts: ArrayLike | None = None,
    initial_parameters: ArrayLike | None = None,
    initial_association: ArrayLike | None = None,
    monotone_efficacy: bool = True,
    monotone_toxicity: bool = True,
    association: float | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    rng: np.random.Generator,
    max_evaluations: int = 2_000_000,
) -> UAROETFit:
    """Fit UAROET using explicit independent normal priors and uniform rho.

    ``counts`` axes are dose, efficacy category, toxicity category. Optional
    ``toxicity_only_counts`` has dose-by-toxicity axes and counts additional
    patients whose toxicity is observed while efficacy is pending. These
    patients contribute the marginal toxicity likelihood and must not also
    appear in ``counts``. Efficacy-only and wholly pending patients contribute
    no likelihood under the current desktop observation contract. Every
    monotone increment has a normal prior truncated below at zero; other
    normal coordinates are untruncated. If association is omitted, rho has
    the Uniform(-1,1) prior and an independent-uniform Metropolis update. With
    no observations, exact independent prior draws replace MCMC burn-in.

    Optional initial_parameters is one parameter vector or a chain-by-parameter
    matrix; initial_association starts an inferred correlation at a supplied
    interior scalar or one value per chain. These affect initialization only, not
    the prior/model. One through twenty chains are supported; a single chain
    has unavailable (NaN) R-hat. Explicit work budgets may exceed the default
    two million evaluations, up to 128 million; retained-cell limits still apply.
    Analytic likelihood upper bounds can reject extreme proposals without
    evaluating unrepresentable cells. Every accepted state still uses the
    actual Gaussian-copula likelihood; unexplained numerical failures propagate.
    """
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit NumPy Generator")
    observed = _counts(counts)
    if toxicity_only_counts is None:
        partial = np.zeros((observed.shape[0], observed.shape[2]), dtype=np.int64)
    else:
        raw_partial = _real_matrix(toxicity_only_counts, "toxicity_only_counts")
        if (
            raw_partial.shape != (observed.shape[0], observed.shape[2])
            or np.any(raw_partial < 0)
            or np.any(raw_partial != np.floor(raw_partial))
            or np.any(raw_partial >= 2**53)
            or raw_partial.sum() + observed.sum() > _MAX_SUBJECTS
        ):
            raise ValueError(
                "toxicity_only_counts must be dose-by-toxicity nonnegative integers; "
                f"combined patient count must not exceed {_MAX_SUBJECTS}"
            )
        partial = raw_partial.astype(np.int64)
    if not isinstance(monotone_efficacy, (bool, np.bool_)) or not isinstance(
        monotone_toxicity, (bool, np.bool_)
    ):
        raise ValueError("monotonicity flags must be boolean")
    names = uaroet_parameter_names(
        observed.shape[0],
        observed.shape[1],
        observed.shape[2],
        monotone_efficacy=monotone_efficacy,
        monotone_toxicity=monotone_toxicity,
    )
    mean = _real_matrix(prior_mean, "prior_mean")
    sd = _real_matrix(prior_sd, "prior_sd")
    if mean.ndim != 1 or sd.shape != mean.shape or mean.size != len(names) or np.any(sd <= 0):
        raise ValueError("prior_mean and positive prior_sd must match uaroet_parameter_names")
    positive = _positive_mask(names)
    fixed_rho: float | None
    if association is None:
        fixed_rho = None
    else:
        rho_raw = np.asarray(association)
        if rho_raw.ndim != 0 or rho_raw.dtype.kind not in "iuf":
            raise ValueError("association must be None or a finite real scalar")
        fixed_rho = float(rho_raw)
        if not np.isfinite(fixed_rho) or not -1 < fixed_rho < 1:
            raise ValueError("fitted association must lie strictly inside (-1,1)")
    draw_count = _integer(draws, "draws", 8, _MAX_DRAWS)
    warmup_count = _integer(warmup, "warmup", 0, _MAX_WARMUP)
    chain_count = _integer(chains, "chains", 1, _MAX_CHAINS)
    evaluation_limit = _integer(max_evaluations, "max_evaluations", 1, _MAX_EVALUATIONS)
    retained_cells = chain_count * draw_count * int(observed.size)
    if retained_cells > _MAX_RETAINED:
        raise ValueError("retained joint posterior exceeds 2 million cells")
    direct = not bool(np.any(observed) or np.any(partial))
    starts: FloatArray | None = None
    if initial_parameters is not None:
        starts = _real_matrix(initial_parameters, "initial_parameters")
        if starts.shape == (len(names),):
            starts = np.broadcast_to(starts, (chain_count, len(names))).copy()
        if starts.shape != (chain_count, len(names)) or np.any(starts[:, positive] < 0):
            raise ValueError(
                "initial_parameters must match chains/coordinates and increment support"
            )
    start_rho: FloatArray | None = None
    if initial_association is not None:
        rho_start = _real_matrix(initial_association, "initial_association")
        if rho_start.ndim == 0:
            rho_start = np.full(chain_count, float(rho_start))
        if rho_start.shape != (chain_count,) or np.any(np.abs(rho_start) >= 1):
            raise ValueError(
                "initial_association must be an interior scalar or one value per chain"
            )
        if fixed_rho is not None:
            raise ValueError("initial_association applies only to an inferred correlation")
        start_rho = rho_start
    if direct and (starts is not None or start_rho is not None):
        raise ValueError("initial values are only meaningful with observed data")
    minimum_likelihood_calls = chain_count * (
        draw_count
        if direct
        else 1 + (1 if fixed_rho is not None else 2) * (warmup_count + draw_count)
    )
    if minimum_likelihood_calls > evaluation_limit:
        raise ValueError("max_evaluations is below the minimum likelihood-call workload")

    rho_for_shape = fixed_rho
    # Allocations follow the validated dimensions above; no data-dependent tensors.
    parameter_draws = np.empty((chain_count, draw_count, len(names)), dtype=float)
    rho_draws = np.empty((chain_count, draw_count), dtype=float)
    joint_draws = np.empty((chain_count, draw_count, *observed.shape), dtype=float)
    likelihood_draws = np.empty((chain_count, draw_count), dtype=float)
    budget = [0, 0]
    likelihood_calls = 0

    def evaluate(
        theta: FloatArray, rho: float, rejection_threshold: float | None = None
    ) -> tuple[float, FloatArray | None]:
        nonlocal likelihood_calls
        if budget[0] >= evaluation_limit:
            raise RuntimeError("UAROET likelihood/rectangle work exceeded max_evaluations")
        budget[0] += 1
        likelihood_calls += 1
        e_theta, t_theta = uaroet_logits(
            theta,
            dose_count=observed.shape[0],
            efficacy_levels=observed.shape[1],
            toxicity_levels=observed.shape[2],
            monotone_efficacy=bool(monotone_efficacy),
            monotone_toxicity=bool(monotone_toxicity),
        )
        log_e = _marginal_log_probabilities(e_theta)
        log_t = _marginal_log_probabilities(t_theta)
        if (
            rejection_threshold is not None
            and _likelihood_upper_bound(log_e, log_t, observed, partial) < rejection_threshold
        ):
            return -np.inf, None
        with np.errstate(under="ignore"):
            e, t = np.exp(log_e), np.exp(log_t)
        e_sums, t_sums = e.sum(axis=1, keepdims=True), t.sum(axis=1, keepdims=True)
        if np.any(e_sums == 0) or np.any(t_sums == 0):
            raise ArithmeticError("ordinal marginal mass is not representable")
        e /= e_sums
        t /= t_sums
        joint, _ = _joint_from_marginals(e, t, rho, budget=budget, max_evaluations=evaluation_limit)
        if rho == 0:
            log_joint = log_e[:, :, None] + log_t[:, None, :]
        else:
            with np.errstate(divide="ignore"):
                log_joint = np.log(joint)
        observed_mask = observed > 0
        missing = observed_mask & ~np.isfinite(log_joint)
        if np.any(missing):
            if (
                rejection_threshold is not None
                and _likelihood_upper_bound(log_e, log_t, observed, partial, rho)
                < rejection_threshold
            ):
                return -np.inf, None
            if abs(rho) < 1:
                raise ArithmeticError(
                    "an observed UAROET outcome has unrepresentable Gaussian-copula probability"
                )
            return -np.inf, joint
        likelihood = float(np.sum(observed[observed_mask] * log_joint[observed_mask]))
        partial_mask = partial > 0
        likelihood += float(np.sum(partial[partial_mask] * log_t[partial_mask]))
        if not np.isfinite(likelihood):
            if likelihood == -np.inf:
                return likelihood, joint
            raise ArithmeticError("UAROET log likelihood is not representable")
        return likelihood, joint

    acceptance = np.zeros(chain_count, dtype=float)
    for chain in range(chain_count):
        if direct:
            parameter_draws[chain] = _draw_prior(mean, sd, positive, draw_count, rng)
            if rho_for_shape is None:
                rho_draws[chain] = 2 * rng.random(draw_count) - 1
                endpoints = np.abs(rho_draws[chain]) == 1
                while np.any(endpoints):
                    rho_draws[chain, endpoints] = 2 * rng.random(int(endpoints.sum())) - 1
                    endpoints = np.abs(rho_draws[chain]) == 1
            else:
                rho_draws[chain] = rho_for_shape
            for draw in range(draw_count):
                likelihood, joint = evaluate(parameter_draws[chain, draw], rho_draws[chain, draw])
                assert joint is not None
                joint_draws[chain, draw] = joint
                likelihood_draws[chain, draw] = likelihood
            continue

        state = (
            _draw_prior(mean, sd, positive, 1, rng)[0] if starts is None else starts[chain].copy()
        )
        rho = (
            float(fixed_rho)
            if fixed_rho is not None
            else (float(start_rho[chain]) if start_rho is not None else float(2 * rng.random() - 1))
        )
        while abs(rho) == 1:
            rho = float(2 * rng.random() - 1)
        likelihood, joint = evaluate(state, rho)
        if not np.isfinite(likelihood):
            raise ArithmeticError("prior initialization has zero likelihood")
        assert joint is not None
        accepted_rho = 0
        total_iterations = warmup_count + draw_count
        for iteration in range(total_iterations):
            centered = state - mean
            direction = rng.normal(size=state.size) * sd
            height = likelihood + np.log1p(-rng.random())
            angle = float(rng.uniform(0, 2 * np.pi))
            lower, upper = angle - 2 * np.pi, angle
            for _ in range(1000):
                proposal = mean + centered * np.cos(angle) + direction * np.sin(angle)
                if np.any(proposal[positive] < 0):
                    proposal_likelihood = -np.inf
                else:
                    proposal_likelihood, proposal_joint = evaluate(proposal, rho, float(height))
                if proposal_likelihood >= height:
                    state = proposal
                    likelihood = proposal_likelihood
                    if not np.any(proposal[positive] < 0):
                        assert proposal_joint is not None
                        joint = proposal_joint
                    break
                if angle < 0:
                    lower = angle
                else:
                    upper = angle
                angle = float(rng.uniform(lower, upper))
            else:
                raise ArithmeticError(f"UAROET elliptical slice failed in chain {chain}")

            if fixed_rho is None:
                proposal_rho = float(2 * rng.random() - 1)
                while abs(proposal_rho) == 1:
                    proposal_rho = float(2 * rng.random() - 1)
                threshold = float(likelihood + np.log1p(-rng.random()))
                proposal_likelihood, proposal_joint = evaluate(state, proposal_rho, threshold)
                if threshold < proposal_likelihood:
                    assert proposal_joint is not None
                    rho, likelihood, joint = proposal_rho, proposal_likelihood, proposal_joint
                    if iteration >= warmup_count:
                        accepted_rho += 1
            if iteration >= warmup_count:
                draw = iteration - warmup_count
                parameter_draws[chain, draw] = state
                rho_draws[chain, draw] = rho
                assert joint is not None
                joint_draws[chain, draw] = joint
                likelihood_draws[chain, draw] = likelihood
        acceptance[chain] = accepted_rho / draw_count

    parameter_summary = _summarize_uaroet_chains(parameter_draws)
    association_summary = _summarize_uaroet_chains(rho_draws)
    return UAROETFit(
        names,
        _freeze(parameter_draws),
        _freeze(rho_draws),
        _freeze(joint_draws),
        _freeze(likelihood_draws),
        _freeze(acceptance),
        parameter_summary,
        association_summary,
        likelihood_calls,
        budget[1],
        budget[0],
        0 if direct else warmup_count,
        bool(monotone_efficacy),
        bool(monotone_toxicity),
        fixed_rho,
        direct,
    )
