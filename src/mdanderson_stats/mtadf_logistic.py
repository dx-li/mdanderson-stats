"""Global and local Bayesian logistic OBD methods from Zang, Lee and Yuan.

The posterior models and priors follow the paper. The sampler and unspecified
conduct choices are explicit Python conventions, not claims of native-app
parity.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from ._cdflib import _freeze
from ._validation import FloatArray, count, finite, scalar
from .hierarchical_binomial import ChainSummary, summarize_chains
from .mtadf import MTADFDecision, MTADFPrior, mtadf_decision

_MAX_DOSES = 20
_MAX_SUBJECTS = 10_000
_MAX_RETAINED_CELLS = 2_000_000


def _positive_int(value: int, name: str, low: int, high: int) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu" or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    result = int(raw)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return result


def _freeze_bool(value: ArrayLike) -> NDArray[np.bool_]:
    result = np.array(value, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _counts(
    subjects: ArrayLike, responses: ArrayLike, *, allow_empty: bool = False
) -> tuple[FloatArray, FloatArray]:
    if len(np.shape(subjects)) != 1 or np.shape(subjects) != np.shape(responses):
        raise ValueError("subjects and responses must be matching one-dimensional vectors")
    if not 1 <= np.shape(subjects)[0] <= _MAX_DOSES:
        raise ValueError(f"subjects and responses must have 1..{_MAX_DOSES} dose entries")
    n = count(subjects, "subjects")
    y = count(responses, "responses")
    if n.ndim != 1 or not 1 <= n.size <= _MAX_DOSES or y.shape != n.shape:
        raise ValueError(
            f"subjects and responses must be matching vectors of 1..{_MAX_DOSES} doses"
        )
    if np.any(y > n) or np.any(n > _MAX_SUBJECTS) or n.sum() > _MAX_SUBJECTS:
        raise ValueError(
            "responses cannot exceed subjects and total subjects are limited to 10,000"
        )
    if not allow_empty and not np.any(n):
        raise ValueError("posterior fitting requires at least one observed subject")
    return n.astype(float), y.astype(float)


def _doses(value: ArrayLike, size: int) -> FloatArray:
    if len(np.shape(value)) != 1 or np.shape(value) != (size,):
        raise ValueError("doses must be a matching one-dimensional vector")
    d = finite(value, "doses")
    if d.ndim != 1 or d.size != size or np.any(np.diff(d) <= 0):
        raise ValueError("doses must be a finite, strictly increasing vector matching counts")
    return d


def _mcmc_options(draws: int, warmup: int, chains: int) -> tuple[int, int, int]:
    draws = _positive_int(draws, "draws", 8, 100_000)
    warmup = _positive_int(warmup, "warmup", 0, 100_000) if warmup != 0 else 0
    chains = _positive_int(chains, "chains", 2, 16)
    return draws, warmup, chains


def _preflight_work(
    draws: int, warmup: int, chains: int, dose_count: int, parameter_count: int
) -> None:
    # Count full retained arrays plus two transient summary arrays and one
    # additional safety margin for sort/interval temporaries.
    if chains * draws * (parameter_count + 3 * dose_count) > _MAX_RETAINED_CELLS:
        raise ValueError("posterior draws and summary temporaries exceed the 2 million cell bound")
    if chains * (draws + warmup) * dose_count > 20_000_000:
        raise ValueError(
            "posterior transition workload exceeds the 20 million dose-evaluation bound"
        )


def _local_window_indices(current: int, length: int, dose_count: int) -> NDArray[np.int64]:
    """Use preceding levels where possible; use the first window at the low boundary."""
    start = min(max(0, current - length + 1), dose_count - length)
    return np.arange(start, start + length, dtype=np.int64)


def _sampler(
    design: FloatArray,
    n: FloatArray,
    y: FloatArray,
    prior_scales: FloatArray,
    *,
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Random-walk Metropolis in prior-scale coordinates; warmup adapts scale."""
    dimension = design.shape[1]
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    if draws * chains * (dimension + design.shape[0]) > _MAX_RETAINED_CELLS:
        raise ValueError("combined retained parameter and probability draws exceed storage bound")
    parameters = np.empty((chains, draws, dimension), dtype=float)
    probabilities = np.empty((chains, draws, design.shape[0]), dtype=float)
    accept_rate = np.empty(chains, dtype=float)
    log_prior_constant = -np.log(np.pi * prior_scales)

    def log_target(z: FloatArray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            theta = z * prior_scales
            eta = design @ theta
        if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(theta)):
            return -np.inf
        ll = np.sum(y * -np.logaddexp(0.0, -eta) + (n - y) * -np.logaddexp(0.0, eta))
        # Independent Cauchy(0, prior_scale) priors in original coefficients.
        lp = np.sum(log_prior_constant - 2 * np.log(np.hypot(1.0, z)))
        return float(ll + lp)

    target = 0.30
    for chain in range(chains):
        state: FloatArray = np.asarray(rng.normal(size=dimension), dtype=float)
        current = log_target(state)
        if not np.isfinite(current):
            raise ArithmeticError("initial posterior state is not representable")
        scale = 0.65
        accepted = 0
        total = warmup + draws
        for iteration in range(total):
            proposal = state + rng.normal(size=dimension) * scale
            candidate = log_target(proposal)
            was_accepted = np.log(rng.random()) < candidate - current
            if was_accepted:
                state, current = proposal, candidate
                if iteration >= warmup:
                    accepted += 1
            if iteration < warmup:
                # Diminishing adaptation, only during warmup.
                rate = 1.0 / np.sqrt(iteration + 10.0)
                scale = float(
                    np.clip(
                        np.exp(np.log(scale) + rate * (float(was_accepted) - target)),
                        0.02,
                        5.0,
                    )
                )
            else:
                index = iteration - warmup
                theta = state * prior_scales
                parameters[chain, index] = theta
                probabilities[chain, index] = expit(design @ theta)
        accept_rate[chain] = accepted / draws
    return _freeze(parameters), _freeze(probabilities), _freeze(accept_rate)


@dataclass(frozen=True)
class MTADFLogisticPosterior:
    """Posterior draws for the global quadratic logistic efficacy model."""

    doses: FloatArray
    subjects: FloatArray
    responses: FloatArray
    parameter_draws: FloatArray
    efficacy_draws: FloatArray
    efficacy_summary: ChainSummary
    acceptance_rate: FloatArray
    warmup: int

    @property
    def posterior_mean_efficacy(self) -> FloatArray:
        return self.efficacy_summary.mean


@dataclass(frozen=True)
class MTADFLocalLogisticPosterior:
    """Posterior draws for one local linear logistic window."""

    window_doses: FloatArray
    window_subjects: FloatArray
    window_responses: FloatArray
    parameter_draws: FloatArray
    efficacy_draws: FloatArray
    parameter_summary: ChainSummary
    efficacy_summary: ChainSummary
    positive_slope_draws: NDArray[np.bool_]
    positive_slope_summary: ChainSummary
    probability_positive_slope: float
    probability_positive_slope_mcse: float
    acceptance_rate: FloatArray
    warmup: int
    window_length: int


def mtadf_logistic_posterior(
    subjects: ArrayLike,
    responses: ArrayLike,
    doses: ArrayLike,
    *,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> MTADFLogisticPosterior:
    """Fit the paper's global quadratic logistic efficacy model.

    Uses logit(p)=alpha+beta*d+gamma*d² with independent Cauchy(0,10),
    Cauchy(0,2.5), Cauchy(0,2.5) priors. Dose values are used exactly as
    supplied because rescaling changes these priors. A random-walk Metropolis
    sampler with warmup-only scale adaptation is a transparent Python choice;
    the paper specifies MCMC but not a particular sampler.
    """
    n, y = _counts(subjects, responses)
    d = _doses(doses, n.size)
    draws, warmup, chains = _mcmc_options(draws, warmup, chains)
    with np.errstate(over="ignore", invalid="ignore"):
        design = np.column_stack((np.ones(d.size), d, d * d))
    if not np.all(np.isfinite(design)):
        raise ValueError("dose coding produces a non-finite quadratic model matrix")
    _preflight_work(draws, warmup, chains, d.size, 3)
    parameters, efficacy, acceptance = _sampler(
        design,
        n,
        y,
        np.array([10.0, 2.5, 2.5]),
        draws=draws,
        warmup=warmup,
        chains=chains,
        rng=rng,
    )
    return MTADFLogisticPosterior(
        _freeze(d),
        _freeze(n),
        _freeze(y),
        parameters,
        efficacy,
        summarize_chains(efficacy),
        acceptance,
        warmup,
    )


def mtadf_local_logistic_posterior(
    subjects: ArrayLike,
    responses: ArrayLike,
    doses: ArrayLike,
    *,
    current_dose: int,
    window_length: int = 2,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator,
) -> MTADFLocalLogisticPosterior:
    """Fit the paper's local linear logistic model at ``current_dose``.

    The local window includes the current dose and the preceding
    ``window_length-1`` dose levels. Only observed counts in that window
    contribute to the likelihood; priors are independent Cauchy(0,10) for
    alpha and Cauchy(0,2.5) for beta.
    """
    n, y = _counts(subjects, responses)
    d = _doses(doses, n.size)
    current = _positive_int(current_dose, "current_dose", 0, n.size - 1)
    length = _positive_int(window_length, "window_length", 2, n.size)
    draws, warmup, chains = _mcmc_options(draws, warmup, chains)
    _preflight_work(draws, warmup, chains, length, 2)
    indices = _local_window_indices(current, length, n.size)
    if np.any(n[indices] == 0):
        raise ValueError("all dose levels in the local window must have observations")
    window = d[indices]
    design = np.column_stack((np.ones(length), window))
    if not np.all(np.isfinite(design)):
        raise ValueError("dose coding produces a non-finite local model matrix")
    parameters, efficacy, acceptance = _sampler(
        design,
        n[indices],
        y[indices],
        np.array([10.0, 2.5]),
        draws=draws,
        warmup=warmup,
        chains=chains,
        rng=rng,
    )
    beta = parameters[:, :, 1]
    positive = beta > 0
    probability = float(np.mean(positive))
    positive_summary = summarize_chains(positive)
    mcse = float(positive_summary.batch_mean_mcse)
    return MTADFLocalLogisticPosterior(
        _freeze(window),
        _freeze(n[indices]),
        _freeze(y[indices]),
        parameters,
        efficacy,
        summarize_chains(parameters),
        summarize_chains(efficacy),
        _freeze_bool(positive),
        positive_summary,
        probability,
        mcse,
        acceptance,
        warmup,
        length,
    )


@dataclass(frozen=True)
class MTADFLogisticDecision:
    """One global-model action plus the posterior and toxicity decision."""

    action: str
    dose: int | None
    reason: str
    efficacy_mean: FloatArray
    toxicity: MTADFDecision
    posterior: MTADFLogisticPosterior | None


@dataclass(frozen=True)
class MTADFLocalLogisticDecision:
    """One local-model action, retaining posterior and safety diagnostics."""

    action: str
    dose: int | None
    reason: str
    toxicity: MTADFDecision
    posterior: MTADFLocalLogisticPosterior | None
    final_decision: MTADFDecision | None


def mtadf_logistic_decision(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    doses: ArrayLike,
    *,
    current_dose: int | None,
    posterior: MTADFLogisticPosterior | None = None,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    prior: MTADFPrior | None = None,
    final: bool = False,
) -> MTADFLogisticDecision:
    """Apply a transparent one-step global-model policy to a posterior fit.

    The policy ranks safe doses by posterior-mean efficacy (lowest index wins
    ties), moves one level toward that target, stops if all doses are unsafe,
    and drops an unsafe current dose to the highest admissible dose. This fills
    policy details omitted by the paper and does not assert native parity.
    """
    n, y = _counts(subjects, responses, allow_empty=True)
    d = _doses(doses, n.size)
    if not np.any(n) and not final:
        toxicity = mtadf_decision(
            n,
            toxicities,
            y,
            current_dose=None,
            final=False,
            toxicity_limit=toxicity_limit,
            safety_cutoff=safety_cutoff,
            prior=prior,
        )
        safe = np.flatnonzero(toxicity.admissible)
        if safe.size == 0:
            return MTADFLogisticDecision(
                "stop", None, "no_admissible_dose", _freeze(np.full(n.size, np.nan)), toxicity, None
            )
        return MTADFLogisticDecision(
            "start",
            int(safe[0]),
            "lowest_admissible_dose",
            _freeze(np.full(n.size, np.nan)),
            toxicity,
            None,
        )
    if posterior is None:
        raise ValueError("posterior fit is required after enrollment has begun")
    if (
        not isinstance(posterior, MTADFLogisticPosterior)
        or not np.array_equal(posterior.doses, d)
        or not np.array_equal(posterior.subjects, n)
        or not np.array_equal(posterior.responses, y)
    ):
        raise ValueError("posterior fit must match dose coding and efficacy counts")
    current = (
        None if current_dose is None else _positive_int(current_dose, "current_dose", 0, n.size - 1)
    )
    anchor = current
    if np.any(n) and (anchor is None or n[anchor] == 0):
        anchor = int(np.flatnonzero(n)[0])
    toxicity = mtadf_decision(
        n,
        toxicities,
        y,
        current_dose=anchor,
        final=final,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
        prior=prior,
    )
    mean = posterior.posterior_mean_efficacy
    safe = np.flatnonzero(toxicity.admissible)
    if safe.size == 0:
        action, dose, reason = "stop", None, "no_admissible_dose"
    elif final:
        action, dose, reason = (
            "select_obd",
            int(safe[np.argmax(mean[safe])]),
            "highest_safe_posterior_mean",
        )
    elif current is None:
        dose = int(safe[0])
        action, reason = "start", "lowest_admissible_dose"
    elif not toxicity.admissible[current]:
        dose = int(safe[-1])
        action, reason = "treat", "unsafe_current_drop_to_highest_admissible"
    else:
        target = int(safe[np.argmax(mean[safe])])
        dose = current + int(np.sign(target - current))
        if not toxicity.admissible[dose]:
            dose = current
        action, reason = ("treat", "one_step_toward_posterior_mean_peak")
    return MTADFLogisticDecision(action, dose, reason, mean, toxicity, posterior)


def mtadf_local_logistic_decision(
    subjects: ArrayLike,
    toxicities: ArrayLike,
    responses: ArrayLike,
    doses: ArrayLike,
    *,
    current_dose: int | None,
    window_length: int = 2,
    efficacy_escalation_cutoff: float = 0.4,
    efficacy_deescalation_cutoff: float = 0.3,
    posterior: MTADFLocalLogisticPosterior | None = None,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 4,
    rng: np.random.Generator | None = None,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    prior: MTADFPrior | None = None,
    final: bool = False,
) -> MTADFLocalLogisticDecision:
    """Apply L-logistic slope actions, toxicity limits and final isotonic OBD.

    The default efficacy cutoffs match the paper's illustration; the caller
    should calibrate them for the intended operating characteristics. At a
    dose boundary movement is clamped. Unsafe doses are never recommended.
    """
    n, y = _counts(subjects, responses, allow_empty=True)
    d = _doses(doses, n.size)
    if final:
        final_fit = mtadf_decision(
            n,
            toxicities,
            y,
            final=True,
            toxicity_limit=toxicity_limit,
            safety_cutoff=safety_cutoff,
            prior=prior,
        )
        return MTADFLocalLogisticDecision(
            final_fit.action,
            final_fit.dose,
            "final_double_sided_isotonic",
            final_fit,
            None,
            final_fit,
        )
    current = (
        None if current_dose is None else _positive_int(current_dose, "current_dose", 0, n.size - 1)
    )
    if not np.any(n) and not final:
        toxicity = mtadf_decision(
            n,
            toxicities,
            y,
            current_dose=None,
            final=False,
            toxicity_limit=toxicity_limit,
            safety_cutoff=safety_cutoff,
            prior=prior,
        )
        safe = np.flatnonzero(toxicity.admissible)
        if safe.size == 0:
            return MTADFLocalLogisticDecision(
                "stop", None, "no_admissible_dose", toxicity, None, None
            )
        start = 0 if toxicity.admissible[0] else int(safe[0])
        return MTADFLocalLogisticDecision(
            "start", start, "initial_lowest_admissible_dose", toxicity, None, None
        )
    if current is None:
        raise ValueError("current_dose is required after enrollment has begun")
    anchor = current if n[current] else int(np.flatnonzero(n)[0])
    toxicity = mtadf_decision(
        n,
        toxicities,
        y,
        current_dose=anchor,
        final=final,
        toxicity_limit=toxicity_limit,
        safety_cutoff=safety_cutoff,
        prior=prior,
    )
    if not np.any(toxicity.admissible):
        return MTADFLocalLogisticDecision("stop", None, "no_admissible_dose", toxicity, None, None)
    ce1, ce2 = (
        scalar(efficacy_escalation_cutoff, "efficacy_escalation_cutoff"),
        scalar(efficacy_deescalation_cutoff, "efficacy_deescalation_cutoff"),
    )
    if not 0 <= ce2 < ce1 <= 1:
        raise ValueError("efficacy cutoffs must satisfy 0 <= de-escalation < escalation <= 1")
    length = _positive_int(window_length, "window_length", 2, n.size)
    if not np.all(n[:length] > 0):
        missing = int(np.flatnonzero(n[:length] == 0)[0])
        dose = missing
        if not toxicity.admissible[dose]:
            return MTADFLocalLogisticDecision(
                "stop", None, "unsafe_initial_ramp_dose", toxicity, None, None
            )
        return MTADFLocalLogisticDecision(
            "start", dose, "initial_local_dose_ramp", toxicity, None, None
        )
    if posterior is None:
        if rng is None:
            raise ValueError("rng is required when posterior is not supplied")
        posterior = mtadf_local_logistic_posterior(
            n,
            y,
            d,
            current_dose=current,
            window_length=length,
            draws=draws,
            warmup=warmup,
            chains=chains,
            rng=rng,
        )
    elif (
        posterior.window_length != length
        or not np.array_equal(
            posterior.window_doses, d[_local_window_indices(current, length, n.size)]
        )
        or not np.array_equal(
            posterior.window_subjects, n[_local_window_indices(current, length, n.size)]
        )
        or not np.array_equal(
            posterior.window_responses, y[_local_window_indices(current, length, n.size)]
        )
    ):
        raise ValueError("posterior fit does not match the current local window")
    p_positive = posterior.probability_positive_slope
    desired = current + 1 if p_positive > ce1 else current - 1 if p_positive < ce2 else current
    if desired > current and current + 1 < n.size and n[current + 1] > 0:
        if rng is None:
            raise ValueError("rng is required to evaluate the local bounce guard")
        next_posterior = mtadf_local_logistic_posterior(
            n,
            y,
            d,
            current_dose=current + 1,
            window_length=length,
            draws=draws,
            warmup=warmup,
            chains=chains,
            rng=rng,
        )
        if next_posterior.probability_positive_slope < ce2:
            desired = current
    desired = min(max(desired, 0), n.size - 1)
    if not toxicity.admissible[current]:
        lower_safe = np.flatnonzero(toxicity.admissible[:current])
        if lower_safe.size:
            desired = int(lower_safe[-1])
        else:
            return MTADFLocalLogisticDecision(
                "stop", None, "unsafe_current_without_safe_lower_dose", toxicity, posterior, None
            )
    elif not toxicity.admissible[desired]:
        desired = current
    return MTADFLocalLogisticDecision(
        "treat", int(desired), "local_slope_policy", toxicity, posterior, None
    )
