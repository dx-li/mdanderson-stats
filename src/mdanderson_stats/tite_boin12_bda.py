"""Bayesian data augmentation for binary TITE-BOIN12 records.

The imputation conditionals follow the paper's working independence assumption
for endpoint event times. The joint Dirichlet prior is an explicit Python
input: the article's total concentration and two marginal means do not identify
the four-cell association parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar
from .boin12 import BOIN12Design, BOIN12Posterior
from .boin12 import posterior as boin12_posterior
from .hierarchical_binomial import ChainSummary, summarize_chains
from .tite_boin12 import _boin12_transition, _endpoint, _inputs, _readonly

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]

_MAX_WORK = 20_000_000
_MAX_RETAINED_VALUES = 2_000_000


@dataclass(frozen=True)
class TITEBOIN12BDADiagnostics:
    """Existing chain summaries for cell probabilities, counts and five metrics.

    The ``boin12_metrics`` final axis is toxicity tail, efficacy-futility tail,
    utility mean, utility probability and quasi-event count, in that order.
    """

    joint_probabilities: ChainSummary
    completed_joint_counts: ChainSummary
    boin12_metrics: ChainSummary


@dataclass(frozen=True)
class TITEBOIN12BDAPosterior:
    """BDA posterior output, with the Dirichlet and BOIN12 layers separated.

    ``joint_probability_draws`` are P-step draws of the four-cell multinomial
    probabilities. ``posterior`` instead averages the existing BOIN12
    quasi-Beta summaries over completed-data imputations. Its ``admissible``
    mask applies strict design cutoffs to the averaged tail probabilities.
    Patient-level imputation traces are not retained.
    """

    joint_probability_draws: FloatArray
    mean_completed_joint_counts: FloatArray
    posterior: BOIN12Posterior
    admissible: NDArray[np.bool_]
    diagnostics: TITEBOIN12BDADiagnostics
    prior_concentrations: FloatArray
    chains: int
    draws_per_chain: int


@dataclass(frozen=True)
class TITEBOIN12BDADecision:
    """One BDA-based TITE-BOIN12 next-dose decision."""

    action: str
    next_dose: int | None
    eliminated: NDArray[np.bool_]
    pending_counts: IntArray
    imputed_toxicity_rate: FloatArray | None
    posterior: TITEBOIN12BDAPosterior | None


def _readonly_bda(value: ArrayLike) -> FloatArray:
    return _readonly(value, np.float64)  # type: ignore[return-value]


def _prior(value: ArrayLike, k: int) -> FloatArray:
    prior = np.asarray(value, dtype=float)
    if prior.shape == (4,):
        prior = np.broadcast_to(prior, (k, 4)).copy()
    if prior.shape != (k, 4) or np.any(~np.isfinite(prior)) or np.any(prior <= 0):
        raise ValueError(
            "prior_concentrations must be positive finite values with shape (4,) or (n_doses,4)"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        totals = np.sum(prior, axis=1)
    if np.any(~np.isfinite(totals)):
        raise ValueError("prior concentration totals must be finite")
    return prior


def _categorical_rows(rng: np.random.Generator, log_weights: FloatArray) -> IntArray:
    """Draw row-wise categorical values without materializing patient traces."""
    row_max = np.max(log_weights, axis=1, keepdims=True)
    if np.any(~np.isfinite(row_max)):
        raise FloatingPointError("pending-outcome conditional has zero or unrepresentable mass")
    probabilities = np.exp(log_weights - row_max)
    row_total = np.sum(probabilities, axis=1, keepdims=True)
    if np.any(~np.isfinite(row_total)) or np.any(row_total <= 0):
        raise FloatingPointError("pending-outcome conditional has zero or unrepresentable mass")
    probabilities /= row_total
    uniforms = np.asarray(rng.random(size=probabilities.shape[0]), dtype=float)
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    return np.sum(uniforms[:, None] >= cumulative, axis=1).astype(np.int64)


def _dirichlet_draws(rng: np.random.Generator, prior: FloatArray, counts: IntArray) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        concentration = prior + counts
    if np.any(~np.isfinite(concentration)) or np.any(concentration <= 0):
        raise FloatingPointError("Dirichlet concentration is not representable")
    result = np.vstack([rng.dirichlet(concentration[j]) for j in range(prior.shape[0])])
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise FloatingPointError(
            "Dirichlet draw lost positive cell mass at floating-point precision"
        )
    return result


def _impute_pending(
    rng: np.random.Generator,
    dose_index: IntArray,
    toxicity: IntArray,
    efficacy: IntArray,
    tox_log_survival: FloatArray,
    eff_log_survival: FloatArray,
    probabilities: FloatArray,
) -> tuple[IntArray, IntArray]:
    """Apply one I step using conditional working independence of event times."""
    t = toxicity.copy()
    e = efficacy.copy()
    both = (t < 0) & (e < 0)
    if np.any(both):
        p = probabilities[dose_index[both]]
        log_wt, log_we = tox_log_survival[both], eff_log_survival[both]
        log_weights = np.column_stack(
            (
                np.log(p[:, 0]) + log_we,
                np.log(p[:, 1]),
                np.log(p[:, 2]) + log_wt + log_we,
                np.log(p[:, 3]) + log_wt,
            )
        )
        cells = _categorical_rows(rng, log_weights)
        t[both] = np.where((cells == 2) | (cells == 3), 1, 0)
        e[both] = np.where((cells == 0) | (cells == 2), 1, 0)

    tox_only = (t < 0) & (e >= 0)
    if np.any(tox_only):
        idx = dose_index[tox_only]
        observed_e = e[tox_only]
        p = probabilities[idx]
        log_wt = tox_log_survival[tox_only]
        no_t = np.log(np.where(observed_e == 1, p[:, 0], p[:, 1]))
        yes_t = np.log(np.where(observed_e == 1, p[:, 2], p[:, 3])) + log_wt
        maximum = np.maximum(no_t, yes_t)
        probability_t = np.exp(yes_t - maximum) / (np.exp(no_t - maximum) + np.exp(yes_t - maximum))
        t[tox_only] = (rng.random(idx.size) < probability_t).astype(np.int64)

    eff_only = (t >= 0) & (e < 0)
    if np.any(eff_only):
        idx = dose_index[eff_only]
        observed_t = t[eff_only]
        p = probabilities[idx]
        log_we = eff_log_survival[eff_only]
        no_e = np.log(np.where(observed_t == 0, p[:, 1], p[:, 3]))
        yes_e = np.log(np.where(observed_t == 0, p[:, 0], p[:, 2])) + log_we
        maximum = np.maximum(no_e, yes_e)
        probability_e = np.exp(yes_e - maximum) / (np.exp(no_e - maximum) + np.exp(yes_e - maximum))
        e[eff_only] = (rng.random(idx.size) < probability_e).astype(np.int64)
    return t, e


def _inputs_valid(
    design: BOIN12Design,
    rng: np.random.Generator,
    draws: int,
    warmup: int,
    chains: int,
    max_work: int,
) -> None:
    if not isinstance(design, BOIN12Design):
        raise ValueError("design must be a BOIN12Design")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    settings = ((draws, "draws", 20), (warmup, "warmup", 0), (chains, "chains", 2))
    for value, name, minimum in settings:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if isinstance(max_work, (bool, np.bool_)) or not isinstance(max_work, (int, np.integer)):
        raise ValueError("max_work must be an integer")
    if max_work < 1 or max_work > _MAX_WORK:
        raise ValueError(f"max_work must lie in [1,{_MAX_WORK}]")


def tite_boin12_bda_posterior(
    design: BOIN12Design,
    doses: ArrayLike,
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    toxicity_followup: ArrayLike,
    efficacy_followup: ArrayLike,
    *,
    toxicity_window: float,
    efficacy_window: float,
    n_doses: int,
    prior_concentrations: ArrayLike,
    rng: np.random.Generator,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    max_work: int = _MAX_WORK,
) -> TITEBOIN12BDAPosterior:
    """Run a bounded Gibbs sampler for pending binary TITE-BOIN12 outcomes.

    Cell order is ``(noT/efficacy, noT/noE, T/efficacy, T/noE)``. The explicit
    Dirichlet concentration is shared across doses as a length-four vector or
    supplied dose by dose as ``(n_doses, 4)``. Each pending event time has a
    uniform assessment-window weight; conditional event-time independence is
    the paper's working imputation assumption. ``rng``, prior and chain settings
    are required/visible Python choices; they are not native defaults.

    Retained P-step Dirichlet draws are returned separately from BOIN12
    quasi-Beta posterior summaries, which are averaged over each retained
    completed-data imputation. At least two chains and twenty retained draws
    are required for the reported split-Rhat and batch-means MCSE diagnostics.
    """
    _inputs_valid(design, rng, draws, warmup, chains, max_work)
    d, t, e, tf, ef, tw, ew, k = _inputs(
        doses,
        toxicity,
        efficacy,
        toxicity_followup,
        efficacy_followup,
        toxicity_window,
        efficacy_window,
        n_doses,
    )
    prior = _prior(prior_concentrations, k)
    n = d.size
    dose_index = d - 1
    tox_log_survival = np.zeros(n, dtype=float)
    eff_log_survival = np.zeros(n, dtype=float)
    tox_pending, eff_pending = t == -1, e == -1
    tox_log_survival[tox_pending] = np.log(tw - tf[tox_pending]) - np.log(tw)
    eff_log_survival[eff_pending] = np.log(ew - ef[eff_pending]) - np.log(ew)
    n_iter = int(warmup) + int(draws)
    work = int(chains) * (n_iter + 1) * (max(1, n) + k)
    if work > max_work:
        raise ValueError(f"BDA work estimate {work} exceeds max_work={max_work}")
    # Includes retained P draws/counts/metrics plus conservative chain-summary
    # sorting, splitting, batch and interval scratch arrays.
    stored_values = int(chains) * int(draws) * k * 40
    if stored_values > _MAX_RETAINED_VALUES:
        raise ValueError(
            f"retained draws and summaries require {stored_values} values; "
            f"limit is {_MAX_RETAINED_VALUES}"
        )
    retained_p = np.empty((chains, draws, k, 4), dtype=float)
    # Metrics 0:5 are BOIN12 summaries; 5:9 are completed joint counts.
    retained_metrics = np.empty((chains, draws, k, 9), dtype=float)
    utilities = np.asarray(design.utilities, dtype=float)
    for chain in range(chains):
        t_chain, e_chain = t.copy(), e.copy()
        with np.errstate(under="ignore", invalid="ignore"):
            initial_p = prior / np.sum(prior, axis=1, keepdims=True)
        if np.any(~np.isfinite(initial_p)) or np.any(initial_p <= 0):
            raise FloatingPointError(
                "Dirichlet prior proportions lose positive cell mass at floating-point precision"
            )
        t_chain, e_chain = _impute_pending(
            rng, dose_index, t_chain, e_chain, tox_log_survival, eff_log_survival, initial_p
        )
        counts = np.zeros((k, 4), dtype=np.int64)
        np.add.at(counts, (dose_index, _cell_index(t_chain, e_chain)), 1)
        p = _dirichlet_draws(rng, prior, counts)
        saved = 0
        for iteration in range(n_iter):
            t_chain, e_chain = _impute_pending(
                rng, dose_index, t, e, tox_log_survival, eff_log_survival, p
            )
            counts.fill(0)
            np.add.at(counts, (dose_index, _cell_index(t_chain, e_chain)), 1)
            p = _dirichlet_draws(rng, prior, counts)
            if iteration < warmup:
                continue
            retained_p[chain, saved] = p
            tox_counts = counts[:, 2] + counts[:, 3]
            eff_counts = counts[:, 0] + counts[:, 2]
            eff_without_tox = counts[:, 0]
            summary = boin12_posterior(
                counts.sum(axis=1),
                tox_counts,
                eff_counts,
                toxicity_limit=design.toxicity_limit,
                efficacy_limit=design.efficacy_limit,
                utilities=utilities,
                efficacy_without_toxicity=eff_without_tox,
            )
            retained_metrics[chain, saved, :, :5] = np.column_stack(
                (
                    summary.toxicity_overdose_probability,
                    summary.efficacy_futility_probability,
                    summary.utility_mean,
                    summary.utility_probability,
                    summary.utility_events,
                )
            )
            retained_metrics[chain, saved, :, 5:] = counts
            saved += 1

    metric_means = np.mean(retained_metrics, axis=(0, 1))
    tox_over = metric_means[:, 0]
    eff_futile = metric_means[:, 1]
    utility_mean = metric_means[:, 2]
    utility_probability = metric_means[:, 3]
    utility_events = metric_means[:, 4]
    posterior = BOIN12Posterior(
        _readonly_bda(tox_over),
        _readonly_bda(eff_futile),
        _readonly_bda(utility_mean),
        _readonly_bda(utility_probability),
        _readonly_bda(utility_events),
    )
    metric_names = retained_metrics[:, :, :, :5]
    count_trace = retained_metrics[:, :, :, 5:]
    with np.errstate(divide="ignore", invalid="ignore"):
        diagnostics = TITEBOIN12BDADiagnostics(
            summarize_chains(retained_p),
            summarize_chains(count_trace),
            summarize_chains(metric_names),
        )
    return TITEBOIN12BDAPosterior(
        _readonly_bda(retained_p),
        _readonly_bda(metric_means[:, 5:]),
        posterior,
        _readonly(
            (tox_over < design.toxicity_cutoff) & (eff_futile < design.efficacy_cutoff),
            np.bool_,
        ),
        diagnostics,
        _readonly_bda(prior),
        int(chains),
        int(draws),
    )


def tite_boin12_bda_decision(
    design: BOIN12Design,
    doses: ArrayLike,
    toxicity: ArrayLike,
    efficacy: ArrayLike,
    toxicity_followup: ArrayLike,
    efficacy_followup: ArrayLike,
    *,
    toxicity_window: float,
    efficacy_window: float,
    n_doses: int,
    current_dose: int,
    prior_concentrations: ArrayLike,
    rng: np.random.Generator,
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    max_work: int = _MAX_WORK,
    eliminated: ArrayLike | None = None,
    max_pending_toxicity: float = 0.5,
    max_pending_efficacy: float = 0.5,
    run_in_3plus3: bool = False,
) -> TITEBOIN12BDADecision:
    """Make a next-dose decision from BDA-averaged completed-data summaries.

    The source's >50% pending gate is applied at the current dose before any
    sampler work. BDA uses the posterior-averaged imputed toxicity rate and
    BOIN12 utility-probability summaries for movement. The optional run-in and
    precision-stop ordering mirror the Python AL conduct policy; the paper does
    not prescribe those BDA-specific computational/conduct conventions.
    """
    _inputs_valid(design, rng, draws, warmup, chains, max_work)
    ct = scalar(design.toxicity_cutoff, "toxicity_cutoff")
    ce = scalar(design.efficacy_cutoff, "efficacy_cutoff")
    if not 0 < ct < 1 or not 0 < ce < 1:
        raise ValueError("toxicity_cutoff and efficacy_cutoff must lie in (0,1)")
    if not isinstance(run_in_3plus3, (bool, np.bool_)):
        raise ValueError("run_in_3plus3 must be a boolean")
    if run_in_3plus3 and design.toxicity_limit != 0.25:
        raise ValueError("the 3+3 run-in is available only when toxicity_limit is 0.25")
    current_value = scalar(current_dose, "current_dose")
    if current_value != int(current_value):
        raise ValueError("current_dose must be an integer dose index in [1,n_doses]")
    d, t, e, tf, ef, tw, ew, k = _inputs(
        doses,
        toxicity,
        efficacy,
        toxicity_followup,
        efficacy_followup,
        toxicity_window,
        efficacy_window,
        n_doses,
    )
    if not 1 <= int(current_value) <= k:
        raise ValueError("current_dose must be an integer dose index in [1,n_doses]")
    prior = _prior(prior_concentrations, k)
    excluded = np.zeros(k, dtype=bool) if eliminated is None else np.asarray(eliminated)
    if excluded.shape != (k,) or not np.all(np.isin(excluded, (False, True, 0, 1))):
        raise ValueError("eliminated must match n_doses")
    excluded = excluded.astype(bool)
    n_t, _, _, pending_t, _, _, _ = _endpoint(t, tf, d, tw, k)
    n_e, _, _, pending_e, _, _, _ = _endpoint(e, ef, d, ew, k)
    pending_counts = np.column_stack((pending_t, pending_e)).astype(np.int64)
    current = int(current_value) - 1
    if np.all(excluded):
        return TITEBOIN12BDADecision(
            "stop_safety",
            None,
            _readonly(excluded, np.bool_),
            _readonly(pending_counts, np.int64),
            None,
            None,
        )
    if n_t[current] == 0 or n_e[current] == 0:
        raise ValueError("current_dose must identify a treated dose")
    max_pending_toxicity = scalar(max_pending_toxicity, "max_pending_toxicity")
    max_pending_efficacy = scalar(max_pending_efficacy, "max_pending_efficacy")
    if not 0 <= max_pending_toxicity <= 1 or not 0 <= max_pending_efficacy <= 1:
        raise ValueError("pending thresholds must lie in [0,1]")
    frac_t = pending_t[current] / n_t[current]
    frac_e = pending_e[current] / n_e[current]
    if frac_t > max_pending_toxicity or frac_e > max_pending_efficacy:
        return TITEBOIN12BDADecision(
            "suspend_pending",
            None,
            _readonly(excluded, np.bool_),
            _readonly(pending_counts, np.int64),
            None,
            None,
        )
    result = tite_boin12_bda_posterior(
        design,
        d,
        t,
        e,
        tf,
        ef,
        toxicity_window=tw,
        efficacy_window=ew,
        n_doses=k,
        prior_concentrations=prior,
        rng=rng,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_work=max_work,
    )
    allowed = result.admissible & ~excluded
    patients = n_t
    completed_toxicities = np.sum(result.mean_completed_joint_counts[:, 2:], axis=1)
    toxicity_rate = np.full(k, np.nan)
    np.divide(completed_toxicities, patients, out=toxicity_rate, where=patients > 0)
    if not np.any(allowed):
        return TITEBOIN12BDADecision(
            "stop_safety",
            None,
            _readonly(~allowed, np.bool_),
            _readonly(pending_counts, np.int64),
            _readonly_bda(toxicity_rate),
            result,
        )
    observed_dlt_count = int(np.count_nonzero((d == current + 1) & (t == 1)))
    action, next_dose = _boin12_transition(
        design,
        current,
        patients,
        toxicity_rate,
        result.posterior.utility_probability,
        allowed,
        observed_dlt_count,
        run_in_3plus3=run_in_3plus3,
    )
    return TITEBOIN12BDADecision(
        action,
        next_dose,
        _readonly(~allowed, np.bool_),
        _readonly(pending_counts, np.int64),
        _readonly_bda(toxicity_rate),
        result,
    )


def _cell_index(toxicity: IntArray, efficacy: IntArray) -> IntArray:
    """Map binary outcomes into the public BOIN12 four-cell order."""
    return np.select(
        (
            (toxicity == 0) & (efficacy == 1),
            (toxicity == 0) & (efficacy == 0),
            (toxicity == 1) & (efficacy == 1),
            (toxicity == 1) & (efficacy == 0),
        ),
        (0, 1, 2, 3),
        default=-1,
    ).astype(np.int64)
