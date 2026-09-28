"""Bounded operating-characteristic simulation for the BaCIS workflow.

This simulates independent binomial subgroup outcomes and runs the existing
two-stage :func:`bacis_fit` analysis once per replication. It does not
calibrate cutoffs or claim parity with the unpublished native simulation
driver. The efficacy decision is ``P(p_i > phi_low) > efficacy_cutoff``;
classification and efficacy are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray, scalar
from .bacis import _integer, _probability, _readonly, bacis_fit

_MAX_GROUPS = 100
_MAX_REPLICATIONS = 1_000
_MAX_TOTAL_SAMPLER_STEPS = 20_000_000
_MAX_RESULT_CELLS = 2_000_000


def _generator(value: int | np.random.Generator | None, name: str) -> np.random.Generator:
    if isinstance(value, np.random.Generator):
        return value
    if value is None or (isinstance(value, (int, np.integer)) and not isinstance(value, bool)):
        return np.random.default_rng(value)
    raise ValueError(f"{name} must be an integer seed, Generator, or None")


@dataclass(frozen=True)
class BaCISOperatingCharacteristics:
    """Compact replication results and empirical operating characteristics.

    ``successes``, ``high_cluster`` and ``efficacious`` retain only subgroup
    counts and binary outcomes, not the per-replication posterior fits. The
    familywise false-positive rate is the chance that at least one subgroup
    with truth ``<= phi_low`` is efficacious; it is ``None`` when there are no
    null subgroups. ``power`` is defined for truth ``> phi_low``, including
    intermediate rates between the two classification centers.
    """

    true_response_rates: FloatArray
    trials_per_group: NDArray[np.int64]
    successes: NDArray[np.int64]
    high_cluster: NDArray[np.bool_]
    efficacious: NDArray[np.bool_]
    null_groups: NDArray[np.bool_]
    classification_high_probability: FloatArray
    classification_high_mcse: FloatArray
    efficacy_probability: FloatArray
    efficacy_mcse: FloatArray
    false_positive_rate: FloatArray
    false_positive_mcse: FloatArray
    power: FloatArray
    power_mcse: FloatArray
    familywise_false_positive_probability: float | None
    familywise_false_positive_mcse: float | None
    single_cluster_probability: float
    single_cluster_mcse: float
    max_split_rhat_by_replication: FloatArray
    max_batch_mean_mcse_by_replication: FloatArray
    nonfinite_diagnostic_by_replication: NDArray[np.bool_]
    replications: int
    phi_low: float
    phi_high: float
    classification_precision: float | None
    classification_cutoff: float | None
    adaptive_weighting: str
    mean_precision: float
    precision_shape: float
    precision_rate: float
    efficacy_cutoff: float
    draws: int
    warmup: int
    chains: int


def simulate_bacis_oc(
    true_response_rates: ArrayLike,
    *,
    trials_per_group: int | ArrayLike = 25,
    replications: int = 100,
    phi_low: float = 0.1,
    phi_high: float = 0.3,
    classification_precision: float | None = None,
    classification_cutoff: float | None = 0.5,
    adaptive_weighting: str = "subgroup",
    mean_precision: float = 0.1,
    precision_shape: float = 50,
    precision_rate: float = 2,
    efficacy_cutoff: float = 0.92,
    draws: int = 2_000,
    warmup: int = 1_000,
    chains: int = 2,
    outcome_rng: int | np.random.Generator | None = None,
    sampler_rng: int | np.random.Generator | None = None,
    max_work: int = 20_000_000,
) -> BaCISOperatingCharacteristics:
    """Simulate independent subgroup outcomes and analyze each replication.

    ``true_response_rates`` is a fixed vector of subgroup response rates.
    Outcome generation and MCMC seed generation use separate random streams;
    explicit Generators must not share a BitGenerator; callers who create
    separate Generators are responsible for choosing distinct initial states.
    Work is serial and bounded by ``max_work`` (itself capped internally),
    measured as ``replications * groups * chains * (draws + warmup)``.
    """
    raw_truth = np.asarray(true_response_rates)
    if raw_truth.ndim != 1 or not 1 <= raw_truth.size <= _MAX_GROUPS:
        raise ValueError("true_response_rates must contain 1..100 subgroup rates")
    if np.iscomplexobj(raw_truth):
        raise ValueError("true_response_rates must be real-valued")
    truth = np.asarray(raw_truth, dtype=np.float64)
    if not np.isfinite(truth).all() or np.any((truth < 0) | (truth > 1)):
        raise ValueError("true_response_rates must be finite probabilities in [0,1]")

    groups = int(truth.size)
    if np.ndim(trials_per_group) == 0:
        raw_trial_scalar = np.asarray(trials_per_group)
        if np.iscomplexobj(raw_trial_scalar):
            raise ValueError("trials_per_group must be real-valued")
        trial_values = np.full(groups, scalar(float(raw_trial_scalar), "trials_per_group"))
    else:
        raw_trials = np.asarray(trials_per_group)
        if raw_trials.ndim != 1 or raw_trials.size != groups or np.iscomplexobj(raw_trials):
            raise ValueError("trials_per_group must be scalar or match the subgroup count")
        trial_values = np.asarray(raw_trials, dtype=np.float64)
    if (
        not np.isfinite(trial_values).all()
        or np.any(trial_values < 1)
        or np.any(trial_values > 10_000)
        or np.any(trial_values != np.floor(trial_values))
    ):
        raise ValueError("trials_per_group values must be integers in [1,10000]")
    trial_counts = trial_values.astype(np.int64)

    reps = _integer(replications, "replications", 1, _MAX_REPLICATIONS)
    low, high = _probability(phi_low, "phi_low"), _probability(phi_high, "phi_high")
    if low >= high:
        raise ValueError("phi_low must be less than phi_high")
    if classification_precision is None:
        separation = (
            np.log(high) - np.log1p(-high) - np.log(low) + np.log1p(-low)
        ) / 6
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            classifier_precision = 1.0 / separation**2
        if not np.isfinite(classifier_precision) or not 1e-6 <= classifier_precision <= 1e6:
            raise ValueError("derived classification_precision must lie in [1e-6,1e6]")
    else:
        classifier_precision = scalar(classification_precision, "classification_precision")
        if not 1e-6 <= classifier_precision <= 1e6:
            raise ValueError("classification_precision must lie in [1e-6,1e6]")
    if classification_cutoff is not None:
        classification_cutoff = _probability(classification_cutoff, "classification_cutoff")
    if adaptive_weighting not in ("subgroup", "patient"):
        raise ValueError("adaptive_weighting must be 'subgroup' or 'patient'")
    mean_prior_precision = scalar(mean_precision, "mean_precision")
    shape_prior = scalar(precision_shape, "precision_shape")
    rate_prior = scalar(precision_rate, "precision_rate")
    if min(mean_prior_precision, shape_prior, rate_prior) <= 0 or max(
        mean_prior_precision, shape_prior, rate_prior
    ) > 1e12:
        raise ValueError("hierarchical precision hyperparameters must lie in (0,1e12]")
    efficacy_threshold = _probability(efficacy_cutoff, "efficacy_cutoff")
    draw_count = _integer(draws, "draws", 8, 10_000)
    warmup_count = _integer(warmup, "warmup", 0, 10_000)
    chain_count = _integer(chains, "chains", 2, 8)
    per_fit_retained = chain_count * draw_count * groups
    per_fit_steps = chain_count * (draw_count + warmup_count) * groups
    if per_fit_retained > 500_000 or per_fit_steps > 1_000_000:
        raise ValueError("per-fit sampler dimensions exceed the bacis_fit work limits")
    sampler_steps = reps * per_fit_steps
    if sampler_steps > _MAX_TOTAL_SAMPLER_STEPS:
        raise ValueError("simulation sampler work exceeds the 20,000,000-step hard limit")
    work_limit = _integer(max_work, "max_work", 1, _MAX_TOTAL_SAMPLER_STEPS)
    if sampler_steps > work_limit:
        raise ValueError("simulation sampler work exceeds max_work")
    if reps * groups * 4 > _MAX_RESULT_CELLS:
        raise ValueError("simulation result arrays exceed the 2,000,000-cell limit")

    if (
        isinstance(outcome_rng, (int, np.integer))
        and not isinstance(outcome_rng, bool)
        and isinstance(sampler_rng, (int, np.integer))
        and not isinstance(sampler_rng, bool)
        and int(outcome_rng) == int(sampler_rng)
    ):
        raise ValueError("outcome_rng and sampler_rng integer seeds must differ")
    if isinstance(outcome_rng, np.random.Generator) and isinstance(
        sampler_rng, np.random.Generator
    ):
        if outcome_rng.bit_generator is sampler_rng.bit_generator:
            raise ValueError("outcome_rng and sampler_rng must use independent BitGenerators")

    outcome_generator = _generator(outcome_rng, "outcome_rng")
    sampler_generator = _generator(sampler_rng, "sampler_rng")
    successes = outcome_generator.binomial(
        trial_counts[None, :], truth[None, :], size=(reps, groups)
    ).astype(np.int64, copy=False)
    high_cluster = np.empty((reps, groups), dtype=bool)
    efficacious = np.empty((reps, groups), dtype=bool)
    max_rhat = np.empty(reps, dtype=float)
    max_mcse = np.empty(reps, dtype=float)
    diagnostic_nonfinite = np.empty(reps, dtype=bool)

    for replication in range(reps):
        seed = int(sampler_generator.integers(0, np.iinfo(np.int64).max))
        fit = bacis_fit(
            successes[replication],
            trial_counts,
            phi_low=low,
            phi_high=high,
            classification_precision=classification_precision,
            classification_cutoff=classification_cutoff,
            adaptive_weighting=adaptive_weighting,
            mean_precision=mean_prior_precision,
            precision_shape=shape_prior,
            precision_rate=rate_prior,
            efficacy_cutoff=efficacy_threshold,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            seed=seed,
        )
        high_cluster[replication] = fit.classification.cluster == 2
        efficacious[replication] = fit.efficacious
        finite_rhat = fit.summary.split_rhat[np.isfinite(fit.summary.split_rhat)]
        finite_mcse = fit.summary.batch_mean_mcse[np.isfinite(fit.summary.batch_mean_mcse)]
        max_rhat[replication] = float(np.max(finite_rhat)) if finite_rhat.size else np.nan
        max_mcse[replication] = float(np.max(finite_mcse)) if finite_mcse.size else np.nan
        diagnostic_nonfinite[replication] = (
            finite_rhat.size != fit.summary.split_rhat.size
            or finite_mcse.size != fit.summary.batch_mean_mcse.size
        )
        del fit

    null_groups = truth <= low
    high_rate = high_cluster.mean(axis=0)
    efficacy_rate = efficacious.mean(axis=0)
    high_mcse = np.sqrt(high_rate * (1 - high_rate) / reps)
    efficacy_mcse = np.sqrt(efficacy_rate * (1 - efficacy_rate) / reps)
    false_positive = np.full(groups, np.nan)
    false_positive_se = np.full(groups, np.nan)
    false_positive[null_groups] = efficacy_rate[null_groups]
    false_positive_se[null_groups] = efficacy_mcse[null_groups]
    power = np.full(groups, np.nan)
    power_se = np.full(groups, np.nan)
    alternative = ~null_groups
    power[alternative] = efficacy_rate[alternative]
    power_se[alternative] = efficacy_mcse[alternative]
    familywise: float | None
    familywise_se: float | None
    if np.any(null_groups):
        familywise_events = np.any(efficacious[:, null_groups], axis=1)
        familywise = float(familywise_events.mean())
        familywise_se = float(np.sqrt(familywise * (1 - familywise) / reps))
    else:
        familywise = familywise_se = None
    single_cluster_events = np.all(high_cluster == high_cluster[:, :1], axis=1)
    single_cluster = float(single_cluster_events.mean())
    single_cluster_se = float(np.sqrt(single_cluster * (1 - single_cluster) / reps))

    return BaCISOperatingCharacteristics(
        _readonly(truth),
        _readonly(trial_counts, dtype=np.int64),
        _readonly(successes, dtype=np.int64),
        _readonly(high_cluster, dtype=np.bool_),
        _readonly(efficacious, dtype=np.bool_),
        _readonly(null_groups, dtype=np.bool_),
        _readonly(high_rate),
        _readonly(high_mcse),
        _readonly(efficacy_rate),
        _readonly(efficacy_mcse),
        _readonly(false_positive),
        _readonly(false_positive_se),
        _readonly(power),
        _readonly(power_se),
        familywise,
        familywise_se,
        single_cluster,
        single_cluster_se,
        _readonly(max_rhat),
        _readonly(max_mcse),
        _readonly(diagnostic_nonfinite, dtype=np.bool_),
        reps,
        low,
        high,
        classification_precision,
        classification_cutoff,
        adaptive_weighting,
        mean_prior_precision,
        shape_prior,
        rate_prior,
        efficacy_threshold,
        draw_count,
        warmup_count,
        chain_count,
    )
