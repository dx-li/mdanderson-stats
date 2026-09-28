"""Bounded complete-outcome UAROET trial replay and serial OC simulation.

Patients' outcomes are available before the next scheduled analysis. This is a
Python scheduling convention; delayed ascertainment and native random-stream
parity are not modeled.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import finite
from .uaroet import _integer, _real_matrix, _scalar, uaroet_parameter_names
from .uaroet_decision import UAROETAllocation, uaroet_allocation
from .uaroet_fit import UAROETFit, fit_uaroet

_MAX_DOSES = 5
_MAX_LEVELS = 4
_MAX_PATIENTS = 10_000
_MAX_LOOKS = 200
_MAX_TRIALS = 10_000
_MAX_FIT_EVALUATIONS = 2_000_000
_MAX_TOTAL_EVALUATIONS = 20_000_000
_MAX_SUMMARY_CELLS = 200_000
type RNGInput = int | np.integer | np.random.Generator | None


def _owned_int(value: ArrayLike) -> NDArray[np.int64]:
    result = np.array(value, dtype=np.int64, copy=True)
    result.flags.writeable = False
    return result


def _owned_bool(value: ArrayLike) -> NDArray[np.bool_]:
    result = np.array(value, dtype=bool, copy=True)
    result.flags.writeable = False
    return result


def _streams(
    rng: RNGInput,
) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator, tuple[int, int, int]]:
    if isinstance(rng, np.random.Generator):
        entropy = rng.integers(0, 2**32, size=4, dtype=np.uint32)
        sequence = np.random.SeedSequence(entropy)
    else:
        if rng is not None and (
            isinstance(rng, (bool, np.bool_))
            or not isinstance(rng, (int, np.integer))
            or int(rng) < 0
        ):
            raise ValueError("rng must be a nonnegative integer, Generator or None")
        sequence = np.random.SeedSequence(None if rng is None else int(rng))
    children = sequence.spawn(3)
    seed_values = tuple(int(child.generate_state(1, dtype=np.uint64)[0]) for child in children)
    seeds = (seed_values[0], seed_values[1], seed_values[2])
    streams = tuple(np.random.default_rng(seed) for seed in seeds)
    return streams[0], streams[1], streams[2], seeds


def _truth(value: ArrayLike) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if (
        raw.ndim != 3
        or not 1 <= raw.shape[0] <= _MAX_DOSES
        or any(not 2 <= width <= _MAX_LEVELS for width in raw.shape[1:])
        or raw.size > _MAX_DOSES * _MAX_LEVELS**2
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError("truth must have shape (dose, efficacy, toxicity) on supported grids")
    result = finite(raw, "truth")
    if np.any((result < 0) | (result > 1)):
        raise ValueError("truth probabilities must lie in [0,1]")
    totals = result.sum(axis=(1, 2))
    if not np.all(np.isclose(totals, 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("each dose truth table must sum to one")
    return result / totals[:, None, None]


def _utility(value: ArrayLike, e: int, t: int) -> NDArray[np.float64]:
    result = _real_matrix(value, "utility")
    if (
        result.shape != (e, t)
        or np.any(result < 0)
        or np.any(np.diff(result, axis=0) < 0)
        or np.any(np.diff(result, axis=1) > 0)
    ):
        raise ValueError(
            "utility must be nonnegative, increasing in efficacy and decreasing in toxicity"
        )
    return result


def _looks(
    value: ArrayLike, tolerance_value: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    raw = np.asarray(value)
    if raw.ndim != 1 or not 1 <= raw.size <= _MAX_LOOKS:
        raise ValueError(f"look_sizes must be a vector of 1..{_MAX_LOOKS} entries")
    numeric = finite(raw, "look_sizes")
    if (
        np.any(numeric < 1)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric > _MAX_PATIENTS)
    ):
        raise ValueError(f"look_sizes must be integers in [1,{_MAX_PATIENTS}]")
    looks = numeric.astype(np.int64)
    if np.any(np.diff(looks) <= 0):
        raise ValueError("look_sizes must be strictly increasing")
    raw_tolerance = np.asarray(tolerance_value)
    if raw_tolerance.ndim != 1 or raw_tolerance.size != looks.size:
        raise ValueError("utility_tolerance must have one entry per analysis look")
    tolerance = finite(raw_tolerance, "utility_tolerance")
    if np.any(tolerance < 0) or np.any(np.diff(tolerance) > 0):
        raise ValueError("utility_tolerance must be nonnegative and nonincreasing")
    return looks, tolerance


@dataclass(frozen=True)
class UAROETTrialStep:
    analyzed_patients: int
    utility_tolerance: float
    action: str
    reason: str
    allocation_probabilities: NDArray[np.float64]
    mean_utility: NDArray[np.float64]
    toxicity_risk: NDArray[np.float64]
    probability_best: NDArray[np.float64]
    acceptable: NDArray[np.bool_]
    eligible: NDArray[np.bool_]
    selected_dose: int | None
    max_split_rhat: float
    max_mcse: float
    mean_association_acceptance: float
    work_evaluations: int


@dataclass(frozen=True)
class UAROETTrial:
    assigned_dose: NDArray[np.int64]
    efficacy_category: NDArray[np.int64]
    toxicity_category: NDArray[np.int64]
    assignment_probabilities: NDArray[np.float64]
    observed_counts: NDArray[np.int64]
    steps: tuple[UAROETTrialStep, ...]
    selected_dose: int | None
    stop_reason: str
    planned_patients: int
    work_evaluations: int
    rng_seeds: tuple[int, int, int]


@dataclass(frozen=True)
class UAROETSimulation:
    selected_dose: NDArray[np.int64]
    enrolled_patients: NDArray[np.int64]
    dose_counts: NDArray[np.int64]
    stop_reason: tuple[str, ...]
    trial_work_evaluations: NDArray[np.int64]
    max_split_rhat: NDArray[np.float64]
    max_mcse: NDArray[np.float64]
    selection_probability: NDArray[np.float64]
    selection_mcse: NDArray[np.float64]
    no_selection_probability: float
    no_selection_mcse: float
    early_stop_probability: float
    early_stop_mcse: float
    mean_enrollment: float
    mean_enrollment_mcse: float
    mean_dose_count: NDArray[np.float64]
    planned_allocation_probability: NDArray[np.float64]
    mean_efficacy_category: NDArray[np.float64]
    mean_toxicity_category: NDArray[np.float64]
    mean_observed_utility: NDArray[np.float64]
    total_work_evaluations: int
    rng_seeds: tuple[int, int, int]


@dataclass(frozen=True)
class _Config:
    truth: NDArray[np.float64]
    utility: NDArray[np.float64]
    looks: NDArray[np.int64]
    tolerance: NDArray[np.float64]
    prior_mean: NDArray[np.float64]
    prior_sd: NDArray[np.float64]
    monotone_efficacy: bool
    monotone_toxicity: bool
    association: float | None
    starting_dose: int
    toxicity_limit: float
    bad_toxicity_level: int
    p_L: float
    p_U: float
    good_utility_cutoff: float
    final_rule: str
    draws: int
    warmup: int
    chains: int
    max_fit_evaluations: int
    max_total_evaluations: int


def _config(
    truth: ArrayLike,
    utility: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    look_sizes: ArrayLike,
    utility_tolerance: ArrayLike,
    starting_dose: int,
    toxicity_limit: float,
    bad_toxicity_level: int,
    p_L: float,
    p_U: float,
    good_utility_cutoff: float,
    final_rule: str,
    monotone_efficacy: bool,
    monotone_toxicity: bool,
    association: float | None,
    draws: int,
    warmup: int,
    chains: int,
    max_fit_evaluations: int,
    max_total_evaluations: int,
) -> _Config:
    true = _truth(truth)
    d, e, t = true.shape
    util = _utility(utility, e, t)
    looks, tolerances = _looks(look_sizes, utility_tolerance)
    start = _integer(starting_dose, "starting_dose", 0, d - 1)
    limit, lower, upper, good_cutoff = (
        _scalar(toxicity_limit, "toxicity_limit"),
        _scalar(p_L, "p_L"),
        _scalar(p_U, "p_U"),
        _scalar(good_utility_cutoff, "good_utility_cutoff"),
    )
    bad_level = _integer(bad_toxicity_level, "bad_toxicity_level", 1, t - 1)
    if (
        not 0 <= limit <= 1
        or not 0 <= lower <= 1
        or not 0 <= upper <= 1
        or not np.isfinite(good_cutoff)
    ):
        raise ValueError("invalid toxicity/probability/utility cutoffs")
    if final_rule not in ("acceptable", "paper_global"):
        raise ValueError("final_rule must be acceptable or paper_global")
    if not isinstance(monotone_efficacy, (bool, np.bool_)) or not isinstance(
        monotone_toxicity, (bool, np.bool_)
    ):
        raise ValueError("monotonicity flags must be boolean")
    names = uaroet_parameter_names(
        d,
        e,
        t,
        monotone_efficacy=bool(monotone_efficacy),
        monotone_toxicity=bool(monotone_toxicity),
    )
    mean, sd = _real_matrix(prior_mean, "prior_mean"), _real_matrix(prior_sd, "prior_sd")
    if mean.ndim != 1 or sd.shape != mean.shape or mean.size != len(names) or np.any(sd <= 0):
        raise ValueError("prior_mean and positive prior_sd must match UAROET parameter names")
    if association is None:
        rho = None
    else:
        raw_rho = np.asarray(association)
        if raw_rho.ndim != 0 or raw_rho.dtype.kind not in "iuf":
            raise ValueError("association must be None or a real scalar")
        rho = float(raw_rho)
        if not np.isfinite(rho) or not -1 < rho < 1:
            raise ValueError("association must lie strictly inside (-1,1)")
    draw_count = _integer(draws, "draws", 8, 100_000)
    warmup_count = _integer(warmup, "warmup", 0, 100_000)
    chain_count = _integer(chains, "chains", 2, 8)
    fit_limit = _integer(max_fit_evaluations, "max_fit_evaluations", 1, _MAX_FIT_EVALUATIONS)
    total_limit = _integer(
        max_total_evaluations, "max_total_evaluations", 1, _MAX_TOTAL_EVALUATIONS
    )
    if chain_count * draw_count * true.size > 2_000_000:
        raise ValueError("retained joint posterior per fit exceeds 2 million cells")
    minimum_calls = chain_count * (1 + (1 if rho is not None else 2) * (warmup_count + draw_count))
    if minimum_calls > fit_limit:
        raise ValueError("max_fit_evaluations is below the minimum per-fit likelihood workload")
    if looks.size * minimum_calls > total_limit:
        raise ValueError("max_total_evaluations is below the planned minimum likelihood workload")
    return _Config(
        true,
        util,
        looks,
        tolerances,
        mean,
        sd,
        bool(monotone_efficacy),
        bool(monotone_toxicity),
        rho,
        start,
        limit,
        bad_level,
        lower,
        upper,
        good_cutoff,
        final_rule,
        draw_count,
        warmup_count,
        chain_count,
        fit_limit,
        total_limit,
    )


def _tape(value: ArrayLike | None, length: int, name: str) -> NDArray[np.float64] | None:
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != length or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector of length {length}")
    result = finite(raw, name)
    if np.any((result < 0) | (result >= 1)):
        raise ValueError(f"{name} values must lie in [0,1)")
    return result


def _diag(fit: UAROETFit) -> tuple[float, float, float]:
    rhat_parts = [np.asarray(fit.parameter_summary.split_rhat).ravel()]
    mcse_parts = [np.asarray(fit.parameter_summary.batch_mean_mcse).ravel()]
    if fit.fixed_association is None:
        rhat_parts.append(np.asarray(fit.association_summary.split_rhat).ravel())
        mcse_parts.append(np.asarray(fit.association_summary.batch_mean_mcse).ravel())
    rhat = np.concatenate(rhat_parts)
    mcse = np.concatenate(mcse_parts)
    return (
        float("nan") if np.isnan(rhat).any() else float(rhat.max()),
        float("nan") if np.isnan(mcse).any() else float(mcse.max()),
        float(np.mean(fit.association_acceptance)),
    )


def _categorical(u: float, cumulative: NDArray[np.float64]) -> int:
    return min(int(np.searchsorted(cumulative, u, side="right")), cumulative.size - 1)


def _run(
    config: _Config,
    *,
    outcome_rng: np.random.Generator,
    allocation_rng: np.random.Generator,
    posterior_rng: np.random.Generator,
    seeds: tuple[int, int, int],
    outcome_uniforms: ArrayLike | None = None,
    allocation_uniforms: ArrayLike | None = None,
) -> UAROETTrial:
    truth, utility = config.truth, config.utility
    d, e, t = truth.shape
    maximum = int(config.looks[-1])
    outcome_tape = _tape(outcome_uniforms, maximum, "outcome_uniforms")
    allocation_tape = _tape(allocation_uniforms, maximum, "allocation_uniforms")
    cumulative_truth = np.cumsum(truth.reshape(d, e * t), axis=1)
    assigned = np.empty(maximum, dtype=np.int64)
    efficacy = np.empty(maximum, dtype=np.int64)
    toxicity = np.empty(maximum, dtype=np.int64)
    assign_prob = np.zeros((maximum, d), dtype=float)
    counts = np.zeros((d, e, t), dtype=np.int64)
    probabilities = np.zeros(d)
    probabilities[config.starting_dose] = 1.0
    has_looked = False
    next_patient = 0
    steps: list[UAROETTrialStep] = []
    total_work = 0
    selected_dose: int | None = None
    stop_reason = "maximum_enrollment"

    for i, look_v in enumerate(config.looks):
        look = int(look_v)
        while next_patient < look:
            j = next_patient
            if has_looked:
                alloc_u = (
                    float(allocation_tape[j])
                    if allocation_tape is not None
                    else float(allocation_rng.random())
                )
                dose = _categorical(alloc_u, np.cumsum(probabilities))
            else:
                dose = config.starting_dose
            assigned[j] = dose
            assign_prob[j] = probabilities
            u = float(outcome_tape[j]) if outcome_tape is not None else float(outcome_rng.random())
            cell = _categorical(u, cumulative_truth[dose])
            ye, yt = divmod(cell, t)
            efficacy[j], toxicity[j] = ye, yt
            counts[dose, ye, yt] += 1
            next_patient += 1

        remaining = config.max_total_evaluations - total_work
        if remaining <= 0:
            raise RuntimeError(f"UAROET work budget exhausted before look {i + 1} (n={look})")
        try:
            fit = fit_uaroet(
                counts,
                prior_mean=config.prior_mean,
                prior_sd=config.prior_sd,
                monotone_efficacy=config.monotone_efficacy,
                monotone_toxicity=config.monotone_toxicity,
                association=config.association,
                draws=config.draws,
                warmup=config.warmup,
                chains=config.chains,
                rng=posterior_rng,
                max_evaluations=min(config.max_fit_evaluations, remaining),
            )
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            raise RuntimeError(
                f"UAROET posterior fit failed at look {i + 1} (n={look})"
            ) from exc
        work = int(fit.work_evaluations)
        if work > remaining:
            raise RuntimeError("UAROET fit exceeded remaining cumulative work budget")
        total_work += work
        final = look == maximum
        decision: UAROETAllocation = uaroet_allocation(
            fit,
            utility,
            counts.sum(axis=(1, 2)),
            toxicity_limit=config.toxicity_limit,
            bad_toxicity_level=config.bad_toxicity_level,
            p_L=config.p_L,
            p_U=config.p_U,
            utility_tolerance=float(config.tolerance[i]),
            good_utility_cutoff=config.good_utility_cutoff,
            starting_dose=config.starting_dose,
            final=final,
            final_rule=config.final_rule,
        )
        max_rhat, max_mcse, acceptance = _diag(fit)
        selected = decision.best_dose if final and decision.action == "select_obd" else None
        steps.append(
            UAROETTrialStep(
                look,
                float(config.tolerance[i]),
                decision.action,
                decision.reason,
                _freeze(decision.probabilities),
                _freeze(decision.mean_utility),
                _freeze(decision.toxicity_risk),
                _freeze(decision.probability_best),
                _owned_bool(decision.acceptable),
                _owned_bool(decision.eligible),
                selected,
                max_rhat,
                max_mcse,
                acceptance,
                work,
            )
        )
        del fit

        if final:
            selected_dose = selected
            stop_reason = "maximum_enrollment"
            break
        if decision.action == "stop":
            stop_reason = decision.reason
            assigned = assigned[:look]
            efficacy = efficacy[:look]
            toxicity = toxicity[:look]
            assign_prob = assign_prob[:look]
            break
        if decision.action != "randomize":
            raise RuntimeError(f"unexpected interim UAROET action {decision.action!r}")
        probabilities = np.asarray(decision.probabilities, dtype=float)
        has_looked = True

    return UAROETTrial(
        _owned_int(assigned),
        _owned_int(efficacy),
        _owned_int(toxicity),
        _freeze(assign_prob),
        _owned_int(counts),
        tuple(steps),
        selected_dose,
        stop_reason,
        maximum,
        total_work,
        seeds,
    )


def run_uaroet_trial(
    truth: ArrayLike,
    utility: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    look_sizes: ArrayLike,
    utility_tolerance: ArrayLike,
    starting_dose: int,
    rng: RNGInput = None,
    toxicity_limit: float = 0.3,
    bad_toxicity_level: int = 1,
    p_L: float = 0.1,
    p_U: float = 0.8,
    good_utility_cutoff: float = 0.5,
    final_rule: str = "acceptable",
    monotone_efficacy: bool = True,
    monotone_toxicity: bool = True,
    association: float | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    max_fit_evaluations: int = 200_000,
    max_total_evaluations: int = 2_000_000,
    outcome_uniforms: ArrayLike | None = None,
    allocation_uniforms: ArrayLike | None = None,
) -> UAROETTrial:
    """Replay one trial with complete outcomes observed before every analysis.

    Patients before the first look use the physician starting dose; each
    later between-look interval uses its preceding posterior allocation.
    Uniform tapes replace patient-level outcome/randomization draws for replay.
    """
    config = _config(
        truth,
        utility,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        look_sizes=look_sizes,
        utility_tolerance=utility_tolerance,
        starting_dose=starting_dose,
        toxicity_limit=toxicity_limit,
        bad_toxicity_level=bad_toxicity_level,
        p_L=p_L,
        p_U=p_U,
        good_utility_cutoff=good_utility_cutoff,
        final_rule=final_rule,
        monotone_efficacy=monotone_efficacy,
        monotone_toxicity=monotone_toxicity,
        association=association,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_fit_evaluations=max_fit_evaluations,
        max_total_evaluations=max_total_evaluations,
    )
    # Validate tapes before deriving streams from a caller-owned Generator.
    outcome_tape = _tape(outcome_uniforms, int(config.looks[-1]), "outcome_uniforms")
    allocation_tape = _tape(allocation_uniforms, int(config.looks[-1]), "allocation_uniforms")
    outcome_rng, allocation_rng, posterior_rng, seeds = _streams(rng)
    return _run(
        config,
        outcome_rng=outcome_rng,
        allocation_rng=allocation_rng,
        posterior_rng=posterior_rng,
        seeds=seeds,
        outcome_uniforms=outcome_tape,
        allocation_uniforms=allocation_tape,
    )


def _mcse(p: float, n: int) -> float:
    return sqrt(max(0.0, p * (1.0 - p)) / n)


def simulate_uaroet(
    truth: ArrayLike,
    utility: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    look_sizes: ArrayLike,
    utility_tolerance: ArrayLike,
    starting_dose: int,
    trials: int = 100,
    rng: RNGInput = None,
    toxicity_limit: float = 0.3,
    bad_toxicity_level: int = 1,
    p_L: float = 0.1,
    p_U: float = 0.8,
    good_utility_cutoff: float = 0.5,
    final_rule: str = "acceptable",
    monotone_efficacy: bool = True,
    monotone_toxicity: bool = True,
    association: float | None = None,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    max_fit_evaluations: int = 200_000,
    max_total_evaluations: int = _MAX_TOTAL_EVALUATIONS,
) -> UAROETSimulation:
    """Run serial trials; return compact OC summaries without posterior histories.

    Selection probabilities are unconditional over all trials, with
    no-selection separate. Planned allocation shares divide mean dose counts
    by maximum planned enrollment so stopped trials contribute zero afterward.
    """
    ntrial = _integer(trials, "trials", 2, _MAX_TRIALS)
    config = _config(
        truth,
        utility,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        look_sizes=look_sizes,
        utility_tolerance=utility_tolerance,
        starting_dose=starting_dose,
        toxicity_limit=toxicity_limit,
        bad_toxicity_level=bad_toxicity_level,
        p_L=p_L,
        p_U=p_U,
        good_utility_cutoff=good_utility_cutoff,
        final_rule=final_rule,
        monotone_efficacy=monotone_efficacy,
        monotone_toxicity=monotone_toxicity,
        association=association,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_fit_evaluations=max_fit_evaluations,
        max_total_evaluations=max_total_evaluations,
    )
    d, _, _ = config.truth.shape
    if ntrial * (d + len(config.looks) + 8) > _MAX_SUMMARY_CELLS:
        raise ValueError("compact UAROET simulation output exceeds the cell limit")
    minimum_calls = config.chains * (
        1 + (1 if config.association is not None else 2) * (config.warmup + config.draws)
    )
    if ntrial * len(config.looks) * minimum_calls > config.max_total_evaluations:
        raise ValueError("total evaluation budget is below the planned minimum work")

    outcome_rng, allocation_rng, posterior_rng, seeds = _streams(rng)
    selected = np.full(ntrial, -1, dtype=np.int64)
    enrolled = np.zeros(ntrial, dtype=np.int64)
    per_trial_dose = np.zeros((ntrial, d), dtype=np.int64)
    work = np.zeros(ntrial, dtype=np.int64)
    worst_rhat = np.full(ntrial, np.nan)
    worst_mcse = np.full(ntrial, np.nan)
    reasons: list[str] = []
    total_work = 0
    efficacy_sum = np.zeros(d)
    toxicity_sum = np.zeros(d)
    outcome_cell_counts = np.zeros(config.truth.shape, dtype=np.int64)
    patients_by_dose = np.zeros(d, dtype=np.int64)

    for trial_idx in range(ntrial):
        remaining = config.max_total_evaluations - total_work
        required = minimum_calls * len(config.looks)
        if remaining < required:
            raise RuntimeError(f"cumulative work budget exhausted before trial {trial_idx + 1}")
        trial_config = _Config(
            config.truth,
            config.utility,
            config.looks,
            config.tolerance,
            config.prior_mean,
            config.prior_sd,
            config.monotone_efficacy,
            config.monotone_toxicity,
            config.association,
            config.starting_dose,
            config.toxicity_limit,
            config.bad_toxicity_level,
            config.p_L,
            config.p_U,
            config.good_utility_cutoff,
            config.final_rule,
            config.draws,
            config.warmup,
            config.chains,
            config.max_fit_evaluations,
            remaining,
        )
        try:
            result = _run(
                trial_config,
                outcome_rng=outcome_rng,
                allocation_rng=allocation_rng,
                posterior_rng=posterior_rng,
                seeds=seeds,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"UAROET simulation trial {trial_idx + 1} failed") from exc
        selected[trial_idx] = -1 if result.selected_dose is None else result.selected_dose
        enrolled[trial_idx] = result.assigned_dose.size
        per_trial_dose[trial_idx] = result.observed_counts.sum(axis=(1, 2))
        work[trial_idx] = result.work_evaluations
        total_work += result.work_evaluations
        reasons.append(result.stop_reason)
        rhats = np.asarray([step.max_split_rhat for step in result.steps])
        mcses = np.asarray([step.max_mcse for step in result.steps])
        worst_rhat[trial_idx] = np.nan if np.isnan(rhats).any() else rhats.max()
        worst_mcse[trial_idx] = np.nan if np.isnan(mcses).any() else mcses.max()
        for dose in range(d):
            mask = result.assigned_dose == dose
            if np.any(mask):
                patients_by_dose[dose] += int(mask.sum())
                efficacy_sum[dose] += float(result.efficacy_category[mask].sum())
                toxicity_sum[dose] += float(result.toxicity_category[mask].sum())
        np.add(outcome_cell_counts, result.observed_counts, out=outcome_cell_counts)
        del result

    select_p = np.asarray([(selected == dose).mean() for dose in range(d)])
    no_select = float(np.mean(selected < 0))
    early = float(np.mean(enrolled < int(config.looks[-1])))
    mean_count = per_trial_dose.mean(axis=0)
    avg_efficacy = np.divide(
        efficacy_sum, patients_by_dose, out=np.full(d, np.nan), where=patients_by_dose > 0
    )
    avg_toxicity = np.divide(
        toxicity_sum, patients_by_dose, out=np.full(d, np.nan), where=patients_by_dose > 0
    )
    utility_scale = float(np.max(config.utility))
    if utility_scale > 0:
        scaled_utility_mean = np.divide(
            np.sum(outcome_cell_counts * (config.utility / utility_scale), axis=(1, 2)),
            patients_by_dose,
            out=np.full(d, np.nan),
            where=patients_by_dose > 0,
        )
        avg_utility = scaled_utility_mean * utility_scale
    else:
        avg_utility = np.where(patients_by_dose > 0, 0.0, np.nan)
    return UAROETSimulation(
        _owned_int(selected),
        _owned_int(enrolled),
        _owned_int(per_trial_dose),
        tuple(reasons),
        _owned_int(work),
        _freeze(worst_rhat),
        _freeze(worst_mcse),
        _freeze(select_p),
        _freeze(np.asarray([_mcse(float(v), ntrial) for v in select_p])),
        no_select,
        _mcse(no_select, ntrial),
        early,
        _mcse(early, ntrial),
        float(enrolled.mean()),
        float(enrolled.std(ddof=1) / sqrt(ntrial)),
        _freeze(mean_count),
        _freeze(mean_count / int(config.looks[-1])),
        _freeze(avg_efficacy),
        _freeze(avg_toxicity),
        _freeze(avg_utility),
        total_work,
        seeds,
    )
