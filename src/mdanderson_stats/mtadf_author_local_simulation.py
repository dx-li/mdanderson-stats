"""Replay and simulate the rendered author-local MTADF conduct rule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cdflib import _freeze
from ._validation import FloatArray
from .mtadf import _scalar
from .mtadf_author import MTADFAuthorDecision, _mtadf_author_admissible_dose_count
from .mtadf_author_local import _mtadf_author_local_decision_with_count, mtadf_author_local_decision
from .mtadf_logistic import MTADFLocalLogisticPosterior, _mcmc_options, _preflight_work
from .mtadf_logistic_simulation import (
    MTADFLogisticSimulation,
    _freeze_uint,
    _record_diagnostic,
)

_MAX_DOSES = 20
_MAX_COHORTS = 1_000
_MAX_PATIENTS = 1_000
_MAX_TRIALS = 10_000
_MAX_TOTAL_WORK = 100_000_000
_MAX_TOTAL_POTENTIAL_CELLS = 2_000_000
_MAX_TOTAL_STORAGE_BYTES = 512_000_000
_MAX_LIVE_DRAW_CELLS = 2_000_000


def _freeze_int(value: ArrayLike) -> NDArray[np.int64]:
    array = np.ascontiguousarray(value, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, np.ndarray)):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    if isinstance(value, np.ndarray) and value.ndim != 0:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    result = int(raw)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return result


def _seed(value: object, name: str) -> int:
    return _integer(value, name, 0, int(np.iinfo(np.uint64).max))


def _sampler_options(draws: int, warmup: int, chains: int) -> tuple[int, int, int]:
    draw_count = _integer(draws, "draws", 8, 100_000)
    warmup_count = _integer(warmup, "warmup", 0, 100_000)
    chain_count = _integer(chains, "chains", 2, 16)
    return _mcmc_options(draw_count, warmup_count, chain_count)


def _probabilities(value: ArrayLike, name: str) -> FloatArray:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= _MAX_DOSES or any(not np.isscalar(item) for item in value):
            raise ValueError(f"{name} must be a scalar vector with 2..{_MAX_DOSES} entries")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded array, list, or tuple")
    if raw.ndim != 1 or not 2 <= raw.size <= _MAX_DOSES or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real vector with 2..{_MAX_DOSES} entries")
    result: FloatArray = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} must contain probabilities in [0,1]")
    return result


def _count_matrix(value: ArrayLike, name: str, cohort_size: int) -> NDArray[np.int64]:
    if isinstance(value, np.ndarray):
        raw = value
    elif isinstance(value, (list, tuple)):
        if not 2 <= len(value) <= _MAX_COHORTS:
            raise ValueError(f"{name} must have 2..{_MAX_COHORTS} cohort rows")
        for row in value:
            if isinstance(row, np.ndarray):
                valid_row = row.ndim == 1 and 2 <= row.size <= _MAX_DOSES
                if valid_row and any(not np.isscalar(cell) for cell in row):
                    raise ValueError(f"{name} must contain scalar count cells")
            elif isinstance(row, (list, tuple)):
                valid_row = 2 <= len(row) <= _MAX_DOSES
                if valid_row and any(not np.isscalar(cell) for cell in row):
                    raise ValueError(f"{name} must contain scalar count cells")
            else:
                valid_row = False
            if not valid_row:
                raise ValueError(f"{name} rows must contain 2..{_MAX_DOSES} scalar entries")
        raw = np.asarray(value)
    else:
        raise ValueError(f"{name} must be a bounded array, list, or tuple")
    if (
        raw.ndim != 2
        or not 2 <= raw.shape[0] <= _MAX_COHORTS
        or not 2 <= raw.shape[1] <= _MAX_DOSES
        or raw.dtype.kind not in "iuf"
    ):
        raise ValueError(f"{name} must be a cohorts-by-doses real matrix within supported limits")
    numeric: FloatArray = np.asarray(raw, dtype=float)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 0)
        or np.any(numeric != np.floor(numeric))
        or np.any(numeric > cohort_size)
        or np.any(numeric >= 2**53)
    ):
        raise ValueError(f"{name} entries must be integer cohort counts in [0, cohort_size]")
    return _freeze_int(numeric)


def _options(
    cohorts: int,
    cohort_size: int,
    doses: int,
    draws: int,
    warmup: int,
    chains: int,
    max_total_work: int,
    max_total_potential_cells: int,
) -> tuple[int, int, int, int, int, int, int]:
    cohort_count = _integer(cohorts, "cohorts", 2, _MAX_COHORTS)
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    if cohort_count * size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    if not 2 <= doses <= _MAX_DOSES:
        raise ValueError(f"dose count must be in [2, {_MAX_DOSES}]")
    draw_count, warmup_count, chain_count = _sampler_options(draws, warmup, chains)
    _preflight_work(draw_count, warmup_count, chain_count, 2, 2)
    worst_case_work = 2 * (cohort_count - 1) * chain_count * (draw_count + warmup_count) * 2
    if worst_case_work > _MAX_TOTAL_WORK:
        raise ValueError("worst-case local posterior work exceeds the single-trial work bound")
    work_limit = _integer(max_total_work, "max_total_work", 1, _MAX_TOTAL_WORK)
    potential_limit = _integer(
        max_total_potential_cells,
        "max_total_potential_cells",
        1,
        _MAX_TOTAL_POTENTIAL_CELLS,
    )
    # At most two distinct adjacent-pair fits can be retained for a review.
    live_fit_cells = 2 * chain_count * draw_count * (2 + 4 + 3 * 2)
    if live_fit_cells > _MAX_LIVE_DRAW_CELLS:
        raise ValueError("two local posterior fits exceed the 2 million cell peak-storage bound")
    return cohort_count, size, draw_count, warmup_count, chain_count, work_limit, potential_limit


@dataclass(frozen=True)
class MTADFAuthorLocalTrialResult:
    """Replay ledger and compact MCMC diagnostics for one author-local trial."""

    subjects: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    responses: NDArray[np.int64]
    assigned_dose: NDArray[np.int64]
    admissible_count_before: NDArray[np.int64]
    admissible_count_after: NDArray[np.int64]
    probability_forward_positive: FloatArray
    probability_backward_nonpositive: FloatArray
    fits_by_cohort: NDArray[np.int64]
    selected_dose: int
    final_decision: MTADFAuthorDecision
    sampler_seed: int
    posterior_fit_count: int
    mean_acceptance_rate: float
    maximum_split_rhat: float
    mean_positive_slope_mcse: float
    maximum_positive_slope_rhat: float
    stop_reason: str


def replay_mtadf_author_local_trial(
    cohort_toxicities: ArrayLike,
    cohort_responses: ArrayLike,
    *,
    sampler_seed: int,
    cohort_size: int = 3,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    ce1: float = 0.3,
    ce2: float = 0.4,
    draws: int = 2_000,
    warmup: int = 1_000,
    chains: int = 4,
) -> MTADFAuthorLocalTrialResult:
    """Replay potential cohort counts under the author's local logistic rule.

    Outcome matrices have shape ``(cohorts, doses)``. At each cohort only the
    assigned-dose counts are observed. The first cohort goes to dose zero; the
    author's subsequent cap-one branch forces dose zero and skips MCMC. Other
    moves use the cap from before that cohort, then refresh it.
    """
    seed_value = _seed(sampler_seed, "sampler_seed")
    size = _integer(cohort_size, "cohort_size", 1, _MAX_PATIENTS)
    tox_tape = _count_matrix(cohort_toxicities, "cohort_toxicities", size)
    response_tape = _count_matrix(cohort_responses, "cohort_responses", size)
    if tox_tape.shape != response_tape.shape:
        raise ValueError("cohort outcome matrices must have matching shapes")
    cohort_count, dose_count = tox_tape.shape
    if cohort_count * size > _MAX_PATIENTS:
        raise ValueError(f"planned patients per trial must not exceed {_MAX_PATIENTS}")
    draw_count, warmup_count, chain_count = _sampler_options(draws, warmup, chains)
    _preflight_work(draw_count, warmup_count, chain_count, 2, 2)
    trial_work = 2 * (cohort_count - 1) * chain_count * (draw_count + warmup_count) * 2
    if trial_work > _MAX_TOTAL_WORK:
        raise ValueError("worst-case local posterior work exceeds the single-trial work bound")
    live_fit_cells = 2 * chain_count * draw_count * (2 + 4 + 6)
    if live_fit_cells > _MAX_LIVE_DRAW_CELLS:
        raise ValueError("two local posterior fits exceed the 2 million cell peak-storage bound")
    phi, cutoff = _scalar(toxicity_limit, "toxicity_limit"), _scalar(safety_cutoff, "safety_cutoff")
    ce1_value, ce2_value = _scalar(ce1, "ce1"), _scalar(ce2, "ce2")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    if not 0 <= ce1_value <= ce2_value <= 1:
        raise ValueError("ce1 and ce2 must satisfy 0 <= ce1 <= ce2 <= 1")
    sampler_rng = np.random.default_rng(seed_value)

    n = np.zeros(dose_count, dtype=np.int64)
    y = np.zeros(dose_count, dtype=np.int64)
    r = np.zeros(dose_count, dtype=np.int64)
    assigned = np.empty(cohort_count, dtype=np.int64)
    cap_before = np.empty(cohort_count, dtype=np.int64)
    cap_after = np.empty(cohort_count, dtype=np.int64)
    forward_positive = np.full(cohort_count, np.nan, dtype=float)
    backward_nonpositive = np.full(cohort_count, np.nan, dtype=float)
    fits_by_cohort = np.zeros(cohort_count, dtype=np.int64)
    acceptance_total = 0.0
    acceptance_count = 0
    max_rhat = float("nan")
    slope_mcse_total = 0.0
    slope_mcse_count = 0
    slope_rhat_max = float("nan")

    def record_fit(posterior: MTADFLocalLogisticPosterior) -> None:
        nonlocal acceptance_total, acceptance_count, max_rhat
        nonlocal slope_mcse_total, slope_mcse_count, slope_rhat_max
        acceptance, rhat, slope_mcse, slope_rhat = _record_diagnostic(posterior)
        acceptance_total += acceptance
        acceptance_count += 1
        if not np.isnan(rhat):
            max_rhat = rhat if np.isnan(max_rhat) else max(max_rhat, rhat)
        if np.isfinite(slope_mcse):
            slope_mcse_total += slope_mcse
            slope_mcse_count += 1
        if not np.isnan(slope_rhat):
            slope_rhat_max = (
                slope_rhat if np.isnan(slope_rhat_max) else max(slope_rhat_max, slope_rhat)
            )

    current = 0
    cap = _mtadf_author_admissible_dose_count(n, y, toxicity_limit=phi, safety_cutoff=cutoff)
    assigned[0] = 0
    cap_before[0] = cap
    n[0] += size
    y[0] += tox_tape[0, 0]
    r[0] += response_tape[0, 0]
    cap = _mtadf_author_admissible_dose_count(n, y, toxicity_limit=phi, safety_cutoff=cutoff)
    cap_after[0] = cap
    current = 1

    for cohort in range(1, cohort_count):
        cap_before[cohort] = cap
        if cap == 1:
            current = 0
            assigned[cohort] = current
            n[current] += size
            y[current] += tox_tape[cohort, current]
            r[current] += response_tape[cohort, current]
            cap = _mtadf_author_admissible_dose_count(
                n, y, toxicity_limit=phi, safety_cutoff=cutoff
            )
            cap_after[cohort] = cap
            continue

        assigned[cohort] = current
        old_cap = cap
        n[current] += size
        y[current] += tox_tape[cohort, current]
        r[current] += response_tape[cohort, current]
        decision = _mtadf_author_local_decision_with_count(
            n,
            y,
            r,
            current_dose=current,
            toxicity_limit=phi,
            safety_cutoff=cutoff,
            ce1=ce1_value,
            ce2=ce2_value,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
            rng=sampler_rng,
            admissibility_count=old_cap,
        )
        assert decision.dose is not None
        current = decision.dose
        cap = decision.admissible_dose_count
        cap_after[cohort] = cap
        fits_by_cohort[cohort] = decision.fit_count
        if decision.probability_forward_positive is not None:
            forward_positive[cohort] = decision.probability_forward_positive
        if decision.probability_backward_nonpositive is not None:
            backward_nonpositive[cohort] = decision.probability_backward_nonpositive
        for posterior in (decision.forward_posterior, decision.backward_posterior):
            if posterior is not None:
                record_fit(posterior)
        del posterior
        del decision

    final_local = mtadf_author_local_decision(
        n,
        y,
        r,
        current_dose=None,
        final=True,
        toxicity_limit=phi,
        safety_cutoff=cutoff,
        ce1=ce1_value,
        ce2=ce2_value,
        draws=draw_count,
        warmup=warmup_count,
        chains=chain_count,
        rng=sampler_rng,
    )
    assert final_local.final_selection is not None
    final = final_local.final_selection
    assert final.dose is not None
    return MTADFAuthorLocalTrialResult(
        _freeze_int(n),
        _freeze_int(y),
        _freeze_int(r),
        _freeze_int(assigned),
        _freeze_int(cap_before),
        _freeze_int(cap_after),
        _freeze(forward_positive),
        _freeze(backward_nonpositive),
        _freeze_int(fits_by_cohort),
        final.dose,
        final,
        seed_value,
        int(fits_by_cohort.sum()),
        acceptance_total / acceptance_count if acceptance_count else float("nan"),
        max_rhat,
        slope_mcse_total / slope_mcse_count if slope_mcse_count else float("nan"),
        slope_rhat_max,
        "maximum_enrollment",
    )


def _seed_sequence(
    value: int | np.integer | np.random.SeedSequence | None,
) -> np.random.SeedSequence:
    if isinstance(value, np.random.SeedSequence):
        return value
    if value is None or (
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    ):
        return np.random.SeedSequence(None if value is None else int(value))
    raise ValueError("rng must be an integer seed, SeedSequence, or None")


def simulate_mtadf_author_local(
    true_toxicity: ArrayLike,
    true_efficacy: ArrayLike,
    *,
    cohorts: int = 10,
    cohort_size: int = 3,
    trials: int = 10,
    toxicity_limit: float = 0.3,
    safety_cutoff: float = 0.8,
    ce1: float = 0.3,
    ce2: float = 0.4,
    draws: int = 2_000,
    warmup: int = 1_000,
    chains: int = 4,
    rng: int | np.integer | np.random.SeedSequence | None = None,
    max_total_work: int = _MAX_TOTAL_WORK,
    max_total_potential_cells: int = _MAX_TOTAL_POTENTIAL_CELLS,
    max_total_storage_bytes: int = _MAX_TOTAL_STORAGE_BYTES,
) -> MTADFLogisticSimulation:
    """Run serial author-local trials with independent replay seed pairs."""
    tox = _probabilities(true_toxicity, "true_toxicity")
    eff = _probabilities(true_efficacy, "true_efficacy")
    if tox.shape != eff.shape:
        raise ValueError("true_toxicity and true_efficacy must have matching dose counts")
    (
        cohort_count,
        size,
        draw_count,
        warmup_count,
        chain_count,
        work_limit,
        potential_limit,
    ) = _options(
        cohorts,
        cohort_size,
        tox.size,
        draws,
        warmup,
        chains,
        max_total_work,
        max_total_potential_cells,
    )
    trial_count = _integer(trials, "trials", 1, _MAX_TRIALS)
    phi, cutoff = _scalar(toxicity_limit, "toxicity_limit"), _scalar(safety_cutoff, "safety_cutoff")
    ce1_value, ce2_value = _scalar(ce1, "ce1"), _scalar(ce2, "ce2")
    if not 0 < phi < 1 or not 0 < cutoff < 1:
        raise ValueError("toxicity_limit and safety_cutoff must lie in (0,1)")
    if not 0 <= ce1_value <= ce2_value <= 1:
        raise ValueError("ce1 and ce2 must satisfy 0 <= ce1 <= ce2 <= 1")
    fits_per_trial = 2 * (cohort_count - 1)
    per_fit_work = chain_count * (draw_count + warmup_count) * 2
    total_work = trial_count * fits_per_trial * per_fit_work
    potential_cells = trial_count * cohort_count * tox.size * 2
    if total_work > work_limit:
        raise ValueError("worst-case local posterior work exceeds max_total_work")
    if potential_cells > potential_limit:
        raise ValueError("potential outcome tables exceed max_total_potential_cells")
    # Summaries retain compact counts/seeds; at most two posterior objects are
    # live within a single review and are released before the next trial.
    bytes_per_trial = 3 * tox.size * 8 + 2 * 8 + 512
    fit_cells = 2 * chain_count * draw_count * (2 + 4 + 6)
    posterior_peak_bytes = fit_cells * np.dtype(np.float64).itemsize
    storage = trial_count * bytes_per_trial + posterior_peak_bytes
    storage += 2 * cohort_count * tox.size * np.dtype(np.int64).itemsize
    storage_limit = _integer(
        max_total_storage_bytes,
        "max_total_storage_bytes",
        1,
        _MAX_TOTAL_STORAGE_BYTES,
    )
    if storage > storage_limit:
        raise ValueError("summary and peak posterior storage exceed max_total_storage_bytes")

    root_seed = _seed_sequence(rng)
    outcome_seed_sequence, sampler_seed_sequence = root_seed.spawn(2)
    outcome_seeds = outcome_seed_sequence.generate_state(trial_count, dtype=np.uint64)
    sampler_seeds = sampler_seed_sequence.generate_state(trial_count, dtype=np.uint64)
    trial_seeds = np.column_stack((outcome_seeds, sampler_seeds))
    subjects = np.zeros((trial_count, tox.size), dtype=np.int64)
    toxicities = np.zeros_like(subjects)
    responses = np.zeros_like(subjects)
    selected: NDArray[np.int64] = np.full(trial_count, -1, dtype=np.int64)
    reasons: list[str] = []
    fit_counts: NDArray[np.int64] = np.zeros(trial_count, dtype=np.int64)
    acceptance: FloatArray = np.full(trial_count, np.nan, dtype=float)
    max_rhat: FloatArray = np.full(trial_count, np.nan, dtype=float)
    slope_mcse: FloatArray = np.full(trial_count, np.nan, dtype=float)
    slope_rhat: FloatArray = np.full(trial_count, np.nan, dtype=float)
    for index in range(trial_count):
        outcome_rng = np.random.default_rng(int(outcome_seeds[index]))
        potential_tox = np.column_stack(
            [outcome_rng.binomial(size, p, size=cohort_count) for p in tox]
        )
        potential_eff = np.column_stack(
            [outcome_rng.binomial(size, p, size=cohort_count) for p in eff]
        )
        result = replay_mtadf_author_local_trial(
            potential_tox,
            potential_eff,
            cohort_size=size,
            sampler_seed=int(sampler_seeds[index]),
            toxicity_limit=phi,
            safety_cutoff=cutoff,
            ce1=ce1_value,
            ce2=ce2_value,
            draws=draw_count,
            warmup=warmup_count,
            chains=chain_count,
        )
        subjects[index] = result.subjects
        toxicities[index] = result.toxicities
        responses[index] = result.responses
        selected[index] = result.selected_dose
        reasons.append(result.stop_reason)
        fit_counts[index] = result.posterior_fit_count
        acceptance[index] = result.mean_acceptance_rate
        max_rhat[index] = result.maximum_split_rhat
        slope_mcse[index] = result.mean_positive_slope_mcse
        slope_rhat[index] = result.maximum_positive_slope_rhat
        del result
        del potential_tox, potential_eff

    selected_frequency: FloatArray = np.bincount(
        selected[selected >= 0], minlength=tox.size
    ).astype(float)
    selected_frequency /= trial_count
    no_selection = float(np.count_nonzero(selected < 0) / trial_count)
    planned = cohort_count * size
    early = float(np.count_nonzero(subjects.sum(axis=1) < planned) / trial_count)
    return MTADFLogisticSimulation(
        _freeze_int(subjects),
        _freeze_int(toxicities),
        _freeze_int(responses),
        _freeze_int(selected),
        tuple(reasons),
        _freeze_uint(trial_seeds),
        _freeze_int(fit_counts),
        _freeze(acceptance),
        _freeze(max_rhat),
        _freeze(slope_mcse),
        _freeze(slope_rhat),
        _freeze(selected_frequency),
        _freeze(np.sqrt(selected_frequency * (1 - selected_frequency) / trial_count)),
        no_selection,
        float(np.sqrt(no_selection * (1 - no_selection) / trial_count)),
        early,
        float(np.sqrt(early * (1 - early) / trial_count)),
        _freeze(np.mean(subjects, axis=0)),
        _freeze(np.mean(toxicities, axis=0)),
        _freeze(np.mean(responses, axis=0)),
    )
