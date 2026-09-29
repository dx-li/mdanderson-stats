"""Bounded Monte Carlo search for MERIT interim and final integer boundaries."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from ._validation import FloatArray, finite
from .boin import _owned
from .merit import MERITDesign, _integer, _isotonic
from .merit_interims import MERITInterims
from .merit_search import _corner_scenarios, _union_count_table
from .merit_simulation import _correlation, _draw

_MAX_PATIENTS = 200
_MAX_TRIALS = 100_000
_MAX_WORK = 500_000_000
_MAX_MEMORY_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MERITInterimSearch:
    """First estimated-feasible interim design and its corner summaries."""

    design: MERITDesign
    interims: MERITInterims
    global_type_one_error: float
    global_power_one: float
    global_power_two: float
    null_error: FloatArray
    null_error_mcse: FloatArray
    alternative_power_one: FloatArray
    alternative_power_one_mcse: FloatArray
    alternative_power_two: FloatArray
    alternative_power_two_mcse: FloatArray
    null_rates: FloatArray
    alternative_rates: FloatArray
    mean_enrolled_null: FloatArray
    mean_enrolled_null_mcse: FloatArray
    mean_enrolled_alternative: FloatArray
    mean_enrolled_alternative_mcse: FloatArray
    trials: int
    correlation: float
    power_definition: int
    max_work_estimate: int
    max_scratch_bytes_estimate: int


def _candidate_probabilities(
    toxicity: NDArray[np.int64],
    efficacy: NDArray[np.int64],
    active: NDArray[np.bool_],
    truth: NDArray[np.bool_] | None,
    n: int,
    *,
    isotonic_toxicity: bool,
    isotonic_efficacy: bool,
) -> tuple[FloatArray, FloatArray | None]:
    """Return any-selection probability and, under alternatives, both powers.

    Active-set patterns are pooled independently, matching the existing MERIT
    interim trial policy. Inclusion-exclusion then evaluates all final integer
    cutoffs without a trial-by-boundary tensor.
    """
    trials, doses = toxicity.shape
    patterns = (active.astype(np.uint8) * (1 << np.arange(doses, dtype=np.uint8))).sum(1)
    any_count = np.zeros((n + 1, n + 1), dtype=np.int64)
    power_one_count = np.zeros_like(any_count) if truth is not None else None
    power_two_count = np.zeros_like(any_count) if truth is not None else None
    for pattern in np.unique(patterns):
        rows = np.flatnonzero(patterns == pattern)
        dose_indices = np.flatnonzero(int(pattern) & (1 << np.arange(doses)))
        if rows.size == 0 or dose_indices.size == 0:
            continue
        t = toxicity[np.ix_(rows, dose_indices)].astype(float)
        e = efficacy[np.ix_(rows, dose_indices)].astype(float)
        if isotonic_toxicity:
            t = _isotonic(t)
        if isotonic_efficacy:
            e = _isotonic(e)
        all_rows = _union_count_table(t, e, n)
        any_count += all_rows
        if truth is None:
            continue
        good_columns = np.flatnonzero(truth[dose_indices])
        bad_columns = np.flatnonzero(~truth[dose_indices])
        if good_columns.size:
            good_rows = _union_count_table(t[:, good_columns], e[:, good_columns], n)
            assert power_two_count is not None
            power_two_count += good_rows
        if bad_columns.size:
            bad_rows = _union_count_table(t[:, bad_columns], e[:, bad_columns], n)
        else:
            bad_rows = np.zeros_like(all_rows)
        assert power_one_count is not None
        power_one_count += all_rows - bad_rows
    if truth is None:
        return any_count / trials, None
    assert power_one_count is not None and power_two_count is not None
    # Under an alternative, Power I is any true dose selected and no false dose
    # selected. Any-all minus any-bad is exactly this event.
    return power_one_count / trials, power_two_count / trials


def merit_interim_sample_size(
    *,
    interims: MERITInterims,
    toxicity_null: float,
    toxicity_alternative: float,
    efficacy_null: float,
    efficacy_alternative: float,
    doses: int = 2,
    alpha: float = 0.1,
    power: float = 0.8,
    power_definition: int = 2,
    max_patients_per_arm: int = 100,
    trials: int = 10000,
    correlation: float = 0.5,
    isotonic_toxicity: bool = True,
    isotonic_efficacy: bool = True,
    rng: int | np.random.Generator | None = None,
    max_work: int = _MAX_WORK,
) -> MERITInterimSearch:
    """Search first feasible maximum sample size and final integer boundaries.

    Interim targets must match the acceptable alternative rates. At each
    scheduled look, the existing raw-count integer boundaries permanently stop
    each arm on toxicity or futility. Stopped arms receive no later patients and
    are excluded from final isotonic pooling. Surviving arms may enroll to the
    candidate per-arm maximum; enrollment is not reallocated.

    The paper's ordered null/alternative corner configurations and Power I/II
    events are used. Final boundaries are searched over every integer pair
    ``0 <= toxicity_max, efficacy_min <= n``. Search stops at the first n with
    estimated error <= alpha and the selected estimated power >= target. Ties
    prefer greater chosen global power, lower global type-I error, smaller
    toxicity maximum, then greater efficacy minimum. These Monte Carlo search
    and stopping conventions are Python choices, not native parity claims.
    """
    if not isinstance(interims, MERITInterims):
        raise ValueError("interims must be a MERITInterims policy")
    d = _integer(doses, "doses", 2, 4)
    maximum = _integer(max_patients_per_arm, "max_patients_per_arm", 1, _MAX_PATIENTS)
    repetitions = _integer(trials, "trials", 100, _MAX_TRIALS)
    definition = _integer(power_definition, "power_definition", 1, 2)
    _integer(max_work, "max_work", 1, _MAX_WORK)
    MERITDesign(1, 0, 0, d, isotonic_toxicity, isotonic_efficacy)
    raw_rates = [
        toxicity_null,
        toxicity_alternative,
        efficacy_null,
        efficacy_alternative,
        alpha,
        power,
    ]
    if np.iscomplexobj(raw_rates):
        raise ValueError("rates and targets must be real")
    rates = finite(raw_rates, "rates and targets")
    if (
        rates.shape != (6,)
        or np.any((rates <= 0) | (rates >= 1))
        or not toxicity_alternative < toxicity_null
        or not efficacy_null < efficacy_alternative
    ):
        raise ValueError(
            "require interior rates/targets, toxicity_alternative < null "
            "and efficacy_null < alternative"
        )
    if not np.isclose(interims.toxicity_target, toxicity_alternative, rtol=0, atol=1e-12):
        raise ValueError("interim toxicity_target must equal toxicity_alternative")
    if not np.isclose(interims.efficacy_target, efficacy_alternative, rtol=0, atol=1e-12):
        raise ValueError("interim efficacy_target must equal efficacy_alternative")
    rho = _correlation(correlation)
    null_rates, alternative_rates, truths = _corner_scenarios(
        toxicity_null,
        toxicity_alternative,
        efficacy_null,
        efficacy_alternative,
        d,
    )
    n_null, n_alternative = null_rates.shape[0], alternative_rates.shape[0]
    scenario_rates = np.concatenate((null_rates, alternative_rates), axis=0)
    scenario_count = scenario_rates.shape[0]
    quantiles = ndtri(scenario_rates)
    tox_looks = np.asarray(interims.toxicity_looks, dtype=np.int64)
    eff_looks = np.asarray(interims.efficacy_looks, dtype=np.int64)
    largest_look = max(
        int(np.max(tox_looks, initial=0)),
        int(np.max(eff_looks, initial=0)),
    )
    first_n = max(1, largest_look + 1)
    if first_n > maximum:
        raise ValueError("maximum sample size must exceed every scheduled interim look")
    boundary_table = interims.boundaries(maximum)
    boundary_at = {int(n): i for i, n in enumerate(boundary_table.patients)}

    boundary_families = n_null + 3 * n_alternative
    candidate_count = maximum - first_n + 1
    grid_sum = sum((n + 1) ** 2 for n in range(first_n, maximum + 1))
    subset_count = 2**d - 1
    pattern_count = 2**d - 1
    histogram_work = repetitions * boundary_families * subset_count * d * candidate_count
    # Sum nonempty dose subsets across all nonempty survivor patterns.
    # Each union table also needs its cumulative grid passes.
    pattern_subset_count = 3**d - 2**d
    grid_work = boundary_families * (pattern_subset_count + 4 * pattern_count) * grid_sum
    state_work = 6 * maximum * scenario_count * repetitions * d
    boundary_work = histogram_work + grid_work
    work_estimate = int(boundary_work + state_work)
    if work_estimate > max_work:
        raise ValueError(f"work estimate {work_estimate} exceeds max_work={max_work}")

    state_cells = scenario_count * repetitions * d
    state_bytes = state_cells * (3 * np.dtype(np.int64).itemsize + np.dtype(np.bool_).itemsize)
    max_grid = (maximum + 1) ** 2
    summary_values = (n_null + 2 * n_alternative + 4) * max_grid
    summary_bytes = summary_values * np.dtype(float).itemsize
    evaluation_scratch_bytes = 128 * repetitions * d + 20 * max_grid * np.dtype(float).itemsize
    draw_scratch_bytes = 24 * repetitions * d
    memory_estimate = int(
        state_bytes + summary_bytes + evaluation_scratch_bytes + draw_scratch_bytes
    )
    if memory_estimate > _MAX_MEMORY_BYTES:
        raise ValueError(
            f"state and boundary summaries need {memory_estimate} bytes; "
            f"limit is {_MAX_MEMORY_BYTES}"
        )

    # All dimensions, corner grids, work and memory are checked before this
    # point. For a supplied Generator, none of the preflight above advances it.
    generator = np.random.default_rng(rng)
    toxicity = np.zeros((scenario_count, repetitions, d), dtype=np.int64)
    efficacy = np.zeros_like(toxicity)
    enrolled = np.zeros_like(toxicity)
    active = np.ones((scenario_count, repetitions, d), dtype=bool)
    truth_by_scenario: list[NDArray[np.bool_] | None] = [None] * n_null + [row for row in truths]

    for n in range(1, maximum + 1):
        z_toxicity, z_efficacy = _draw(generator, repetitions, d, rho)
        for scenario in range(scenario_count):
            current = active[scenario]
            enrolled[scenario] += current
            toxicity[scenario] += (z_toxicity <= quantiles[scenario, :, 0]) & current
            efficacy[scenario] += (z_efficacy <= quantiles[scenario, :, 1]) & current
            if n in boundary_at:
                look = boundary_at[n]
                stop_t = boundary_table.assess_toxicity[look] & (
                    toxicity[scenario] >= boundary_table.toxicity_stop_min[look]
                )
                stop_e = boundary_table.assess_efficacy[look] & (
                    efficacy[scenario] <= boundary_table.efficacy_stop_max[look]
                )
                active[scenario] &= ~(stop_t | stop_e)

        if n < first_n:
            continue
        boundaries = (n + 1) ** 2
        null_error = np.empty((n_null, boundaries), dtype=float)
        alt_one = np.empty((n_alternative, boundaries), dtype=float)
        alt_two = np.empty_like(alt_one)
        for scenario in range(scenario_count):
            truth = truth_by_scenario[scenario]
            first, second = _candidate_probabilities(
                toxicity[scenario],
                efficacy[scenario],
                active[scenario],
                truth,
                n,
                isotonic_toxicity=isotonic_toxicity,
                isotonic_efficacy=isotonic_efficacy,
            )
            if scenario < n_null:
                null_error[scenario] = first.ravel()
            else:
                assert second is not None
                alt_one[scenario - n_null] = first.ravel()
                alt_two[scenario - n_null] = second.ravel()

        worst_error = null_error.max(axis=0).reshape(n + 1, n + 1)
        worst_one = alt_one.min(axis=0).reshape(n + 1, n + 1)
        worst_two = alt_two.min(axis=0).reshape(n + 1, n + 1)
        chosen_power = worst_one if definition == 1 else worst_two
        candidate_t, candidate_e = np.nonzero((worst_error <= alpha) & (chosen_power >= power))
        if candidate_t.size == 0:
            continue
        ordering = np.lexsort(
            (
                -candidate_e,
                candidate_t,
                worst_error[candidate_t, candidate_e],
                -chosen_power[candidate_t, candidate_e],
            )
        )
        mt, me = int(candidate_t[ordering[0]]), int(candidate_e[ordering[0]])
        index = mt * (n + 1) + me
        null_rates_at_choice = null_error[:, index]
        one_rates_at_choice = alt_one[:, index]
        two_rates_at_choice = alt_two[:, index]
        enrolled_means = np.empty((scenario_count, d), dtype=float)
        enrolled_mcse = np.empty_like(enrolled_means)
        for scenario in range(scenario_count):
            scenario_enrolled = enrolled[scenario].astype(float)
            enrolled_means[scenario] = scenario_enrolled.mean(axis=0)
            enrolled_mcse[scenario] = scenario_enrolled.std(axis=0, ddof=1) / np.sqrt(repetitions)
        return MERITInterimSearch(
            design=MERITDesign(n, mt, me, d, isotonic_toxicity, isotonic_efficacy),
            interims=interims,
            global_type_one_error=float(worst_error[mt, me]),
            global_power_one=float(worst_one[mt, me]),
            global_power_two=float(worst_two[mt, me]),
            null_error=_owned(null_rates_at_choice),
            null_error_mcse=_owned(
                np.sqrt(null_rates_at_choice * (1.0 - null_rates_at_choice) / repetitions)
            ),
            alternative_power_one=_owned(one_rates_at_choice),
            alternative_power_one_mcse=_owned(
                np.sqrt(one_rates_at_choice * (1.0 - one_rates_at_choice) / repetitions)
            ),
            alternative_power_two=_owned(two_rates_at_choice),
            alternative_power_two_mcse=_owned(
                np.sqrt(two_rates_at_choice * (1.0 - two_rates_at_choice) / repetitions)
            ),
            null_rates=_owned(null_rates),
            alternative_rates=_owned(alternative_rates),
            mean_enrolled_null=_owned(enrolled_means[:n_null]),
            mean_enrolled_null_mcse=_owned(enrolled_mcse[:n_null]),
            mean_enrolled_alternative=_owned(enrolled_means[n_null:]),
            mean_enrolled_alternative_mcse=_owned(enrolled_mcse[n_null:]),
            trials=repetitions,
            correlation=rho,
            power_definition=definition,
            max_work_estimate=work_estimate,
            max_scratch_bytes_estimate=memory_estimate,
        )

    raise ValueError(
        f"no interim design meets the estimated constraints through {maximum} patients per arm"
    )
