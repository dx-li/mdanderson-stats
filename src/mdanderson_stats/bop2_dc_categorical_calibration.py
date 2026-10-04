"""Finite candidate comparison for categorical BOP2-DC using Monte Carlo OCs."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import scalar
from .bop2_dc_categorical import BOP2DCCategoricalDesign
from .bop2_dc_categorical_simulation import (
    BOP2DCCategoricalSimulation,
    _seed,
    _truth,
    simulate_bop2_dc_categorical,
)

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
_MAX_CANDIDATES = 100
_MAX_AGGREGATE_PATHS = 2_000_000
_MAX_AGGREGATE_WORK = 5_000_000_000
_INFEASIBLE_TOL = 64 * np.finfo(np.float64).eps


class BOP2DCCategoricalInfeasibleError(ValueError):
    """No supplied finite candidate meets the requested estimated OC limits."""


def _freeze(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BOP2DCCategoricalCalibration:
    """Monte Carlo OC comparison of finite compatible candidate designs."""

    selected_design: BOP2DCCategoricalDesign
    selected_index: int
    objective: str
    scenario_names: tuple[str, str]
    terminal_names: tuple[str, ...]
    terminal_probabilities: FloatArray
    terminal_mcse: FloatArray
    trial_seeds: NDArray[np.uint64]
    look_action_names: tuple[str, ...]
    look_action_probabilities: FloatArray
    look_action_mcse: FloatArray
    look_reached_counts: IntArray
    false_go_rate: FloatArray
    false_go_mcse: FloatArray
    false_no_go_rate: FloatArray
    false_no_go_mcse: FloatArray
    correct_go_rate: FloatArray
    correct_go_mcse: FloatArray
    false_consider_rate: FloatArray
    false_consider_mcse: FloatArray
    false_consider_rate_by_scenario: FloatArray
    false_consider_mcse_by_scenario: FloatArray
    futile_expected_sample_size: FloatArray
    futile_expected_sample_size_mcse: FloatArray
    feasible: NDArray[np.bool_]
    false_go_limit: float
    false_no_go_limit: float
    false_consider_limit: float | None
    n_trials: int
    rng_seed: int | None
    candidate_count: int
    limitations: tuple[str, ...]


def _limits(value: float, name: str) -> float:
    if not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a probability scalar")
    result = scalar(float(np.asarray(value, dtype=np.float64)), name)
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must lie in [0,1]")
    return result


def _same_candidate_family(candidates: tuple[BOP2DCCategoricalDesign, ...]) -> None:
    reference = candidates[0]
    for candidate in candidates[1:]:
        same_assignments = (
            candidate.arm_assignments is None and reference.arm_assignments is None
        ) or (
            candidate.arm_assignments is not None
            and reference.arm_assignments is not None
            and np.array_equal(candidate.arm_assignments, reference.arm_assignments)
        )
        fixed = (
            candidate.max_subjects == reference.max_subjects
            and np.array_equal(candidate.indicators, reference.indicators)
            and candidate.combination == reference.combination
            and candidate.directions == reference.directions
            and np.array_equal(candidate.lrv, reference.lrv)
            and np.array_equal(candidate.cmv, reference.cmv)
            and candidate.prior == reference.prior
            and candidate.control_prior == reference.control_prior
            and same_assignments
            and np.array_equal(candidate.looks, reference.looks)
            and candidate.graduate_at_interim == reference.graduate_at_interim
            and candidate.comparison_tolerance == reference.comparison_tolerance
        )
        if not fixed:
            raise ValueError("candidates must differ only in lambda and gamma cutoff settings")


def _summary_rate(
    simulation: BOP2DCCategoricalSimulation,
    indices: tuple[str, ...],
) -> tuple[float, float]:
    successes = sum(
        int(simulation.terminal_counts[simulation.terminal_names.index(name)]) for name in indices
    )
    trials = len(simulation.trial_seeds)
    probability = successes / trials
    # The combined event is Bernoulli at the trial level, including early stop.
    return probability, sqrt(probability * (1 - probability) / trials)


def calibrate_bop2_dc_categorical(
    candidates: Sequence[BOP2DCCategoricalDesign],
    futile_truth: ArrayLike,
    effective_truth: ArrayLike,
    *,
    n_trials: int = 1_000,
    false_go_limit: float = 0.1,
    false_no_go_limit: float = 0.1,
    false_consider_limit: float | None = None,
    objective: str = "cgr",
    rng: np.random.Generator | int | None = None,
) -> BOP2DCCategoricalCalibration:
    """Compare caller-supplied candidates against explicit categorical truths.

    Feasibility is assessed from Monte Carlo point estimates and is not a
    guarantee of operating-characteristic control. Objectives follow paper
    §2.3: maximize correct-go probability (`cgr`) or minimize expected sample
    size under the declared futile truth (`futile_ess`), subject to the supplied
    false-go/no-go limits and optional false-consider limit.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("candidates must be a finite sequence of designs")
    if not 1 <= len(candidates) <= _MAX_CANDIDATES:
        raise ValueError(f"candidates must contain 1..{_MAX_CANDIDATES} designs")
    designs = tuple(candidates)
    if any(not isinstance(item, BOP2DCCategoricalDesign) for item in designs):
        raise TypeError("each candidate must be a BOP2DCCategoricalDesign")
    _same_candidate_family(designs)
    n_raw = scalar(n_trials, "n_trials")
    n = int(n_raw)
    if n_raw != n or not 2 <= n <= 5_000:
        raise ValueError("n_trials must be an integer in [2,5000]")
    if objective not in ("cgr", "futile_ess"):
        raise ValueError("objective must be 'cgr' or 'futile_ess'")
    fg_limit = _limits(false_go_limit, "false_go_limit")
    fn_limit = _limits(false_no_go_limit, "false_no_go_limit")
    fc_limit = (
        None
        if false_consider_limit is None
        else _limits(false_consider_limit, "false_consider_limit")
    )
    _truth(futile_truth, designs[0], "futile_truth")
    _truth(effective_truth, designs[0], "effective_truth")
    aggregate_paths = len(designs) * 2 * n * designs[0].max_subjects
    if aggregate_paths > _MAX_AGGREGATE_PATHS:
        raise ValueError("candidate comparison exceeds its aggregate patient-path budget")
    # Conservative compare-beta workload, no dependence on observed paths.
    from .bop2_dc_categorical_simulation import _PAIR_WORK, _cache_entry_bound

    aggregate_work = sum(_cache_entry_bound(design, n) * 2 * _PAIR_WORK for design in designs) * 2
    if aggregate_work > _MAX_AGGREGATE_WORK:
        raise ValueError("candidate comparison exceeds its aggregate posterior-work budget")
    retained_cells = (
        len(designs) * 2 * n
        + len(designs) * 2 * designs[0].looks.size * 6
        + len(designs)
        * 2
        * len(
            terminal_names := (
                "graduate",
                "stop_no_go",
                "final_go",
                "final_consider",
                "final_no_go",
            )
        )
    )
    if retained_cells > 2_000_000:
        raise ValueError("candidate comparison exceeds its retained-output cell bound")
    master, rng_seed = _seed(rng)
    terminal_names = ("graduate", "stop_no_go", "final_go", "final_consider", "final_no_go")
    simulations: list[tuple[BOP2DCCategoricalSimulation, BOP2DCCategoricalSimulation]] = []
    for design in designs:
        futile = simulate_bop2_dc_categorical(design, futile_truth, n_trials=n, rng=master)
        effective = simulate_bop2_dc_categorical(design, effective_truth, n_trials=n, rng=master)
        simulations.append((futile, effective))
    shape = (len(designs), 2)
    terminal_p = np.empty((*shape, len(terminal_names)))
    terminal_e = np.empty_like(terminal_p)
    trial_seeds: NDArray[np.uint64] = np.empty((len(designs), 2, n), dtype=np.uint64)
    look_p = np.empty((len(designs), 2, designs[0].looks.size, 6))
    look_e = np.empty_like(look_p)
    reached = np.empty((len(designs), 2, designs[0].looks.size), dtype=np.int64)
    false_go = np.empty(len(designs))
    false_go_e = np.empty(len(designs))
    false_no_go = np.empty(len(designs))
    false_no_go_e = np.empty(len(designs))
    correct_go = np.empty(len(designs))
    correct_go_e = np.empty(len(designs))
    false_consider = np.empty(len(designs))
    false_consider_e = np.empty(len(designs))
    false_consider_by_scenario = np.empty((len(designs), 2))
    false_consider_e_by_scenario = np.empty((len(designs), 2))
    futile_n = np.empty(len(designs))
    futile_n_e = np.empty(len(designs))
    for i, (futile, effective) in enumerate(simulations):
        for scenario, simulation in enumerate((futile, effective)):
            terminal_p[i, scenario] = simulation.terminal_probabilities
            terminal_e[i, scenario] = simulation.terminal_mcse
            trial_seeds[i, scenario] = simulation.trial_seeds
            look_p[i, scenario] = simulation.look_action_probabilities
            look_e[i, scenario] = simulation.look_action_mcse
            reached[i, scenario] = simulation.look_reached_counts
        false_go[i], false_go_e[i] = _summary_rate(futile, ("graduate", "final_go"))
        false_no_go[i], false_no_go_e[i] = _summary_rate(effective, ("stop_no_go", "final_no_go"))
        correct_go[i], correct_go_e[i] = _summary_rate(effective, ("graduate", "final_go"))
        futile_consider = float(
            futile.terminal_probabilities[terminal_names.index("final_consider")]
        )
        effective_consider = float(
            effective.terminal_probabilities[terminal_names.index("final_consider")]
        )
        false_consider_by_scenario[i] = (futile_consider, effective_consider)
        false_consider_e_by_scenario[i] = (
            float(futile.terminal_mcse[terminal_names.index("final_consider")]),
            float(effective.terminal_mcse[terminal_names.index("final_consider")]),
        )
        false_consider[i] = max(futile_consider, effective_consider)
        # This is the larger component Monte Carlo error, reported only as a
        # scale for the maximum-of-two estimate; the joint maximum has no single
        # binomial standard error.
        false_consider_e[i] = max(false_consider_e_by_scenario[i])
        futile_n[i], futile_n_e[i] = futile.expected_sample_size, futile.expected_sample_size_mcse
    feasible = (false_go <= fg_limit + _INFEASIBLE_TOL) & (
        false_no_go <= fn_limit + _INFEASIBLE_TOL
    )
    if fc_limit is not None:
        feasible &= false_consider <= fc_limit + _INFEASIBLE_TOL
    if not np.any(feasible):
        raise BOP2DCCategoricalInfeasibleError("no finite candidate meets the estimated OC limits")
    eligible = np.flatnonzero(feasible)
    if objective == "cgr":
        selected = min(eligible, key=lambda i: (-correct_go[i], futile_n[i], i))
    else:
        selected = min(eligible, key=lambda i: (futile_n[i], -correct_go[i], i))
    return BOP2DCCategoricalCalibration(
        designs[selected],
        int(selected),
        objective,
        ("futile", "effective"),
        terminal_names,
        _freeze(terminal_p),
        _freeze(terminal_e),
        _freeze(trial_seeds, dtype=np.uint64),
        simulations[0][0].look_action_names,
        _freeze(look_p),
        _freeze(look_e),
        _freeze(reached, dtype=np.int64),
        _freeze(false_go),
        _freeze(false_go_e),
        _freeze(false_no_go),
        _freeze(false_no_go_e),
        _freeze(correct_go),
        _freeze(correct_go_e),
        _freeze(false_consider),
        _freeze(false_consider_e),
        _freeze(false_consider_by_scenario),
        _freeze(false_consider_e_by_scenario),
        _freeze(futile_n),
        _freeze(futile_n_e),
        _freeze(feasible, dtype=np.bool_),
        fg_limit,
        fn_limit,
        fc_limit,
        n,
        rng_seed,
        len(designs),
        (
            (
                "Candidate feasibility uses Monte Carlo point estimates, not "
                "confidence-bound guarantees."
            ),
            (
                "Using the same simulated trials to select and report a candidate "
                "creates selection optimism; rerun the selected candidate with an "
                "independent seed for an unbiased confirmation estimate."
            ),
            (
                "Truth vectors are caller-declared joint category distributions; "
                "their clinical futile/effective interpretation is not inferred "
                "or validated."
            ),
        ),
    )
