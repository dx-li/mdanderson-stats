"""Serial Monte Carlo operating characteristics for categorical BOP2-DC."""

from dataclasses import dataclass
from math import fsum, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count, finite, scalar
from .bop2_dc_categorical import BOP2DCCategoricalDesign

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
_MAX_TRIALS = 5_000
_MAX_PATIENT_PATHS = 1_000_000
_MAX_COMPARISON_CACHE = 10_000
_MAX_COMPARISON_WORK = 1_500_000_000
_MAX_RESULT_CELLS = 2_000_000
_PAIR_WORK = 4 * 3 * 21 * 599
_TERMINAL_NAMES = ("graduate", "stop_no_go", "final_go", "final_consider", "final_no_go")
_LOOK_NAMES = ("continue", "stop_no_go", "graduate", "final_go", "final_consider", "final_no_go")


def _readonly(value: ArrayLike, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _seed(value: np.random.Generator | int | None) -> tuple[np.random.Generator, int | None]:
    if isinstance(value, np.random.Generator):
        return value, None
    if value is None:
        return np.random.default_rng(), None
    if not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError("rng must be a NumPy Generator, integer seed, or None")
    raw = count(value, "rng")
    if raw.ndim != 0 or raw >= 2**53:
        raise ValueError("rng must be a NumPy Generator, integer seed, or None")
    seed = int(raw)
    return np.random.default_rng(seed), seed


def _truth(value: ArrayLike, design: BOP2DCCategoricalDesign, name: str) -> FloatArray:
    expected = (2, design.n_categories) if design.randomized else (design.n_categories,)
    if isinstance(value, np.ndarray):
        if (
            value.shape != expected
            or value.size > 2 * design.n_categories
            or np.iscomplexobj(value)
        ):
            raise ValueError(f"{name} must have shape {expected}")
    elif isinstance(value, (list, tuple)):
        if design.randomized:
            if len(value) != 2 or any(
                not isinstance(row, (list, tuple, np.ndarray)) or len(row) != design.n_categories
                for row in value
            ):
                raise ValueError(f"{name} must have shape {expected}")
            if any(
                (
                    isinstance(row, np.ndarray)
                    and (row.shape != (design.n_categories,) or np.iscomplexobj(row))
                )
                or (
                    not isinstance(row, np.ndarray)
                    and any(not np.isscalar(item) or np.iscomplexobj(item) for item in row)
                )
                for row in value
            ):
                raise ValueError(f"{name} must contain real scalar probabilities")
        elif len(value) != design.n_categories or any(
            not np.isscalar(x) or np.iscomplexobj(x) for x in value
        ):
            raise ValueError(f"{name} must have shape {expected}")
    else:
        raise ValueError(f"{name} must have shape {expected}")
    result = finite(value, name)
    if result.shape != expected or np.any((result < 0) | (result > 1)):
        raise ValueError(f"{name} entries must be probabilities in [0,1]")
    rows = result.reshape((-1, design.n_categories))
    for row in rows:
        if abs(fsum(float(x) for x in row) - 1.0) > 1e-12:
            raise ValueError(f"each row of {name} must sum to one")
    return _readonly(rows / rows.sum(axis=1, keepdims=True) if design.randomized else rows[0])


@dataclass(frozen=True)
class BOP2DCCategoricalSimulation:
    """Compact unconditional operating characteristics and Monte Carlo errors."""

    truth_probabilities: FloatArray
    rng_seed: int | None
    trial_seeds: NDArray[np.uint64]
    terminal_names: tuple[str, ...]
    terminal_counts: IntArray
    terminal_probabilities: FloatArray
    terminal_mcse: FloatArray
    look_action_names: tuple[str, ...]
    look_reached_counts: IntArray
    look_action_counts: IntArray
    look_action_probabilities: FloatArray
    look_action_mcse: FloatArray
    terminal_sample_sizes: IntArray
    terminal_decisions: NDArray[np.str_]
    expected_sample_size: float
    expected_sample_size_mcse: float
    unique_comparisons: int


def _cache_entry_bound(design: BOP2DCCategoricalDesign, trials: int) -> int:
    if not design.randomized:
        return 0
    assert design.arm_assignments is not None
    total = 0
    for look in design.looks:
        n = int(look)
        nc = int(np.count_nonzero(design.arm_assignments[:n] == 0))
        nt = n - nc
        total += design.n_endpoints * min(trials, (nc + 1) * (nt + 1))
    return total


def simulate_bop2_dc_categorical(
    design: BOP2DCCategoricalDesign,
    truth_probabilities: ArrayLike,
    *,
    n_trials: int = 100,
    rng: np.random.Generator | int | None = None,
) -> BOP2DCCategoricalSimulation:
    """Simulate joint category outcomes serially under one declared truth.

    The truth is a K-cell vector for a single arm or a (2,K) control/treatment
    matrix for fixed-allocation randomized mode. Returned trial seeds replay
    each generated outcome tape with one categorical draw per assigned patient.
    """
    if not isinstance(design, BOP2DCCategoricalDesign):
        raise TypeError("design must be BOP2DCCategoricalDesign")
    trials_value = scalar(n_trials, "n_trials")
    trials = int(trials_value)
    if trials_value != trials or not 1 <= trials <= _MAX_TRIALS:
        raise ValueError(f"n_trials must be an integer in [1,{_MAX_TRIALS}]")
    truth = _truth(truth_probabilities, design, "truth_probabilities")
    paths = trials * design.max_subjects
    if paths > _MAX_PATIENT_PATHS:
        raise ValueError("categorical simulation exceeds its patient-path work bound")
    look_count = int(design.looks.size)
    if trials * (12 + look_count * 6) > _MAX_RESULT_CELLS:
        raise ValueError("categorical simulation summaries exceed the retained-cell bound")
    cache_bound = _cache_entry_bound(design, trials)
    tail_work = cache_bound * 2 * _PAIR_WORK
    if cache_bound > _MAX_COMPARISON_CACHE or tail_work > _MAX_COMPARISON_WORK:
        raise ValueError("categorical simulation exceeds its posterior comparison/cache budget")

    master, rng_seed = _seed(rng)
    trial_seeds: NDArray[np.uint64] = master.integers(0, 2**63, size=trials, dtype=np.uint64)
    terminal_decisions: NDArray[np.str_] = np.empty(trials, dtype="U16")
    terminal_sizes: IntArray = np.empty(trials, dtype=np.int64)
    terminal_counts: IntArray = np.zeros(len(_TERMINAL_NAMES), dtype=np.int64)
    look_reached: IntArray = np.zeros(look_count, dtype=np.int64)
    look_counts: IntArray = np.zeros((look_count, len(_LOOK_NAMES)), dtype=np.int64)
    comparison_cache: dict[tuple[float | int, ...], tuple[float, float, float, float]] = {}
    lookup = {name: i for i, name in enumerate(_LOOK_NAMES)}
    for trial_index, raw_seed in enumerate(trial_seeds):
        local = np.random.default_rng(int(raw_seed))
        if design.randomized:
            assert design.arm_assignments is not None
            outcome: IntArray = np.empty(design.max_subjects, dtype=np.int64)
            for arm in (0, 1):
                indices = np.flatnonzero(design.arm_assignments == arm)
                outcome[indices] = local.choice(
                    design.n_categories, size=indices.size, p=truth[arm]
                )
        else:
            outcome = local.choice(design.n_categories, size=design.max_subjects, p=truth)
        counts: IntArray = (
            np.zeros((2, design.n_categories), dtype=np.int64)
            if design.randomized
            else np.zeros(design.n_categories, dtype=np.int64)
        )
        look_index = 0
        terminal = "continue"
        for patient, category in enumerate(outcome, start=1):
            if design.randomized:
                assert design.arm_assignments is not None
                counts[int(design.arm_assignments[patient - 1]), int(category)] += 1
            else:
                counts[int(category)] += 1
            if patient != int(design.looks[look_index]):
                continue
            state = design._monitor(counts, comparison_cache)
            look_reached[look_index] += 1
            look_counts[look_index, lookup[state.decision]] += 1
            terminal = state.decision
            if terminal != "continue":
                break
            look_index += 1
        terminal_decisions[trial_index] = terminal
        terminal_sizes[trial_index] = patient
        terminal_counts[_TERMINAL_NAMES.index(terminal)] += 1
    terminal_probabilities = terminal_counts / trials
    terminal_mcse = np.sqrt(terminal_probabilities * (1 - terminal_probabilities) / trials)
    look_probabilities = look_counts / trials
    look_mcse = np.sqrt(look_probabilities * (1 - look_probabilities) / trials)
    mean_n = float(np.mean(terminal_sizes))
    mcse_n = float(np.std(terminal_sizes, ddof=1) / sqrt(trials)) if trials > 1 else float("nan")
    return BOP2DCCategoricalSimulation(
        truth,
        rng_seed,
        _readonly(trial_seeds, dtype=np.uint64),
        _TERMINAL_NAMES,
        _readonly(terminal_counts, dtype=np.int64),
        _readonly(terminal_probabilities),
        _readonly(terminal_mcse),
        _LOOK_NAMES,
        _readonly(look_reached, dtype=np.int64),
        _readonly(look_counts, dtype=np.int64),
        _readonly(look_probabilities),
        _readonly(look_mcse),
        _readonly(terminal_sizes, dtype=np.int64),
        _readonly(terminal_decisions, dtype=np.dtype("U16")),
        mean_n,
        mcse_n,
        len(comparison_cache),
    )
